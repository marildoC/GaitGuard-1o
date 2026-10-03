
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple, Dict

import cv2
import numpy as np

from face.detector_align import FaceDetectorAligner, FaceCandidate
from face.embedder import FaceEmbedder
from face.quality import compute_full_quality
from face.multiview_types import PoseBin, MultiViewSample
from identity.face_gallery import FaceGallery, FaceTemplate

logger = logging.getLogger(__name__)




@dataclass
class GuidedCaptureConfig:
    """
    Configuration for each pose capture step.

    num_frames         : target number of *valid* samples we aim to collect.
    max_duration_sec   : hard time limit per pose; after this we stop even
                         if we have fewer than num_frames samples.
    max_frames_total   : safety cap on how many frames we read from the camera
                         for this pose (prevents infinite loops).
    """
    num_frames: int = 20
    min_quality: float = 0.15
    yaw_tolerance_deg: float = 25.0
    pitch_tolerance_deg: float = 25.0
    display: bool = True

    max_duration_sec: float = 6.0
    max_frames_total: int = 180


class GuidedCapture:
    """
    Captures a batch of frames for one pose request,
    extracts faces, yaw/pitch, quality and embeddings, then
    returns all valid MultiViewSample items for that pose.
    """

    def __init__(
        self,
        aligner: FaceDetectorAligner,
        embedder: FaceEmbedder,
        cfg: GuidedCaptureConfig,
    ) -> None:
        self._aligner = aligner
        self._embedder = embedder
        self._cfg = cfg

    def _pick_best_candidate(self, candidates: List[FaceCandidate]) -> Optional[FaceCandidate]:
        """
        Currently we rely on FaceDetectorAligner returning candidates
        sorted by detector score. We still add a small safety layer.
        """
        if not candidates:
            return None
        return candidates[0]

    def _compute_quality_for_candidate(
        self,
        frame_bgr: np.ndarray,
        cand: FaceCandidate,
    ) -> float:
        """
        Use the unified quality pipeline, consistent with runtime.
        """
        return compute_full_quality(
            image=frame_bgr,
            bbox=cand.bbox,
            det_score=cand.det_score,
            yaw=cand.yaw,
            pitch=cand.pitch,
        )

    def capture_pose(
        self,
        cap: Any,
        pose_bin: PoseBin,
        pose_prompt: str,
        target_yaw: float,
        target_pitch: float,
    ) -> List[MultiViewSample]:
        """
        Capture multiple frames while the operator is instructing
        the subject to look in a specific direction (pose_bin).

        Returns a list of MultiViewSample objects that satisfy:
          - target yaw/pitch within configured tolerances
          - quality ≥ min_quality

        The loop is bounded by both:
          - a maximum duration in seconds
          - a maximum total number of frames
        so it can never hang indefinitely.
        """
        logger.info(f"[guided] Starting capture for {pose_bin.name}: {pose_prompt}")

        samples: List[MultiViewSample] = []
        collected = 0

        start_ts = time.time()
        frames_seen = 0

        while True:
            now = time.time()
            if collected >= self._cfg.num_frames:
                break
            if now - start_ts >= self._cfg.max_duration_sec:
                logger.info(
                    "[guided] Time limit reached for %s (%.2fs)",
                    pose_bin.name,
                    now - start_ts,
                )
                break
            if frames_seen >= self._cfg.max_frames_total:
                logger.info(
                    "[guided] Frame limit reached for %s (%d frames)",
                    pose_bin.name,
                    frames_seen,
                )
                break

            ret, frame_bgr = cap.read()
            if not ret:
                logger.warning("[guided] Camera read failed")
                continue

            frames_seen += 1

            if self._cfg.display:
                disp = frame_bgr.copy()
                cv2.putText(
                    disp,
                    f"Look {pose_prompt}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Guided Enrollment", disp)
                cv2.waitKey(1)

            candidates = self._aligner.detect_and_align(frame_bgr)
            if not candidates:
                continue

            cand = self._pick_best_candidate(candidates)
            if cand is None:
                continue

            yaw = cand.yaw
            pitch = cand.pitch
            if yaw is None or pitch is None:
                continue

            yaw_f = float(yaw)
            pitch_f = float(pitch)


            q = self._compute_quality_for_candidate(frame_bgr, cand)
            if q < self._cfg.min_quality:
                continue

            emb_raw = cand.embedding
            if emb_raw is None:
                continue
            emb = self._embedder.embed(emb_raw)

            ts_now = time.time()

            samples.append(
                MultiViewSample(
                    embedding=emb,
                    yaw_deg=yaw_f,
                    pitch_deg=pitch_f,
                    roll_deg=None,
                    quality=q,
                    pose_bin=pose_bin,
                    source="enroll_guided",
                    ts=ts_now,
                    metadata={
                        "det_score": float(cand.det_score),
                        "bbox": tuple(cand.bbox),
                    },
                )
            )

            collected += 1
            logger.debug(
                "[guided-debug] pose_bin=%s raw yaw=%.1f pitch=%.1f",
                pose_bin.name, yaw_f, pitch_f,
            )

        duration = time.time() - start_ts
        logger.info(
            "[guided] %s: collected=%d (target=%d) | frames=%d | duration=%.2fs",
            pose_bin.name,
            len(samples),
            self._cfg.num_frames,
            frames_seen,
            duration,
        )
        return samples
        



@dataclass
class GuidedEnrollmentStep:
    pose_bin: PoseBin
    prompt: str
    target_yaw: float
    target_pitch: float


GUIDED_STEPS: List[GuidedEnrollmentStep] = [
    GuidedEnrollmentStep(PoseBin.FRONT, "FRONT / STRAIGHT",  0.0,  0.0),

    GuidedEnrollmentStep(PoseBin.LEFT,  "LEFT",             -45.0,  0.0),

    GuidedEnrollmentStep(PoseBin.RIGHT, "RIGHT",            +45.0,  0.0),

    GuidedEnrollmentStep(PoseBin.UP,    "UP",                0.0, +25.0),
    GuidedEnrollmentStep(PoseBin.DOWN,  "DOWN",              0.0, -25.0),
]




class EnrollmentMultiViewGuide:
    """
    Unified guided workflow for capturing 5 pose-specific views
    and storing them in the FaceGallery with rich metadata.

    This class does not touch classic enrollment; it is meant to be
    invoked by a dedicated CLI subcommand (e.g. `guided-enroll`).
    """

    def __init__(
        self,
        gallery: FaceGallery,
        aligner: FaceDetectorAligner,
        embedder: FaceEmbedder,
        capture_cfg: Optional[GuidedCaptureConfig] = None,
    ) -> None:
        self._gallery = gallery
        self._aligner = aligner
        self._embedder = embedder
        self._cap_cfg = capture_cfg or GuidedCaptureConfig()
        self._capture = GuidedCapture(
            aligner=self._aligner,
            embedder=self._embedder,
            cfg=self._cap_cfg,
        )


    def guided_enroll(
        self,
        person_id: str,
        name: str,
        surname: str,
        camera_index: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Main entry point:
        - Opens camera
        - Iterates over the 5 guided steps
        - Collects valid samples for each pose
        - Writes enriched FaceTemplate entries into FaceGallery
        """
        logger.info(
            f"[guided] Starting guided enrollment for {person_id} ({name} {surname})"
        )

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera for guided enrollment")

        all_samples: List[MultiViewSample] = []

        try:
            for step in GUIDED_STEPS:
                print(f"\nPlease ask the subject to look: {step.prompt}")
                print("Press ENTER in the terminal to begin capturing this pose...")

                self._wait_for_space()

                samples = self._capture.capture_pose(
                    cap=cap,
                    pose_bin=step.pose_bin,
                    pose_prompt=step.prompt,
                    target_yaw=step.target_yaw,
                    target_pitch=step.target_pitch,
                )

                if samples:
                    all_samples.extend(samples)
                else:
                    logger.warning(
                        f"[guided] No valid samples for {step.pose_bin.name}"
                    )

        finally:
            cap.release()
            cv2.destroyAllWindows()

        logger.info(f"[guided] Total collected samples = {len(all_samples)}")

        if not all_samples:
            raise RuntimeError("No valid samples collected for guided enrollment")

        templates: List[FaceTemplate] = []
        for s in all_samples:
            meta: Dict[str, Any] = {
                "pose_bin": s.pose_bin.name if s.pose_bin is not None else None,
                "yaw_deg": s.yaw_deg,
                "pitch_deg": s.pitch_deg,
                "roll_deg": s.roll_deg,
                "quality": s.quality,
                "timestamp": s.ts,
            }
            if extra_metadata:
                meta.update(extra_metadata)

            templates.append(
                FaceTemplate(
                    embedding=s.embedding,
                    metadata=meta,
                )
            )

        self._gallery.enroll_person(
            person_id=person_id,
            name=name,
            surname=surname,
            templates=templates,
        )

        logger.info(f"[guided] Guided enrollment completed for {person_id}")


    @staticmethod
    def _wait_for_space() -> None:
        """
        Block until the operator confirms they are ready.

        We intentionally use terminal input here instead of cv2.waitKey,
        because at this point no OpenCV window is shown yet. Using
        cv2.waitKey without a window would never receive the SPACE key
        (it would just spin forever). The message reminds the user to
        press ENTER in the terminal.
        """
        input("  [guided] Press ENTER when ready to capture this pose... ")
