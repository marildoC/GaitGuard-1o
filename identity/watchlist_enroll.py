"""
identity/watchlist_enroll.py

Logic for enrolling WATCHLIST persons from external images.

Design goals
------------
- NO user interaction here (no input(), no prints()). Only pure logic + logging.
- Use the SAME face backend as live enrollment:
    * FaceConfig
    * FaceDetectorAligner
    * FaceEmbedder
    * compute_full_quality
    * FaceGallery / FaceTemplate
- Do NOT touch the runtime core (identity engine, main_loop).
  We only:
    * prepare a watchlist person draft,
    * turn images into FaceTemplate proposals,
    * commit them into the FaceGallery.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from face.config import FaceConfig, default_face_config
from face.detector_align import FaceDetectorAligner
from face.embedder import FaceEmbedder
from face.quality import compute_full_quality

from .face_gallery import FaceGallery, FaceTemplate, Category

logger = logging.getLogger(__name__)


QUALITY_HIGH_THRESHOLD = 0.75
QUALITY_MEDIUM_THRESHOLD = 0.40

MIN_FACE_AREA_HARD_REJECT = 20 * 20
MIN_FACE_AREA_WARN = 40 * 40



@dataclass
class WatchlistPersonDraft:
    """
    In-memory representation of a watchlist person BEFORE committing to the gallery.
    """
    person_id: str
    name: str
    surname: Optional[str] = None
    country: Optional[str] = None
    notes: Optional[str] = None
    category: Category = "watchlist"
    source: str = "IMAGE_ONLY"  # LIVE_ENROLLMENT / IMAGE_ONLY / HYBRID
    created_at: float = field(default_factory=lambda: time.time())

    def full_name(self) -> str:
        if self.surname:
            return f"{self.name} {self.surname}"
        return self.name


@dataclass
class FaceTemplateProposal:
    """
    Proposed face template derived from one external image (not yet persisted).
    """
    person_id: str
    image_path: Path

    bbox_xyxy: Tuple[int, int, int, int]
    face_area: int
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    pose_bin: str

    quality: float
    quality_level: str

    embedding: np.ndarray
    det_score: float
    timestamp: float = field(default_factory=lambda: time.time())

    warnings: List[str] = field(default_factory=list)


@dataclass
class DetectedFaceDebug:
    """
    Summary of a detected face in an image for CLI selection in multi-face cases.
    """
    index: int
    bbox_xyxy: Tuple[int, int, int, int]
    face_area: int
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    det_score: float


@dataclass
class MultiFaceImageResult:
    """
    Returned when an image contains multiple faces and no face_index was specified.
    """
    image_path: Path
    faces: List[DetectedFaceDebug]


@dataclass
class NoFaceResult:
    """
    Hard rejection: the image contains no usable face.
    """
    image_path: Path
    reason: str = "no_face_detected"


@dataclass
class ImageReadError:
    """
    Hard rejection: the image could not be read from disk.
    """
    image_path: Path
    reason: str = "image_read_error"


ProcessImageOutcome = Union[
    FaceTemplateProposal,
    MultiFaceImageResult,
    NoFaceResult,
    ImageReadError,
]



def _allocate_watchlist_person_id(gallery: FaceGallery) -> str:
    """
    Allocate a new watchlist person_id with prefix 'p_w_'.
    """
    max_idx = 0
    for pid in gallery.persons.keys():
        if isinstance(pid, str) and pid.startswith("p_w_"):
            try:
                n = int(pid.split("_")[-1])
                max_idx = max(max_idx, n)
            except Exception:
                continue
    return f"p_w_{max_idx + 1:04d}"


def _classify_quality(score: float) -> str:
    if score >= QUALITY_HIGH_THRESHOLD:
        return "HIGH"
    if score >= QUALITY_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _guess_pose_bin(yaw_deg: float, pitch_deg: float) -> str:
    """
    Simple pose binning: FRONT / LEFT / RIGHT / UP / DOWN / UNKNOWN.
    """
    try:
        y = float(yaw_deg)
        p = float(pitch_deg)
    except Exception:
        return "UNKNOWN"

    if abs(y) <= 15 and -15 <= p <= 15:
        return "FRONT"
    if y > 15 and abs(y) <= 60:
        return "RIGHT"
    if y < -15 and abs(y) <= 60:
        return "LEFT"
    if p >= 20:
        return "UP"
    if p <= -20:
        return "DOWN"
    return "UNKNOWN"


def _face_area_from_bbox(bbox_xyxy: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox_xyxy
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h


def _extract_bbox_xyxy(det: Any, img_shape: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int, int, int]:
    """
    Robustly extract (x1, y1, x2, y2) from FaceCandidate-like object.

    Supports:
      - det.bbox      (actual field in your FaceCandidate),
      - det.bbox_xyxy (future/alt naming).
    """
    raw_bbox = getattr(det, "bbox", None)
    if raw_bbox is None:
        raw_bbox = getattr(det, "bbox_xyxy", None)

    if raw_bbox is None:
        if img_shape is not None:
            h, w = img_shape[:2]
            logger.warning(
                "FaceCandidate has no bbox/bbox_xyxy; falling back to full image bbox."
            )
            return 0, 0, int(w - 1), int(h - 1)
        raise AttributeError("FaceCandidate has neither 'bbox' nor 'bbox_xyxy'.")

    if len(raw_bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {raw_bbox!r}")

    x1, y1, x2, y2 = raw_bbox
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))
    return x1, y1, x2, y2


def _extract_pose_and_score(det: Any) -> Tuple[float, float, float, float]:
    """
    Extract yaw, pitch, roll, score from FaceCandidate-like object.

    Supports:
      - yaw_deg / pitch_deg / roll_deg  OR yaw / pitch / roll
      - score OR det_score
    """
    yaw = getattr(det, "yaw_deg", None)
    if yaw is None:
        yaw = getattr(det, "yaw", 0.0)

    pitch = getattr(det, "pitch_deg", None)
    if pitch is None:
        pitch = getattr(det, "pitch", 0.0)

    roll = getattr(det, "roll_deg", None)
    if roll is None:
        roll = getattr(det, "roll", 0.0)

    score = getattr(det, "score", None)
    if score is None:
        score = getattr(det, "det_score", 0.0)

    return float(yaw), float(pitch), float(roll), float(score)


def _extract_aligned_and_embedding(
    det: Any,
    embedder: FaceEmbedder,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Extract aligned crop (optional) and embedding for the candidate.

    For THIS project:
      - FaceDetectorAligner already computes .embedding (InsightFace).
      - FaceEmbedder is in "passthrough" mode (expects 512-D vectors).

    So the correct path is:
      embedding = embedder.embed(det.embedding)

    If det.embedding is missing, we raise, because that indicates a
    misconfiguration for this watchlist path.
    """
    aligned_bgr = getattr(det, "aligned_bgr", None)
    raw_emb = getattr(det, "embedding", None)

    if raw_emb is None:
        raise RuntimeError(
            "FaceCandidate has no 'embedding'; configure FaceDetectorAligner "
            "to produce embeddings for watchlist_enroll."
        )

    emb_arr = np.asarray(raw_emb, dtype=np.float32).reshape(-1)
    embedding = embedder.embed(emb_arr)
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return aligned_bgr, embedding




def create_watchlist_person_draft(
    gallery: FaceGallery,
    *,
    name: str,
    surname: Optional[str] = None,
    country: Optional[str] = None,
    notes: Optional[str] = None,
) -> WatchlistPersonDraft:
    """
    Create an in-memory WatchlistPersonDraft with a new unique person_id.
    """
    pid = _allocate_watchlist_person_id(gallery)
    draft = WatchlistPersonDraft(
        person_id=pid,
        name=name,
        surname=surname,
        country=country,
        notes=notes,
    )
    logger.info(
        "Created WatchlistPersonDraft: %s (%s, country=%s, source=%s)",
        draft.person_id,
        draft.full_name(),
        draft.country,
        draft.source,
    )
    return draft


def _ensure_backends(
    cfg: Optional[FaceConfig],
    detector: Optional[FaceDetectorAligner],
    embedder: Optional[FaceEmbedder],
) -> Tuple[FaceConfig, FaceDetectorAligner, FaceEmbedder]:
    """
    Ensure we have a FaceConfig, FaceDetectorAligner and FaceEmbedder.
    """
    if cfg is None:
        cfg = default_face_config()
    if detector is None:
        detector = FaceDetectorAligner(cfg)
    if embedder is None:
        embedder = FaceEmbedder(cfg)
    return cfg, detector, embedder


def process_image(
    person: WatchlistPersonDraft,
    image_path: Union[str, Path],
    *,
    face_index: Optional[int] = None,
    cfg: Optional[FaceConfig] = None,
    detector: Optional[FaceDetectorAligner] = None,
    embedder: Optional[FaceEmbedder] = None,
) -> ProcessImageOutcome:
    """
    Turn an external image into:
      - ImageReadError
      - NoFaceResult
      - MultiFaceImageResult
      - FaceTemplateProposal
    """
    img_path = Path(image_path)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        logger.warning("Failed to read image for watchlist enrollment: %s", img_path)
        return ImageReadError(image_path=img_path)

    cfg, detector, embedder = _ensure_backends(cfg, detector, embedder)

    detections = detector.detect_and_align(img_bgr)

    if not detections:
        logger.info("No face detected in image %s", img_path)
        return NoFaceResult(image_path=img_path)

    if face_index is None and len(detections) > 1:
        faces_dbg: List[DetectedFaceDebug] = []
        for idx, det in enumerate(detections):
            try:
                bbox_xyxy = _extract_bbox_xyxy(det, img_bgr.shape)
            except Exception as exc:
                logger.warning(
                    "Failed to extract bbox for face #%d in %s: %s",
                    idx,
                    img_path,
                    exc,
                )
                continue

            face_area = _face_area_from_bbox(bbox_xyxy)
            yaw_deg, pitch_deg, roll_deg, det_score = _extract_pose_and_score(det)

            faces_dbg.append(
                DetectedFaceDebug(
                    index=idx,
                    bbox_xyxy=bbox_xyxy,
                    face_area=face_area,
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                    roll_deg=roll_deg,
                    det_score=det_score,
                )
            )

        if not faces_dbg:
            logger.info(
                "Multiple detections in %s but none had valid bbox; treating as no face.",
                img_path,
            )
            return NoFaceResult(image_path=img_path, reason="no_valid_bbox")

        logger.info(
            "Image %s has %d faces; returning MultiFaceImageResult.",
            img_path,
            len(faces_dbg),
        )
        return MultiFaceImageResult(image_path=img_path, faces=faces_dbg)

    if face_index is None:
        det = detections[0]
    else:
        if face_index < 0 or face_index >= len(detections):
            logger.warning(
                "face_index %d out of range for image %s (faces=%d).",
                face_index,
                img_path,
                len(detections),
            )
            return NoFaceResult(image_path=img_path, reason="invalid_face_index")
        det = detections[face_index]

    bbox_xyxy = _extract_bbox_xyxy(det, img_bgr.shape)
    face_area = _face_area_from_bbox(bbox_xyxy)
    yaw_deg, pitch_deg, roll_deg, det_score = _extract_pose_and_score(det)

    if face_area < MIN_FACE_AREA_HARD_REJECT:
        logger.info(
            "Face too small in image %s (area=%d < %d), hard rejecting.",
            img_path,
            face_area,
            MIN_FACE_AREA_HARD_REJECT,
        )
        return NoFaceResult(image_path=img_path, reason="face_too_small")

    aligned_bgr, embedding = _extract_aligned_and_embedding(det, embedder)

    try:
        q_val = compute_full_quality(
            image=img_bgr,
            bbox=bbox_xyxy,
            det_score=det_score,
            yaw=yaw_deg,
            pitch=pitch_deg,
            cfg=cfg,
        )
    except TypeError:
        try:
            q_val = compute_full_quality(
                image=img_bgr,
                bbox=bbox_xyxy,
                det_score=det_score,
                yaw=yaw_deg,
                pitch=pitch_deg,
            )
        except TypeError:
            logger.warning(
                "compute_full_quality signature mismatch in watchlist_enroll; "
                "falling back to q_overall=0.5 for %s",
                img_path,
            )
            q_val = 0.5

    q_overall = float(q_val)
    quality_level = _classify_quality(q_overall)

    pose_bin = _guess_pose_bin(yaw_deg, pitch_deg)
    warnings: List[str] = []

    if face_area < MIN_FACE_AREA_WARN:
        warnings.append(
            f"Face is small (area={face_area}); template may be noisy."
        )
    if quality_level == "LOW":
        warnings.append(
            f"LOW quality (q_overall={q_overall:.2f}); may cause false matches."
        )

    proposal = FaceTemplateProposal(
        person_id=person.person_id,
        image_path=img_path,
        bbox_xyxy=bbox_xyxy,
        face_area=face_area,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        pose_bin=pose_bin,
        quality=q_overall,
        quality_level=quality_level,
        embedding=embedding,
        det_score=det_score,
        warnings=warnings,
    )

    logger.info(
        "Created FaceTemplateProposal for %s from %s | pose=%s, q=%.3f (%s), "
        "area=%d, warnings=%d",
        person.person_id,
        img_path.name,
        pose_bin,
        q_overall,
        quality_level,
        face_area,
        len(warnings),
    )

    return proposal


def commit_watchlist_enrollment(
    gallery: FaceGallery,
    person: WatchlistPersonDraft,
    proposals: Sequence[FaceTemplateProposal],
    *,
    copy_images: bool = True,
    raw_root: Optional[Union[str, Path]] = None,
) -> str:
    """
    Commit a watchlist person + their image-based templates into the gallery.
    """
    if not proposals:
        raise ValueError(
            "commit_watchlist_enrollment requires at least one FaceTemplateProposal."
        )

    meta: Dict[str, Union[str, float]] = {
        "source": person.source,
        "created_at": person.created_at,
    }
    if person.country:
        meta["country"] = person.country
    if person.notes:
        meta["notes"] = person.notes

    if copy_images:
        if raw_root is None:
            storage_root = Path("data") / "watchlist_raw"
        else:
            storage_root = Path(raw_root)
        person_dir = storage_root / person.person_id
        person_dir.mkdir(parents=True, exist_ok=True)
    else:
        storage_root = None
        person_dir = None

    templates: List[FaceTemplate] = []

    for idx, prop in enumerate(proposals):
        origin_file = prop.image_path.name
        storage_path_str: Optional[str] = None

        if copy_images and storage_root is not None:
            dest_name = f"{idx:03d}_{origin_file}"
            dest_path = person_dir / dest_name
            try:
                shutil.copy2(prop.image_path, dest_path)
                storage_path_str = str(dest_path)
            except Exception as exc:
                logger.warning(
                    "Failed to copy watchlist image %s -> %s: %s",
                    prop.image_path,
                    dest_path,
                    exc,
                )

        tmpl_meta: Dict[str, Union[str, float, List[str]]] = {
            "pose_bin_hint": prop.pose_bin,
            "yaw_deg": prop.yaw_deg,
            "pitch_deg": prop.pitch_deg,
            "roll_deg": prop.roll_deg,
            "quality": prop.quality,
            "quality_level": prop.quality_level,
            "source_type": "EXTERNAL_IMAGE",
            "origin_file": origin_file,
            "timestamp": prop.timestamp,
            "det_score": prop.det_score,
        }
        if storage_path_str is not None:
            tmpl_meta["storage_path"] = storage_path_str
        if prop.warnings:
            tmpl_meta["warnings"] = prop.warnings

        tmpl = FaceTemplate(
            embedding=prop.embedding,
            condition="neutral",
            metadata=tmpl_meta,
        )
        templates.append(tmpl)

    full_name = person.full_name()
    person_id = gallery.enroll_person(
        templates=templates,
        category="watchlist",
        name=full_name,
        metadata=meta,
        person_id=person.person_id,
        surname=person.surname,
    )

    gallery.save()

    logger.info(
        "Committed watchlist enrollment for %s (%s) with %d templates.",
        person_id,
        full_name,
        len(templates),
    )
    return person_id 
