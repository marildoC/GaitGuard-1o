
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from schemas import Frame, Tracklet
from .config import FaceConfig, default_face_config
from .detector_align import FaceDetectorAligner, FaceCandidate
from .quality import compute_full_quality
from identity.evidence_gate import EvidenceGate, GateDecision

logger = logging.getLogger(__name__)




class FaceEvidence:
    """
    One single high-quality observation for a track.

    IdentityEngine performs temporal smoothing / multi-sample stability.
    FaceRoute only generates per-frame evidence when available.

    Fields:
      - track_id       : person track identifier (Tracklet.track_id)
      - ts             : timestamp of the frame (monotonic / Frame.ts)
      - frame_id       : sequential frame index
      - quality        : scalar 0–1 face quality (full model)
      - embedding      : 1D L2-normalised face descriptor (e.g. 512-D)
      - bbox_in_frame  : (x1, y1, x2, y2) of head crop in full frame coords
      - yaw, pitch, roll:
            approximate pose angles in degrees; used by 3D/multiview
            identity engines to select appropriate prototypes.
      - det_score      : raw detector confidence for the chosen face (optional)
      - landmarks_2d   : (5, 2) 2D landmarks in head-crop coordinates, if
                         available. This is used by SourceAuth / 3D logic
                         to analyse motion and background.
    """

    __slots__ = (
        "track_id",
        "ts",
        "frame_id",
        "quality",
        "embedding",
        "bbox_in_frame",
        "yaw",
        "pitch",
        "roll",
        "det_score",
        "landmarks_2d",
    )

    def __init__(
        self,
        track_id: int,
        ts: float,
        frame_id: int,
        quality: float,
        embedding: np.ndarray,
        bbox_in_frame: Tuple[float, float, float, float],
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        det_score: Optional[float] = None,
        landmarks_2d: Optional[np.ndarray] = None,
    ) -> None:
        self.track_id = track_id
        self.ts = ts
        self.frame_id = frame_id
        self.quality = quality
        self.embedding = embedding
        self.bbox_in_frame = bbox_in_frame
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.det_score = det_score
        self.landmarks_2d = landmarks_2d

    def __repr__(self) -> str:
        lm_flag = "lm" if self.landmarks_2d is not None else "no-lm"
        return (
            f"FaceEvidence(track_id={self.track_id}, frame_id={self.frame_id}, "
            f"q={self.quality:.3f}, yaw={self.yaw}, pitch={self.pitch}, "
            f"roll={self.roll}, det_score={self.det_score}, {lm_flag})"
        )




class FaceRoute:
    """
    Wave-3 FaceRoute:
      - Multi-throttled (N frames + min_interval_ms)
      - Strong geometric filtering
      - Robust head crop logic
      - Full quality model (size, blur, pose penalties)
      - Per-track evidence buffer with pruning
      - Emits evidence only when q_face >= runtime gate

    It is 3D-ready:
      - Each FaceEvidence carries yaw / pitch / roll (+ det_score)
      - Higher layers (IdentityEngine / multiview engine / SourceAuth) can query
        recent evidence via get_best_evidence() / get_all_best() /
        get_last_evidence() / get_latest_landmarks().
    """

    def __init__(
        self,
        cfg: Optional[FaceConfig] = None,
        detector: Optional[FaceDetectorAligner] = None,
        evidence_gate: Optional[EvidenceGate] = None,
    ) -> None:
        self.cfg: FaceConfig = cfg or default_face_config()
        self.detector = detector or FaceDetectorAligner(self.cfg)
        
        self.evidence_gate = evidence_gate
        if self.evidence_gate is None:
            try:
                from core.config import load_config
                from core.governance_metrics import get_metrics_collector
                cfg_loaded = load_config()
                metrics_coll = get_metrics_collector()
                self.evidence_gate = EvidenceGate(cfg_loaded, metrics_coll)
            except Exception:
                self.evidence_gate = EvidenceGate(cfg, None)

        self._buffers: Dict[int, Deque[FaceEvidence]] = {}

        self._last_processed_frame: Dict[int, int] = {}

        self._last_processed_ts: Dict[int, float] = {}

        route_cfg = self.cfg.route
        logger.info(
            "FaceRoute initialised | lookback=%.1fs | max_entries=%d | "
            "every_n_frames=%d | min_interval_ms=%d",
            getattr(
                route_cfg,
                "lookback_seconds",
                getattr(route_cfg, "max_seconds_lookback", 2.0),
            ),
            route_cfg.max_entries_per_track,
            route_cfg.process_every_n_frames,
            getattr(route_cfg, "min_interval_ms", 0),
        )


    def reset(self) -> None:
        """Clear all track buffers + throttle state."""
        self._buffers.clear()
        self._last_processed_frame.clear()
        self._last_processed_ts.clear()

    def run(
        self,
        frame: Frame,
        tracklets: List[Tracklet],
    ) -> Dict[int, FaceEvidence]:
        """
        Main entry point. Called once per frame.

        Returns
        -------
        Dict[int, FaceEvidence]
            Mapping track_id → *new* FaceEvidence captured in this frame
            and passing the runtime quality gate.
        """
        if frame.image is None or frame.image.size == 0:
            return {}

        img = frame.image
        th = self.cfg.thresholds
        route_cfg = self.cfg.route
        ts_now = float(frame.ts)

        min_q_runtime = float(
            getattr(th, "min_quality_runtime", th.min_quality_for_embed)
        )

        lookback_seconds = float(
            getattr(
                route_cfg,
                "lookback_seconds",
                getattr(route_cfg, "max_seconds_lookback", 2.0),
            )
        )
        max_entries = int(route_cfg.max_entries_per_track)
        process_every_n_frames = int(route_cfg.process_every_n_frames)
        min_interval_ms = int(getattr(route_cfg, "min_interval_ms", 0))

        new_evidences: Dict[int, FaceEvidence] = {}

        for trk in tracklets:
            tid = int(getattr(trk, "track_id", -1))

            try:
                x1, y1, x2, y2 = trk.last_box
            except Exception:
                continue

            box_h = float(y2 - y1)
            if box_h < float(th.min_box_height_for_face_px):
                continue

            last_fid = self._last_processed_frame.get(tid)
            if (
                last_fid is not None
                and frame.frame_id - last_fid < process_every_n_frames
            ):
                continue

            last_ts = self._last_processed_ts.get(tid)
            if (
                last_ts is not None
                and min_interval_ms > 0
                and (ts_now - last_ts) * 1000.0 < float(min_interval_ms)
            ):
                continue

            head_crop, head_box = self._crop_head_region(img, (x1, y1, x2, y2))
            if head_crop is None:
                self._last_processed_frame[tid] = frame.frame_id
                self._last_processed_ts[tid] = ts_now
                continue

            try:
                candidates = self.detector.detect_and_align(head_crop)
            except Exception:
                logger.exception(
                    "FaceRoute: detect_and_align failed for track_id=%d", tid
                )
                self._last_processed_frame[tid] = frame.frame_id
                self._last_processed_ts[tid] = ts_now
                continue

            if not candidates:
                self._last_processed_frame[tid] = frame.frame_id
                self._last_processed_ts[tid] = ts_now
                continue

            best_cand, best_q = self._select_best_candidate(head_crop, candidates)

            if best_cand is None:
                self._last_processed_frame[tid] = frame.frame_id
                self._last_processed_ts[tid] = ts_now
                continue

            if best_cand.embedding is None:
                logger.warning(
                    "FaceRoute: best candidate for track_id=%d has no embedding; "
                    "skipping.",
                    tid,
                )
                self._last_processed_frame[tid] = frame.frame_id
                self._last_processed_ts[tid] = ts_now
                continue

            if best_q < min_q_runtime:
                self._last_processed_frame[tid] = frame.frame_id
                self._last_processed_ts[tid] = ts_now
                logger.debug(
                    "FaceRoute: track=%d frame=%d rejected q=%.3f < min_q_runtime=%.3f",
                    tid,
                    frame.frame_id,
                    best_q,
                    min_q_runtime,
                )
                continue

            emb = self._ensure_embedding(best_cand.embedding)

            lm_2d = getattr(best_cand, "landmarks_2d", None)
            if lm_2d is not None:
                lm_2d = np.asarray(lm_2d, dtype=np.float32).reshape(-1, 2)

            ev = FaceEvidence(
                track_id=tid,
                ts=frame.ts,
                frame_id=frame.frame_id,
                quality=float(best_q),
                embedding=emb,
                bbox_in_frame=head_box,
                yaw=best_cand.yaw,
                pitch=best_cand.pitch,
                roll=best_cand.roll,
                det_score=getattr(best_cand, "det_score", None),
                landmarks_2d=lm_2d,
            )

            if self.evidence_gate and self.evidence_gate.enabled:
                track_context = {
                    'track_id': tid,
                    'binding_state': 'UNKNOWN',  # Phase C will update this
                    'track_age_sec': frame.ts - trk.create_ts if hasattr(trk, 'create_ts') else 0.0,
                }
                
                gate_decision, gate_reason = self.evidence_gate.decide(
                    face_sample=ev,
                    track_context=track_context
                )
                
                if gate_decision == GateDecision.REJECT:
                    logger.debug(
                        "FaceRoute: track=%d rejected by gate (reason=%s)",
                        tid, gate_reason
                    )
                    self._last_processed_frame[tid] = frame.frame_id
                    self._last_processed_ts[tid] = ts_now
                    continue
                
                elif gate_decision == GateDecision.HOLD:
                    logger.debug(
                        "FaceRoute: track=%d held by gate (reason=%s)",
                        tid, gate_reason
                    )
                    self._last_processed_frame[tid] = frame.frame_id
                    self._last_processed_ts[tid] = ts_now
                    continue
                

            buf = self._buffers.setdefault(tid, deque())
            buf.append(ev)
            self._prune_buffer(tid, frame.ts, lookback_seconds)

            if len(buf) > max_entries:
                buf.popleft()

            self._last_processed_frame[tid] = frame.frame_id
            self._last_processed_ts[tid] = ts_now

            new_evidences[tid] = ev

        return new_evidences


    def get_best_evidence(
        self,
        track_id: int,
        current_ts: Optional[float] = None,
    ) -> Optional[FaceEvidence]:
        """
        Return the best-quality FaceEvidence for a track within the
        configured lookback window, or None if none exists.
        """
        buf = self._buffers.get(track_id)
        if not buf:
            return None

        route_cfg = self.cfg.route
        lookback_seconds = float(
            getattr(
                route_cfg,
                "lookback_seconds",
                getattr(route_cfg, "max_seconds_lookback", 2.0),
            )
        )

        if current_ts is None:
            current_ts = buf[-1].ts

        min_ts = float(current_ts) - lookback_seconds

        best_ev: Optional[FaceEvidence] = None
        best_q = -1.0

        for ev in buf:
            if ev.ts < min_ts:
                continue
            if ev.quality > best_q:
                best_q = ev.quality
                best_ev = ev

        return best_ev

    def get_all_best(
        self,
        current_ts: Optional[float] = None,
    ) -> Dict[int, FaceEvidence]:
        """
        Return best FaceEvidence for all tracks that have any evidence
        within the lookback window.
        """
        out: Dict[int, FaceEvidence] = {}
        for tid, buf in self._buffers.items():
            if not buf:
                continue

            ts = current_ts if current_ts is not None else buf[-1].ts
            ev = self.get_best_evidence(tid, ts)
            if ev is not None:
                out[tid] = ev

        return out

    def get_last_evidence(self, track_id: int) -> Optional[FaceEvidence]:
        """
        Lightweight helper: return the *most recent* FaceEvidence for
        a track (if any), without applying lookback or quality logic.

        This is useful for debugging and for components that only care
        about the latest pose / embedding snapshot.
        """
        buf = self._buffers.get(track_id)
        if not buf:
            return None
        return buf[-1]

    def get_latest_landmarks(
        self,
        track_id: int,
    ) -> Optional[Tuple[Tuple[float, float, float, float], np.ndarray, float]]:
        """
        Convenience accessor for SourceAuth and other motion-based modules.

        Returns
        -------
        Optional[Tuple[bbox_in_frame, landmarks_2d, quality]]

        Where:
          - bbox_in_frame : (x1, y1, x2, y2) head crop in frame coordinates
          - landmarks_2d  : (5, 2) numpy array in head-crop coordinates
          - quality       : scalar 0–1 face quality for this evidence

        If there is no evidence for this track, or landmarks_2d is missing,
        returns None.
        """
        buf = self._buffers.get(track_id)
        if not buf:
            return None

        ev = buf[-1]
        if ev.landmarks_2d is None:
            return None

        return ev.bbox_in_frame, ev.landmarks_2d, ev.quality


    def _prune_buffer(
        self,
        track_id: int,
        current_ts: float,
        lookback_seconds: float,
    ) -> None:
        """
        Drop entries older than lookback_seconds from a track buffer.
        """
        buf = self._buffers.get(track_id)
        if not buf:
            return

        max_age = float(lookback_seconds)
        while buf and (current_ts - buf[0].ts) > max_age:
            buf.popleft()

    def _crop_head_region(
        self,
        image: np.ndarray,
        box: Tuple[float, float, float, float],
    ) -> Tuple[Optional[np.ndarray], Tuple[float, float, float, float]]:
        """
        Crop a robust head+shoulders region from a person box.

        We bias towards the top of the person box:
          - extend a bit above y1
          - include down to ~60% of the body height
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box

        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 1.0 or bh <= 1.0:
            return None, (0.0, 0.0, 0.0, 0.0)

        head_top = y1 - 0.15 * bh
        head_bottom = y1 + 0.60 * bh

        hx1 = max(0, int(np.floor(x1)))
        hx2 = min(w, int(np.ceil(x2)))
        hy1 = max(0, int(np.floor(head_top)))
        hy2 = min(h, int(np.ceil(head_bottom)))

        if hx2 <= hx1 or hy2 <= hy1:
            return None, (0.0, 0.0, 0.0, 0.0)

        crop = image[hy1:hy2, hx1:hx2]
        bbox_in_frame = (float(hx1), float(hy1), float(hx2), float(hy2))
        return crop, bbox_in_frame

    def _select_best_candidate(
        self,
        head_crop: np.ndarray,
        candidates: List[FaceCandidate],
    ) -> Tuple[Optional[FaceCandidate], float]:
        """
        Among detected faces in the head crop, choose the one with the
        highest quality score according to compute_full_quality().

        No thresholding is applied here; the caller enforces the runtime gate.
        """
        if not candidates:
            return None, 0.0

        best_cand: Optional[FaceCandidate] = None
        best_q = -1.0

        for cand in candidates:
            q = compute_full_quality(
                image=head_crop,
                bbox=cand.bbox,
                det_score=cand.det_score,
                yaw=cand.yaw,
                pitch=cand.pitch,
                cfg=self.cfg,
            )

            if q > best_q:
                best_q = q
                best_cand = cand

        return best_cand, float(best_q)

    def _ensure_embedding(self, emb: np.ndarray) -> np.ndarray:
        """
        Ensure embedding is float32, 1-D and L2-normalised.

        buffalo_l usually already outputs L2-normalised 512-D vectors,
        but we enforce normalisation once more for safety.
        """
        e = np.asarray(emb, dtype=np.float32).reshape(-1)

        gallery = getattr(self.cfg, "gallery", None)
        dim_attr = getattr(gallery, "dim", None) if gallery is not None else None
        if isinstance(dim_attr, (int, float)) and dim_attr > 0:
            if e.size != int(dim_attr):
                logger.warning(
                    "FaceRoute: embedding dim mismatch (got %d, expected %d).",
                    e.size,
                    int(dim_attr),
                )

        norm = float(np.linalg.norm(e))
        if norm > 1e-6:
            e /= norm
        else:
            e[:] = 0.0
        return e
