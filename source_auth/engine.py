
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from schemas import Frame, Tracklet, IdSignals
from source_auth.config import SourceAuthConfig
from source_auth.types import (
    SourceAuthScores,
    SourceAuthComponentScores,
    SourceAuthReliabilityFlags,
    SourceAuthState,
    LandmarkFrame,
    BBoxTuple,
)
from source_auth.motion import compute_3d_motion_score
from source_auth.screen_artifacts import compute_screen_artifact_score
from source_auth.background import compute_background_consistency
from source_auth.fusion import fuse_source_auth

logger = logging.getLogger(__name__)




@dataclass
class FrameWindowEntry:
    """
    One frame-level observation for SourceAuth for a given track.

    Fields:
      - ts            : timestamp (monotonic, same domain as Frame.ts)
      - bbox_in_frame : (x1, y1, x2, y2) in full-frame coordinates (if known)
      - frame         : schemas.Frame reference (holds the actual image)
    """

    ts: float
    bbox_in_frame: Optional[BBoxTuple]
    frame: Frame


@dataclass
class SourceAuthTrackState:
    """
    Persistent SourceAuth *identity-level* state for one track_id.

    Landmark / motion and frame / bbox histories are stored separately as
    time-series in _landmark_history / _frame_history and pruned independently.
    """

    track_id: int
    last_seen_ts: float = field(default_factory=lambda: time.time())
    num_frames_seen: int = 0

    smoothed_score: float = 0.5
    frames_above_real: int = 0
    frames_below_spoof: int = 0
    last_state: SourceAuthState = "UNCERTAIN"




class SourceAuthEngine:
    """
    Main entry point for source authenticity (SourceAuth).

        engine = SourceAuthEngine(cfg)
        scores_by_track = engine.update(frame, tracks, id_signals)
    """

    def __init__(self, cfg: Optional[SourceAuthConfig] = None) -> None:
        self.cfg: SourceAuthConfig = cfg or SourceAuthConfig()

        self._tracks: Dict[int, SourceAuthTrackState] = {}

        self._landmark_history: Dict[int, Deque[LandmarkFrame]] = {}

        self._frame_history: Dict[int, Deque[FrameWindowEntry]] = {}

        self._score_history: Dict[int, float] = {}

        self._last_scores: Dict[int, SourceAuthScores] = {}

        self._last_debug: Dict[int, Dict[str, Dict[str, float]]] = {}

        self._last_log_ts: float = time.time()

        logger.info(
            "SourceAuthEngine initialised | window_sec=%.2f neutral=%.3f "
            "| motion_window_sec=%.2f screen_window_sec=%.2f background_window_sec=%.2f",
            float(getattr(self.cfg, "window_sec", 1.5)),
            float(getattr(self.cfg, "neutral_score", 0.5)),
            float(
                getattr(
                    self.cfg,
                    "motion_window_sec",
                    getattr(self.cfg, "window_sec", 1.5),
                )
            ),
            float(
                getattr(
                    self.cfg,
                    "screen_window_sec",
                    getattr(self.cfg, "window_sec", 1.5),
                )
            ),
            float(
                getattr(
                    self.cfg,
                    "background_window_sec",
                    getattr(self.cfg, "window_sec", 1.5),
                )
            ),
        )


    def reset(self) -> None:
        """
        Reset all SourceAuth per-track state and time-series histories.
        """
        self._tracks.clear()
        self._landmark_history.clear()
        self._frame_history.clear()
        self._score_history.clear()
        self._last_scores.clear()
        self._last_debug.clear()
        logger.info("SourceAuthEngine: state + landmark + frame history reset.")


    def update(
        self,
        frame: Frame,
        tracks: List[Tracklet],
        id_signals: List[IdSignals],
    ) -> Dict[int, SourceAuthScores]:
        """
        Main per-frame update.

        Behaviour:
          1) Tracks are the authoritative active track_ids.
          2) For each track:
               - ensure track state,
               - update bookkeeping (frames_seen, last_seen_ts),
               - update landmark + frame windows,
               - compute motion / screen / background cues,
               - fuse into a real-likelihood score + state,
               - store per-track scores for hysteresis/EMA.
          3) Prune stale tracks.
          4) Periodically log telemetry.
        """
        now = float(getattr(frame, "ts", time.time()))

        sig_by_tid: Dict[int, IdSignals] = {}
        for sig in id_signals:
            try:
                tid = int(sig.track_id)
            except Exception:
                continue
            sig_by_tid[tid] = sig

        results: Dict[int, SourceAuthScores] = {}

        for trk in tracks:
            track_id = getattr(trk, "track_id", getattr(trk, "id", None))
            if track_id is None:
                continue
            track_id = int(track_id)

            state = self._tracks.get(track_id)
            if state is None:
                state = SourceAuthTrackState(track_id=track_id)
                self._tracks[track_id] = state

            state.last_seen_ts = now
            state.num_frames_seen += 1

            sig = sig_by_tid.get(track_id)

            self._maybe_update_landmark_history(track_id, trk, sig, now)

            self._maybe_update_frame_history(track_id, trk, sig, frame, now)

            motion_score, motion_reliable, motion_debug = self._compute_motion_cue(
                track_id
            )

            (
                screen_score,
                screen_reliable,
                screen_debug,
                bg_score,
                bg_reliable,
                bg_debug,
            ) = self._compute_screen_and_background_cues(track_id)

            dbg_bucket = self._last_debug.setdefault(track_id, {})
            dbg_bucket["motion"] = motion_debug
            dbg_bucket["screen"] = screen_debug
            dbg_bucket["background"] = bg_debug

            scores = self._build_scores_with_motion_screen_background(
                track_id=track_id,
                state=state,
                sig=sig,
                motion_score=motion_score,
                motion_reliable=motion_reliable,
                motion_debug=motion_debug,
                screen_score=screen_score,
                screen_reliable=screen_reliable,
                screen_debug=screen_debug,
                background_score=bg_score,
                background_reliable=bg_reliable,
                background_debug=bg_debug,
            )

            self._last_scores[track_id] = scores
            self._score_history[track_id] = float(scores.source_auth_score)
            state.smoothed_score = float(scores.source_auth_score)
            state.last_state = scores.state

            results[track_id] = scores

        self._prune_stale_tracks(now)

        self._maybe_log_telemetry(now, results)

        return results


    def _maybe_update_landmark_history(
        self,
        track_id: int,
        trk: Tracklet,
        sig: Optional[IdSignals],
        ts: float,
    ) -> None:
        """
        Append a LandmarkFrame for this track if IdSignals carries
        usable 2D landmarks + sufficient quality.
        """
        if sig is None:
            return

        landmarks = getattr(sig, "landmarks_2d", None)
        if landmarks is None:
            return

        try:
            lm_arr = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        except Exception:
            logger.debug(
                "SourceAuthEngine: landmarks_2d for track_id=%d could not be "
                "reshaped to (N,2); skipping.",
                track_id,
            )
            return

        if lm_arr.size == 0:
            return

        q = getattr(sig, "face_quality", None)
        if q is None:
            q = getattr(sig, "quality", None)

        try:
            q_val = float(q) if q is not None else float(
                getattr(self.cfg, "neutral_score", 0.5)
            )
        except Exception:
            q_val = float(getattr(self.cfg, "neutral_score", 0.5))

        motion_q_min = float(
            getattr(
                self.cfg,
                "motion_min_quality",
                getattr(self.cfg, "motion_min_quality_default", 0.55),
            )
        )
        if q_val < motion_q_min:
            return

        bbox = getattr(sig, "face_bbox_in_frame", None)
        if bbox is None:
            bbox = getattr(sig, "bbox_in_frame", None)

        if bbox is None:
            try:
                bbox = tuple(trk.last_box)
            except Exception:
                bbox = None

        if bbox is not None:
            try:
                x1, y1, x2, y2 = bbox
                bbox_tuple: Optional[BBoxTuple] = (
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                )
            except Exception:
                bbox_tuple = None
        else:
            bbox_tuple = None

        hist = self._landmark_history.setdefault(track_id, deque())
        hist.append(
            LandmarkFrame(
                ts=ts,
                bbox=bbox_tuple,
                landmarks_2d=lm_arr,
                quality=q_val,
            )
        )

        self._prune_landmark_history(track_id, ts)

    def _prune_landmark_history(self, track_id: int, now: float) -> None:
        """
        Drop landmark frames older than motion_window_sec for a track.
        """
        hist = self._landmark_history.get(track_id)
        if not hist:
            return

        window_sec = float(
            getattr(
                self.cfg,
                "motion_window_sec",
                getattr(self.cfg, "window_sec", 1.5),
            )
        )

        max_age = float(window_sec)
        while hist and (now - hist[0].ts) > max_age:
            hist.popleft()

    def get_landmark_history(self, track_id: int) -> List[LandmarkFrame]:
        """
        Shallow copy of the landmark history for diagnostics / motion tools.
        """
        hist = self._landmark_history.get(track_id)
        if not hist:
            return []
        return list(hist)


    def _maybe_update_frame_history(
        self,
        track_id: int,
        trk: Tracklet,
        sig: Optional[IdSignals],
        frame: Frame,
        ts: float,
    ) -> None:
        """
        Maintain per-track sliding window of FrameWindowEntry for
        screen / background cues.
        """
        bbox = None

        if sig is not None:
            bbox = getattr(sig, "face_bbox_in_frame", None)
            if bbox is None:
                bbox = getattr(sig, "bbox_in_frame", None)

        if bbox is None:
            try:
                bbox = tuple(trk.last_box)
            except Exception:
                bbox = None

        if bbox is not None:
            try:
                x1, y1, x2, y2 = bbox
                bbox_tuple: Optional[BBoxTuple] = (
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                )
            except Exception:
                bbox_tuple = None
        else:
            bbox_tuple = None

        hist = self._frame_history.setdefault(track_id, deque())
        hist.append(
            FrameWindowEntry(
                ts=ts,
                bbox_in_frame=bbox_tuple,
                frame=frame,
            )
        )

        self._prune_frame_history(track_id, ts)

    def _prune_frame_history(self, track_id: int, now: float) -> None:
        """
        Drop frame entries older than the maximum of screen/background windows.
        """
        hist = self._frame_history.get(track_id)
        if not hist:
            return

        base_window = float(getattr(self.cfg, "window_sec", 1.5))
        screen_window = float(
            getattr(self.cfg, "screen_window_sec", base_window)
        )
        background_window = float(
            getattr(self.cfg, "background_window_sec", base_window)
        )

        window_sec = max(base_window, screen_window, background_window)
        max_age = float(window_sec)

        while hist and (now - hist[0].ts) > max_age:
            hist.popleft()

    def get_frame_history(self, track_id: int) -> List[FrameWindowEntry]:
        """
        Shallow copy of the frame window history for diagnostics.
        """
        hist = self._frame_history.get(track_id)
        if not hist:
            return []
        return list(hist)


    def _compute_motion_cue(
        self,
        track_id: int,
    ) -> Tuple[float, bool, Dict[str, float]]:
        """
        Wrapper around compute_3d_motion_score(...) for a single track.
        Returns:
          - motion_score   : [0,1] (0 planar, 1 strong 3D)
          - motion_reliable: bool
          - motion_debug   : dict with diagnostic stats (floats)
        """
        neutral = float(getattr(self.cfg, "neutral_score", 0.5))
        hist = self.get_landmark_history(track_id)

        if not hist:
            return neutral, False, {}

        try:
            score, reliable, debug = compute_3d_motion_score(hist, self.cfg)
        except Exception:
            logger.exception(
                "SourceAuthEngine: compute_3d_motion_score failed for track_id=%d",
                track_id,
            )
            return neutral, False, {}

        score = max(0.0, min(1.0, float(score)))
        return score, bool(reliable), {k: float(v) for k, v in debug.items()}


    def _compute_screen_and_background_cues(
        self,
        track_id: int,
    ) -> Tuple[float, bool, Dict[str, float], float, bool, Dict[str, float]]:
        """
        Compute screen-artifact and background-consistency cues for a track.
        Returns:
          - screen_score     : [0,1] (0 no screen, 1 strong screen evidence)
          - screen_reliable  : bool
          - screen_debug     : dict with diagnostic stats (floats)
          - background_score : [0,1]
                1 = consistent same-world background (real head)
                0 = strong mismatch (phone / different scene)
          - background_rel   : bool
          - background_debug : dict with diagnostic stats (floats)
        """
        neutral = float(getattr(self.cfg, "neutral_score", 0.5))
        hist = self.get_frame_history(track_id)

        if not hist:
            return neutral, False, {}, neutral, False, {}

        base_window = float(getattr(self.cfg, "window_sec", 1.5))
        screen_window = float(
            getattr(self.cfg, "screen_window_sec", base_window)
        )
        background_window = float(
            getattr(self.cfg, "background_window_sec", base_window)
        )
        window_sec = max(screen_window, background_window)

        ts_ref = hist[-1].ts
        frames_window: List[Tuple[np.ndarray, Optional[BBoxTuple]]] = []

        for entry in hist:
            if ts_ref - entry.ts > window_sec:
                continue

            img = getattr(entry.frame, "image", None)
            if img is None:
                continue

            frames_window.append((img, entry.bbox_in_frame))

        if not frames_window:
            return neutral, False, {}, neutral, False, {}

        try:
            screen_score, screen_reliable, screen_debug = compute_screen_artifact_score(
                frames_window, self.cfg
            )
        except Exception:
            logger.exception(
                "SourceAuthEngine: compute_screen_artifact_score failed for track_id=%d",
                track_id,
            )
            screen_score, screen_reliable, screen_debug = neutral, False, {}

        try:
            bg_score, bg_reliable, bg_debug = compute_background_consistency(
                frames_window, self.cfg
            )
        except Exception:
            logger.exception(
                "SourceAuthEngine: compute_background_consistency failed for track_id=%d",
                track_id,
            )
            bg_score, bg_reliable, bg_debug = neutral, False, {}

        screen_score = max(0.0, min(1.0, float(screen_score)))
        bg_score = max(0.0, min(1.0, float(bg_score)))

        return (
            screen_score,
            bool(screen_reliable),
            {k: float(v) for k, v in screen_debug.items()},
            bg_score,
            bool(bg_reliable),
            {k: float(v) for k, v in bg_debug.items()},
        )


    def _build_scores_with_motion_screen_background(
        self,
        track_id: int,
        state: SourceAuthTrackState,
        sig: Optional[IdSignals],
        motion_score: float,
        motion_reliable: bool,
        motion_debug: Dict[str, float],
        screen_score: float,
        screen_reliable: bool,
        screen_debug: Dict[str, float],
        background_score: float,
        background_reliable: bool,
        background_debug: Dict[str, float],
    ) -> SourceAuthScores:
        """
        Build SourceAuthScores using motion, screen and background cues.

        Semantics:
          - motion_score        : 1 = strong 3D parallax, 0 = planar card
          - screen_score        : 1 = strong screen-like evidence, 0 = no screen
          - background_score    : 1 = same-world background, 0 = mismatch
        """
        neutral = float(getattr(self.cfg, "neutral_score", 0.5))

        planar_3d_score = motion_score if motion_reliable else neutral
        screen_component_score = screen_score if screen_reliable else neutral
        background_component_score = (
            background_score if background_reliable else neutral
        )

        components = SourceAuthComponentScores(
            planar_3d=planar_3d_score,
            screen_artifacts=screen_component_score,
            background_consistency=background_component_score,
        )

        reliability = SourceAuthReliabilityFlags(
            enough_motion=motion_reliable,
            enough_landmarks=motion_reliable,
            enough_background=(screen_reliable or background_reliable),
        )

        prev_scores = self._last_scores.get(track_id)

        scores = fuse_source_auth(
            track_id=track_id,
            components=components,
            reliability=reliability,
            cfg=self.cfg,
            prev_scores=prev_scores,
        )

        base_debug: Dict[str, Any] = {
            "phase": "source_auth_phase5_fusion_state_machine",
            "frames_seen": state.num_frames_seen,
            "has_id_signals": sig is not None,
            "landmark_frames": len(self._landmark_history.get(track_id, ())),
            "frame_window_frames": len(self._frame_history.get(track_id, ())),
        }

        scores.update_debug("", base_debug)
        scores.update_debug("motion_", motion_debug)
        scores.update_debug("screen_", screen_debug)
        scores.update_debug("background_", background_debug)

        return scores


    def _apply_temporal_smoothing_and_state(
        self,
        track_id: int,
        state: SourceAuthTrackState,
        new_likelihood: float,
    ) -> Tuple[float, SourceAuthState]:
        """
        Legacy EMA + discrete state machine.
        Currently not used by fusion (handled in source_auth.fusion),
        but kept for compatibility / possible future reuse.
        """
        neutral = float(getattr(self.cfg, "neutral_score", 0.5))

        alpha = float(getattr(self.cfg, "ema_alpha", 0.6))
        alpha = max(0.0, min(1.0, alpha))

        prev = self._score_history.get(track_id, state.smoothed_score)
        if prev is None:
            prev = neutral

        smoothed = alpha * new_likelihood + (1.0 - alpha) * prev
        smoothed = max(0.0, min(1.0, float(smoothed)))

        self._score_history[track_id] = smoothed
        state.smoothed_score = smoothed

        real_min = float(getattr(self.cfg, "real_min_score", 0.9))
        likely_real_min = float(getattr(self.cfg, "likely_real_min_score", 0.7))
        spoof_max = float(getattr(self.cfg, "spoof_max_score", 0.1))
        likely_spoof_max = float(getattr(self.cfg, "likely_spoof_max_score", 0.3))

        frames_to_confirm_real = int(getattr(self.cfg, "frames_to_confirm_real", 5))
        frames_to_confirm_spoof = int(
            getattr(self.cfg, "frames_to_confirm_spoof", 5)
        )

        if smoothed >= real_min:
            state.frames_above_real += 1
        else:
            state.frames_above_real = 0

        if smoothed <= spoof_max:
            state.frames_below_spoof += 1
        else:
            state.frames_below_spoof = 0

        if state.frames_above_real >= frames_to_confirm_real and smoothed >= real_min:
            new_state: SourceAuthState = "REAL"
        elif (
            state.frames_below_spoof >= frames_to_confirm_spoof
            and smoothed <= spoof_max
        ):
            new_state = "SPOOF"
        elif smoothed >= real_min or smoothed >= likely_real_min:
            new_state = "LIKELY_REAL"
        elif smoothed <= spoof_max or smoothed <= likely_spoof_max:
            new_state = "LIKELY_SPOOF"
        else:
            new_state = "UNCERTAIN"

        state.last_state = new_state
        return smoothed, new_state


    def _build_neutral_scores(
        self,
        track_id: int,
        state: SourceAuthTrackState,
        sig: Optional[IdSignals],
    ) -> SourceAuthScores:
        """
        Legacy neutral builder retained for completeness / possible
        future use. Behaviour unchanged; now uses update_debug().
        """
        neutral = float(getattr(self.cfg, "neutral_score", 0.5))

        components = SourceAuthComponentScores(
            planar_3d=neutral,
            screen_artifacts=neutral,
            background_consistency=neutral,
        )

        reliability = SourceAuthReliabilityFlags(
            enough_motion=False,
            enough_landmarks=False,
            enough_background=False,
        )

        sa_state: SourceAuthState = "UNCERTAIN"

        debug_info = {
            "phase": "source_auth_neutral_legacy",
            "frames_seen": state.num_frames_seen,
            "has_id_signals": sig is not None,
            "landmark_frames": len(self._landmark_history.get(track_id, ())),
            "frame_window_frames": len(self._frame_history.get(track_id, ())),
        }

        scores = SourceAuthScores(
            track_id=track_id,
            source_auth_score=neutral,
            state=sa_state,
            components=components,
            reliability=reliability,
        )
        scores.update_debug("", debug_info)
        return scores


    def _prune_stale_tracks(self, now: float) -> None:
        """
        Remove very old track states and their histories to avoid
        unbounded growth.
        """
        max_idle = float(getattr(self.cfg, "max_idle_sec", 10.0))

        to_delete: List[int] = []
        for tid, st in self._tracks.items():
            if now - st.last_seen_ts > max_idle:
                to_delete.append(tid)

        for tid in to_delete:
            self._tracks.pop(tid, None)
            self._landmark_history.pop(tid, None)
            self._frame_history.pop(tid, None)
            self._score_history.pop(tid, None)
            self._last_scores.pop(tid, None)
            self._last_debug.pop(tid, None)

        if to_delete:
            logger.debug(
                "SourceAuthEngine: pruned %d stale track states (max_idle=%.1fs).",
                len(to_delete),
                max_idle,
            )


    def _maybe_log_telemetry(
        self,
        now: float,
        results: Dict[int, SourceAuthScores],
    ) -> None:
        """
        Periodically log a compact summary for each active track:
          - track_id, component scores, reliability flags,
          - final source_auth_score + state,
          - a couple of canonical debug metrics if present.
        """
        interval = float(
            getattr(
                self.cfg,
                "log_interval_sec",
                getattr(self.cfg, "metrics_log_interval_sec", 5.0),
            )
        )
        if interval <= 0.0:
            return
        if now - self._last_log_ts < interval:
            return

        self._last_log_ts = now

        if not results:
            return

        for tid, sa in results.items():
            d = sa.debug
            try:
                planar = sa.components.planar_3d
                screen = sa.components.screen_artifacts
                bg = sa.components.background_consistency

                motion_parallax = float(d.get("motion_parallax_ratio", 0.0))
                screen_border = float(d.get("screen_border_strength", 0.0))

                logger.info(
                    "SourceAuthTelemetry | track_id=%d | score=%.3f state=%s | "
                    "planar_3d=%.3f screen_artifacts=%.3f background_consistency=%.3f | "
                    "motion_parallax_ratio=%.3f screen_border_strength=%.3f",
                    tid,
                    sa.source_auth_score,
                    sa.state,
                    planar,
                    screen,
                    bg,
                    motion_parallax,
                    screen_border,
                )
            except Exception:
                logger.debug(
                    "SourceAuthTelemetry: failed to log telemetry for track_id=%d",
                    tid,
                )
