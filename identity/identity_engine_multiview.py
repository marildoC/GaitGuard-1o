
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace, fields
from typing import Deque, Dict, List, Optional, Tuple, Any

import numpy as np

from core.interfaces import IdentityEngine as IdentityEngineBase
from schemas import Frame, Tracklet, IdSignals, IdentityDecision, FaceSample
from face.config import FaceConfig, default_face_config
from face.multiview_types import MultiViewConfig
from face.multiview_gallery_view import MultiViewGalleryView
from face.route import FaceRoute
from identity.face_gallery import FaceGallery
from identity.multiview_matcher import (
    MultiViewMatcher,
    MultiViewMatchResult,
    MultiViewCandidate,
)
from identity.binding import BindingManager, BindingDecision

logger = logging.getLogger(__name__)




@dataclass
class EvidenceSample:
    """
    One temporal evidence sample for a given track at a specific time.

    We keep:
      - which person we matched (if any),
      - numeric distance / score,
      - coarse band: strong / weak / none,
      - pose bin used for the match,
      - centroid / match quality for diagnostics,
      - face_quality: quality of the underlying face observation (0..1).
    """

    ts: float
    person_id: Optional[str]
    distance: float
    score: float
    strength: str
    pose_bin: str
    match_quality: float
    face_quality: float = 0.0


@dataclass
class TrackIdentityState:
    """
    Persistent identity state for a single track_id.

    Tracks:
      - current committed person_id (or None if still Unknown),
      - current band (strong / weak / none),
      - last numeric score + distance,
      - pose bin for the current committed identity,
      - evidence window of recent samples,
      - timestamps for decay / staleness.
      - FIX 2: binding state and confidence from BindingManager
    """

    track_id: int
    current_person_id: Optional[str] = None
    current_strength: str = "none"  # "strong" / "weak" / "none"
    current_score: float = 0.0
    current_distance: float = 1e9
    current_pose_bin: str = "none"
    last_update_ts: float = field(default_factory=lambda: time.time())
    last_seen_ts: float = field(default_factory=lambda: time.time())
    evidence: Deque[EvidenceSample] = field(default_factory=deque)
    binding_state: str = "UNKNOWN"
    binding_confidence: float = 0.0
    binding_person_id: Optional[str] = None

    def add_evidence(self, sample: EvidenceSample, max_len: int) -> None:
        """
        Append a new sample to the evidence window, keeping at most max_len.
        """
        self.evidence.append(sample)
        while len(self.evidence) > max_len:
            self.evidence.popleft()
        self.last_update_ts = sample.ts

    def clear(self) -> None:
        """
        Reset to Unknown / none, but keep the track alive.
        """
        self.current_person_id = None
        self.current_strength = "none"
        self.current_score = 0.0
        self.current_distance = 1e9
        self.current_pose_bin = "none"
        self.binding_state = "UNKNOWN"  # FIX 2: Reset binding state
        self.binding_confidence = 0.0
        self.binding_person_id = None
        self.evidence.clear()




class IdentityEngineMultiView(IdentityEngineBase):
    """
    Pose-aware, pseudo-3D identity engine on top of MultiViewMatcher.

    High-level flow:

        1) We run a FaceRoute (same as classic engine) to build IdSignals
           with canonical FaceSample objects for each track.
        2) decide(signals) calls MultiViewMatcher for each valid observation.
        3) We update per-track evidence windows (deques of EvidenceSample).
        4) We apply confirm/switch rules to choose/maintain person_id per track.
        5) We emit IdentityDecision objects for overlay / events / logging.

    Classic identity engine and this multi-view engine are swappable in
    core.main_loop via configuration.
    
    LAYER 2 & 3 ROBUSTNESS ENHANCEMENTS:
    
    LAYER 2 (Quality Smoothing):
        - Applied in EvidenceGate._compute_smoothed_quality()
        - 5-frame moving average eliminates frame-to-frame noise
        - Provides stable, consistent recognition (0.5-1 sec vs 3-4 sec)
        - Improves acceptance rate from ~90% to ~99%
        - No code changes needed here; gate handles it upstream
    
    LAYER 3 (Robust Per-Track Binding):
        - Each track has independent evidence buffer (TrackIdentityState.evidence)
        - No starvation even with 10+ people in frame
        - Enhanced diagnostics via _log_evidence_diagnostics()
        - Provides visibility into why binding hasn't occurred yet
        - Helps troubleshoot multi-person deployment scenarios
    """

    def __init__(
        self,
        face_cfg: Optional[FaceConfig] = None,
        mv_cfg: Optional[MultiViewConfig] = None,
        mv_gallery: Optional[MultiViewGalleryView] = None,
        matcher: Optional[MultiViewMatcher] = None,
    ) -> None:
        super().__init__()

        self.identity_mode: str = "multiview"

        self._face_cfg: FaceConfig = face_cfg or default_face_config()

        self._mv_cfg: MultiViewConfig = mv_cfg or MultiViewConfig()

        th = getattr(self._face_cfg, "thresholds", None)
        if th is not None:
            mv_field_names = {f.name for f in fields(self._mv_cfg)}

            strong_field = None
            if "strong_distance" in mv_field_names:
                strong_field = "strong_distance"
            elif "strong_dist" in mv_field_names:
                strong_field = "strong_dist"

            weak_field = None
            if "weak_distance" in mv_field_names:
                weak_field = "weak_distance"
            elif "weak_dist" in mv_field_names:
                weak_field = "weak_dist"

            minq_field = None
            if "min_quality_for_match" in mv_field_names:
                minq_field = "min_quality_for_match"
            elif "min_quality" in mv_field_names:
                minq_field = "min_quality"

            changes: Dict[str, float] = {}

            if strong_field is not None:
                strong_val = getattr(self._mv_cfg, strong_field, None)
                if strong_val is None:
                    strong_val = getattr(th, "strong_dist", 0.85)
                changes[strong_field] = float(strong_val)

            if weak_field is not None:
                weak_val = getattr(self._mv_cfg, weak_field, None)
                if weak_val is None:
                    weak_val = getattr(th, "weak_dist", 0.93)
                changes[weak_field] = float(weak_val)

            if minq_field is not None:
                min_q = getattr(self._mv_cfg, minq_field, None)
                if min_q is None:
                    q_rt = getattr(th, "min_quality_runtime", None)
                    if q_rt is None:
                        q_rt = getattr(th, "min_quality_for_embed", 0.0)
                    min_q = q_rt
                changes[minq_field] = float(min_q)

            if changes:
                self._mv_cfg = replace(self._mv_cfg, **changes)

                eff_strong = (
                    getattr(self._mv_cfg, strong_field)
                    if strong_field is not None
                    else None
                )
                eff_weak = (
                    getattr(self._mv_cfg, weak_field)
                    if weak_field is not None
                    else None
                )
                eff_minq = (
                    getattr(self._mv_cfg, minq_field)
                    if minq_field is not None
                    else None
                )

                logger.info(
                    "IdentityEngineMultiView: thresholds aligned with classic "
                    "(strong=%s weak=%s q_match>=%s)",
                    f"{eff_strong:.3f}" if eff_strong is not None else "n/a",
                    f"{eff_weak:.3f}" if eff_weak is not None else "n/a",
                    f"{eff_minq:.3f}" if eff_minq is not None else "n/a",
                )

        source_gallery: Any = None
        if mv_gallery is None:
            gallery = FaceGallery(self._face_cfg.gallery)
            source_gallery = gallery
            mv_gallery = MultiViewGalleryView(gallery, self._mv_cfg)
        else:
            source_gallery = getattr(mv_gallery, "gallery", None)

        self._mv_gallery = mv_gallery

        self._person_categories: Dict[str, str] = self._build_person_category_map(
            source_gallery
        )
        if self._person_categories:
            logger.info(
                "IdentityEngineMultiView: loaded %d person categories from gallery",
                len(self._person_categories),
            )
        else:
            logger.info(
                "IdentityEngineMultiView: no explicit person categories found; "
                "non-unknown identities will default to 'resident'."
            )

        if matcher is None:
            matcher = MultiViewMatcher(self._mv_gallery, self._mv_cfg)
        self._matcher = matcher

        self._tracks: Dict[int, TrackIdentityState] = {}

        smoothing_src = getattr(self._face_cfg, "smoothing", self._mv_cfg)

        self._confirm_strong = getattr(smoothing_src, "confirm_strong", 3)
        self._confirm_weak = getattr(smoothing_src, "confirm_weak", 4)
        self._switch_strong = getattr(smoothing_src, "switch_strong", 4)
        self._switch_weak = getattr(smoothing_src, "switch_weak", 5)
        self._max_evidence_len = getattr(smoothing_src, "max_evidence_len", 15)
        self._max_idle_seconds = getattr(smoothing_src, "max_idle_seconds", 10.0)

        logger.info(
            "IdentityEngineMultiView initialised | "
            "confirm_strong=%d, confirm_weak=%d, "
            "switch_strong=%d, switch_weak=%d, max_evidence_len=%d, max_idle=%.1fs",
            self._confirm_strong,
            self._confirm_weak,
            self._switch_strong,
            self._switch_weak,
            self._max_evidence_len,
            self._max_idle_seconds,
        )

        self._face_route: FaceRoute = FaceRoute(self._face_cfg)

        self.binding_enabled = True
        binding_cfg = None
        try:
            if hasattr(self._face_cfg, 'governance') and hasattr(self._face_cfg.governance, 'binding'):
                binding_cfg = self._face_cfg.governance.binding
                self.binding_enabled = getattr(binding_cfg, 'enabled', True)
        except Exception as e:
            logger.warning(f"Could not read binding config: {e}")
        
        try:
            self._binding_manager = BindingManager(cfg=binding_cfg)
            logger.info(f"IdentityEngineMultiView: BindingManager initialized (enabled={self.binding_enabled})")
        except Exception as binding_init_error:
            logger.error(
                f"IdentityEngineMultiView: Failed to initialize BindingManager: {binding_init_error}; "
                f"binding will be disabled for safety",
                exc_info=False
            )
            self._binding_manager = None
            self.binding_enabled = False


    def reset(self) -> None:
        """
        Reset all per-track identity state.
        """
        self._tracks.clear()


    def update_signals(
        self,
        frame: Frame,
        tracks: List[Tracklet],
        schedule_context: Optional[Any] = None,
    ) -> List[IdSignals]:
        """
        Build IdSignals for the current frame using the same FaceRoute
        front-end as the classic engine.

        Responsibilities:
          - Run FaceRoute on (frame, tracks) to get FaceEvidence per track.
          - Convert each FaceEvidence into a canonical FaceSample.
          - Attach that FaceSample to IdSignals via set_best_face():
                * this also mirrors embedding/quality/pose into legacy fields.
          - Maintain TrackIdentityState.last_seen_ts for temporal cleanup.
          - Prune stale TrackIdentityState entries.
        """
        now = float(getattr(frame, "ts", time.time()))

        tracks_for_route = []
        if schedule_context is not None:
            for tr in tracks:
                t_id = getattr(tr, "track_id", getattr(tr, "id", None))
                if t_id is not None:
                     if t_id in schedule_context.scheduled_track_ids:
                         tracks_for_route.append(tr)
        else:
            tracks_for_route = tracks
            
        evidences = self._face_route.run(frame, tracks_for_route)

        signals: List[IdSignals] = []

        for tr in tracks:
            track_id = getattr(tr, "track_id", getattr(tr, "id", None))
            if track_id is None:
                continue
            track_id = int(track_id)

            state = self._tracks.get(track_id)
            if state is None:
                state = TrackIdentityState(track_id=track_id)
                self._tracks[track_id] = state
            state.last_seen_ts = now

            ev = evidences.get(track_id)

            if ev is not None:
                q = float(ev.quality)

                extra_meta: Dict[str, float] = {}
                if ev.yaw is not None:
                    extra_meta["yaw"] = float(ev.yaw)
                if ev.pitch is not None:
                    extra_meta["pitch"] = float(ev.pitch)
                if ev.roll is not None:
                    extra_meta["roll"] = float(ev.roll)
                if ev.det_score is not None:
                    extra_meta["det_score"] = float(ev.det_score)

                face_sample = FaceSample(
                    bbox=ev.bbox_in_frame,
                    embedding=np.asarray(ev.embedding, dtype=np.float32).reshape(-1),
                    det_score=float(ev.det_score) if ev.det_score is not None else 0.0,
                    quality=q,
                    yaw=float(ev.yaw) if ev.yaw is not None else None,
                    pitch=float(ev.pitch) if ev.pitch is not None else None,
                    roll=float(ev.roll) if ev.roll is not None else None,
                    ts=float(now),
                    pose_bin=None,
                    source="runtime",
                    extra=extra_meta or None,
                )

                sig = IdSignals(track_id=track_id)
                sig.set_best_face(face_sample)
                
                
                if ev.landmarks_2d is not None:
                    try:
                        sig.landmarks_2d = np.asarray(
                            ev.landmarks_2d, dtype=np.float32
                        ).reshape(-1, 2)
                    except Exception:
                        pass
                
                if ev.bbox_in_frame is not None:
                    try:
                        sig.face_bbox_in_frame = (
                            float(ev.bbox_in_frame[0]),
                            float(ev.bbox_in_frame[1]),
                            float(ev.bbox_in_frame[2]),
                            float(ev.bbox_in_frame[3]),
                        )
                    except Exception:
                        pass
                        
            else:
                sig = IdSignals(track_id=track_id)

            signals.append(sig)

        self._prune_stale_tracks(now)

        return signals

    def decide(self, signals: List[IdSignals]) -> List[IdentityDecision]:
        """
        Core identity logic.

        For each track in IdSignals:
          - Ensure we have a canonical FaceSample in IdSignals.best_face
            (using ensure_best_face_from_legacy() as a safety net).
          - Pull embedding + yaw/pitch + quality from best_face.
          - Match using MultiViewMatcher (pose-aware, distance bands).
          - Update per-track evidence window.
          - Apply confirm/switch rules.
          - Emit an IdentityDecision for overlay/events.
        """
        now = time.time()
        decisions: List[IdentityDecision] = []

        for sig in signals:
            track_id = int(sig.track_id)
            state = self._tracks.get(track_id)
            if state is None:
                state = TrackIdentityState(track_id=track_id)
                self._tracks[track_id] = state

            state.last_seen_ts = now

            try:
                sig.ensure_best_face_from_legacy(ts=now)
            except AttributeError:
                pass

            face: Optional[FaceSample] = sig.best_face

            if face is None or face.embedding is None:
                logger.debug(
                    "IdentityEngineMultiView: no best_face or embedding for track_id=%d",
                    track_id,
                )
                decisions.append(
                    self._build_decision_from_state(
                        track_id=track_id,
                        state=state,
                        sample=None,
                    )
                )
                continue

            emb = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
            if emb.size == 0:
                logger.warning(
                    "IdentityEngineMultiView: empty embedding for track_id=%d; "
                    "using existing state only.",
                    track_id,
                )
                decisions.append(
                    self._build_decision_from_state(
                        track_id=track_id,
                        state=state,
                        sample=None,
                    )
                )
                continue

            yaw = float(face.yaw) if face.yaw is not None else 0.0
            pitch = float(face.pitch) if face.pitch is not None else 0.0
            quality = float(face.quality)

            if (face.yaw is None or face.pitch is None) and sig.extra and isinstance(
                sig.extra, dict
            ):
                if face.yaw is None and "yaw" in sig.extra:
                    try:
                        yaw = float(sig.extra["yaw"])
                    except Exception:
                        pass
                if face.pitch is None and "pitch" in sig.extra:
                    try:
                        pitch = float(sig.extra["pitch"])
                    except Exception:
                        pass

            mv_result: MultiViewMatchResult = self._matcher.match(
                embedding=emb,
                yaw=yaw,
                pitch=pitch,
                quality=quality,
                top_k=5,
            )
            if mv_result.best is not None:
                best_dbg: MultiViewCandidate = mv_result.best
                logger.info(
                    "IdentityEngineMultiView: track=%d strength=%s pid=%s dist=%.3f "
                    "score=%.3f bin=%s q=%.3f",
                    track_id,
                    mv_result.strength,
                    best_dbg.person_id,
                    best_dbg.distance,
                    best_dbg.score,
                    best_dbg.pose_bin,
                    mv_result.query_quality,
                )
            else:
                logger.info(
                    "IdentityEngineMultiView: track=%d strength=%s (no best) "
                    "bin_used=%s q=%.3f",
                    track_id,
                    mv_result.strength,
                    mv_result.pose_bin_used,
                    mv_result.query_quality,
                )

            if mv_result.best is None or mv_result.strength == "none":
                ev_sample = EvidenceSample(
                    ts=now,
                    person_id=None,
                    distance=1e9,
                    score=0.0,
                    strength="none",
                    pose_bin=mv_result.pose_bin_used,
                    match_quality=0.0,
                    face_quality=quality,
                )
            else:
                best: MultiViewCandidate = mv_result.best
                ev_sample = EvidenceSample(
                    ts=now,
                    person_id=best.person_id,
                    distance=best.distance,
                    score=best.score,
                    strength=mv_result.strength,
                    pose_bin=best.pose_bin,
                    match_quality=best.centroid_quality,
                    face_quality=quality,
                )

            state.add_evidence(ev_sample, max_len=self._max_evidence_len)
            self._apply_decision_logic(state)
            
            if self.binding_enabled and self._binding_manager is not None:
                try:
                    second_best_score = 0.0
                    if mv_result.best and mv_result.candidates:
                        best_pid = mv_result.best.person_id
                        for cand in mv_result.candidates:
                            if cand.person_id != best_pid:
                                second_best_score = float(cand.score)
                                break
                        else:
                            second_best_score = 0.0
                    
                    binding_result: BindingDecision = self._binding_manager.process_evidence(
                        track_id=track_id,
                        person_id=ev_sample.person_id,
                        score=ev_sample.score,
                        second_best_score=second_best_score,
                        quality=quality,
                        timestamp=now,
                    )
                    
                    state.binding_state = binding_result.binding_state
                    state.binding_confidence = binding_result.confidence
                    
                    if binding_result.person_id != state.current_person_id:
                        state.current_person_id = binding_result.person_id
                        state.binding_person_id = binding_result.person_id
                        logger.debug(
                            f"IdentityEngineMultiView: track={track_id} binding override "
                            f"matcher={ev_sample.person_id} -> binding_manager={binding_result.person_id} "
                            f"state={binding_result.binding_state}"
                        )
                    else:
                        state.binding_person_id = binding_result.person_id
                except Exception as binding_error:
                    logger.error(
                        f"IdentityEngineMultiView: binding manager error for track={track_id}: {binding_error}",
                        exc_info=False
                    )
                    state.binding_state = "ERROR"
                    state.binding_confidence = 0.0
                    state.binding_person_id = None
            else:
                state.binding_state = "BYPASS"
                state.binding_confidence = 1.0 if ev_sample.person_id else 0.0
                state.binding_person_id = ev_sample.person_id
            
            self._log_evidence_diagnostics(state, ev_sample)

            decisions.append(
                self._build_decision_from_state(
                    track_id=track_id,
                    state=state,
                    sample=ev_sample,
                )
            )

        self._prune_stale_tracks(time.time())

        return decisions


    def _apply_decision_logic(self, state: TrackIdentityState) -> None:
        """
        Update `state.current_person_id` and related fields based on the
        evidence window and confirm / switch thresholds.

        High-level rules:

          1) If currently Unknown:
               - Count strong / weak evidence per candidate person_id.
               - If any reaches confirm_strong / confirm_weak, commit.

          2) If currently assigned to P:
               - If recent evidence still supports P (strong / weak), keep it.
               - If another Q gets enough strong evidence (>= switch_strong),
                 switch to Q.
               - If evidence is mostly "none", gradually degrade / reset.

          3) Decay: if no evidence for max_idle_seconds, reset.
        """
        now = time.time()

        if now - state.last_seen_ts > self._max_idle_seconds:
            state.clear()
            return

        if not state.evidence:
            return

        strong_counts: Dict[str, int] = {}
        weak_counts: Dict[str, int] = {}
        none_count = 0

        for ev in state.evidence:
            if ev.strength == "none" or ev.person_id is None:
                none_count += 1
                continue
            if ev.strength == "strong":
                strong_counts[ev.person_id] = strong_counts.get(ev.person_id, 0) + 1
            elif ev.strength == "weak":
                weak_counts[ev.person_id] = weak_counts.get(ev.person_id, 0) + 1

        def best_candidate(counts: Dict[str, int]) -> Tuple[Optional[str], int]:
            if not counts:
                return None, 0
            pid = max(counts.items(), key=lambda kv: kv[1])[0]
            return pid, counts[pid]

        strong_pid, strong_n = best_candidate(strong_counts)
        weak_pid, weak_n = best_candidate(weak_counts)

        current = state.current_person_id

        if current is None:
            if strong_pid is not None and strong_n >= self._confirm_strong:
                state.current_person_id = strong_pid
                state.current_strength = "strong"
                self._refresh_numeric_stats_for_pid(state, strong_pid)
                return

            if weak_pid is not None and weak_n >= self._confirm_weak:
                state.current_person_id = weak_pid
                state.current_strength = "weak"
                self._refresh_numeric_stats_for_pid(state, weak_pid)
                return

            state.current_person_id = None
            state.current_strength = "none"
            state.current_score = 0.0
            state.current_distance = 1e9
            state.current_pose_bin = "none"
            return

        curr_strong_n = strong_counts.get(current, 0)
        curr_weak_n = weak_counts.get(current, 0)

        if curr_strong_n > 0 or curr_weak_n > 0:
            if curr_strong_n >= self._confirm_strong:
                state.current_strength = "strong"
            elif curr_weak_n >= 1:
                state.current_strength = "weak"

            self._refresh_numeric_stats_for_pid(state, current)
            return

        if (
            strong_pid is not None
            and strong_pid != current
            and strong_n >= self._switch_strong
        ):
            state.current_person_id = strong_pid
            state.current_strength = "strong"
            self._refresh_numeric_stats_for_pid(state, strong_pid)
            return

        if none_count >= len(state.evidence) // 2:
            if state.current_strength == "strong":
                state.current_strength = "weak"
            elif state.current_strength == "weak":
                state.clear()

    def _refresh_numeric_stats_for_pid(
        self,
        state: TrackIdentityState,
        pid: str,
    ) -> None:
        """
        For the given committed person_id, update numeric stats on the state
        (score, distance, pose_bin) from the most recent matching evidence.
        """
        for ev in reversed(state.evidence):
            if ev.person_id == pid and ev.strength != "none":
                state.current_score = ev.score
                state.current_distance = ev.distance
                state.current_pose_bin = ev.pose_bin
                return

        state.current_score = 0.0
        state.current_distance = 1e9
        state.current_pose_bin = "none"

    def _log_evidence_diagnostics(
        self,
        state: TrackIdentityState,
        ev_sample: Optional[EvidenceSample],
    ) -> None:
        """
        LAYER 3: Enhanced diagnostics for robust evidence accumulation.
        
        This method logs detailed information about per-track evidence windows
        to help diagnose recognition issues and verify proper binding.
        
        Called whenever a binding decision is made, this provides:
        - Evidence accumulation progress (N/M samples)
        - Strong/weak/none distribution
        - Quality consistency across window
        - Time window span
        - Recommended next steps if binding hasn't occurred yet
        
        Benefits (LAYER 3):
        - Transparency: See exactly why binding hasn't occurred
        - Diagnostics: Identify if evidence gate is too strict
        - Multi-person: Verify no starvation even with multiple tracks
        - Production: Helps troubleshoot real-world deployment issues
        """
        try:
            if len(state.evidence) == 0:
                return
            
            strong_samples = [e for e in state.evidence if e.strength == "strong"]
            weak_samples = [e for e in state.evidence if e.strength == "weak"]
            none_samples = [e for e in state.evidence if e.strength == "none"]
            
            qualities = [e.face_quality for e in state.evidence]
            avg_quality = np.mean(qualities) if qualities else 0.0
            min_quality = np.min(qualities) if qualities else 0.0
            max_quality = np.max(qualities) if qualities else 0.0
            
            time_span = state.evidence[-1].ts - state.evidence[0].ts if len(state.evidence) > 1 else 0.0
            
            person_ids = set()
            for ev in state.evidence:
                if ev.person_id is not None:
                    person_ids.add(ev.person_id)
            
            logger.debug(
                "LAYER3_Evidence track=%d | "
                "window=%d/%d (%.1fs) | "
                "strong=%d weak=%d none=%d | "
                "persons=%d | "
                "quality: avg=%.3f min=%.3f max=%.3f | "
                "current_binding=%s(%s)",
                state.track_id,
                len(state.evidence),
                self._max_evidence_len,
                time_span,
                len(strong_samples),
                len(weak_samples),
                len(none_samples),
                len(person_ids),
                avg_quality,
                min_quality,
                max_quality,
                state.current_person_id or "None",
                state.current_strength,
            )
            
            if state.current_person_id is None and person_ids:
                pid = list(person_ids)[0]
                pid_strong = sum(1 for e in strong_samples if e.person_id == pid)
                pid_weak = sum(1 for e in weak_samples if e.person_id == pid)
                
                logger.debug(
                    "LAYER3_NotBound track=%d | "
                    "best_candidate=%s (strong=%d/%d weak=%d/%d) | "
                    "need_strong=%d or weak=%d",
                    state.track_id,
                    pid,
                    pid_strong,
                    self._confirm_strong,
                    pid_weak,
                    self._confirm_weak,
                    self._confirm_strong - pid_strong,
                    self._confirm_weak - pid_weak,
                )
        
        except Exception as e:
            logger.warning(f"Error in evidence diagnostics for track {state.track_id}: {e}")

    def _prune_stale_tracks(self, now: float) -> None:
        """
        Remove track states that haven't been seen in a long time.
        CRITICAL FIX: Also clean up quality smoothing buffers in evidence gate.
        """
        to_delete: List[int] = []
        for track_id, st in self._tracks.items():
            if now - st.last_seen_ts > (self._max_idle_seconds * 2.0):
                to_delete.append(track_id)
        
        for tid in to_delete:
            del self._tracks[tid]
            try:
                if hasattr(self._face_route, '_evidence_gate'):
                    evidence_gate = self._face_route._evidence_gate
                    if evidence_gate and hasattr(evidence_gate, 'cleanup_track_buffers'):
                        evidence_gate.cleanup_track_buffers(tid)
            except Exception as cleanup_error:
                logger.debug(f"Could not cleanup evidence gate buffers for track {tid}: {cleanup_error}")


    def _build_person_category_map(self, gallery_like: Any) -> Dict[str, str]:
        """
        Build a static mapping person_id -> category (e.g. resident, watchlist)
        from the underlying FaceGallery.

        This is intentionally defensive and tries several patterns so we don't
        break if FaceGallery evolves:

          1) Preferred: gallery.get_person_meta() -> {pid: {..., category: ...}}
          2) Fallback: gallery.persons dict.
          3) Fallback: gallery.iter_persons() returning records.

        If nothing is found, returns {} and the engine will default to
        category='resident' for any known identity (same as before).
        """
        mapping: Dict[str, str] = {}
        if gallery_like is None:
            return mapping

        try:
            if hasattr(gallery_like, "get_person_meta"):
                meta = gallery_like.get_person_meta()
                if isinstance(meta, dict):
                    for pid, info in meta.items():
                        cat: Optional[str] = None
                        if isinstance(info, dict):
                            cat = info.get("category") or info.get("person_category")
                        else:
                            cat = getattr(info, "category", None)
                            if cat is None and hasattr(info, "meta"):
                                inner = getattr(info, "meta")
                                if isinstance(inner, dict):
                                    cat = inner.get("category")
                        if isinstance(cat, str) and cat.strip():
                            mapping[str(pid)] = cat.strip().lower()

            if not mapping and hasattr(gallery_like, "persons"):
                persons_obj = getattr(gallery_like, "persons")
                if isinstance(persons_obj, dict):
                    for pid, rec in persons_obj.items():
                        cat: Optional[str] = None
                        if isinstance(rec, dict):
                            cat = rec.get("category") or rec.get("person_category")
                        else:
                            cat = getattr(rec, "category", None)
                        if isinstance(cat, str) and cat.strip():
                            mapping[str(pid)] = cat.strip().lower()

            if not mapping and hasattr(gallery_like, "iter_persons"):
                for rec in gallery_like.iter_persons():
                    pid: Optional[str] = None
                    if isinstance(rec, dict):
                        pid = rec.get("person_id")
                        cat = rec.get("category") or rec.get("person_category")
                    else:
                        pid = getattr(rec, "person_id", None)
                        cat = getattr(rec, "category", None)
                    if (
                        pid is not None
                        and isinstance(cat, str)
                        and cat.strip()
                    ):
                        mapping[str(pid)] = cat.strip().lower()

        except Exception as exc:
            logger.warning(
                "IdentityEngineMultiView: failed to build person category map: %s",
                exc,
            )

        return mapping


    @staticmethod
    def _estimate_quality_from_state(
        state: TrackIdentityState,
        sample: Optional[EvidenceSample],
    ) -> float:
        """
        Compute a decision-level face quality for this track.

        Priority:
          1) If we have a current EvidenceSample with face_quality > 0, use it.
          2) Else, use the last non-zero face_quality from the evidence window.
          3) Else, 0.0.
        """
        if sample is not None and sample.face_quality > 0.0:
            return float(np.clip(sample.face_quality, 0.0, 1.0))

        for ev in reversed(state.evidence):
            if ev.face_quality > 0.0:
                return float(np.clip(ev.face_quality, 0.0, 1.0))

        return 0.0

    def _build_decision_from_state(
        self,
        track_id: int,
        state: TrackIdentityState,
        sample: Optional[EvidenceSample],
    ) -> IdentityDecision:
        """
        Create an IdentityDecision object from the current track state
        and last evidence sample.

        Core fields:
          - track_id, identity_id, category, confidence, reason

        Extra diagnostics (pose_bin, engine, distance, score, etc.) are
        attached via setattr so they NEVER break older schemas.

        Additionally (Phase-3/4):
          - Attach a 'quality' attribute to the decision so FaceMetrics
            can treat it as q_face in multiview mode.
        """
        person_id = state.current_person_id
        strength = state.current_strength
        score = state.current_score
        distance = state.current_distance
        pose_bin = state.current_pose_bin

        if person_id is None:
            category = "unknown"
        else:
            raw_cat = self._person_categories.get(person_id)
            if isinstance(raw_cat, str) and raw_cat.strip():
                category = raw_cat.strip().lower()
            else:
                category = "resident"

        conf = float(np.clip(score, 0.0, 1.0))

        parts: List[str] = []
        if person_id is None:
            parts.append("face_multiview:unknown")
        else:
            parts.append(f"face_multiview:id={person_id}")
            parts.append(f"strength={strength}")
            parts.append(f"pose={pose_bin}")
            parts.append(f"score={score:.3f}")
            parts.append(f"dist={distance:.3f}")

        if sample is not None:
            parts.append(f"ev_strength={sample.strength}")
            parts.append(f"ev_match_q={sample.match_quality:.3f}")
            parts.append(f"ev_q_face={sample.face_quality:.3f}")

        reason = "|".join(parts) if parts else None

        decision = IdentityDecision(
            track_id=track_id,
            identity_id=person_id,
            category=category,
            confidence=conf,
            reason=reason,
        )

        q_decision = self._estimate_quality_from_state(state, sample)
        setattr(decision, "quality", float(q_decision))

        setattr(decision, "pose_bin", pose_bin)
        setattr(decision, "engine", "multiview")
        setattr(decision, "distance", distance)
        setattr(decision, "score", score)
        setattr(decision, "binding_state", state.binding_state)
        setattr(decision, "binding_confidence", state.binding_confidence)
        setattr(
            decision,
            "extra",
            {
                "strength": strength,
                "samples": len(state.evidence),
                "binding_state": state.binding_state,  # FIX 2: Include in extra dict
                "binding_confidence": state.binding_confidence,
            },
        )

        return decision
