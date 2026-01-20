"""
Gait Engine: Integrates GaitExtractor and GaitGallery to process Tracklets
with pose data and produce identity recognition decisions based on gait.

DEEP ROBUST DESIGN:
1. Clear separation: Extract → Match → Decide
2. Uses main system's schemas for interoperability
3. Quality-gated processing (skip low-quality sequences)
4. Detailed logging for debugging
"""
from __future__ import annotations
import logging
from typing import List, Optional, Dict
import numpy as np

from schemas import Frame, Tracklet
from schemas.id_signals import IdSignal
from schemas.identity_decision import IdentityDecision

from gait_subsystem.gait.config import GaitConfig, default_gait_config, GaitRobustConfig
from gait_subsystem.gait.gait_extractor import GaitExtractor
from gait_subsystem.gait.gait_gallery import GaitGallery
from gait_subsystem.gait.state import GaitTrackState, GaitState, GaitReason
import time

logger = logging.getLogger(__name__)


class GaitEngine:
    """
    Main orchestrator for gait-based identity recognition.
    
    DEEP ROBUST DESIGN:
    1. Lazy initialization of heavy models
    2. Per-track quality checks before processing
    3. Threshold-based accept/reject decisions
    4. Integration with main system's IdentityDecision format
    """
    
    def __init__(self, config: Optional[GaitConfig] = None):
        """
        Initialize the GaitEngine with configuration.
        
        Args:
            config: Optional custom configuration. Uses default if None.
        """
        self.config = config or default_gait_config()
        self.extractor = GaitExtractor(self.config)
        self.gallery = GaitGallery(self.config)
        self._track_states: Dict[int, GaitTrackState] = {} 
        logger.info("✅ GaitEngine initialized with Deep Robust logic")

    def _get_or_create_state(self, track_id: int) -> GaitTrackState:
        if track_id not in self._track_states:
            self._track_states[track_id] = GaitTrackState(track_id=track_id)
        return self._track_states[track_id]

    def _compute_quality(self, track: Tracklet) -> float:
        """
        Compute robust sequence quality scalar (q_seq).
        Gold Spec D2: Visibility ratio + Core Anchors presence.
        """
        if not track.gait_sequence_data:
            return 0.0
            
        recent = track.gait_sequence_data[-30:]
        total_score = 0.0
        
        for kp in recent:
            hits = (kp[:, 2] > 0.4).sum()
            vis_ratio = hits / 17.0
            
            core_conf = kp[[5,6,11,12], 2].mean()

            leg_conf = kp[[13,14,15,16], 2].mean()
            
            base_score = 0.5 * vis_ratio + 0.3 * core_conf + 0.2 * leg_conf
            
            
            if leg_conf < 0.45:
                frame_score = 0.1
            else:
                frame_score = base_score
            
            total_score += frame_score
            
        return total_score / len(recent)

    def _check_regime(self, track: Tracklet) -> bool:
        """
        Regime Gate: Check for Motion and Validity.
        Gold Spec D3.
        Returns True if regime is valid (moving), False if STILL/CHAOS.
        """
        if not track.gait_sequence_data or len(track.gait_sequence_data) < 5:
            return False
            
        
        hist = track.gait_sequence_data[-15:]
        if len(hist) < 5: return False
        
        diffs = []
        
        
        for frame_kp in hist:
            hip_y = frame_kp[[11,12], 1].mean()
            
            la_y = frame_kp[15, 1] - hip_y 
            ra_y = frame_kp[16, 1] - hip_y
            
            diffs.append(la_y - ra_y)

        diffs = np.array(diffs)
        
        curr = track.gait_sequence_data[-1]
        shoulders = curr[[5,6], :2].mean(axis=0)
        hips = curr[[11,12], :2].mean(axis=0)
        torso_h = np.linalg.norm(shoulders - hips)
        
        if torso_h < 10.0: return False
        
        motion_energy = np.std(diffs)
        min_energy = 0.02 * torso_h
        
        
        
        is_moving = motion_energy > min_energy
        
        
        return is_moving

    def update_signals(self, frame: Frame, tracks: List[Tracklet]) -> List[IdSignal]:
        """
        Execute Gold Spec Pipeline: Evidence -> Gate -> Evaluate -> Decide.
        """
        signals = []
        now = time.perf_counter()
        cfg_robust = self.config.robust
        
        for track in tracks:
            state = self._get_or_create_state(track.track_id)
            seq_len = len(track.gait_sequence_data)
            state.q_seq = self._compute_quality(track)
            
            motion_valid = self._check_regime(track)
            quality_ok = state.q_seq >= cfg_robust.quality_min
            len_ok = seq_len >= cfg_robust.min_seq_len
            
            if state.state == GaitState.COLLECTING:
                if len_ok and quality_ok and motion_valid:
                    state.state = GaitState.EVALUATING
                    state.set_reason(GaitReason.HOLD_NEUTRAL)
                    
            elif state.state == GaitState.EVALUATING:
                if not quality_ok:
                    state.set_reason(GaitReason.SKIP_LOW_QUALITY)
                elif not motion_valid:
                    state.set_reason(GaitReason.SKIP_STILL)

            should_eval = (
                state.state in [GaitState.EVALUATING, GaitState.CONFIRMED, GaitState.UNSURE] and
                state.can_evaluate(now, cfg_robust.eval_period) and
                quality_ok and
                motion_valid
            )
            
            match_id = None
            confidence = 0.0
            
            if should_eval:
                state.last_eval_ts = now
                
                embedding, q = self.extractor.extract_gait_embedding_and_quality(track.gait_sequence_data)
                
                prev_state = state.state
                
                if embedding is not None and q >= cfg_robust.quality_min:
                    anthro_stats = self.extractor.extract_anthropometry(track.gait_sequence_data)
                    
                    match_id, confidence, details = self.gallery.search(embedding, anthro_query=anthro_stats)
                    
                    current_best = details.get("best_pid", "Unknown")
                    prev_best = state.best_id
                    
                    if prev_best and current_best != prev_best:
                         state.reset_streak()
                         if prev_state == GaitState.CONFIRMED:
                             state.state = GaitState.UNSURE
                             state.set_reason(GaitReason.UNSURE_CONFLICT)
                    
                    state.update_match(
                        best_id=current_best, 
                        best_sim=details.get("best_sim", 0.0), 
                        best_dist=details.get("best_dist", 1.0)
                    )
                    
                    margin = details.get("margin", 0.0) # Might be missing if 1 candidate
                    if "margin" not in details and confidence > 0:
                        margin = 1.0
                        
                    sim = state.best_sim
                    
                    is_confirm_candidate = (
                        sim >= cfg_robust.threshold_confirm and
                        margin >= cfg_robust.margin_confirm and
                        state.q_seq >= cfg_robust.quality_confirm
                    )
                    
                    if is_confirm_candidate:
                        state.increment_streak()
                        state.bad_eval_streak = 0
                        
                        if state.confirm_streak >= cfg_robust.confirm_streak:
                            state.state = GaitState.CONFIRMED
                            state.set_reason(GaitReason.CONFIRM_STRONG)
                        else:
                            state.set_reason(GaitReason.HOLD_STREAK)
                    else:
                        
                        if sim < cfg_robust.threshold_candidate:
                            state.reset_streak()
                            state.set_reason(GaitReason.REJECT_LOW_SIM)
                            
                            if prev_state == GaitState.CONFIRMED:
                                state.bad_eval_streak += 1
                                if state.bad_eval_streak >= 2:
                                    state.state = GaitState.UNSURE
                                    state.set_reason(GaitReason.REJECT_LOW_SIM)
                            else:
                                state.state = GaitState.EVALUATING 
                                
                        elif sim < cfg_robust.threshold_confirm:
                            state.reset_streak()
                            state.set_reason(GaitReason.HOLD_BORDERLINE)
                            if prev_state == GaitState.CONFIRMED:
                                state.bad_eval_streak += 1
                                if state.bad_eval_streak >= 4:
                                    state.state = GaitState.UNSURE
                                    
                        else:
                            
                            state.set_reason(GaitReason.HOLD_LOW_MARGIN)
                            
                            if prev_state == GaitState.CONFIRMED:
                                if margin < (cfg_robust.margin_confirm * 0.5):
                                    state.bad_eval_streak += 2 
                                else:
                                    state.bad_eval_streak += 1
                                    
                                if state.bad_eval_streak >= 3:
                                    state.state = GaitState.UNSURE
                                    state.set_reason(GaitReason.UNSURE_CONFLICT)
                            
            
             
            if state.state == GaitState.CONFIRMED and not should_eval:
                if not quality_ok:
                    state.bad_eval_streak += 1
                    state.set_reason(GaitReason.UNSURE_QUALITY_DROP)
                elif not motion_valid:
                     state.bad_eval_streak += 0.5
                
                if state.bad_eval_streak >= 5:
                    state.state = GaitState.UNSURE
                    
            
            final_id = None
            final_conf = 0.0
            
            if state.state == GaitState.CONFIRMED:
                final_id = state.best_id
                final_conf = state.best_sim
            elif state.state == GaitState.EVALUATING:
                 if state.best_sim >= cfg_robust.threshold_candidate:
                     final_id = state.best_id
                     final_conf = state.best_sim
            
            
            
            meta = {
                "gait_state": state.state.value,
                "reason": state.reason.value,
                "q_seq": state.q_seq,
                "best_pid": state.best_id,
                "best_sim": state.best_sim,
                "best_dist": state.best_dist,
                "streak": state.confirm_streak,
                "bad_streak": state.bad_eval_streak,
            }
            
            if should_eval and 'details' in locals():
                 meta["second_pid"] = details.get("second_pid")
                 meta["second_sim"] = details.get("second_sim")
                 meta["margin"] = details.get("margin")
                 meta["status"] = details.get("status")
                 if "match_anthro_dist" in details:
                     meta["anthro_dist"] = f"{details['match_anthro_dist']:.2f}"
            
            signal = IdSignal(
                track_id=track.track_id, 
                identity_id=final_id, 
                confidence=final_conf,
                method="gait",
                extra=meta
            )
            signals.append(signal)

        return signals

    def decide(self, signals: List[IdSignal]) -> List[IdentityDecision]:
        """
        Convert raw IdSignals into final IdentityDecisions.
        
        Enriches decisions with category information from gallery
        for display in UI/Overlay.
        
        Args:
            signals: List of gait identification signals
            
        Returns:
            List of IdentityDecision objects for overlay rendering
        """
        decisions = []
        
        for signal in signals:
            identity_id = None
            category = "unknown"
            confidence = 0.0
            
            if signal.identity_id:
                identity_id = signal.identity_id
                confidence = signal.confidence
                
                category = self.gallery.get_category(identity_id)
            
            decision = IdentityDecision(
                track_id=signal.track_id,
                identity_id=identity_id,
                category=category, 
                confidence=confidence,
                reason=f"gait:{confidence:.2f}",
                extra=signal.extra
            )
            decisions.append(decision)
            
        return decisions