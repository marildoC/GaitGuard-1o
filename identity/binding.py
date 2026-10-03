
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Deque, Any, Tuple, List
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)



class BindingState(Enum):
    """Binding state machine states."""
    UNKNOWN = "UNKNOWN"
    PENDING = "PENDING"
    CONFIRMED_WEAK = "CONFIRMED_WEAK"
    CONFIRMED_STRONG = "CONFIRMED_STRONG"
    SWITCH_PENDING = "SWITCH_PENDING"
    STALE = "STALE"


@dataclass
class EvidenceRecord:
    """One face observation with identity information."""
    person_id: Optional[str]
    score: float
    second_best_score: float
    margin: float
    quality: float
    timestamp: float


@dataclass
class BindingDecision:
    """Output of binding state machine."""
    track_id: int
    person_id: Optional[str]
    binding_state: str
    confidence: float
    reason: str
    margin: float = 0.0
    state_changed: bool = False
    extra: Optional[Dict[str, Any]] = None


@dataclass
class TrackBindingState:
    """Per-track binding state."""
    track_id: int
    state: BindingState
    person_id: Optional[str]
    confidence: float
    evidence_buffer: Deque[EvidenceRecord] = field(default_factory=deque)
    contradiction_counter: int = 0
    last_update_ts: float = 0.0
    state_enter_ts: float = 0.0
    pending_person_id: Optional[str] = None
    last_margin: float = 0.0



class BindingManager:
    """
    State machine for identity binding.
    
    Maintains per-track binding state and enforces evidence-based
    transitions with margin protection and contradiction detection.
    
    Never crashes (all exceptions caught).
    Configuration-driven (all thresholds tunable).
    Metrics-integrated (all events recorded).
    """
    
    def __init__(
        self,
        cfg: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
    ) -> None:
        """
        Initialize Binding Manager.
        
        Args:
            cfg: Config object with governance.binding section
            metrics_collector: MetricsCollector for recording events
        """
        self.cfg = cfg
        self.metrics_collector = metrics_collector
        
        try:
            binding_config = None
            
            if cfg and hasattr(cfg, 'governance') and hasattr(cfg.governance, 'binding'):
                binding_config = cfg.governance.binding
            
            elif cfg and hasattr(cfg, 'confirmation') and hasattr(cfg, 'switching'):
                 binding_config = cfg
            
            elif cfg and hasattr(cfg, 'enabled') and not hasattr(cfg, 'governance'):
                 binding_config = cfg

            if binding_config:
                self.enabled = getattr(binding_config, 'enabled', True)
                self.confirmation = getattr(binding_config, 'confirmation', None)
                self.switching = getattr(binding_config, 'switching', None)
                self.contradiction = getattr(binding_config, 'contradiction', None)
            else:
                logger.info("BindingManager: No specific config found, defaulting to ENABLED")
                self.enabled = True
                self.confirmation = None
                self.switching = None
                self.contradiction = None
                
        except Exception as e:
            logger.warning(f"BindingManager init config error: {e}")
            self.enabled = False
            self.confirmation = None
            self.switching = None
            self.contradiction = None
        
        if self.confirmation is None:
            self.confirmation = self._default_confirmation_config()
        if self.switching is None:
            self.switching = self._default_switching_config()
        if self.contradiction is None:
            self.contradiction = self._default_contradiction_config()
        
        self._track_states: Dict[int, TrackBindingState] = {}
        
        logger.info(
            f"BindingManager initialized | enabled={self.enabled} | "
            f"min_samples_strong={self._get_threshold('confirmation.min_samples_strong', 3)}"
        )
    
    
    def process_evidence(
        self,
        track_id: int,
        person_id: Optional[str],
        score: float,
        second_best_score: float,
        quality: float,
        timestamp: float,
    ) -> BindingDecision:
        """
        Process a new face sample for a track.
        
        Args:
            track_id: Track identifier
            person_id: Gallery match result (or None)
            score: Match score (0-1)
            second_best_score: Next best score
            quality: Face quality from Phase B
            timestamp: Frame timestamp
        
        Returns:
            BindingDecision with state, person_id, confidence, reason
        
        Safety:
            - Never crashes (all exceptions caught)
            - Returns safe default on errors
            - Always records metrics
        """
        try:
            if not self.enabled:
                return BindingDecision(
                    track_id=track_id,
                    person_id=person_id,
                    binding_state="BYPASS",
                    confidence=1.0 if person_id else 0.0,
                    reason="binding_disabled"
                )
            
            track_state = self._get_or_create_track_state(track_id, timestamp)
            
            margin = score - second_best_score
            evidence = EvidenceRecord(
                person_id=person_id,
                score=score,
                second_best_score=second_best_score,
                margin=margin,
                quality=quality,
                timestamp=timestamp
            )
            
            track_state.evidence_buffer.append(evidence)
            self._prune_evidence_buffer(track_state, timestamp)
            
            old_state = track_state.state
            self._apply_state_transitions(track_state, timestamp)
            new_state = track_state.state
            
            decision = BindingDecision(
                track_id=track_id,
                person_id=track_state.person_id,
                binding_state=new_state.value,
                confidence=track_state.confidence,
                reason=self._get_transition_reason(old_state, new_state),
                margin=margin,
                state_changed=(old_state != new_state)
            )
            
            self._record_metrics(decision, old_state, new_state)
            
            return decision
        
        except Exception as e:
            logger.error(f"BindingManager.process_evidence exception: {e}", exc_info=True)
            return BindingDecision(
                track_id=track_id,
                person_id=None,
                binding_state="ERROR",
                confidence=0.0,
                reason=f"error_exception: {str(e)}"
            )
    
    def get_binding_state(self, track_id: int) -> str:
        """Get current binding state for track."""
        if track_id not in self._track_states:
            return BindingState.UNKNOWN.value
        return self._track_states[track_id].state.value
    
    def get_person_id(self, track_id: int) -> Optional[str]:
        """Get bound person_id for track."""
        if track_id not in self._track_states:
            return None
        return self._track_states[track_id].person_id
    
    def get_confidence(self, track_id: int) -> float:
        """Get confidence in binding."""
        if track_id not in self._track_states:
            return 0.0
        return self._track_states[track_id].confidence
    
    def get_all_states(self) -> Dict[int, str]:
        """
        Phase D: Get binding states for all tracked identities.
        
        Used by scheduler to prioritize which tracks to process.
        Tracks with PENDING state are processed more frequently.
        
        Returns:
            Dict[int, str]: mapping of track_id -> binding_state
                States: "UNKNOWN", "PENDING", "CONFIRMED_WEAK", "CONFIRMED_STRONG"
        """
        return {
            track_id: state.state.value
            for track_id, state in self._track_states.items()
        }
    
    def cleanup_stale_tracks(self, current_ts: float, max_age_sec: float = 30.0) -> None:
        """Remove tracks that haven't been updated in max_age_sec."""
        stale_tracks = [
            tid for tid, state in self._track_states.items()
            if (current_ts - state.last_update_ts) > max_age_sec
        ]
        for tid in stale_tracks:
            del self._track_states[tid]
        if stale_tracks:
            logger.debug(f"Cleaned up {len(stale_tracks)} stale binding states")
    
    def reset(self) -> None:
        """Reset all binding states."""
        self._track_states.clear()
        logger.info("BindingManager reset")
    
    
    def _apply_state_transitions(
        self,
        track_state: TrackBindingState,
        current_ts: float,
    ) -> None:
        """Apply state transition rules."""
        try:
            current_state = track_state.state
            
            if current_state == BindingState.UNKNOWN:
                self._transition_from_unknown(track_state, current_ts)
            elif current_state == BindingState.PENDING:
                self._transition_from_pending(track_state, current_ts)
            elif current_state == BindingState.CONFIRMED_WEAK:
                self._transition_from_confirmed_weak(track_state, current_ts)
            elif current_state == BindingState.CONFIRMED_STRONG:
                self._transition_from_confirmed_strong(track_state, current_ts)
            elif current_state == BindingState.SWITCH_PENDING:
                self._transition_from_switch_pending(track_state, current_ts)
            
            self._check_contradiction_downgrade(track_state, current_ts)
        
        except Exception as e:
            logger.error(f"State transition error: {e}", exc_info=True)
    
    def _transition_from_unknown(
        self,
        track_state: TrackBindingState,
        current_ts: float,
    ) -> None:
        """UNKNOWN → PENDING or CONFIRMED_WEAK"""
        try:
            buffer = track_state.evidence_buffer
            if not buffer:
                return
            
            
            strong_samples = []
            for e in buffer:
                if e.person_id is None:
                    continue
                
                is_connected = False
                
                if (e.score >= self._get_threshold('confirmation.min_avg_score', 0.45) and 
                    e.margin >= self._get_threshold('confirmation.min_avg_margin', 0.02)):
                    is_connected = True
                    
                elif (e.score >= 0.45 and e.margin >= 0.10):
                    is_connected = True
                
                if is_connected and e.quality >= self._get_threshold('confirmation.min_quality_for_strong', 0.40):
                     strong_samples.append(e)
            
            min_samples = self._get_threshold('confirmation.min_samples_strong', 1)
            window_sec = self._get_threshold('confirmation.window_seconds', 3.0)
            
            if len(strong_samples) >= min_samples:
                min_ts = current_ts - window_sec
                recent_strong = [e for e in strong_samples if e.timestamp >= min_ts]
                
                if len(recent_strong) >= min_samples:
                    persons = set(e.person_id for e in recent_strong)
                    if len(persons) == 1:
                        person_id = recent_strong[0].person_id
                        avg_margin = sum(e.margin for e in recent_strong) / len(recent_strong)
                        avg_score = sum(e.score for e in recent_strong) / len(recent_strong)
                        
                        if avg_margin >= self._get_threshold('confirmation.min_avg_margin', 0.05):
                            track_state.state = BindingState.PENDING
                            track_state.person_id = person_id
                            track_state.confidence = 0.5
                            track_state.state_enter_ts = current_ts
                            logger.info(
                                f"TRANSITION SUCCESS: Track {track_state.track_id} UNKNOWN -> PENDING {person_id} "
                                f"(score={avg_score:.3f}, margin={avg_margin:.3f}, samples={len(recent_strong)})"
                            )
                        else:
                            logger.info(f"TRANSITION FAIL: Margin too low ({avg_margin:.3f} < 0.05)")
                    else:
                        logger.info(f"TRANSITION FAIL: Mixed identities in window: {persons}")
                else:
                    logger.info(f"TRANSITION FAIL: Not enough recent strong samples ({len(recent_strong)} < {min_samples})")
            else:
                logger.info(f"TRANSITION FAIL: Not enough total strong samples ({len(strong_samples)} < {min_samples})")
            
            
            
            '''
            # OPTION C: Close-up adaptive thresholds
            # Detect if this is a close-up scenario based on quality (close faces have higher quality)
            avg_quality_in_buffer = sum(e.quality for e in buffer) / len(buffer) if buffer else 0.0
            is_close_up = avg_quality_in_buffer >= 0.60  # High quality = close to camera
            
            # Choose thresholds based on close-up detection
            if is_close_up:
                # RELAXED thresholds for close-up faces
                CLOSE_UP_MIN_SCORE = 0.55
                CLOSE_UP_MIN_MARGIN = 0.03
                CLOSE_UP_MIN_QUALITY = 0.45
                CLOSE_UP_MIN_SAMPLES = 2
                logger.debug(f"CLOSE-UP MODE: Track {track_state.track_id} avg_quality={avg_quality_in_buffer:.3f}")
            else:
                # STANDARD thresholds for normal distance
                CLOSE_UP_MIN_SCORE = self._get_threshold('confirmation.min_avg_score', 0.60)
                CLOSE_UP_MIN_MARGIN = self._get_threshold('confirmation.min_avg_margin', 0.05)
                CLOSE_UP_MIN_QUALITY = self._get_threshold('confirmation.min_quality_for_strong', 0.50)
                CLOSE_UP_MIN_SAMPLES = int(self._get_threshold('confirmation.min_samples_strong', 3))
            
            # Count strong samples using adaptive thresholds
            strong_samples = []
            for e in buffer:
                if e.person_id is None:
                    continue
                
                is_connected = False
                
                # Tier 1: Match using adaptive thresholds
                if (e.score >= CLOSE_UP_MIN_SCORE and e.margin >= CLOSE_UP_MIN_MARGIN):
                    is_connected = True
                
                # Tier 2: High margin fallback (unchanged)
                elif (e.score >= 0.50 and e.margin >= 0.20):
                    is_connected = True
                
                # Quality gate using adaptive threshold
                if is_connected and e.quality >= CLOSE_UP_MIN_QUALITY:
                    strong_samples.append(e)
            
            min_samples = CLOSE_UP_MIN_SAMPLES
            window_sec = self._get_threshold('confirmation.window_seconds', 3.0)
            
            # Check if all strong samples are recent and same person
            if len(strong_samples) >= min_samples:
                min_ts = current_ts - window_sec
                recent_strong = [e for e in strong_samples if e.timestamp >= min_ts]
                
                if len(recent_strong) >= min_samples:
                    persons = set(e.person_id for e in recent_strong)
                    if len(persons) == 1:
                        person_id = recent_strong[0].person_id
                        avg_margin = sum(e.margin for e in recent_strong) / len(recent_strong)
                        avg_score = sum(e.score for e in recent_strong) / len(recent_strong)
                        
                        if avg_margin >= CLOSE_UP_MIN_MARGIN:
                            track_state.state = BindingState.PENDING
                            track_state.person_id = person_id
                            track_state.confidence = 0.5
                            track_state.state_enter_ts = current_ts
                            mode_tag = "[CLOSE-UP]" if is_close_up else "[NORMAL]"
                            logger.info(
                                f"TRANSITION SUCCESS {mode_tag}: Track {track_state.track_id} UNKNOWN -> PENDING {person_id} "
                                f"(score={avg_score:.3f}, margin={avg_margin:.3f}, samples={len(recent_strong)})"
                            )
                        else:
                            logger.info(f"TRANSITION FAIL: Margin too low ({avg_margin:.3f} < {CLOSE_UP_MIN_MARGIN})")
                    else:
                        logger.info(f"TRANSITION FAIL: Mixed identities in window: {persons}")
                else:
                    logger.info(f"TRANSITION FAIL: Not enough recent strong samples ({len(recent_strong)} < {min_samples})")
            else:
                logger.info(f"TRANSITION FAIL: Not enough total strong samples ({len(strong_samples)} < {min_samples})")
            '''
        
        except Exception as e:
            logger.error(f"Transition UNKNOWN error: {e}", exc_info=True)
    
    def _transition_from_pending(
        self,
        track_state: TrackBindingState,
        current_ts: float,
    ) -> None:
        """PENDING → CONFIRMED_WEAK or CONFIRMED_STRONG"""
        try:
            buffer = track_state.evidence_buffer
            if not buffer or track_state.person_id is None:
                return
            
            confirm_time = self._get_threshold('confirmation.window_seconds', 3.0)
            min_ts = current_ts - confirm_time
            
            recent_matches = []
            for e in buffer:
                if e.timestamp < min_ts or e.person_id != track_state.person_id:
                    continue
                    
                is_valid = False
                if (e.score >= self._get_threshold('confirmation.min_avg_score', 0.60)):
                    is_valid = True
                elif (e.score >= 0.50 and e.margin >= 0.20):
                    is_valid = True
                    
                if is_valid:
                    recent_matches.append(e)
            
            min_samples = self._get_threshold('confirmation.min_samples_strong', 3)
            
            if len(recent_matches) >= min_samples:
                avg_margin = sum(e.margin for e in recent_matches) / len(recent_matches)
                avg_score = sum(e.score for e in recent_matches) / len(recent_matches)
                
                if avg_margin >= self._get_threshold('confirmation.min_avg_margin', 0.05):
                    track_state.state = BindingState.CONFIRMED_WEAK
                    track_state.confidence = min(0.75, avg_score)
                    track_state.state_enter_ts = current_ts
                    logger.debug(
                        f"Track {track_state.track_id} PENDING → CONFIRMED_WEAK "
                        f"{track_state.person_id} (conf={track_state.confidence:.3f})"
                    )
        
        except Exception as e:
            logger.error(f"Transition PENDING error: {e}", exc_info=True)
    
    def _transition_from_confirmed_weak(
        self,
        track_state: TrackBindingState,
        current_ts: float,
    ) -> None:
        """CONFIRMED_WEAK → CONFIRMED_STRONG or SWITCH_PENDING"""
        try:
            buffer = track_state.evidence_buffer
            if not buffer or track_state.person_id is None:
                return
            
            confirm_time = self._get_threshold('confirmation.window_seconds', 3.0)
            min_ts = current_ts - confirm_time
            
            recent_matches = [
                e for e in buffer
                if (e.timestamp >= min_ts and
                    e.person_id == track_state.person_id)
            ]
            
            min_samples = self._get_threshold('confirmation.min_samples_strong', 3)
            
            if len(recent_matches) >= min_samples:
                avg_score = sum(e.score for e in recent_matches) / len(recent_matches)
                if avg_score >= self._get_threshold('confirmation.min_avg_score', 0.75):
                    track_state.state = BindingState.CONFIRMED_STRONG
                    track_state.confidence = min(0.95, avg_score)
                    track_state.state_enter_ts = current_ts
                    logger.debug(
                        f"Track {track_state.track_id} CONFIRMED_WEAK → CONFIRMED_STRONG "
                        f"(conf={track_state.confidence:.3f})"
                    )
            
            self._check_switch_candidate(track_state, current_ts)
        
        except Exception as e:
            logger.error(f"Transition CONFIRMED_WEAK error: {e}", exc_info=True)
    
    def _transition_from_confirmed_strong(
        self,
        track_state: TrackBindingState,
        current_ts: float,
    ) -> None:
        """CONFIRMED_STRONG → SWITCH_PENDING (or stay)"""
        self._check_switch_candidate(track_state, current_ts)
    
    def _transition_from_switch_pending(
        self,
        track_state: TrackBindingState,
        current_ts: float,
    ) -> None:
        """SWITCH_PENDING → CONFIRMED_STRONG (new person) or fallback"""
        try:
            if track_state.pending_person_id is None:
                return
            
            switch_time = self._get_threshold('switching.window_seconds', 2.0)
            min_ts = current_ts - switch_time
            
            pending_matches = [
                e for e in track_state.evidence_buffer
                if (e.timestamp >= min_ts and
                    e.person_id == track_state.pending_person_id)
            ]
            
            min_samples = self._get_threshold('switching.min_sustained_samples', 4)
            
            if len(pending_matches) >= min_samples:
                track_state.person_id = track_state.pending_person_id
                track_state.pending_person_id = None
                track_state.state = BindingState.CONFIRMED_STRONG
                track_state.confidence = 0.75
                track_state.state_enter_ts = current_ts
                track_state.contradiction_counter = 0
                logger.debug(
                    f"Track {track_state.track_id} switched to {track_state.person_id}"
                )
            else:
                timeout_sec = self._get_threshold('switching.timeout_seconds', 5.0)
                if (current_ts - track_state.state_enter_ts) > timeout_sec:
                    track_state.pending_person_id = None
                    track_state.state = BindingState.CONFIRMED_WEAK
                    logger.debug(
                        f"Track {track_state.track_id} switch attempt timed out"
                    )
        
        except Exception as e:
            logger.error(f"Transition SWITCH_PENDING error: {e}", exc_info=True)
    
    def _check_switch_candidate(
        self,
        track_state: TrackBindingState,
        current_ts: float,
    ) -> None:
        """Check if alternative person deserves switch attempt."""
        try:
            if track_state.person_id is None:
                return
            
            alt_person_id = None
            alt_samples = []
            
            for person_id, samples in self._group_by_person(track_state.evidence_buffer).items():
                if person_id != track_state.person_id and person_id is not None:
                    alt_samples = samples
                    alt_person_id = person_id
                    break
            
            if not alt_samples:
                return
            
            required_margin = self._get_threshold('switching.margin_advantage', 0.12)
            min_samples = self._get_threshold('switching.min_sustained_samples', 4)
            
            switch_time = self._get_threshold('switching.window_seconds', 2.0)
            min_ts = current_ts - switch_time
            recent_alt = [e for e in alt_samples if e.timestamp >= min_ts]
            
            if len(recent_alt) >= min_samples:
                avg_alt_score = sum(e.score for e in recent_alt) / len(recent_alt)
                
                current_samples = [
                    e for e in track_state.evidence_buffer
                    if (e.timestamp >= min_ts and
                        e.person_id == track_state.person_id)
                ]
                
                if current_samples:
                    avg_current = sum(e.score for e in current_samples) / len(current_samples)
                    margin_advantage = avg_alt_score - avg_current
                    
                    if margin_advantage > required_margin:
                        track_state.state = BindingState.SWITCH_PENDING
                        track_state.pending_person_id = alt_person_id
                        track_state.state_enter_ts = current_ts
                        logger.debug(
                            f"Track {track_state.track_id} SWITCH_PENDING → {alt_person_id} "
                            f"(margin={margin_advantage:.3f})"
                        )
        
        except Exception as e:
            logger.error(f"Switch candidate check error: {e}", exc_info=True)
    
    def _check_contradiction_downgrade(
        self,
        track_state: TrackBindingState,
        current_ts: float,
    ) -> None:
        """Apply contradiction counter logic (anti-lock-in)."""
        try:
            if track_state.state in [BindingState.UNKNOWN, BindingState.PENDING]:
                return
            
            if track_state.person_id is None:
                return
            
            contradiction_time = self._get_threshold('contradiction.decay_per_second', 1.0)
            min_ts = current_ts - contradiction_time
            recent = [e for e in track_state.evidence_buffer if e.timestamp >= min_ts]
            
            if not recent:
                return
            
            threshold = self._get_threshold('contradiction.threshold', 0.15)
            current_matches = [
                e for e in recent
                if e.person_id == track_state.person_id
            ]
            
            current_avg = sum(e.score for e in current_matches) / len(current_matches) if current_matches else 0.0
            
            if current_avg < threshold:
                track_state.contradiction_counter += 1
            else:
                track_state.contradiction_counter = max(0, track_state.contradiction_counter - 1)
            
            counter_max = self._get_threshold('contradiction.counter_max', 5)
            if track_state.contradiction_counter > counter_max:
                if track_state.state == BindingState.CONFIRMED_STRONG:
                    track_state.state = BindingState.CONFIRMED_WEAK
                    track_state.confidence *= 0.8
                    logger.debug(
                        f"Track {track_state.track_id} downgraded due to contradiction"
                    )
                elif track_state.state == BindingState.CONFIRMED_WEAK:
                    track_state.state = BindingState.PENDING
                    track_state.contradiction_counter = 0
                    logger.debug(
                        f"Track {track_state.track_id} downgraded WEAK → PENDING"
                    )
        
        except Exception as e:
            logger.error(f"Contradiction downgrade error: {e}", exc_info=True)
    
    
    def _get_or_create_track_state(
        self,
        track_id: int,
        current_ts: float,
    ) -> TrackBindingState:
        """Get or create track state."""
        if track_id not in self._track_states:
            self._track_states[track_id] = TrackBindingState(
                track_id=track_id,
                state=BindingState.UNKNOWN,
                person_id=None,
                confidence=0.0,
                last_update_ts=current_ts,
                state_enter_ts=current_ts
            )
        
        self._track_states[track_id].last_update_ts = current_ts
        return self._track_states[track_id]
    
    def _prune_evidence_buffer(
        self,
        track_state: TrackBindingState,
        current_ts: float,
    ) -> None:
        """Remove old evidence from buffer."""
        try:
            max_age = self._get_threshold('confirmation.window_seconds', 5.0)
            min_ts = current_ts - max_age
            
            while track_state.evidence_buffer and track_state.evidence_buffer[0].timestamp < min_ts:
                track_state.evidence_buffer.popleft()
            
            max_size = 8
            while len(track_state.evidence_buffer) > max_size:
                track_state.evidence_buffer.popleft()
        
        except Exception:
            pass
    
    def _group_by_person(
        self,
        buffer: Deque[EvidenceRecord]
    ) -> Dict[Optional[str], List[EvidenceRecord]]:
        """Group evidence records by person."""
        result: Dict[Optional[str], List[EvidenceRecord]] = {}
        for record in buffer:
            if record.person_id not in result:
                result[record.person_id] = []
            result[record.person_id].append(record)
        return result
    
    def _get_threshold(self, path: str, default: float) -> float:
        """Get threshold value with safe defaults."""
        try:
            parts = path.split('.')
            obj = getattr(self, parts[0])
            for part in parts[1:]:
                obj = getattr(obj, part, None)
                if obj is None:
                    return default
            return float(obj)
        except Exception:
            return default
    
    def _get_transition_reason(
        self,
        old_state: BindingState,
        new_state: BindingState,
    ) -> str:
        """Generate reason code for state transition."""
        if old_state == new_state:
            return "no_transition"
        return f"{old_state.value} → {new_state.value}"
    
    def _record_metrics(
        self,
        decision: BindingDecision,
        old_state: BindingState,
        new_state: BindingState,
    ) -> None:
        """Record metrics for this decision."""
        try:
            if not self.metrics_collector:
                return
            
            metrics = self.metrics_collector.metrics
            
            metrics.binding_state_counts[new_state.value] = \
                metrics.binding_state_counts.get(new_state.value, 0) + 1
            
            if old_state != new_state:
                if new_state in [BindingState.CONFIRMED_WEAK, BindingState.CONFIRMED_STRONG]:
                    metrics.record_binding_confirmation()
                
                if (old_state in [BindingState.CONFIRMED_WEAK, BindingState.CONFIRMED_STRONG] and
                    new_state in [BindingState.PENDING, BindingState.UNKNOWN]):
                    metrics.record_binding_downgrade()
                
                if old_state == BindingState.SWITCH_PENDING and new_state == BindingState.CONFIRMED_STRONG:
                    metrics.record_binding_switch(success=True)
        
        except Exception as e:
            logger.warning(f"BindingManager: failed to record metrics: {e}")
    
    
    def _default_confirmation_config(self) -> Any:
        """Default confirmation thresholds."""
        
        class ConfirmationConfig:
            min_samples_strong = 3
            min_samples_weak = 5
            window_seconds = 3.0
            min_avg_score = 0.75
            min_avg_margin = 0.08
            min_quality_for_strong = 0.60
        return ConfirmationConfig()
        
        
        '''
        class ConfirmationConfig:
            min_samples_strong = 2       # CHANGED: 3 → 2 (faster confirmation)
            min_samples_weak = 3         # CHANGED: 5 → 3 (faster fallback)
            window_seconds = 3.0         # UNCHANGED
            min_avg_score = 0.55         # CHANGED: 0.75 → 0.55 (matches multiview weak)
            min_avg_margin = 0.03        # CHANGED: 0.08 → 0.03 (more tolerant)
            min_quality_for_strong = 0.45  # CHANGED: 0.60 → 0.45 (accepts lower quality)
        return ConfirmationConfig()
        '''
    
    def _default_switching_config(self) -> Any:
        """Default switching thresholds."""
        class SwitchingConfig:
            min_sustained_samples = 4
            margin_advantage = 0.12
            window_seconds = 2.0
            timeout_seconds = 5.0
        return SwitchingConfig()
    
    def _default_contradiction_config(self) -> Any:
        """Default contradiction thresholds."""
        class ContradictionConfig:
            threshold = 0.15
            counter_max = 5
            decay_per_second = 1.0
        return ContradictionConfig()
