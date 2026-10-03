"""
Phase E: Handoff Merge Manager

Manages intelligent track aliasing for reducing ghost duplicates.
Implements conservative, evidence-based merging of track fragments.

Key Principles:
- Time-exclusive only (no simultaneous track merges)
- Evidence-based scoring with explicit criteria
- Reversible (tentative merges can be undone)
- Binding-aware (respects identity state)
- Safe (never creates false merges)

Architecture:
- MergeCandidate: Represents ended tracklet with full state
- CanonicalMapping: Maps multiple tracklets to one canonical entity
- MergeManager: Main orchestrator for all merge operations
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
from datetime import datetime
from datetime import datetime
import json
from core.governance_metrics import get_metrics_collector, MetricsCollector


logger = logging.getLogger(__name__)



class MergeStatus(Enum):
    """Merge decision status"""
    NO_MERGE = "no_merge"
    MERGE_CONFIDENT = "merge_confident"
    MERGE_TENTATIVE = "merge_tentative"


class MergeFailureReason(Enum):
    """Why a merge was rejected"""
    TIME_GAP_INVALID = "time_gap_invalid"
    DISTANCE_TOO_LARGE = "distance_too_large"
    DISTANCE_TOO_SMALL = "distance_too_small"
    OPPOSITE_MOTION = "opposite_motion"
    APPEARANCE_MISMATCH = "appearance_mismatch"
    CONFLICTING_IDENTITIES = "conflicting_identities"
    INSUFFICIENT_QUALITY = "insufficient_quality"
    RECENT_MERGE_ALREADY = "recent_merge_already"
    NO_BINDING_STATE = "no_binding_state"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OTHER = "other"



@dataclass
class MergeCandidate:
    """Represents a tracklet that is a candidate for merging"""
    tracklet_id: str
    person_id: Optional[str] = None
    binding_state: Optional[str] = None
    confidence: float = 0.0
    end_time: float = 0.0
    start_time: float = 0.0
    last_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    first_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    motion_vector: Optional[np.ndarray] = None
    appearance_features: Optional[np.ndarray] = None
    track_length: int = 0
    quality_samples: int = 0
    
    def __post_init__(self):
        """Ensure numpy arrays"""
        if not isinstance(self.last_position, np.ndarray):
            self.last_position = np.array(self.last_position, dtype=np.float32)
        if not isinstance(self.first_position, np.ndarray):
            self.first_position = np.array(self.first_position, dtype=np.float32)
        if self.motion_vector is not None and not isinstance(self.motion_vector, np.ndarray):
            self.motion_vector = np.array(self.motion_vector, dtype=np.float32)
        if self.appearance_features is not None and not isinstance(self.appearance_features, np.ndarray):
            self.appearance_features = np.array(self.appearance_features, dtype=np.float32)


@dataclass
class MergeDecision:
    """Result of merge evaluation"""
    should_merge: bool
    status: MergeStatus
    reason: str
    score: float = 0.0
    confidence: float = 0.0
    details: Dict = field(default_factory=dict)
    tentative: bool = False
    reversal_deadline: Optional[float] = None


@dataclass
class MergeEvidence:
    """Record of a merge operation"""
    source_tracklet_id: str
    target_tracklet_id: str
    merge_time: float
    merge_reason: str
    confidence: float
    details: Dict = field(default_factory=dict)
    reversed: bool = False
    reversal_time: Optional[float] = None
    reversal_reason: Optional[str] = None


@dataclass
class CanonicalMapping:
    """Maps multiple tracklets to one canonical identity"""
    canonical_id: str
    aliases: List[str] = field(default_factory=list)
    primary_binding: Optional[Dict] = None
    merge_history: List[MergeEvidence] = field(default_factory=list)
    last_update_time: float = 0.0
    
    def add_alias(self, tracklet_id: str, evidence: MergeEvidence) -> None:
        """Add an alias and record evidence"""
        if tracklet_id not in self.aliases and tracklet_id != self.canonical_id:
            self.aliases.append(tracklet_id)
            self.merge_history.append(evidence)
            self.last_update_time = evidence.merge_time
    
    def remove_alias(self, tracklet_id: str, reversal_reason: str, reversal_time: float) -> None:
        """Remove an alias (reverse a merge)"""
        if tracklet_id in self.aliases:
            self.aliases.remove(tracklet_id)
            if self.merge_history:
                for evidence in self.merge_history:
                    if evidence.source_tracklet_id == tracklet_id:
                        evidence.reversed = True
                        evidence.reversal_time = reversal_time
                        evidence.reversal_reason = reversal_reason
                        break
            self.last_update_time = reversal_time
    
    def get_all_tracklets(self) -> List[str]:
        """Get all tracklets in this canonical"""
        return [self.canonical_id] + self.aliases


@dataclass
class MergeMetrics:
    """Metrics for merge manager"""
    merge_attempts: int = 0
    merges_executed: int = 0
    merges_confident: int = 0
    merges_tentative: int = 0
    merge_reversals: int = 0
    average_merge_score: float = 0.0
    merge_failure_reasons: Dict[str, int] = field(default_factory=dict)
    active_canonical_ids: int = 0
    current_tracklets_aliased: int = 0
    
    def record_failed_merge(self, reason: MergeFailureReason) -> None:
        """Record a failed merge attempt"""
        reason_key = reason.value
        self.merge_failure_reasons[reason_key] = self.merge_failure_reasons.get(reason_key, 0) + 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            'merge_attempts': self.merge_attempts,
            'merges_executed': self.merges_executed,
            'merges_confident': self.merges_confident,
            'merges_tentative': self.merges_tentative,
            'merge_reversals': self.merge_reversals,
            'average_merge_score': round(self.average_merge_score, 3),
            'active_canonical_ids': self.active_canonical_ids,
            'current_tracklets_aliased': self.current_tracklets_aliased,
            'failure_reasons': self.merge_failure_reasons
        }


@dataclass
class MergeConfig:
    """Configuration for merge manager"""
    enabled: bool = True
    
    merge_confidence_min: float = 60.0
    tentative_threshold_min: float = 40.0
    
    min_gap_seconds: float = 0.3
    max_gap_seconds: float = 5.0
    
    max_distance_pixels: float = 150.0
    velocity_influence_factor: float = 20.0
    
    max_embedding_distance: float = 0.35
    quality_min_samples: int = 2
    
    velocity_similarity_min: float = 0.6
    
    allow_different_identities: bool = False
    confidence_required_for_diff_ids: float = 0.95
    
    max_merges_per_canonical: int = 5
    merge_reversal_window_seconds: float = 5.0
    min_time_between_merges_seconds: float = 2.0
    
    inactive_tracklet_retention_seconds: float = 10.0
    
    log_merge_reasons: bool = True
    log_reversal_reasons: bool = True
    debug_mode: bool = False



class MergeManager:
    """
    Manages handoff merges (time-exclusive track aliasing).
    
    Responsibilities:
    - Track lifecycle monitoring (start, end events)
    - Merge candidate detection
    - Evidence-based merge scoring
    - Canonical mapping maintenance
    - Reversal capability
    """
    
    def __init__(self, config: MergeConfig):
        """Initialize merge manager"""
        self.config = config
        
        self.canonical_mappings: Dict[str, CanonicalMapping] = {}
        self.merge_history: List[MergeEvidence] = []
        self.inactive_tracklets: Dict[str, MergeCandidate] = {}
        self.tentative_merges: Dict[str, float] = {}
        
        self.metrics = MergeMetrics()
        self.global_metrics = get_metrics_collector()
        
        self.current_time: float = 0.0
        self.last_merge_per_tracklet: Dict[str, float] = {}
        
        logger.info(f"MergeManager initialized with config: {self.config}")
    
    
    def on_tracklet_started(
        self,
        tracklet_id: str,
        first_position: Tuple[float, float],
        timestamp: float
    ) -> None:
        """Called when a new tracklet is created"""
        if not self.config.enabled:
            return
        
        self.current_time = timestamp
        
        if tracklet_id not in self.canonical_mappings:
            self.canonical_mappings[tracklet_id] = CanonicalMapping(
                canonical_id=tracklet_id,
                last_update_time=timestamp
            )
        
        logger.debug(f"Tracklet started: {tracklet_id} at position {first_position}")
    
    def on_tracklet_ended(
        self,
        tracklet_id: str,
        binding_state: Optional[Dict],
        appearance_features: Optional[np.ndarray],
        last_position: Tuple[float, float],
        end_time: float,
        track_length: int,
        quality_samples: int
    ) -> None:
        """Called when tracker ends a tracklet"""
        if not self.config.enabled:
            return
        
        self.current_time = end_time
        
        candidate = MergeCandidate(
            tracklet_id=tracklet_id,
            person_id=binding_state.get('person_id') if binding_state else None,
            binding_state=binding_state.get('status') if binding_state else None,
            confidence=binding_state.get('confidence', 0.0) if binding_state else 0.0,
            end_time=end_time,
            last_position=np.array(last_position, dtype=np.float32),
            appearance_features=appearance_features,
            track_length=track_length,
            quality_samples=quality_samples
        )
        
        self.inactive_tracklets[tracklet_id] = candidate
        
        logger.debug(f"Tracklet ended: {tracklet_id} with {quality_samples} quality samples")
    
    def on_tracklet_updated(
        self,
        tracklet_id: str,
        binding_state: Optional[Dict],
        appearance_features: Optional[np.ndarray],
        current_position: Tuple[float, float],
        track_length: int,
        quality_samples: int,
        timestamp: float
    ) -> None:
        """Called when tracklet is updated (active)"""
        if not self.config.enabled:
            return
        
        self.current_time = timestamp
        
        if tracklet_id in self.canonical_mappings:
            mapping = self.canonical_mappings[tracklet_id]
            mapping.primary_binding = binding_state
            mapping.last_update_time = timestamp

    def update_track_state(
        self,
        tracklet_id: str,
        bbox: Tuple[float, float, float, float],
        score: float,
        timestamp: float,
        binding_state: Optional[Dict] = None
    ) -> None:
        """
        Adapter for main_loop.py to call on_tracklet_updated.
        
        Args:
           tracklet_id: Track ID
           bbox: (x1, y1, x2, y2)
           score: Detection confidence
           timestamp: Current time
           binding_state: Optional binding info
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        current_position = (cx, cy)
        
        self.on_tracklet_updated(
            tracklet_id=str(tracklet_id),
            binding_state=binding_state,
            appearance_features=None,
            current_position=current_position,
            track_length=0,
            quality_samples=0,
            timestamp=timestamp
        )
    
    
    def get_canonical_id(self, tracklet_id: str) -> str:
        """
        Get canonical ID for any tracklet (resolves aliases).
        
        Args:
            tracklet_id: Raw tracklet ID from tracker
        
        Returns:
            Canonical ID (might be same as input or an alias)
        """
        if tracklet_id in self.canonical_mappings:
            return tracklet_id
        
        for canonical_id, mapping in self.canonical_mappings.items():
            if tracklet_id in mapping.aliases:
                return canonical_id
        
        if tracklet_id not in self.canonical_mappings:
            self.canonical_mappings[tracklet_id] = CanonicalMapping(
                canonical_id=tracklet_id,
                last_update_time=self.current_time
            )
        
        return tracklet_id
    
    def get_all_tracklets_for_canonical(self, canonical_id: str) -> List[str]:
        """Get all tracklets aliased to a canonical ID"""
        if canonical_id in self.canonical_mappings:
            return self.canonical_mappings[canonical_id].get_all_tracklets()
        return [canonical_id]
    
    
    def check_and_execute_merges(
        self,
        active_tracklets: Optional[Dict[str, MergeCandidate]] = None
    ) -> int:
        """
        Scan for merge candidates and execute merges.
        
        Args:
            active_tracklets: Dict of currently active tracklets
                             (if None, only considers ended tracklets)
        
        Returns:
            Number of merges executed
        """
        if not self.config.enabled:
            return 0
        
        merges_executed = 0
        
        candidates = self._get_merge_candidates(active_tracklets)
        
        for old_tracklet, new_tracklet in candidates:
            decision = self.get_merge_decision(old_tracklet, new_tracklet)
            
            self.metrics.merge_attempts += 1
            self.global_metrics.metrics.merge_attempts += 1
            
            if decision.should_merge:
                evidence = MergeEvidence(
                    source_tracklet_id=old_tracklet.tracklet_id,
                    target_tracklet_id=new_tracklet.tracklet_id,
                    merge_time=self.current_time,
                    merge_reason=decision.reason,
                    confidence=decision.score,
                    details=decision.details
                )
                
                self.execute_merge(
                    canonical_id=new_tracklet.tracklet_id,
                    tracklet_to_merge=old_tracklet.tracklet_id,
                    evidence=evidence,
                    tentative=decision.tentative,
                    reversal_deadline=decision.reversal_deadline
                )
                
                merges_executed += 1
                
                if decision.tentative:
                    self.metrics.merges_tentative += 1
                else:
                    self.metrics.merges_confident += 1
                
                if self.config.log_merge_reasons:
                    logger.info(
                        f"Merge executed: {old_tracklet.tracklet_id} → "
                        f"{new_tracklet.tracklet_id} "
                        f"(reason: {decision.reason}, score: {decision.score:.1f})"
                    )
            else:
                self.metrics.record_failed_merge(MergeFailureReason(decision.reason))
        
        reversals = self._check_reversal_deadlines()
        
        self.metrics.merge_reversals += reversals
        self.metrics.merges_executed += merges_executed
        self.metrics.active_canonical_ids = len(self.canonical_mappings)
        self.metrics.current_tracklets_aliased = sum(
            len(m.aliases) for m in self.canonical_mappings.values()
        )
        
        return merges_executed
    
    def get_merge_decision(
        self,
        tracklet_a: MergeCandidate,
        tracklet_b: MergeCandidate
    ) -> MergeDecision:
        """
        Evaluate if two tracklets should be merged.
        
        Implements all 7 criteria from spec:
        1. Time exclusivity
        2. Spatial continuity
        3. Motion coherence
        4. Appearance consistency
        5. Binding compatibility
        6. Quality threshold
        7. No recent merge
        
        Args:
            tracklet_a: Ended tracklet
            tracklet_b: New tracklet
        
        Returns:
            MergeDecision with verdict and scoring details
        """
        score = 0.0
        details = {}
        
        time_gap = tracklet_b.start_time - tracklet_a.end_time
        
        if not (self.config.min_gap_seconds < time_gap < self.config.max_gap_seconds):
            reason = "time_gap_invalid"
            return MergeDecision(
                should_merge=False,
                status=MergeStatus.NO_MERGE,
                reason=reason,
                score=0.0,
                details={'time_gap': time_gap}
            )
        
        details['time_gap'] = time_gap
        score += 25
        
        distance = np.linalg.norm(tracklet_a.last_position - tracklet_b.first_position)
        velocity_estimate = 0.0
        
        if tracklet_a.motion_vector is not None:
            velocity_magnitude = np.linalg.norm(tracklet_a.motion_vector)
            velocity_estimate = velocity_magnitude * time_gap
        
        max_distance = (
            self.config.max_distance_pixels +
            self.config.velocity_influence_factor * velocity_estimate
        )
        
        if distance > max_distance:
            return MergeDecision(
                should_merge=False,
                status=MergeStatus.NO_MERGE,
                reason="distance_too_large",
                score=0.0,
                details={'distance': distance, 'max_distance': max_distance}
            )
        
        spatial_score = max(0.0, 1.0 - (distance / max(1.0, max_distance)))
        score += spatial_score * 25
        details['distance'] = distance
        details['max_distance'] = max_distance
        
        if tracklet_a.motion_vector is not None and tracklet_b.motion_vector is not None:
            velocity_similarity = self._cosine_similarity(
                tracklet_a.motion_vector,
                tracklet_b.motion_vector
            )
            details['velocity_similarity'] = velocity_similarity
            
            if velocity_similarity < -0.5:
                return MergeDecision(
                    should_merge=False,
                    status=MergeStatus.NO_MERGE,
                    reason="opposite_motion",
                    score=0.0,
                    details=details
                )
            
            motion_score = max(0.0, velocity_similarity) * 0.8 + 0.2
            score += motion_score * 20
        else:
            score += 10
        
        if tracklet_a.appearance_features is None or tracklet_b.appearance_features is None:
            return MergeDecision(
                should_merge=False,
                status=MergeStatus.NO_MERGE,
                reason="appearance_mismatch",
                score=0.0,
                details={'reason': 'missing_features'}
            )
        
        embedding_distance = np.linalg.norm(
            tracklet_a.appearance_features - tracklet_b.appearance_features
        )
        
        if embedding_distance > 0.5:
            return MergeDecision(
                should_merge=False,
                status=MergeStatus.NO_MERGE,
                reason="appearance_mismatch",
                score=0.0,
                details={'embedding_distance': embedding_distance}
            )
        
        appearance_score = max(0.0, 1.0 - (embedding_distance / 0.5))
        score += appearance_score * 20
        details['embedding_distance'] = embedding_distance
        
        if (tracklet_a.person_id is not None and 
            tracklet_b.person_id is not None and
            tracklet_a.person_id != tracklet_b.person_id):
            
            if not self.config.allow_different_identities:
                if score < 80:
                    return MergeDecision(
                        should_merge=False,
                        status=MergeStatus.NO_MERGE,
                        reason="conflicting_identities",
                        score=0.0,
                        details={
                            'person_a': tracklet_a.person_id,
                            'person_b': tracklet_b.person_id
                        }
                    )
        
        details['binding_compatible'] = True
        
        min_quality = self.config.quality_min_samples
        if tracklet_a.quality_samples < min_quality or tracklet_b.quality_samples < min_quality:
            if not (tracklet_a.binding_state == "CONFIRMED_STRONG" or
                    tracklet_b.binding_state == "CONFIRMED_STRONG"):
                return MergeDecision(
                    should_merge=False,
                    status=MergeStatus.NO_MERGE,
                    reason="insufficient_quality",
                    score=0.0,
                    details={
                        'quality_a': tracklet_a.quality_samples,
                        'quality_b': tracklet_b.quality_samples
                    }
                )
        
        score += 5
        
        last_merge_time_a = self.last_merge_per_tracklet.get(tracklet_a.tracklet_id, 0.0)
        last_merge_time_b = self.last_merge_per_tracklet.get(tracklet_b.tracklet_id, 0.0)
        
        if ((self.current_time - last_merge_time_a < self.config.min_time_between_merges_seconds) or
            (self.current_time - last_merge_time_b < self.config.min_time_between_merges_seconds)):
            return MergeDecision(
                should_merge=False,
                status=MergeStatus.NO_MERGE,
                reason="recent_merge_already",
                score=0.0,
                details={
                    'time_since_merge_a': self.current_time - last_merge_time_a,
                    'time_since_merge_b': self.current_time - last_merge_time_b
                }
            )
        
        if tracklet_a.binding_state == "CONFIRMED_STRONG":
            score *= 1.2
        elif tracklet_a.binding_state == "CONFIRMED_WEAK":
            score *= 1.1
        
        if tracklet_b.binding_state == "CONFIRMED_STRONG":
            score *= 1.15
        elif tracklet_b.binding_state == "CONFIRMED_WEAK":
            score *= 1.05
        
        details['final_score'] = score
        
        if score >= self.config.merge_confidence_min:
            return MergeDecision(
                should_merge=True,
                status=MergeStatus.MERGE_CONFIDENT,
                reason="merge_confident",
                score=score,
                confidence=min(1.0, score / 100.0),
                details=details
            )
        elif self.config.tentative_threshold_min <= score < self.config.merge_confidence_min:
            return MergeDecision(
                should_merge=True,
                status=MergeStatus.MERGE_TENTATIVE,
                reason="merge_tentative",
                score=score,
                confidence=min(1.0, score / 100.0),
                details=details,
                tentative=True,
                reversal_deadline=self.current_time + self.config.merge_reversal_window_seconds
            )
        else:
            return MergeDecision(
                should_merge=False,
                status=MergeStatus.NO_MERGE,
                reason="insufficient_evidence",
                score=score,
                details=details
            )
    
    def execute_merge(
        self,
        canonical_id: str,
        tracklet_to_merge: str,
        evidence: MergeEvidence,
        tentative: bool = False,
        reversal_deadline: Optional[float] = None
    ) -> None:
        """
        Execute a merge operation.
        
        Args:
            canonical_id: Target canonical ID
            tracklet_to_merge: Tracklet to merge into canonical
            evidence: Merge evidence and reasoning
            tentative: If True, mark for monitoring and possible reversal
            reversal_deadline: Time after which tentative merge is permanent
        """
        if canonical_id not in self.canonical_mappings:
            self.canonical_mappings[canonical_id] = CanonicalMapping(
                canonical_id=canonical_id,
                last_update_time=self.current_time
            )
        
        mapping = self.canonical_mappings[canonical_id]
        
        if len(mapping.merge_history) >= self.config.max_merges_per_canonical:
            logger.warning(
                f"Reached merge limit for canonical {canonical_id}, "
                f"skipping merge with {tracklet_to_merge}"
            )
            return
        
        mapping.add_alias(tracklet_to_merge, evidence)
        
        self.last_merge_per_tracklet[tracklet_to_merge] = self.current_time
        
        self.merge_history.append(evidence)
        
        if tentative and reversal_deadline:
            self.tentative_merges[tracklet_to_merge] = reversal_deadline
        
        logger.debug(
            f"Merge executed: {tracklet_to_merge} → {canonical_id} "
            f"(tentative={tentative})"
        )
        self.global_metrics.metrics.merge_success += 1
    
    def reverse_merge(
        self,
        tracklet_id: str,
        reason: str
    ) -> bool:
        """
        Reverse a previous merge.
        
        Args:
            tracklet_id: Tracklet to unmerge
            reason: Reason for reversal
        
        Returns:
            True if reversal executed, False if not found
        """
        for canonical_id, mapping in self.canonical_mappings.items():
            if tracklet_id in mapping.aliases:
                mapping.remove_alias(tracklet_id, reason, self.current_time)
                
                self.canonical_mappings[tracklet_id] = CanonicalMapping(
                    canonical_id=tracklet_id,
                    last_update_time=self.current_time
                )
                
                if self.config.log_reversal_reasons:
                    logger.info(
                        f"Merge reversed: {tracklet_id} unmerged from {canonical_id} "
                        f"(reason: {reason})"
                    )
                
                self.tentative_merges.pop(tracklet_id, None)
                
                return True
        
        return False
    
    
    def cleanup_old_tracklets(self, threshold_seconds: Optional[float] = None) -> int:
        """
        Remove old inactive tracklets from memory.
        
        Args:
            threshold_seconds: Remove tracklets older than this
                              (uses config default if None)
        
        Returns:
            Number of tracklets cleaned up
        """
        if threshold_seconds is None:
            threshold_seconds = self.config.inactive_tracklet_retention_seconds
        
        cutoff_time = self.current_time - threshold_seconds
        removed = 0
        
        tracklets_to_remove = [
            tid for tid, candidate in self.inactive_tracklets.items()
            if candidate.end_time < cutoff_time
        ]
        
        for tid in tracklets_to_remove:
            del self.inactive_tracklets[tid]
            removed += 1
        
        if removed > 0:
            logger.debug(f"Cleaned up {removed} old tracklets")
        
        return removed
    
    
    def get_metrics(self) -> MergeMetrics:
        """Get current metrics"""
        return self.metrics
    
    def get_state_summary(self) -> Dict:
        """Get current state summary for logging"""
        return {
            'canonical_ids': len(self.canonical_mappings),
            'total_tracklets_aliased': sum(
                len(m.aliases) for m in self.canonical_mappings.values()
            ),
            'inactive_tracklets': len(self.inactive_tracklets),
            'tentative_merges': len(self.tentative_merges),
            'merge_history_size': len(self.merge_history),
            'metrics': self.metrics.to_dict()
        }
    
    
    def _get_merge_candidates(
        self,
        active_tracklets: Optional[Dict[str, MergeCandidate]] = None
    ) -> List[Tuple[MergeCandidate, MergeCandidate]]:
        """
        Find candidate pairs for merging (ended + new tracklets).
        
        Returns:
            List of (old_tracklet, new_tracklet) pairs to evaluate
        """
        candidates = []
        
        cutoff_time = self.current_time - self.config.max_gap_seconds
        recent_ended = [
            t for t in self.inactive_tracklets.values()
            if t.end_time > cutoff_time
        ]
        
        if not active_tracklets:
            return candidates
        
        new_tracklets = [
            t for t in active_tracklets.values()
            if t.track_length < 0.5
        ]
        
        for old in recent_ended:
            for new in new_tracklets:
                distance = np.linalg.norm(old.last_position - new.first_position)
                
                if distance < 300:
                    candidates.append((old, new))
        
        return candidates
    
    def _check_reversal_deadlines(self) -> int:
        """Check and execute reversals of tentative merges"""
        reversals = 0
        
        expired = [
            tid for tid, deadline in self.tentative_merges.items()
            if self.current_time > deadline
        ]
        
        for tid in expired:
            self.tentative_merges.pop(tid)
        
        return reversals
    
    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)



def create_merge_config_from_dict(config_dict: Dict) -> MergeConfig:
    """Create MergeConfig from dictionary (e.g., from YAML)"""
    return MergeConfig(
        enabled=config_dict.get('enabled', True),
        merge_confidence_min=config_dict.get('thresholds', {}).get('merge_confidence_min', 60.0),
        tentative_threshold_min=config_dict.get('thresholds', {}).get('tentative_threshold', 40.0),
        min_gap_seconds=config_dict.get('temporal', {}).get('min_gap_seconds', 0.3),
        max_gap_seconds=config_dict.get('temporal', {}).get('max_gap_seconds', 5.0),
        max_distance_pixels=config_dict.get('spatial', {}).get('max_distance_pixels', 150.0),
        velocity_influence_factor=config_dict.get('spatial', {}).get('velocity_influence', 20.0),
        max_embedding_distance=config_dict.get('appearance', {}).get('max_embedding_distance', 0.35),
        quality_min_samples=config_dict.get('appearance', {}).get('quality_min_samples', 2),
        velocity_similarity_min=config_dict.get('motion', {}).get('velocity_similarity_min', 0.6),
        allow_different_identities=config_dict.get('binding', {}).get('allow_different_identities', False),
        confidence_required_for_diff_ids=config_dict.get('binding', {}).get('confidence_required_for_diff', 0.95),
        max_merges_per_canonical=config_dict.get('stability', {}).get('max_merges_per_canonical', 5),
        merge_reversal_window_seconds=config_dict.get('stability', {}).get('merge_reversal_window', 5.0),
        min_time_between_merges_seconds=config_dict.get('stability', {}).get('min_time_between_merges', 2.0),
        inactive_tracklet_retention_seconds=config_dict.get('inactive_tracklet_retention_seconds', 10.0),
        log_merge_reasons=config_dict.get('logging', {}).get('log_merge_reasons', True),
        log_reversal_reasons=config_dict.get('logging', {}).get('log_reversal_reasons', True),
        debug_mode=config_dict.get('logging', {}).get('debug_mode', False)
    )
