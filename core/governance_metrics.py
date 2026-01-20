"""
Governance Metrics Collection (Phase A)

Collects structured data on all governance decisions:
- Evidence gate (accept/hold/reject counts)
- Binding state transitions
- Scheduler budget usage
- Merge attempts

Designed for:
1. Real-time monitoring (telemetry)
2. Post-hoc analysis (tuning thresholds)
3. Production debugging (understanding failures)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class GovernanceMetrics:
    """
    Per-second aggregation of governance decisions.
    
    Reset each second after emission; acts as accumulator for frame-by-frame events.
    """
    timestamp: float = field(default_factory=lambda: time.time())
    
    faces_total: int = 0
    faces_accepted: int = 0
    faces_held: int = 0
    faces_rejected: int = 0
    
    reject_reason_counts: Dict[str, int] = field(default_factory=dict)
    
    hold_reason_counts: Dict[str, int] = field(default_factory=dict)
    
    binding_state_counts: Dict[str, int] = field(default_factory=dict)
    
    binding_confirmations: int = 0
    binding_downgrades: int = 0
    binding_switches: int = 0
    binding_switch_failures: int = 0
    binding_contradiction_events: int = 0
    
    scheduler_budget_available: int = 0
    scheduler_selected: int = 0
    scheduler_skipped: int = 0
    scheduler_starved: int = 0
    
    merge_attempts: int = 0
    merge_success: int = 0
    merge_collision_risk: int = 0
    
    fps_estimate: float = 0.0
    track_count: int = 0
    unknown_rate: float = 0.0
    pending_rate: float = 0.0
    confirmed_rate: float = 0.0
    
    def record_face_accepted(self) -> None:
        """Record an accepted face sample"""
        self.faces_total += 1
        self.faces_accepted += 1
    
    def record_face_held(self, reason: str) -> None:
        """Record a held (borderline) face sample"""
        self.faces_total += 1
        self.faces_held += 1
        self.hold_reason_counts[reason] = self.hold_reason_counts.get(reason, 0) + 1
    
    def record_face_rejected(self, reason: str) -> None:
        """Record a rejected face sample"""
        self.faces_total += 1
        self.faces_rejected += 1
        self.reject_reason_counts[reason] = self.reject_reason_counts.get(reason, 0) + 1
    
    def record_binding_confirmation(self) -> None:
        """Record a track confirmed to an identity"""
        self.binding_confirmations += 1
    
    def record_binding_downgrade(self) -> None:
        """Record identity confidence downgrade (contradiction)"""
        self.binding_downgrades += 1
    
    def record_binding_switch(self, success: bool = True) -> None:
        """Record identity switch attempt"""
        if success:
            self.binding_switches += 1
        else:
            self.binding_switch_failures += 1
    
    def record_binding_contradiction(self) -> None:
        """Record contradiction counter event"""
        self.binding_contradiction_events += 1
    
    def record_merge_attempt(self, success: bool = True, collision_risk: bool = False) -> None:
        """Record merge attempt"""
        self.merge_attempts += 1
        if success:
            self.merge_success += 1
        elif collision_risk:
            self.merge_collision_risk += 1
    
    def compute_accept_rate(self) -> float:
        """Compute accept/hold/reject ratio"""
        if self.faces_total == 0:
            return 0.0
        return self.faces_accepted / self.faces_total
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/telemetry"""
        return {
            "timestamp": self.timestamp,
            "faces": {
                "total": self.faces_total,
                "accepted": self.faces_accepted,
                "held": self.faces_held,
                "rejected": self.faces_rejected,
                "accept_rate": self.compute_accept_rate(),
                "reject_reasons": dict(self.reject_reason_counts),
                "hold_reasons": dict(self.hold_reason_counts),
            },
            "binding": {
                "state_counts": dict(self.binding_state_counts),
                "confirmations": self.binding_confirmations,
                "downgrades": self.binding_downgrades,
                "switches": self.binding_switches,
                "switch_failures": self.binding_switch_failures,
                "contradiction_events": self.binding_contradiction_events,
            },
            "scheduler": {
                "budget_available": self.scheduler_budget_available,
                "selected": self.scheduler_selected,
                "skipped": self.scheduler_skipped,
                "starved": self.scheduler_starved,
            },
            "merge": {
                "attempts": self.merge_attempts,
                "success": self.merge_success,
                "collision_risk_rejected": self.merge_collision_risk,
            },
            "system": {
                "fps": self.fps_estimate,
                "track_count": self.track_count,
                "unknown_rate": self.unknown_rate,
                "pending_rate": self.pending_rate,
                "confirmed_rate": self.confirmed_rate,
            }
        }
    
    def reset(self) -> None:
        """Reset all counters (called each second after emission)"""
        self.timestamp = time.time()
        self.faces_total = 0
        self.faces_accepted = 0
        self.faces_held = 0
        self.faces_rejected = 0
        self.reject_reason_counts.clear()
        self.hold_reason_counts.clear()
        
        self.binding_confirmations = 0
        self.binding_downgrades = 0
        self.binding_switches = 0
        self.binding_switch_failures = 0
        self.binding_contradiction_events = 0
        
        self.scheduler_budget_available = 0
        self.scheduler_selected = 0
        self.scheduler_skipped = 0
        self.scheduler_starved = 0
        
        self.merge_attempts = 0
        self.merge_success = 0
        self.merge_collision_risk = 0
        


class MetricsCollector:
    """
    Global metrics collector with thread-safe emission.
    
    Usage:
        collector = MetricsCollector(interval_sec=1.0)
        
        # Record events
        collector.metrics.record_face_accepted()
        
        # Periodic emission
        collector.maybe_emit()
    """
    
    def __init__(self, interval_sec: float = 1.0):
        self.metrics = GovernanceMetrics()
        self.interval_sec = interval_sec
        self.last_emit_time = time.time()
    
    def maybe_emit(self) -> bool:
        """
        Check if it's time to emit metrics.
        
        Returns True if emitted, False otherwise.
        """
        now = time.time()
        if now - self.last_emit_time >= self.interval_sec:
            metrics_dict = self.metrics.to_dict()
            logger.info(
                "Governance Metrics: faces=%d (accept=%d, hold=%d, reject=%d) | "
                "binding: %s | scheduler: %d/%d | merge: %d/%d | "
                "system: fps=%.1f, tracks=%d",
                self.metrics.faces_total,
                self.metrics.faces_accepted,
                self.metrics.faces_held,
                self.metrics.faces_rejected,
                dict(self.metrics.binding_state_counts),
                self.metrics.scheduler_selected,
                self.metrics.scheduler_budget_available,
                self.metrics.merge_success,
                self.metrics.merge_attempts,
                self.metrics.fps_estimate,
                self.metrics.track_count,
            )
            
            logger.debug("Governance Metrics (JSON): %s", metrics_dict)
            
            self.metrics.reset()
            self.last_emit_time = now
            return True
        
        return False


_global_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector(interval_sec=1.0)
    return _global_collector

