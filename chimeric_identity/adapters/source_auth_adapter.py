
from __future__ import annotations

import logging
import time
from typing import Optional, Any, Dict

from source_auth.types import SourceAuthScores, SourceAuthState as SAState

from chimeric_identity.types import (
    SourceAuthEvidence,
    SourceAuthState,
)

logger = logging.getLogger(__name__)



class SourceAuthAdapter:
    """
    Converts SourceAuthEngine outputs to chimeric evidence format.
    
    Responsibility:
        - Read SourceAuthScores from SourceAuthEngine
        - Normalize to SourceAuthEvidence schema
        - Extract realness_score and state
        - Handle missing/failed spoof detection gracefully
    
    Design:
        - Returns None if SourceAuth not available
        - Returns SourceAuthEvidence.UNCERTAIN if score unclear
        - Conservative: Prefers UNCERTAIN over risky REAL
    
    Key Methods:
        - adapt_scores(source_auth_scores): Convert to SourceAuthEvidence
        - extract_state(score): Map realness score to state enum
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize source auth adapter.
        
        Args:
            enabled: Whether spoof detection is enabled
        """
        self.enabled = enabled
        if enabled:
            logger.info("[SOURCE-AUTH-ADAPTER] Initialized (enabled)")
        else:
            logger.info("[SOURCE-AUTH-ADAPTER] Initialized (disabled)")
    
    def adapt_scores(
        self,
        source_auth_scores: Optional[Any],
        now: Optional[float] = None
    ) -> Optional[SourceAuthEvidence]:
        """
        Convert SourceAuthScores to SourceAuthEvidence.
        
        Args:
            source_auth_scores: From SourceAuthEngine (any type)
            now: Current timestamp (or use time.time())
        
        Returns:
            SourceAuthEvidence or None if not applicable
        
        Design:
            - If adapter disabled → return None
            - If source_auth_scores is None → return None
            - If score extraction fails → return UNCERTAIN
            - Otherwise → return evidence with state
        
        Defensive Approach:
            Try to extract score from various attribute names:
            1. .realness_score (preferred)
            2. .score (fallback)
            3. .probability (fallback)
        """
        if now is None:
            now = time.time()
        
        if not self.enabled or source_auth_scores is None:
            return None
        
        realness_score = None
        
        for attr_name in ["realness_score", "score", "probability"]:
            if hasattr(source_auth_scores, attr_name):
                try:
                    realness_score = float(getattr(source_auth_scores, attr_name))
                    break
                except (ValueError, TypeError):
                    continue
        
        if realness_score is None:
            logger.warning(
                "[SOURCE-AUTH-ADAPTER] Could not extract realness_score, "
                "returning UNCERTAIN"
            )
            return SourceAuthEvidence(
                realness_score=0.5,
                state=SourceAuthState.UNCERTAIN,
                timestamp=now,
                reason="score_extraction_failed"
            )
        
        realness_score = max(0.0, min(1.0, realness_score))
        
        state = SourceAuthAdapter._extract_state(realness_score)
        
        source_auth_ev = SourceAuthEvidence(
            realness_score=realness_score,
            state=state,
            timestamp=now,
            reason=None,
            extra={
                "original_scores": str(source_auth_scores)[:100]  # For debugging
            }
        )
        
        logger.debug(
            f"[SOURCE-AUTH-ADAPTER] Adapted: "
            f"realness={realness_score:.3f}, state={state.value}"
        )
        
        return source_auth_ev
    
    @staticmethod
    def _extract_state(realness_score: float) -> SourceAuthState:
        """
        Map realness score to SourceAuthState.
        
        Thresholds (conservative approach):
            0.0-0.2 → SPOOF (very likely fake)
            0.2-0.4 → LIKELY_SPOOF (probably fake)
            0.4-0.6 → UNCERTAIN (unclear)
            0.6-0.8 → LIKELY_REAL (probably real)
            0.8-1.0 → REAL (very likely real)
        
        Design Rationale:
            - Wide UNCERTAIN zone (0.4-0.6) for boundary cases
            - Conservative: Threshold for REAL is high (0.8)
            - Threshold for SPOOF is low (0.2)
            - Encourages caution when unclear
        """
        if realness_score < 0.2:
            return SourceAuthState.SPOOF
        elif realness_score < 0.4:
            return SourceAuthState.LIKELY_SPOOF
        elif realness_score < 0.6:
            return SourceAuthState.UNCERTAIN
        elif realness_score < 0.8:
            return SourceAuthState.LIKELY_REAL
        else:
            return SourceAuthState.REAL
    
    @staticmethod
    def should_block_decision(
        source_auth_ev: Optional[SourceAuthEvidence]
    ) -> bool:
        """
        Check if source auth evidence should block identity decision.
        
        Blocks if:
            - State is SPOOF or LIKELY_SPOOF
            - Realness score is very low (< 0.3)
        
        Args:
            source_auth_ev: SourceAuthEvidence (or None)
        
        Returns:
            True if decision should be blocked, False otherwise
        
        Design:
            Spoof detection is safety gate. If triggered, halt identity
            decision and require manual review or re-detection.
        """
        if source_auth_ev is None:
            return False
        
        if source_auth_ev.is_spoof():
            return True
        
        if source_auth_ev.realness_score < 0.25:
            return True
        
        return False
    
    @staticmethod
    def should_block_learning(
        source_auth_ev: Optional[SourceAuthEvidence]
    ) -> bool:
        """
        Check if source auth evidence should block template learning.
        
        Blocks learning if:
            - State is SPOOF, LIKELY_SPOOF, or UNCERTAIN
            - Only learn when LIKELY_REAL or REAL
        
        Args:
            source_auth_ev: SourceAuthEvidence (or None)
        
        Returns:
            True if learning should be blocked, False otherwise
        
        Design:
            More conservative than decision blocking. Don't train on
            questionable data (includes UNCERTAIN).
        """
        if source_auth_ev is None:
            return False
        
        return not source_auth_ev.is_real()



def get_source_auth_evidence(
    source_auth_scores: Optional[Any],
    enabled: bool = True,
    now: Optional[float] = None
) -> Optional[SourceAuthEvidence]:
    """
    Convenience function: Get source auth evidence.
    
    This is the main entry point for adapting spoof detection.
    
    Args:
        source_auth_scores: From SourceAuthEngine
        enabled: Whether spoof detection is enabled
        now: Current timestamp
    
    Returns:
        SourceAuthEvidence or None
    """
    adapter = SourceAuthAdapter(enabled=enabled)
    return adapter.adapt_scores(source_auth_scores, now)


