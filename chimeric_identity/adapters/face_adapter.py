
from __future__ import annotations

import logging
import time
from typing import Optional

from schemas.identity_decision import IdentityDecision
from schemas.tracklet import Tracklet

from chimeric_identity.types import (
    FaceEvidence,
    EvidenceStatus,
    SourceAuthState,
)

logger = logging.getLogger(__name__)



class FaceAdapter:
    """
    Converts face engine outputs to chimeric evidence format.
    
    Responsibility:
        - Read IdentityDecision from FaceIdentityEngine
        - Normalize to FaceEvidence schema
        - Extract quality, similarity, margin, binding state
        - Handle edge cases (None, stale, quality below threshold)
    
    Design:
        - Stateless (no per-track memory)
        - Returns None if face evidence not applicable
        - Preserves source_auth metadata if present
    
    Key Methods:
        - adapt_decision(tracklet, identity_decision): Convert to FaceEvidence
        - extract_evidence_status(binding_state, distance): Map states
    """
    
    def __init__(self):
        """Initialize face adapter (stateless)."""
        logger.debug("[FACE-ADAPTER] Initialized (stateless)")
    
    @staticmethod
    def adapt_decision(
        tracklet: Tracklet,
        identity_decision: Optional[IdentityDecision],
        now: Optional[float] = None
    ) -> Optional[FaceEvidence]:
        """
        Convert IdentityDecision to FaceEvidence.
        
        Args:
            tracklet: Original tracklet (for context)
            identity_decision: Output from FaceIdentityEngine (or None)
            now: Current timestamp (or use time.time())
        
        Returns:
            FaceEvidence or None if no valid face evidence
        
        Design Rationale:
            - If identity_decision is None → return None (no face detection)
            - If quality too low → EvidenceStatus.UNKNOWN (but return evidence for audit)
            - If binding_state is UNKNOWN → low confidence
            - If binding_state is CONFIRMED_STRONG → high confidence
        """
        if now is None:
            now = time.time()
        
        if identity_decision is None:
            logger.debug(
                f"[FACE-ADAPTER] track_id={tracklet.track_id} → "
                f"identity_decision=None, returning None"
            )
            return None
        
        track_id = identity_decision.track_id
        identity_id = identity_decision.identity_id
        confidence = identity_decision.confidence
        quality = identity_decision.quality
        binding_state = identity_decision.binding_state
        
        distance = identity_decision.distance or 0.0
        similarity = max(0.0, 1.0 - distance)
        
        margin = 0.0
        second_best_id = None
        second_best_similarity = 0.0
        
        if identity_decision.extra:
            margin = identity_decision.extra.get("margin", 0.0)
            second_best_id = identity_decision.extra.get("second_best_id")
            second_best_similarity = identity_decision.extra.get("second_best_sim", 0.0)
        
        evidence_status = FaceAdapter._map_binding_state(binding_state)
        
        if quality < 0.45:
            evidence_status = EvidenceStatus.UNKNOWN
        
        source_auth_score = identity_decision.source_auth_score
        source_auth_state = None
        if identity_decision.source_auth_state:
            try:
                source_auth_state = SourceAuthState[
                    identity_decision.source_auth_state.upper()
                ]
            except (KeyError, AttributeError):
                source_auth_state = None
        
        face_ev = FaceEvidence(
            identity_id=identity_id,
            similarity=similarity,
            quality=quality,
            status=evidence_status,
            binding_state=binding_state,
            margin=margin,
            second_best_id=second_best_id,
            second_best_similarity=second_best_similarity,
            timestamp=now,
            freshness_window_sec=2.0,
            source_auth_score=source_auth_score,
            source_auth_state=source_auth_state,
            extra={
                "original_confidence": confidence,
                "original_distance": distance,
                "original_reason": identity_decision.reason,
            }
        )
        
        logger.debug(
            f"[FACE-ADAPTER] track_id={track_id} adapted: "
            f"id={identity_id}, sim={similarity:.3f}, quality={quality:.2f}, "
            f"status={evidence_status.value}, binding={binding_state}"
        )
        
        return face_ev
    
    @staticmethod
    def _map_binding_state(binding_state: Optional[str]) -> EvidenceStatus:
        """
        Map face binding state to chimeric evidence status.
        
        Binding State Mapping:
            UNKNOWN → UNKNOWN (no identity hypothesis)
            PENDING → TENTATIVE (gathering evidence, not confirmed)
            CONFIRMED_WEAK → CONFIRMED_WEAK (low margin/quality confirm)
            CONFIRMED_STRONG → CONFIRMED_STRONG (high margin/quality confirm)
            SWITCH_PENDING → TENTATIVE (switching, not settled)
            STALE → STALE (evidence too old)
        
        Design:
            - CONFIRMED states stay CONFIRMED (preserve strength)
            - Other states map intuitively
            - UNKNOWN is default (safe assumption)
        """
        if binding_state is None:
            return EvidenceStatus.UNKNOWN
        
        state_upper = binding_state.upper()
        
        mapping = {
            "UNKNOWN": EvidenceStatus.UNKNOWN,
            "PENDING": EvidenceStatus.TENTATIVE,
            "CONFIRMED_WEAK": EvidenceStatus.CONFIRMED_WEAK,
            "CONFIRMED_STRONG": EvidenceStatus.CONFIRMED_STRONG,
            "SWITCH_PENDING": EvidenceStatus.TENTATIVE,
            "STALE": EvidenceStatus.STALE,
        }
        
        return mapping.get(state_upper, EvidenceStatus.UNKNOWN)
    
    @staticmethod
    def extract_confidence_from_similarity(
        similarity: float,
        quality: float,
        margin: float
    ) -> float:
        """
        Extract face confidence from similarity, quality, and margin.
        
        Formula:
            confidence = similarity * quality_weight * margin_weight
        
        Where:
            - quality_weight = quality^1 (higher quality → more confident)
            - margin_weight = 1.0 if margin > 0.10, else margin / 0.10 (margin safety)
        
        Args:
            similarity: Cosine similarity (0-1)
            quality: Face quality (0-1)
            margin: sim(best) - sim(2nd_best)
        
        Returns:
            Adjusted confidence (0-1)
        
        Design Rationale:
            - Quality is multiplicative (poor quality reduces confidence)
            - Margin provides safety margin (close matches are risky)
            - Result is never > similarity (conservative)
        """
        quality_weight = (quality ** 0.5)
        
        if margin >= 0.10:
            margin_weight = 1.0
        else:
            margin_weight = margin / 0.10
            margin_weight = max(0.1, margin_weight)
        
        confidence = similarity * quality_weight * margin_weight
        
        return min(1.0, max(0.0, confidence))



def get_face_evidence_from_engine(
    tracklet: Tracklet,
    identity_decision: Optional[IdentityDecision],
    now: Optional[float] = None
) -> Optional[FaceEvidence]:
    """
    Convenience function: Get face evidence from identity decision.
    
    This is the main entry point for adapting face evidence.
    
    Args:
        tracklet: Tracklet from perception
        identity_decision: From FaceIdentityEngine
        now: Current timestamp
    
    Returns:
        FaceEvidence or None
    """
    return FaceAdapter.adapt_decision(tracklet, identity_decision, now)


