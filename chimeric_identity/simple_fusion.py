
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

from chimeric_identity.identity_registry import (
    IdentityRegistry, 
    get_identity_registry,
    PersonRecord
)

logger = logging.getLogger(__name__)



class FusionState(str, Enum):
    """Simple fusion states - no complex state machine needed!"""
    UNKNOWN = "UNKNOWN"           # No recognition yet
    FACE_ONLY = "FACE_ONLY"       # Only face recognized
    GAIT_ONLY = "GAIT_ONLY"       # Only gait recognized
    FUSED = "FUSED"               # Both recognized, same person
    CONFLICT = "CONFLICT"         # Both recognized, different persons


@dataclass
class FaceInput:
    """Input from face engine (read-only, don't modify face engine)."""
    identity_id: Optional[str]
    confidence: float
    quality: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    
    def is_recognized(self) -> bool:
        return self.identity_id is not None and self.confidence > 0.5


@dataclass
class GaitInput:
    """Input from gait engine (read-only, don't modify gait engine)."""
    identity_id: Optional[str]
    confidence: float
    quality: float
    confirm_streak: int = 0
    
    def is_recognized(self) -> bool:
        return self.identity_id is not None and self.confidence > 0.5


@dataclass
class FusionResult:
    """
    Output from chimeric fusion - the combined decision.
    
    This is what gets displayed and used for alerts.
    """
    display_name: Optional[str]
    
    combined_confidence: float
    face_contribution: float
    gait_contribution: float
    
    state: FusionState
    
    face_id: Optional[str] = None
    gait_id: Optional[str] = None
    
    color_rgb: Tuple[int, int, int] = (255, 255, 255)
    
    timestamp: float = 0.0
    track_id: int = 0
    
    def is_recognized(self) -> bool:
        """Check if person is recognized."""
        return (
            self.state in [FusionState.FACE_ONLY, FusionState.FUSED] and
            self.display_name is not None and
            self.combined_confidence > 0.5
        )



@dataclass
class FusionWeights:
    """
    Configurable weights for biometric fusion.
    
    Face is stronger biometric (more accurate) → higher weight.
    Gait adds confidence when available → lower but meaningful weight.
    """
    face_weight: float = 0.75
    gait_weight: float = 0.25
    
    min_face_quality: float = 0.5
    min_gait_quality: float = 0.5
    
    recognition_threshold: float = 0.5
    
    def normalize(self) -> None:
        """Ensure weights sum to 1.0."""
        total = self.face_weight + self.gait_weight
        if total > 0:
            self.face_weight /= total
            self.gait_weight /= total



class SimpleFusionEngine:
    """
    Simple weighted fusion of face and gait recognition.
    
    THIS IS THE CORRECT APPROACH:
    1. Read face engine output (don't modify it)
    2. Read gait engine output (don't modify it)
    3. Check if they refer to same person via Identity Registry
    4. Combine confidences with weights
    5. Output unified result for visualization
    
    That's it! No complex state machines, no evidence accumulators,
    no governance engines - just simple, clean fusion!
    """
    
    def __init__(
        self,
        weights: Optional[FusionWeights] = None,
        registry: Optional[IdentityRegistry] = None
    ):
        """
        Initialize simple fusion engine.
        
        Args:
            weights: Fusion weight configuration
            registry: Identity registry for face/gait ID mapping
        """
        self.weights = weights or FusionWeights()
        self.weights.normalize()
        
        self.registry = registry or get_identity_registry()
        
        logger.info(
            f"[SIMPLE-FUSION] Initialized: "
            f"face_weight={self.weights.face_weight:.2f}, "
            f"gait_weight={self.weights.gait_weight:.2f}"
        )
    
    def fuse(
        self,
        track_id: int,
        face_input: Optional[FaceInput],
        gait_input: Optional[GaitInput],
        timestamp: Optional[float] = None
    ) -> FusionResult:
        """
        Fuse face and gait recognition results.
        
        THIS IS THE MAIN FUSION LOGIC - SIMPLE AND CLEAN!
        
        Args:
            track_id: Track identifier
            face_input: Output from face engine (or None if no face)
            gait_input: Output from gait engine (or None if no gait)
            timestamp: Current timestamp
        
        Returns:
            FusionResult with combined decision
        """
        if timestamp is None:
            timestamp = time.time()
        
        face_recognized = face_input and face_input.is_recognized()
        gait_recognized = gait_input and gait_input.is_recognized()
        
        face_id = face_input.identity_id if face_input else None
        gait_id = gait_input.identity_id if gait_input else None
        
        face_conf = face_input.confidence if face_input else 0.0
        gait_conf = gait_input.confidence if gait_input else 0.0
        
        if not face_recognized and not gait_recognized:
            return FusionResult(
                display_name=None,
                combined_confidence=0.0,
                face_contribution=0.0,
                gait_contribution=0.0,
                state=FusionState.UNKNOWN,
                face_id=face_id,
                gait_id=gait_id,
                color_rgb=(255, 255, 255),
                timestamp=timestamp,
                track_id=track_id
            )
        
        if face_recognized and not gait_recognized:
            person = self.registry.lookup_by_face(face_id)
            display_name = person.display_name if person else face_id
            
            return FusionResult(
                display_name=display_name,
                combined_confidence=face_conf,
                face_contribution=face_conf,
                gait_contribution=0.0,
                state=FusionState.FACE_ONLY,
                face_id=face_id,
                gait_id=gait_id,
                color_rgb=(0, 255, 0),
                timestamp=timestamp,
                track_id=track_id
            )
        
        if not face_recognized and gait_recognized:
            person = self.registry.lookup_by_gait(gait_id)
            display_name = person.display_name if person else gait_id
            
            adjusted_conf = gait_conf * 0.8
            
            return FusionResult(
                display_name=display_name,
                combined_confidence=adjusted_conf,
                face_contribution=0.0,
                gait_contribution=gait_conf,
                state=FusionState.GAIT_ONLY,
                face_id=face_id,
                gait_id=gait_id,
                color_rgb=(0, 255, 255),
                timestamp=timestamp,
                track_id=track_id
            )
        
        same_person = self.registry.are_same_person(face_id, gait_id)
        
        if same_person:
            face_contribution = face_conf * self.weights.face_weight
            gait_contribution = gait_conf * self.weights.gait_weight
            combined = face_contribution + gait_contribution
            
            person = self.registry.lookup_by_face(face_id)
            display_name = person.display_name if person else face_id
            
            logger.debug(
                f"[SIMPLE-FUSION] track={track_id} FUSED: {display_name} "
                f"face={face_conf:.2f}*{self.weights.face_weight:.2f}={face_contribution:.2f} + "
                f"gait={gait_conf:.2f}*{self.weights.gait_weight:.2f}={gait_contribution:.2f} = "
                f"{combined:.2f}"
            )
            
            return FusionResult(
                display_name=display_name,
                combined_confidence=min(combined, 1.0),
                face_contribution=face_contribution,
                gait_contribution=gait_contribution,
                state=FusionState.FUSED,
                face_id=face_id,
                gait_id=gait_id,
                color_rgb=(0, 255, 0),
                timestamp=timestamp,
                track_id=track_id
            )
        
        else:
            person = self.registry.lookup_by_face(face_id)
            display_name = person.display_name if person else face_id
            
            logger.warning(
                f"[SIMPLE-FUSION] track={track_id} CONFLICT: "
                f"face says {face_id}, gait says {gait_id}"
            )
            
            return FusionResult(
                display_name=display_name,
                combined_confidence=face_conf * 0.7,
                face_contribution=face_conf,
                gait_contribution=0.0,
                state=FusionState.CONFLICT,
                face_id=face_id,
                gait_id=gait_id,
                color_rgb=(255, 0, 0),
                timestamp=timestamp,
                track_id=track_id
            )
    
    def get_color_for_state(self, state: FusionState) -> Tuple[int, int, int]:
        """Get BGR color for fusion state (for OpenCV visualization)."""
        colors = {
            FusionState.UNKNOWN: (255, 255, 255),
            FusionState.FACE_ONLY: (0, 255, 0),
            FusionState.GAIT_ONLY: (0, 255, 255),
            FusionState.FUSED: (0, 255, 0),
            FusionState.CONFLICT: (0, 0, 255),
        }
        return colors.get(state, (255, 255, 255))



def create_face_input_from_decision(identity_decision) -> Optional[FaceInput]:
    """
    Create FaceInput from FaceIdentityEngine's IdentityDecision.
    
    This is the ADAPTER that reads face engine output without modifying it.
    """
    if identity_decision is None:
        return None
    
    return FaceInput(
        identity_id=identity_decision.identity_id,
        confidence=identity_decision.confidence or 0.0,
        quality=identity_decision.quality or 0.0,
        bbox=getattr(identity_decision, 'bbox', None)
    )


def create_gait_input_from_signal(gait_signal, gait_state=None) -> Optional[GaitInput]:
    """
    Create GaitInput from GaitEngine's IdSignal and GaitTrackState.
    
    This is the ADAPTER that reads gait engine output without modifying it.
    """
    if gait_signal is None:
        return None
    
    identity_id = gait_signal.identity_id
    confidence = gait_signal.confidence or 0.0
    
    quality = 0.0
    confirm_streak = 0
    
    if gait_state is not None:
        quality = getattr(gait_state, 'q_seq', 0.0) or 0.0
        confirm_streak = getattr(gait_state, 'confirm_streak', 0) or 0
    
    return GaitInput(
        identity_id=identity_id,
        confidence=confidence,
        quality=quality,
        confirm_streak=confirm_streak
    )



_fusion_engine: Optional[SimpleFusionEngine] = None


def get_fusion_engine(
    weights: Optional[FusionWeights] = None
) -> SimpleFusionEngine:
    """Get global fusion engine singleton."""
    global _fusion_engine
    
    if _fusion_engine is None:
        _fusion_engine = SimpleFusionEngine(weights=weights)
    
    return _fusion_engine



def format_fusion_result(result: FusionResult) -> str:
    """
    Format fusion result for console output.
    
    Shows exactly what you described:
    - Face: 70% Marildo
    - Gait: +10% 
    - Total: 80% Marildo ✓
    """
    if not result.is_recognized():
        return f"Track {result.track_id}: Scanning..."
    
    lines = []
    
    if result.face_contribution > 0:
        face_pct = int(result.face_contribution * 100)
        lines.append(f"Face: {face_pct}% {result.display_name}")
    
    if result.gait_contribution > 0:
        gait_pct = int(result.gait_contribution * 100)
        lines.append(f"Gait: +{gait_pct}%")
    
    total_pct = int(result.combined_confidence * 100)
    state_emoji = {
        FusionState.FUSED: "✓",
        FusionState.FACE_ONLY: "👤",
        FusionState.GAIT_ONLY: "🚶",
        FusionState.CONFLICT: "⚠️",
        FusionState.UNKNOWN: "?"
    }
    emoji = state_emoji.get(result.state, "")
    lines.append(f"Total: {total_pct}% {result.display_name} {emoji}")
    
    return f"Track {result.track_id}: " + " | ".join(lines)



if __name__ == "__main__":
    """Example demonstrating the simple fusion logic."""
    
    registry = get_identity_registry()
    fusion = get_fusion_engine()
    
    registry.register_person(
        display_name="Marildo",
        face_id="p_0001",
        gait_id="Marildo"
    )
    
    print("=" * 60)
    print("SIMPLE CHIMERIC FUSION DEMO")
    print("=" * 60)
    
    print("\n--- Test 1: Face only ---")
    result1 = fusion.fuse(
        track_id=1,
        face_input=FaceInput(identity_id="p_0001", confidence=0.85, quality=0.9),
        gait_input=None
    )
    print(format_fusion_result(result1))
    
    print("\n--- Test 2: Gait only ---")
    result2 = fusion.fuse(
        track_id=2,
        face_input=None,
        gait_input=GaitInput(identity_id="Marildo", confidence=0.75, quality=0.8)
    )
    print(format_fusion_result(result2))
    
    print("\n--- Test 3: Both modalities, same person ---")
    result3 = fusion.fuse(
        track_id=3,
        face_input=FaceInput(identity_id="p_0001", confidence=0.85, quality=0.9),
        gait_input=GaitInput(identity_id="Marildo", confidence=0.75, quality=0.8)
    )
    print(format_fusion_result(result3))
    
    registry.register_person(display_name="Francesco", face_id="p_0002", gait_id="Francesco")
    print("\n--- Test 4: Conflict (face=Marildo, gait=Francesco) ---")
    result4 = fusion.fuse(
        track_id=4,
        face_input=FaceInput(identity_id="p_0001", confidence=0.85, quality=0.9),
        gait_input=GaitInput(identity_id="Francesco", confidence=0.75, quality=0.8)
    )
    print(format_fusion_result(result4))
    
    print("\n" + "=" * 60)
