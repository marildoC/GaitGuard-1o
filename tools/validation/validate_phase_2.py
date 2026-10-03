"""
validate_phase_2.py

Comprehensive validation script for Phase 2: Quality-Weighted Adaptive Fusion.

Tests:
1. Quality boost function (smooth degradation, no cliffs)
2. Adaptive weight computation (quality-aware weighting)
3. Temporal reliability (gait accumulation scaling)
4. Conflict detection and penalization
5. State modulation
6. Integration: full confidence synthesis

Usage:
    python validate_phase_2.py
"""

import sys
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

try:
    sys.path.insert(0, '.')
    from chimeric_identity.fusion_engine import ChimericFusionEngine
    from chimeric_identity.config import ChimericConfig
    from chimeric_identity.types import (
        FaceEvidence, GaitEvidence, SourceAuthEvidence,
        EvidenceStatus, SourceAuthState, ChimericState
    )
    logger.info("[VALIDATION] ✓ All imports successful")
except ImportError as e:
    logger.error(f"[VALIDATION] ✗ Import failed: {e}")
    sys.exit(1)



def assert_close(actual: float, expected: float, tolerance: float = 0.01, msg: str = "") -> bool:
    """Assert that actual ≈ expected within tolerance."""
    if abs(actual - expected) > tolerance:
        logger.error(f"[ASSERT] ✗ {msg}")
        logger.error(f"    Expected: {expected:.4f}, Got: {actual:.4f}, Diff: {abs(actual - expected):.4f}")
        return False
    else:
        logger.info(f"[ASSERT] ✓ {msg}")
        return True


def assert_greater(actual: float, threshold: float, msg: str = "") -> bool:
    """Assert that actual > threshold."""
    if actual <= threshold:
        logger.error(f"[ASSERT] ✗ {msg}")
        logger.error(f"    Expected > {threshold:.4f}, Got: {actual:.4f}")
        return False
    else:
        logger.info(f"[ASSERT] ✓ {msg}")
        return True


def assert_less(actual: float, threshold: float, msg: str = "") -> bool:
    """Assert that actual < threshold."""
    if actual >= threshold:
        logger.error(f"[ASSERT] ✗ {msg}")
        logger.error(f"    Expected < {threshold:.4f}, Got: {actual:.4f}")
        return False
    else:
        logger.info(f"[ASSERT] ✓ {msg}")
        return True


def assert_between(actual: float, low: float, high: float, msg: str = "") -> bool:
    """Assert that low <= actual <= high."""
    if actual < low or actual > high:
        logger.error(f"[ASSERT] ✗ {msg}")
        logger.error(f"    Expected between {low:.4f} and {high:.4f}, Got: {actual:.4f}")
        return False
    else:
        logger.info(f"[ASSERT] ✓ {msg}")
        return True



class Phase2Validator:
    """Phase 2 validation suite."""
    
    def __init__(self):
        self.config = ChimericConfig()
        self.engine = ChimericFusionEngine(config=self.config)
        self.results = []
        
        logger.info("[VALIDATION] Initialized Phase2Validator")
    
    def record_result(self, test_name: str, passed: bool):
        """Record test result."""
        self.results.append((test_name, passed))
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"[TEST] {test_name}: {status}")
    
    def test_quality_boost_above_threshold(self):
        """Test: Quality boost at/above threshold should be 1.0."""
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Quality Boost - Above Threshold")
        logger.info("="*70)
        
        boost_at = self.engine._compute_quality_boost(
            quality_score=0.70, quality_threshold=0.70, penalty_slope=2.0
        )
        result1 = assert_close(boost_at, 1.0, tolerance=0.01, msg="boost(0.70, threshold=0.70) = 1.0")
        
        boost_above = self.engine._compute_quality_boost(
            quality_score=0.85, quality_threshold=0.70, penalty_slope=2.0
        )
        result2 = assert_close(boost_above, 1.0, tolerance=0.01, msg="boost(0.85, threshold=0.70) = 1.0")
        
        self.record_result("Quality Boost - Above Threshold", result1 and result2)
        return result1 and result2
    
    def test_quality_boost_below_threshold(self):
        """Test: Quality boost below threshold should degrade smoothly."""
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Quality Boost - Below Threshold (Smooth Degradation)")
        logger.info("="*70)
        
        boost_mid = self.engine._compute_quality_boost(
            quality_score=0.60, quality_threshold=0.70, penalty_slope=2.0
        )
        logger.info(f"    boost(0.60, threshold=0.70) = {boost_mid:.4f}")
        result1 = assert_greater(boost_mid, 0.9, msg="boost(0.60) is still high (>0.9) - close to threshold")
        
        boost_low = self.engine._compute_quality_boost(
            quality_score=0.30, quality_threshold=0.70, penalty_slope=2.0
        )
        logger.info(f"    boost(0.30, threshold=0.70) = {boost_low:.4f}")
        result2 = assert_between(boost_low, 0.55, 0.75, msg="boost(0.30) is lower (0.55-0.75) - degraded from threshold")
        
        result3 = assert_greater(boost_mid, boost_low, msg="Smooth: boost(0.60) > boost(0.30)")
        
        self.record_result("Quality Boost - Below Threshold", result1 and result2 and result3)
        return result1 and result2 and result3
    
    def test_quality_boost_zero_quality(self):
        """Test: Zero quality should give zero boost."""
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Quality Boost - Zero Quality")
        logger.info("="*70)
        
        boost_zero = self.engine._compute_quality_boost(
            quality_score=0.0, quality_threshold=0.70, penalty_slope=2.0
        )
        result = assert_close(boost_zero, 0.0, tolerance=0.01, msg="boost(0.0) = 0.0")
        
        self.record_result("Quality Boost - Zero Quality", result)
        return result
    
    def test_adaptive_weights_face_only(self):
        """Test: Face-only (no gait) → face weight should be ~0.90."""
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Adaptive Weights - Face Only")
        logger.info("="*70)
        
        face_ev = FaceEvidence(
            identity_id="alice", similarity=0.85, quality=0.80,
            status=EvidenceStatus.CONFIRMED_STRONG, margin=0.12
        )
        
        weights = self.engine._compute_adaptive_weights(face_ev, None, None)
        
        logger.info(f"    w_face={weights['w_face']:.3f}, w_gait={weights['w_gait']:.3f}, w_auth={weights['w_auth']:.3f}")
        
        result1 = assert_greater(weights['w_face'], 0.85, msg="w_face > 0.85 (face-only)")
        result2 = assert_close(weights['w_gait'], 0.0, tolerance=0.01, msg="w_gait ≈ 0.0 (no gait)")
        result3 = assert_close(
            weights['w_face'] + weights['w_gait'] + weights['w_auth'], 1.0, tolerance=0.01,
            msg="Weights sum to 1.0"
        )
        
        self.record_result("Adaptive Weights - Face Only", result1 and result2 and result3)
        return result1 and result2 and result3
    
    def test_adaptive_weights_gait_only(self):
        """Test: Gait-only (no face) → gait weight should be ~0.90."""
        logger.info("\n" + "="*70)
        logger.info("TEST 5: Adaptive Weights - Gait Only")
        logger.info("="*70)
        
        gait_ev = GaitEvidence(
            identity_id="bob", similarity=0.78, quality=0.72,
            status=EvidenceStatus.CONFIRMED_STRONG, margin=0.10,
            sequence_length=120, confirm_streak=8
        )
        
        weights = self.engine._compute_adaptive_weights(None, gait_ev, None)
        
        logger.info(f"    w_face={weights['w_face']:.3f}, w_gait={weights['w_gait']:.3f}, w_auth={weights['w_auth']:.3f}")
        
        result1 = assert_close(weights['w_face'], 0.0, tolerance=0.01, msg="w_face ≈ 0.0 (no face)")
        result2 = assert_greater(weights['w_gait'], 0.85, msg="w_gait > 0.85 (gait-only)")
        result3 = assert_close(
            weights['w_face'] + weights['w_gait'] + weights['w_auth'], 1.0, tolerance=0.01,
            msg="Weights sum to 1.0"
        )
        
        self.record_result("Adaptive Weights - Gait Only", result1 and result2 and result3)
        return result1 and result2 and result3
    
    def test_adaptive_weights_high_quality_face(self):
        """Test: High quality face → w_face should be high."""
        logger.info("\n" + "="*70)
        logger.info("TEST 6: Adaptive Weights - High Quality Face")
        logger.info("="*70)
        
        face_ev = FaceEvidence(
            identity_id="alice", similarity=0.90, quality=0.85,
            status=EvidenceStatus.CONFIRMED_STRONG, margin=0.15
        )
        gait_ev = GaitEvidence(
            identity_id="alice", similarity=0.75, quality=0.60,
            status=EvidenceStatus.TENTATIVE, margin=0.08,
            sequence_length=80, confirm_streak=3
        )
        
        weights = self.engine._compute_adaptive_weights(face_ev, gait_ev, None)
        
        logger.info(f"    face_q=0.85, gait_q=0.60 → w_face={weights['w_face']:.3f}, w_gait={weights['w_gait']:.3f}")
        
        result1 = assert_greater(weights['w_face'], 0.70, msg="w_face > 0.70 (high quality)")
        result2 = assert_less(weights['w_gait'], 0.20, msg="w_gait < 0.20 (low quality)")
        
        self.record_result("Adaptive Weights - High Quality Face", result1 and result2)
        return result1 and result2
    
    def test_adaptive_weights_low_face_high_gait(self):
        """Test: Low face quality + high gait quality → gait weight boosted."""
        logger.info("\n" + "="*70)
        logger.info("TEST 7: Adaptive Weights - Low Face, High Gait (Graceful Degradation)")
        logger.info("="*70)
        
        face_ev = FaceEvidence(
            identity_id="alice", similarity=0.65, quality=0.50,
            status=EvidenceStatus.TENTATIVE, margin=0.05
        )
        gait_ev = GaitEvidence(
            identity_id="alice", similarity=0.82, quality=0.80,
            status=EvidenceStatus.CONFIRMED_STRONG, margin=0.12,
            sequence_length=120, confirm_streak=10
        )
        
        weights = self.engine._compute_adaptive_weights(face_ev, gait_ev, None)
        
        logger.info(f"    face_q=0.50, gait_q=0.80 → w_face={weights['w_face']:.3f}, w_gait={weights['w_gait']:.3f}")
        
        result1 = assert_less(weights['w_face'], 0.70, msg="w_face < 0.70 (low quality)")
        result2 = assert_greater(weights['w_gait'], 0.25, msg="w_gait > 0.25 (boosted)")
        
        self.record_result("Adaptive Weights - Low Face High Gait", result1 and result2)
        return result1 and result2
    
    def test_temporal_reliability_short_sequence(self):
        """Test: Short gait sequence → low temporal reliability."""
        logger.info("\n" + "="*70)
        logger.info("TEST 8: Temporal Reliability - Short Sequence")
        logger.info("="*70)
        
        gait_ev = GaitEvidence(
            identity_id="bob", similarity=0.80, quality=0.75,
            status=EvidenceStatus.TENTATIVE, margin=0.10,
            sequence_length=30, confirm_streak=2
        )
        
        temporal_rel = self.engine._compute_temporal_reliability(gait_ev, track_id="track_1")
        
        logger.info(f"    seq_len=30 → temporal_reliability={temporal_rel:.3f}")
        
        result1 = assert_between(temporal_rel, 0.25, 0.5, msg="Short sequence → low reliability (0.25-0.5)")
        
        self.record_result("Temporal Reliability - Short Sequence", result1)
        return result1
    
    def test_temporal_reliability_long_sequence(self):
        """Test: Long gait sequence → high temporal reliability."""
        logger.info("\n" + "="*70)
        logger.info("TEST 9: Temporal Reliability - Long Sequence")
        logger.info("="*70)
        
        gait_ev = GaitEvidence(
            identity_id="bob", similarity=0.82, quality=0.78,
            status=EvidenceStatus.CONFIRMED_STRONG, margin=0.12,
            sequence_length=150, confirm_streak=12
        )
        
        temporal_rel = self.engine._compute_temporal_reliability(gait_ev, track_id="track_1")
        
        logger.info(f"    seq_len=150 → temporal_reliability={temporal_rel:.3f}")
        
        result = assert_greater(temporal_rel, 0.85, msg="Long sequence → high reliability (>0.85)")
        
        self.record_result("Temporal Reliability - Long Sequence", result)
        return result
    
    def test_confidence_synthesis_high_quality_face(self):
        """Test: High quality face → high chimeric confidence."""
        logger.info("\n" + "="*70)
        logger.info("TEST 10: Confidence Synthesis - High Quality Face")
        logger.info("="*70)
        
        face_ev = FaceEvidence(
            identity_id="alice", similarity=0.90, quality=0.85,
            status=EvidenceStatus.CONFIRMED_STRONG, margin=0.15
        )
        
        conf = self.engine._synthesize_confidence_quality_weighted(
            state=ChimericState.CONFIRMED,
            face_ev=face_ev,
            gait_ev=None,
            source_auth_ev=None,
            track_id="track_1"
        )
        
        logger.info(f"    High quality face: confidence={conf:.3f}")
        
        result = assert_greater(conf, 0.80, msg="High quality face → conf > 0.80")
        
        self.record_result("Confidence Synthesis - High Quality Face", result)
        return result
    
    def test_confidence_synthesis_low_face_high_gait(self):
        """Test: Low face + high gait → medium confidence (graceful degradation)."""
        logger.info("\n" + "="*70)
        logger.info("TEST 11: Confidence Synthesis - Low Face High Gait")
        logger.info("="*70)
        
        face_ev = FaceEvidence(
            identity_id="bob", similarity=0.65, quality=0.50,
            status=EvidenceStatus.TENTATIVE, margin=0.05
        )
        gait_ev = GaitEvidence(
            identity_id="bob", similarity=0.82, quality=0.80,
            status=EvidenceStatus.CONFIRMED_STRONG, margin=0.12,
            sequence_length=120, confirm_streak=10
        )
        
        conf = self.engine._synthesize_confidence_quality_weighted(
            state=ChimericState.CONFIRMED,
            face_ev=face_ev,
            gait_ev=gait_ev,
            source_auth_ev=None,
            track_id="track_1"
        )
        
        logger.info(f"    Low face + high gait: confidence={conf:.3f}")
        
        result1 = assert_greater(conf, 0.65, msg="Gait boost helps → conf > 0.65")
        result2 = assert_less(conf, 0.85, msg="Still not as good as high face → conf < 0.85")
        
        self.record_result("Confidence Synthesis - Low Face High Gait", result1 and result2)
        return result1 and result2
    
    def test_confidence_synthesis_conflict(self):
        """Test: Face ≠ gait identity → confidence reduced."""
        logger.info("\n" + "="*70)
        logger.info("TEST 12: Confidence Synthesis - Conflict Penalty")
        logger.info("="*70)
        
        face_ev = FaceEvidence(
            identity_id="alice", similarity=0.88, quality=0.80,
            status=EvidenceStatus.CONFIRMED_STRONG, margin=0.13
        )
        gait_ev = GaitEvidence(
            identity_id="bob", similarity=0.80, quality=0.75,  # Different ID!
            status=EvidenceStatus.CONFIRMED_STRONG, margin=0.11,
            sequence_length=110, confirm_streak=9
        )
        
        conf_with_conflict = self.engine._synthesize_confidence_quality_weighted(
            state=ChimericState.CONFIRMED,
            face_ev=face_ev,
            gait_ev=gait_ev,
            source_auth_ev=None,
            track_id="track_1"
        )
        
        gait_ev_same = GaitEvidence(
            identity_id="alice", similarity=0.80, quality=0.75,  # Same ID!
            status=EvidenceStatus.CONFIRMED_STRONG, margin=0.11,
            sequence_length=110, confirm_streak=9
        )
        conf_no_conflict = self.engine._synthesize_confidence_quality_weighted(
            state=ChimericState.CONFIRMED,
            face_ev=face_ev,
            gait_ev=gait_ev_same,
            source_auth_ev=None,
            track_id="track_1"
        )
        
        logger.info(f"    Conflict (alice vs bob): confidence={conf_with_conflict:.3f}")
        logger.info(f"    No conflict (alice vs alice): confidence={conf_no_conflict:.3f}")
        
        result = assert_less(conf_with_conflict, conf_no_conflict, msg="Conflict penalty applied")
        
        self.record_result("Confidence Synthesis - Conflict Penalty", result)
        return result
    
    def run_all_tests(self):
        """Run all validation tests."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 2 VALIDATION - COMPREHENSIVE TEST SUITE")
        logger.info("="*70)
        
        self.test_quality_boost_above_threshold()
        self.test_quality_boost_below_threshold()
        self.test_quality_boost_zero_quality()
        self.test_adaptive_weights_face_only()
        self.test_adaptive_weights_gait_only()
        self.test_adaptive_weights_high_quality_face()
        self.test_adaptive_weights_low_face_high_gait()
        self.test_temporal_reliability_short_sequence()
        self.test_temporal_reliability_long_sequence()
        self.test_confidence_synthesis_high_quality_face()
        self.test_confidence_synthesis_low_face_high_gait()
        self.test_confidence_synthesis_conflict()
        
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)
        
        passed = sum(1 for _, result in self.results if result)
        total = len(self.results)
        
        for test_name, result in self.results:
            status = "✓" if result else "✗"
            logger.info(f"{status} {test_name}")
        
        logger.info(f"\nTotal: {passed}/{total} passed")
        
        if passed == total:
            logger.info("\n✅ ALL TESTS PASSED - Phase 2 Implementation Valid")
            return 0
        else:
            logger.error(f"\n❌ {total - passed} TEST(S) FAILED")
            return 1



if __name__ == "__main__":
    validator = Phase2Validator()
    exit_code = validator.run_all_tests()
    sys.exit(exit_code)
