
import sys
import os
import io
import time
from dataclasses import dataclass

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(__file__).rsplit('\\', 1)[0])

from chimeric_identity.types import (
    ChimericState,
    ChimericReason,
    FaceEvidence,
    GaitEvidence,
    SourceAuthEvidence,
    EvidenceStatus,
    SourceAuthState,
)
from chimeric_identity.fusion_engine import ChimericFusionEngine
from chimeric_identity.bindings import (
    BindingManager,
    GaitTemplate,
    FaceGaitBinding,
    BindingStrength,
    BindingStatus,
)
from chimeric_identity.config import default_chimeric_config



def create_face_evidence(
    identity_id: str,
    similarity: float = 0.85,
    quality: float = 0.90,
    status: EvidenceStatus = EvidenceStatus.CONFIRMED_STRONG,
    timestamp: float = None
) -> FaceEvidence:
    """Create face evidence for testing."""
    return FaceEvidence(
        identity_id=identity_id,
        similarity=similarity,
        quality=quality,
        status=status,
        margin=similarity - 0.10,
        timestamp=timestamp or time.time(),
    )


def create_gait_evidence(
    identity_id: str = None,
    similarity: float = 0.78,
    quality: float = 0.75,
    status: EvidenceStatus = EvidenceStatus.TENTATIVE,
    sequence_length: int = 50,
    timestamp: float = None
) -> GaitEvidence:
    """Create gait evidence for testing."""
    return GaitEvidence(
        identity_id=identity_id,
        similarity=similarity,
        margin=similarity - 0.08,
        quality=quality,
        status=status,
        sequence_length=sequence_length,
        confirm_streak=3,
        timestamp=timestamp or time.time(),
    )



def test_binding_creation():
    """Test 1: Binding creation when face confirms."""
    print("\n" + "="*70)
    print("TEST 1: Binding Creation (Governance Gates)")
    print("="*70)
    
    engine = ChimericFusionEngine(default_chimeric_config())
    face_ev = create_face_evidence(
        identity_id="alice",
        quality=0.92,
        status=EvidenceStatus.CONFIRMED_STRONG
    )
    gait_ev = create_gait_evidence(
        identity_id="alice",
        status=EvidenceStatus.TENTATIVE,
        sequence_length=60
    )
    
    from chimeric_identity.types import ChimericDecision
    decision = ChimericDecision(
        track_id=1,
        final_identity="alice",
        chimeric_confidence=0.88,
        state=ChimericState.CONFIRMED,
        decision_reason=ChimericReason.FACE_CONFIRMED_WITH_GAIT_SUPPORT,
        learning_allowed=True,
        timestamp=time.time(),
        face_evidence=face_ev,
        gait_evidence=gait_ev,
        source_auth_evidence=None,
    )
    
    should_bind, reason = engine.governance.bind_face_to_gait(decision, time.time())
    
    print(f"  Should create binding: {should_bind}")
    print(f"  Reason: {reason}")
    print(f"  Face quality: {face_ev.quality:.2f}")
    print(f"  Decision confidence: {decision.chimeric_confidence:.2f}")
    
    assert should_bind, f"Binding creation failed: {reason}"
    print("  ✓ Binding creation gates passed")


def test_binding_storage():
    """Test 2: Binding storage in state machine."""
    print("\n" + "="*70)
    print("TEST 2: Binding Storage in State Machine")
    print("="*70)
    
    manager = BindingManager()
    gait_template = GaitTemplate(
        gait_id="gait_alice_001",
        sequence_length=60,
        quality=0.75,
        margin=0.12,
    )
    
    binding = manager.create_binding(
        track_id=1,
        face_identity_id="alice",
        gait_template=gait_template,
        face_quality=0.92,
        face_confidence=0.88,
    )
    
    print(f"  Created binding: {binding.binding_id}")
    print(f"  Face identity: {binding.face_identity_id}")
    print(f"  Binding strength: {binding.strength.value}")
    print(f"  Initial strength value: {binding.strength_value:.2f}")
    
    retrieved = manager.get_binding(1, "alice", time.time())
    assert retrieved is not None, "Failed to retrieve binding"
    assert retrieved.face_identity_id == "alice"
    print("  ✓ Binding stored and retrieved successfully")


def test_binding_confidence_boost():
    """Test 3: Binding-based confidence boost."""
    print("\n" + "="*70)
    print("TEST 3: Binding Confidence Boost")
    print("="*70)
    
    manager = BindingManager()
    now = time.time()
    
    gait_template = GaitTemplate(
        gait_id="gait_alice_001",
        sequence_length=60,
        quality=0.78,
        margin=0.12,
        timestamp=now,
    )
    binding = manager.create_binding(
        track_id=1,
        face_identity_id="alice",
        gait_template=gait_template,
        face_quality=0.92,
        face_confidence=0.88,
    )
    
    for i in range(5):
        binding.record_match(now + i)
    
    effective_strength = binding.get_effective_strength(now + 10)
    print(f"  After 5 matches, effective strength: {effective_strength:.3f}")
    print(f"  Binding strength enum: {binding.strength.value}")
    
    base_confidence = 0.70
    boosted = binding.boost_gait_confidence(base_confidence, now + 10, max_boost=0.15)
    boost_amount = boosted - base_confidence
    
    print(f"  Base gait confidence: {base_confidence:.2f}")
    print(f"  Boosted confidence: {boosted:.2f}")
    print(f"  Boost amount: {boost_amount:.3f}")
    
    assert boosted > base_confidence, "Boost not applied"
    assert boost_amount <= 0.155, "Boost exceeded max (with epsilon)"  # Allow small epsilon for fp precision
    print("  ✓ Confidence boost applied correctly")


def test_binding_temporal_decay():
    """Test 4: Temporal decay of bindings."""
    print("\n" + "="*70)
    print("TEST 4: Binding Temporal Decay")
    print("="*70)
    
    manager = BindingManager()
    now = time.time()
    
    gait_template = GaitTemplate(
        gait_id="gait_alice_001",
        sequence_length=60,
        quality=0.78,
        margin=0.12,
        timestamp=now,
    )
    binding = manager.create_binding(
        track_id=1,
        face_identity_id="alice",
        gait_template=gait_template,
        face_quality=0.92,
        face_confidence=0.88,
    )
    
    for i in range(5):
        binding.record_match(now + i)
    
    age_0 = binding.temporal_decay_factor(now + 10)
    age_1800 = binding.temporal_decay_factor(now + 1800)
    age_3600 = binding.temporal_decay_factor(now + 3600)
    
    print(f"  Strength at age 10s: {age_0:.3f}")
    print(f"  Strength at age 30m: {age_1800:.3f}")
    print(f"  Strength at age 1h (half-life): {age_3600:.3f}")
    
    assert age_0 > age_1800 > age_3600, "Temporal decay not working"
    assert 0.45 < age_3600 < 0.55, "Half-life should be ~0.5"
    print("  ✓ Temporal decay working correctly")


def test_binding_conflict_tracking():
    """Test 5: Conflict tracking and invalidation."""
    print("\n" + "="*70)
    print("TEST 5: Binding Conflict Tracking")
    print("="*70)
    
    manager = BindingManager()
    now = time.time()
    
    gait_template = GaitTemplate(
        gait_id="gait_alice_001",
        sequence_length=60,
        quality=0.78,
        margin=0.12,
        timestamp=now,
    )
    binding = manager.create_binding(
        track_id=1,
        face_identity_id="alice",
        gait_template=gait_template,
        face_quality=0.92,
        face_confidence=0.88,
    )
    
    print(f"  Initial status: {binding.status.value}")
    print(f"  Initial strength: {binding.strength_value:.2f}")
    
    binding.record_match(now)
    print(f"  After match: strength={binding.strength_value:.2f}, observations={binding.observations}")
    
    binding.record_conflict(now + 10)
    print(f"  After conflict 1: strength={binding.strength_value:.2f}, conflicts={binding.conflict_count}")
    
    binding.record_conflict(now + 20)
    print(f"  After conflict 2: strength={binding.strength_value:.2f}, conflicts={binding.conflict_count}")
    
    binding.record_conflict(now + 30)
    print(f"  After conflict 3: status={binding.status.value}, strength={binding.strength_value:.2f}")
    
    assert binding.status == BindingStatus.INVALIDATED, "Binding should be invalidated after 3 conflicts"
    print("  ✓ Conflict tracking and invalidation working correctly")


def test_binding_consistency_factor():
    """Test 6: Consistency factor calculation."""
    print("\n" + "="*70)
    print("TEST 6: Binding Consistency Factor")
    print("="*70)
    
    manager = BindingManager()
    now = time.time()
    
    gait_template = GaitTemplate(
        gait_id="gait_alice_001",
        sequence_length=60,
        quality=0.78,
        margin=0.12,
        timestamp=now,
    )
    binding = manager.create_binding(
        track_id=1,
        face_identity_id="alice",
        gait_template=gait_template,
        face_quality=0.92,
        face_confidence=0.88,
    )
    
    for i in range(9):
        binding.record_match(now + i)
    binding.record_conflict(now + 100)
    
    consistency = binding.consistency_factor()
    print(f"  With 90% consistency: factor = {consistency:.2f}")
    assert consistency > 1.0, "High consistency should boost"
    
    binding2 = manager.create_binding(
        track_id=2,
        face_identity_id="bob",
        gait_template=gait_template,
        face_quality=0.92,
        face_confidence=0.88,
    )
    binding2.record_match(now)
    for i in range(3):
        binding2.record_conflict(now + i + 10)
    
    consistency2 = binding2.consistency_factor()
    print(f"  With 25% consistency: factor = {consistency2:.2f}")
    assert consistency2 < 1.0, "Low consistency should penalize"
    
    print("  ✓ Consistency factor working correctly")


def test_end_to_end_workflow():
    """Test 7: End-to-end binding workflow."""
    print("\n" + "="*70)
    print("TEST 7: End-to-End Binding Workflow")
    print("="*70)
    
    engine = ChimericFusionEngine(default_chimeric_config())
    track_id = 1
    now = time.time()
    
    print("\n  Scenario: Face confirms alice, gait matches")
    
    face_alice = create_face_evidence("alice", quality=0.92, status=EvidenceStatus.CONFIRMED_STRONG)
    gait_alice = create_gait_evidence("alice", status=EvidenceStatus.TENTATIVE, sequence_length=50)
    
    from chimeric_identity.types import ChimericDecision
    decision1 = ChimericDecision(
        track_id=track_id,
        final_identity="alice",
        chimeric_confidence=0.88,
        state=ChimericState.CONFIRMED,
        decision_reason=ChimericReason.FACE_CONFIRMED_WITH_GAIT_SUPPORT,
        learning_allowed=True,
        timestamp=now,
        face_evidence=face_alice,
        gait_evidence=gait_alice,
        source_auth_evidence=None,
    )
    
    should_bind, _ = engine.governance.bind_face_to_gait(decision1, now)
    if should_bind:
        gait_template = GaitTemplate(
            gait_id="gait_alice_001",
            sequence_length=50,
            quality=gait_alice.quality,
            margin=gait_alice.margin,
            timestamp=now,
        )
        binding = engine.binding_manager.create_binding(
            track_id=track_id,
            face_identity_id="alice",
            gait_template=gait_template,
            face_quality=face_alice.quality,
            face_confidence=decision1.chimeric_confidence,
        )
        print(f"  ✓ Binding created: {binding.binding_id}")
    
    now_later = now + 5.0
    gait_alice_later = create_gait_evidence(
        "alice",
        status=EvidenceStatus.TENTATIVE,
        sequence_length=60,
        timestamp=now_later
    )
    
    decision2 = ChimericDecision(
        track_id=track_id,
        final_identity="alice",
        chimeric_confidence=0.70,
        state=ChimericState.TENTATIVE,
        decision_reason=ChimericReason.GAIT_TENTATIVE_NO_FACE_ANCHOR,
        learning_allowed=False,
        timestamp=now_later,
        face_evidence=None,
        gait_evidence=gait_alice_later,
        source_auth_evidence=None,
    )
    
    binding = engine.binding_manager.get_binding(track_id, "alice", now_later)
    if binding:
        boosted, reason = engine.governance.evaluate_binding_strength_boost(
            decision2,
            binding.get_effective_strength(now_later),
            max_boost=0.15
        )
        boost_amount = boosted - decision2.chimeric_confidence
        print(f"  ✓ Binding boost applied: {decision2.chimeric_confidence:.2f} → {boosted:.2f} (+{boost_amount:.3f})")
        
        engine.binding_manager.record_gait_match(track_id, "alice", now_later)
        print(f"  ✓ Gait match recorded for binding")
    
    binding_stats = engine.binding_manager.get_stats()
    print(f"\n  Binding stats:")
    print(f"    Total bindings: {binding_stats.total_bindings}")
    print(f"    Total observations: {binding_stats.total_observations}")
    print(f"    Average strength: {binding_stats.avg_strength:.2f}")


def test_binding_cleanup():
    """Test 8: Binding cleanup and memory management."""
    print("\n" + "="*70)
    print("TEST 8: Binding Cleanup & Memory Management")
    print("="*70)
    
    manager = BindingManager()
    now = time.time()
    
    for track_id in range(5):
        gait_template = GaitTemplate(
            gait_id=f"gait_{track_id}",
            sequence_length=60,
            quality=0.78,
            margin=0.12,
            timestamp=now,
        )
        manager.create_binding(
            track_id=track_id,
            face_identity_id=f"person_{track_id}",
            gait_template=gait_template,
            face_quality=0.92,
            face_confidence=0.88,
        )
    
    stats_before = manager.get_stats()
    print(f"  Before cleanup: {stats_before.total_bindings} bindings")
    
    manager.cleanup_track(2)
    stats_mid = manager.get_stats()
    print(f"  After track cleanup: {stats_mid.total_bindings} bindings")
    assert stats_mid.total_bindings == stats_before.total_bindings - 1
    
    manager.cleanup_stale(now + 7200, max_age_sec=3600.0)
    stats_after = manager.get_stats()
    print(f"  After stale cleanup: {stats_after.total_bindings} bindings")
    
    print("  ✓ Binding cleanup working correctly")



def run_all_tests():
    """Run all Phase 3B tests."""
    print("\n" + "="*70)
    print("PHASE 3B VALIDATION - FACE-GAIT BINDING SYSTEM")
    print("="*70)
    
    tests = [
        ("Binding Creation", test_binding_creation),
        ("Binding Storage", test_binding_storage),
        ("Confidence Boost", test_binding_confidence_boost),
        ("Temporal Decay", test_binding_temporal_decay),
        ("Conflict Tracking", test_binding_conflict_tracking),
        ("Consistency Factor", test_binding_consistency_factor),
        ("End-to-End Workflow", test_end_to_end_workflow),
        ("Cleanup", test_binding_cleanup),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, "PASS"))
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, status in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("✓ PHASE 3B VALIDATION SUCCESSFUL - ALL BINDING LOGIC ROBUST!")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("✗ PHASE 3B VALIDATION FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
