"""
Phase E: Merge Manager Integration Validation Script

Validates that Phase E (Handoff Merge Manager) is correctly integrated and working.

Validation Tests:
1. Merge Manager Import & API
2. Configuration Loading
3. Merge Scoring Algorithm
4. Canonical ID Mapping
5. Tracklet Lifecycle Events
6. Main Loop Integration
7. Merging in Realistic Scenario
8. Merge Reversal
9. Metrics Collection
"""

import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("phase_e_validation")



class TestResults:
    """Track validation test results"""
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def record_pass(self, test_name):
        """Record a passed test"""
        self.tests_run += 1
        self.tests_passed += 1
        log.info(f"✅ {test_name}")
    
    def record_fail(self, test_name, error):
        """Record a failed test"""
        self.tests_run += 1
        self.tests_failed += 1
        self.failures.append((test_name, str(error)))
        log.error(f"❌ {test_name}: {error}")
    
    def print_summary(self):
        """Print test summary"""
        log.info("\n" + "="*80)
        log.info("PHASE E VALIDATION SUMMARY")
        log.info("="*80)
        log.info(f"Tests Run: {self.tests_run}")
        log.info(f"Tests Passed: {self.tests_passed} ✅")
        log.info(f"Tests Failed: {self.tests_failed} ❌")
        
        if self.tests_failed > 0:
            log.info("\nFailed Tests:")
            for test_name, error in self.failures:
                log.info(f"  - {test_name}: {error}")
        
        log.info("="*80)
        if self.tests_failed == 0:
            log.info(f"✅ All {self.tests_passed}/{self.tests_run} validation tests passed!")
        else:
            log.info(f"❌ {self.tests_failed} test(s) failed. Review above for details.")
        log.info("="*80 + "\n")
        
        return self.tests_failed == 0



def test_merge_manager_import(results):
    """Test 1: Merge Manager Import & API"""
    try:
        from identity.merge_manager import (
            MergeManager,
            MergeConfig,
            MergeCandidate,
            CanonicalMapping,
            MergeEvidence,
            MergeMetrics,
            create_merge_config_from_dict
        )
        log.info("✅ Merge Manager imports successful")
        results.record_pass("Merge Manager Import")
        return True
    except Exception as e:
        results.record_fail("Merge Manager Import", str(e))
        return False


def test_merge_manager_creation(results):
    """Test 2: Merge Manager Creation"""
    try:
        from identity.merge_manager import MergeManager, MergeConfig
        
        cfg = MergeConfig(
            enabled=True,
            merge_confidence_min=60.0,
            max_gap_seconds=5.0,
            max_distance_pixels=150.0,
            max_embedding_distance=0.35
        )
        mm = MergeManager(cfg)
        
        assert mm is not None
        assert mm.config.enabled is True
        assert len(mm.canonical_mappings) == 0
        
        log.info("✅ Merge Manager instantiated successfully")
        results.record_pass("Merge Manager Creation")
        return True
    except Exception as e:
        results.record_fail("Merge Manager Creation", str(e))
        return False


def test_merge_manager_api(results):
    """Test 3: Merge Manager Core API"""
    try:
        from identity.merge_manager import MergeManager, MergeConfig
        
        cfg = MergeConfig(enabled=True)
        mm = MergeManager(cfg)
        
        assert hasattr(mm, 'get_canonical_id')
        assert hasattr(mm, 'on_tracklet_started')
        assert hasattr(mm, 'on_tracklet_ended')
        assert hasattr(mm, 'get_merge_decision')
        assert hasattr(mm, 'execute_merge')
        assert hasattr(mm, 'get_metrics')
        
        canonical_id = mm.get_canonical_id("track_001")
        assert canonical_id == "track_001"
        
        mm.on_tracklet_started("track_002", (100.0, 100.0), 1000.0)
        assert "track_002" in mm.canonical_mappings
        
        log.info("✅ Merge Manager API working correctly")
        results.record_pass("Merge Manager API")
        return True
    except Exception as e:
        results.record_fail("Merge Manager API", str(e))
        return False


def test_config_loading(results):
    """Test 4: Configuration Loading from YAML"""
    try:
        from identity.merge_manager import create_merge_config_from_dict
        
        config_dict = {
            'enabled': True,
            'thresholds': {
                'merge_confidence_min': 60,
                'tentative_threshold': 40
            },
            'temporal': {
                'min_gap_seconds': 0.3,
                'max_gap_seconds': 5.0
            },
            'spatial': {
                'max_distance_pixels': 150,
                'velocity_influence': 20
            },
            'appearance': {
                'max_embedding_distance': 0.35,
                'quality_min_samples': 2
            },
            'logging': {
                'log_merge_reasons': True
            }
        }
        
        cfg = create_merge_config_from_dict(config_dict)
        
        assert cfg.enabled is True
        assert cfg.merge_confidence_min == 60
        assert cfg.max_gap_seconds == 5.0
        assert cfg.max_distance_pixels == 150
        
        log.info("✅ Config loading working correctly")
        results.record_pass("Config Loading")
        return True
    except Exception as e:
        results.record_fail("Config Loading", str(e))
        return False


def test_canonical_id_mapping(results):
    """Test 5: Canonical ID Mapping"""
    try:
        from identity.merge_manager import MergeManager, MergeConfig
        
        cfg = MergeConfig(enabled=True)
        mm = MergeManager(cfg)
        
        canon_id_1 = mm.get_canonical_id("track_001")
        canon_id_2 = mm.get_canonical_id("track_001")
        assert canon_id_1 == canon_id_2 == "track_001"
        
        tracklets = mm.get_all_tracklets_for_canonical("track_001")
        assert "track_001" in tracklets
        assert len(tracklets) == 1
        
        log.info("✅ Canonical ID mapping working correctly")
        results.record_pass("Canonical ID Mapping")
        return True
    except Exception as e:
        results.record_fail("Canonical ID Mapping", str(e))
        return False


def test_merge_scoring(results):
    """Test 6: Merge Scoring Algorithm"""
    try:
        from identity.merge_manager import MergeManager, MergeConfig, MergeCandidate
        
        cfg = MergeConfig(
            enabled=True,
            merge_confidence_min=60.0,
            min_gap_seconds=0.3,
            max_gap_seconds=5.0
        )
        mm = MergeManager(cfg)
        
        tracklet_a = MergeCandidate(
            tracklet_id="track_001",
            end_time=1000.0,
            start_time=990.0,
            last_position=np.array([100.0, 150.0], dtype=np.float32),
            first_position=np.array([100.0, 150.0], dtype=np.float32),
            appearance_features=np.random.rand(512).astype(np.float32),
            quality_samples=5
        )
        
        tracklet_b = MergeCandidate(
            tracklet_id="track_002",
            start_time=1000.5,
            end_time=1015.0,
            first_position=np.array([105.0, 152.0], dtype=np.float32),
            last_position=np.array([105.0, 152.0], dtype=np.float32),
            appearance_features=np.random.rand(512).astype(np.float32),
            quality_samples=5
        )
        
        decision = mm.get_merge_decision(tracklet_a, tracklet_b)
        
        assert decision is not None
        assert hasattr(decision, 'should_merge')
        assert hasattr(decision, 'score')
        assert hasattr(decision, 'reason')
        
        log.info(f"✅ Merge scoring working (score={decision.score:.1f}, reason={decision.reason})")
        results.record_pass("Merge Scoring")
        return True
    except Exception as e:
        results.record_fail("Merge Scoring", str(e))
        return False


def test_merge_execution(results):
    """Test 7: Merge Execution"""
    try:
        from identity.merge_manager import (
            MergeManager, MergeConfig, MergeEvidence
        )
        
        cfg = MergeConfig(enabled=True)
        mm = MergeManager(cfg)
        
        mm.current_time = 1001.5
        evidence = MergeEvidence(
            source_tracklet_id="track_001",
            target_tracklet_id="track_002",
            merge_time=1001.5,
            merge_reason="test_merge",
            confidence=75.0,
            details={}
        )
        
        mm.execute_merge(
            canonical_id="track_002",
            tracklet_to_merge="track_001",
            evidence=evidence
        )
        
        mapping = mm.canonical_mappings["track_002"]
        assert "track_001" in mapping.aliases
        assert len(mm.merge_history) == 1
        
        canonical_id = mm.get_canonical_id("track_001")
        assert canonical_id == "track_002"
        
        log.info("✅ Merge execution working correctly")
        results.record_pass("Merge Execution")
        return True
    except Exception as e:
        results.record_fail("Merge Execution", str(e))
        return False


def test_merge_reversal(results):
    """Test 8: Merge Reversal"""
    try:
        from identity.merge_manager import (
            MergeManager, MergeConfig, MergeEvidence
        )
        
        cfg = MergeConfig(enabled=True)
        mm = MergeManager(cfg)
        
        mm.current_time = 1001.5
        evidence = MergeEvidence(
            source_tracklet_id="track_001",
            target_tracklet_id="track_002",
            merge_time=1001.5,
            merge_reason="test_merge",
            confidence=45.0,
            details={}
        )
        
        mm.execute_merge("track_002", "track_001", evidence)
        assert "track_001" in mm.canonical_mappings["track_002"].aliases
        
        mm.current_time = 1003.0
        success = mm.reverse_merge("track_001", "contradiction")
        
        assert success is True
        assert "track_001" not in mm.canonical_mappings["track_002"].aliases
        
        log.info("✅ Merge reversal working correctly")
        results.record_pass("Merge Reversal")
        return True
    except Exception as e:
        results.record_fail("Merge Reversal", str(e))
        return False


def test_tracklet_lifecycle(results):
    """Test 9: Tracklet Lifecycle Events"""
    try:
        from identity.merge_manager import MergeManager, MergeConfig
        
        cfg = MergeConfig(enabled=True)
        mm = MergeManager(cfg)
        
        mm.on_tracklet_started("track_001", (100.0, 100.0), 1000.0)
        assert "track_001" in mm.canonical_mappings
        
        mm.on_tracklet_ended(
            tracklet_id="track_001",
            binding_state={'person_id': 'person_A', 'status': 'CONFIRMED'},
            appearance_features=np.random.rand(512).astype(np.float32),
            last_position=(150.0, 150.0),
            end_time=1010.0,
            track_length=10,
            quality_samples=5
        )
        assert "track_001" in mm.inactive_tracklets
        
        mm.on_tracklet_updated(
            tracklet_id="track_001",
            binding_state=None,
            appearance_features=None,
            current_position=(150.0, 150.0),
            track_length=10,
            quality_samples=5,
            timestamp=1009.0
        )
        
        log.info("✅ Tracklet lifecycle events working correctly")
        results.record_pass("Tracklet Lifecycle")
        return True
    except Exception as e:
        results.record_fail("Tracklet Lifecycle", str(e))
        return False


def test_metrics_collection(results):
    """Test 10: Metrics Collection"""
    try:
        from identity.merge_manager import MergeManager, MergeConfig
        
        cfg = MergeConfig(enabled=True)
        mm = MergeManager(cfg)
        
        metrics = mm.get_metrics()
        assert metrics is not None
        assert metrics.merge_attempts == 0
        assert metrics.merges_executed == 0
        
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert 'merge_attempts' in metrics_dict
        assert 'merges_executed' in metrics_dict
        
        log.info("✅ Metrics collection working correctly")
        results.record_pass("Metrics Collection")
        return True
    except Exception as e:
        results.record_fail("Metrics Collection", str(e))
        return False


def test_main_loop_integration(results):
    """Test 11: Main Loop Integration (Config Loading)"""
    try:
        from core.config import load_config
        
        cfg = load_config()
        
        assert hasattr(cfg, 'governance')
        assert hasattr(cfg.governance, 'merge')
        
        merge_cfg = cfg.governance.merge
        if hasattr(merge_cfg, '__dict__'):
            merge_enabled = getattr(merge_cfg, 'enabled', False)
        else:
            merge_enabled = merge_cfg.get('enabled', False) if isinstance(merge_cfg, dict) else False
        
        assert merge_enabled is not None
        
        log.info("✅ Main loop integration config accessible")
        results.record_pass("Main Loop Integration")
        return True
    except Exception as e:
        results.record_fail("Main Loop Integration", str(e))
        return False


def test_identity_schema_update(results):
    """Test 12: IdentityDecision Schema Update"""
    try:
        from schemas.identity_decision import IdentityDecision
        
        decision = IdentityDecision(
            track_id=1,
            identity_id="person_A",
            category="resident",
            confidence=0.95,
            canonical_id=1,
            binding_state="CONFIRMED_STRONG"
        )
        
        assert decision.track_id == 1
        assert decision.canonical_id == 1
        assert decision.binding_state == "CONFIRMED_STRONG"
        
        log.info("✅ IdentityDecision schema updated correctly")
        results.record_pass("Identity Schema Update")
        return True
    except Exception as e:
        results.record_fail("Identity Schema Update", str(e))
        return False



def run_all_validations():
    """Run all Phase E validation tests"""
    results = TestResults()
    
    log.info("\n" + "="*80)
    log.info("PHASE E VALIDATION - Starting")
    log.info("="*80 + "\n")
    
    tests = [
        ("Merge Manager Import", test_merge_manager_import),
        ("Merge Manager Creation", test_merge_manager_creation),
        ("Merge Manager API", test_merge_manager_api),
        ("Configuration Loading", test_config_loading),
        ("Canonical ID Mapping", test_canonical_id_mapping),
        ("Merge Scoring", test_merge_scoring),
        ("Merge Execution", test_merge_execution),
        ("Merge Reversal", test_merge_reversal),
        ("Tracklet Lifecycle", test_tracklet_lifecycle),
        ("Metrics Collection", test_metrics_collection),
        ("Main Loop Integration", test_main_loop_integration),
        ("Identity Schema Update", test_identity_schema_update),
    ]
    
    for test_name, test_func in tests:
        try:
            test_func(results)
        except Exception as e:
            log.exception(f"Unexpected error in {test_name}: {e}")
            results.record_fail(test_name, f"Unexpected exception: {e}")
    
    all_passed = results.print_summary()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = run_all_validations()
    sys.exit(exit_code)
