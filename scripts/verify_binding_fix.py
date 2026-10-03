
import logging
from identity.binding import BindingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_binding")

def test_binding_init_default():
    logger.info("--- Test 1: BindingManager Init with None ---")
    bm = BindingManager(cfg=None)
    
    if bm.enabled:
        logger.info("PASS: BindingManager is ENABLED by default.")
    else:
        logger.error("FAIL: BindingManager is DISABLED (fallback failed).")

def test_binding_process_evidence():
    logger.info("\n--- Test 2: BindingManager.process_evidence ---")
    bm = BindingManager(cfg=None)
    if not bm.enabled:
        logger.error("SKIP: Cannot test process_evidence because verified disabled.")
        return

    from dataclasses import dataclass
    @dataclass
    class MockEvidence:
        person_id: str = "p_001"
        confidence: float = 0.9
        ts: float = 100.0
        
    try:
        decision = bm.process_evidence(
            track_id=123,
            person_id="p_001",
            score=0.9,
            second_best_score=0.1,
            quality=0.8,
            timestamp=100.0
        )
        logger.info(f"PASS: process_evidence returned: {decision}")
    except Exception as e:
        logger.error(f"FAIL: process_evidence crashed: {e}")

if __name__ == "__main__":
    test_binding_init_default()
    test_binding_process_evidence()
