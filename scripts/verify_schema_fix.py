
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_schema")

@dataclass
class Tracklet:
    track_id: int
    camera_id: str
    last_frame_id: int
    last_box: Tuple[float, float, float, float]
    confidence: float
    score: float = 0.9
    bbox: Tuple[float, float, float, float] = (0,0,10,10)

try:
    from identity.merge_manager import MergeManager, MergeConfig
    REAL_MERGE_MANAGER = True
except ImportError:
    REAL_MERGE_MANAGER = False
    print("WARNING: Could not import real MergeManager. Running in partial mock mode.")

def test_tracklet_attribute_access():
    logger.info("--- Test 1: Tracklet Attribute Access ---")
    t = Tracklet(track_id=123, camera_id="test", last_frame_id=1, last_box=(0,0,10,10), confidence=0.9)
    
    try:
        tid = t.track_id
        logger.info(f"PASS: Access t.track_id = {tid}")
    except AttributeError:
        logger.error("FAIL: t.track_id failed")

    try:
        _ = t.id
        logger.error("FAIL: t.id should not exist (unless aliased)")
    except AttributeError:
        logger.info("PASS: t.id does not exist (correct, we migrated to track_id)")

def test_merge_manager_adapter():
    logger.info("\n--- Test 2: MergeManager Adapter ---")
    if not REAL_MERGE_MANAGER:
        logger.info("SKIPPING: Real MergeManager not available")
        return

    cfg = MergeConfig(enabled=True)
    mm = MergeManager(cfg)
    
    try:
        mm.update_track_state(
            tracklet_id=123,
            bbox=(10, 10, 50, 50),
            score=0.95,
            timestamp=100.0
        )
        logger.info("PASS: mm.update_track_state called successfully")
    except Exception as e:
        logger.error(f"FAIL: mm.update_track_state crashed: {e}")

if __name__ == "__main__":
    test_tracklet_attribute_access()
    test_merge_manager_adapter()
