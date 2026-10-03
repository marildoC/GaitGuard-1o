
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class Tracklet:
    """
    One moving person/object tracked over time.

    Attributes
    ----------
    track_id      : int
        Unique ID assigned by the tracker for this track.
    camera_id     : str
        Which camera this track is from.
    last_frame_id : int
        ID of the last Frame where it was seen.
    last_box      : (x1, y1, x2, y2)
        Last known bounding box in pixel coordinates.
    confidence    : float
        Tracker/detector confidence (0–1) for this track.

    age_frames    : int
        How many frames this track has existed.
    lost_frames   : int
        How many frames since it was last observed (for "lost" logic).
    history_boxes : list of boxes
        Optional past boxes for this track, useful for motion trails.

    last_ts       : Optional[float]
        Optional timestamp (time.time()) for the last observation.
        Not required by existing code but useful for time-based logic.
    
    # Gait Recognition Fields (Francesco integration)
    gait_sequence_data : List[np.ndarray]
        Buffer of pose keypoints (17 joints × 3 values per frame).
        Each entry is shape (17, 3) where [x, y, confidence] for each joint.
    gait_embedding : Optional[np.ndarray]
        256-dim gait embedding vector for identity comparison.
    gait_quality : float
        Quality score (0-1) for gait data reliability.
    gait_identity_id : Optional[str]
        Person ID recognized through gait analysis.
    gait_confidence : Optional[float]
        Confidence score (0-1) for gait recognition.
    """

    track_id: int
    camera_id: str
    last_frame_id: int
    last_box: Tuple[float, float, float, float]
    confidence: float

    age_frames: int = 0
    lost_frames: int = 0
    history_boxes: List[Tuple[float, float, float, float]] = field(
        default_factory=list
    )

    last_ts: Optional[float] = None

    gait_sequence_data: List[np.ndarray] = field(default_factory=list)
    
    gait_embedding: Optional[np.ndarray] = None
    
    gait_quality: float = 0.0
    
    gait_identity_id: Optional[str] = None
    
    gait_confidence: Optional[float] = None
