from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class Tracklet:
    """
    Represents a single moving person/object tracked over time.

    How it works:
    This dataclass aggregates various pieces of information related to a tracked entity,
    from its basic tracking metadata (ID, bounding box, age) to more advanced
    identity-related data such as gait pose sequences and recognition results.

    - track_id      : unique ID assigned by tracker
    - camera_id     : which camera this track is from
    - last_frame_id : id of the last Frame where it was seen
    - last_box      : (x1, y1, x2, y2) in pixel coordinates
    - confidence    : tracker/detector confidence (0–1)
    - age_frames    : how many frames this track has existed
    - lost_frames   : how many frames since it was last seen
    - history_boxes : optional past boxes for this track
    - gait_sequence_data (List[np.ndarray]): A list containing sequences of smoothed pose keypoints
                                             (e.g., YOLOv8-pose format) extracted for this person over time.
                                             Each element is a NumPy array representing a pose.
    - gait_embedding (Optional[np.ndarray]): The numeric feature vector (embedding) derived from
                                              the person's gait, used for identity comparison. Can be None.
    - gait_quality (float): A quality score (0-1) indicating the reliability or clarity of the
                            extracted gait data for recognition. Defaults to 0.0.
    - gait_identity_id (Optional[str]): The unique identifier of the person recognized through
                                         gait analysis. Can be None if no identity is assigned.
    - gait_confidence (Optional[float]): The confidence score (0-1) for the gait-based
                                         identity recognition result. Can be None.
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

    """
    NEW FIELD FOR GAIT_RECOGNITION(Francesco and Vittorio)
    This field will contain the sequences of poses of this person
    Each element of the list will be a np.ndarray which represent a pose.
    """
    gait_sequence_data: List[np.ndarray] = field(default_factory=list)

    gait_embedding: Optional[np.ndarray] = None
    
    gait_quality: float=0.0
    
    gait_identity_id: Optional[str] = None

    gait_confidence: Optional[float]=None