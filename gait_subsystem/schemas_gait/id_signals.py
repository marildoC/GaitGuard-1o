
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IdSignals:
    """
    All identity-related features for a given track.
    How it works:
    This dataclass serves as a container for different biometric or visual cues
    that can contribute to identifying a tracked person. Each field represents
    an identity signal (e.g., face, gait, appearance), typically storing a
    feature vector (embedding) and an associated quality score.

    Embeddings are typically 1D NumPy arrays (e.g. shape (512,)).

    - track_id            : link to Tracklet.track_id
    - face_embedding      : face feature vector (or None if not available)
    - face_quality        : 0–1 quality score for face
    - gait_embedding      : gait feature vector
    - gait_quality        : 0–1 quality score for gait
    - appearance_embedding: clothing / color features
    - appearance_quality  : 0–1 quality score for appearance
    """
    track_id: int

    face_embedding: Optional[np.ndarray] = None
    face_quality: float = 0.0

    gait_embedding: Optional[np.ndarray] = None
    gait_quality: float = 0.0

    appearance_embedding: Optional[np.ndarray] = None
    appearance_quality: float = 0.0

@dataclass
class IdSignal:
    """
    Represents a single identity signal produced by an identity recognition module.

    How it works:
    This dataclass encapsulates a single suggestion for a track's identity,
    along with the confidence level of that suggestion and the method used to derive it.
    It's typically used when a specific recognition module (e.g., face recognition, gait recognition)
    makes a decision about an identity.

    Attributes:
    - track_id (int): The ID of the track to which this identity signal pertains.
    - identity_id (Optional[str]): The suggested unique identifier for the person. Can be None if
                                    the signal suggests an unknown identity.
    - confidence (float): The confidence score (0-1) associated with the identity suggestion.
    - method (str): A string indicating the method or modality that generated this signal
                    (e.g., "face", "gait", "appearance").
    """
    track_id:int
    identity_id: Optional[str]
    confidence:float
    method:str