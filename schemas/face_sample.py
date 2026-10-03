
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class FaceSample:
    """
    One face observation tied to a single frame and track.

    Fields
    ------
    bbox : (x1, y1, x2, y2) or None
        Face / head crop in full-frame coordinates.
    embedding : np.ndarray or None
        L2-normalised face descriptor (e.g. 512-D).
    det_score : float
        Raw detector confidence for the chosen face (0..1 typically).
    quality : float
        Overall face quality (0..1) after penalties (pose, blur, size, etc.).
    yaw, pitch, roll : Optional[float]
        Approximate pose angles in degrees; used by multiview matcher.
    ts : Optional[float]
        Timestamp (seconds) when this sample was captured.
    pose_bin : Optional[str]
        Discrete view bin: FRONT / LEFT / RIGHT / UP / DOWN / OCCLUDED / ...
    source : str
        Origin of this sample: "runtime", "enroll", "legacy", etc.
    extra : Optional[Dict[str, Any]]
        Free-form diagnostics/metadata (e.g. {"yaw_raw": ..., "blur": ...}).
    """

    bbox: Optional[Tuple[float, float, float, float]] = None
    embedding: Optional[np.ndarray] = None

    det_score: float = 0.0
    quality: float = 0.0

    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    ts: Optional[float] = None
    pose_bin: Optional[str] = None

    source: str = "runtime"
    extra: Optional[Dict[str, Any]] = None

    def as_embedding_1d(self) -> Optional[np.ndarray]:
        """
        Return embedding as 1-D float32 array, or None if missing.
        """
        if self.embedding is None:
            return None
        e = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
        return e

    def clamped_quality(self) -> float:
        """
        Return quality clamped to [0, 1].
        """
        q = float(self.quality)
        if q < 0.0:
            return 0.0
        if q > 1.0:
            return 1.0
        return q
