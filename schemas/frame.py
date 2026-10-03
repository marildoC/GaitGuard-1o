
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Frame:
    """
    One video frame coming from a camera.

    Attributes
    ----------
    frame_id : int
        Incremental counter (0, 1, 2, ...) for this camera stream.
    ts       : float
        Timestamp in seconds (time.time()) when the frame was captured.
    camera_id: str
        Identifier of the camera (e.g. "cam0", "entrance").
    size     : (width, height) or None.
        Logical size of the frame in pixels. If None, it can be inferred
        from image.shape if image is not None.
    image    : np.ndarray or None
        Raw BGR image as a NumPy array (H, W, 3) in OpenCV format.

    NOTE:
    - This class intentionally stays simple. Higher-level components
      (perception, face route, etc.) attach their own metadata rather
      than mutating Frame directly.
    """

    frame_id: int
    ts: float
    camera_id: str = "cam0"
    size: Optional[Tuple[int, int]] = None
    image: Optional[np.ndarray] = None
