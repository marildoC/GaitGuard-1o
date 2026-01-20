from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Frame:
    """
    One video frame coming from a camera.

    - frame_id : incremental counter (0, 1, 2, ...)
    - ts       : timestamp in seconds (time.time())
    - camera_id: which camera this frame comes from (e.g. "cam0")
    - size     : (width, height). Can be left None and inferred from image.
    - image    : raw BGR image as a NumPy array (H, W, 3)
    """
    frame_id: int
    ts: float
    camera_id: str = "cam0"
    size: Optional[Tuple[int, int]] = None
    image: Optional[np.ndarray] = None
