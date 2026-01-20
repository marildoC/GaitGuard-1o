from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from schemas import Frame
from .detector import Detection

logger = logging.getLogger(__name__)



@dataclass
class AppearanceConfig:
    """
    Configuration for the lightweight appearance descriptor.

    The idea:
      - Crop the person bounding box.
      - Resize to a small fixed size.
      - Optionally blur (to reduce noise).
      - Convert to HSV (more robust to lighting).
      - Split into a small grid (e.g., 4x4).
      - For each cell, compute mean H, S, V.
      - Concatenate → feature vector.
      - L2-normalize so cosine similarity can be used.

    This is intentionally cheap but relatively stable for person re-ID.
    """
    resize_width: int = 64
    resize_height: int = 128
    grid_rows: int = 4
    grid_cols: int = 4
    blur_kernel: Optional[Tuple[int, int]] = (3, 3)
    use_hsv: bool = True



def _safe_crop(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Optional[np.ndarray]:
    """
    Crop a region from the image, safely clamping to image bounds.

    Returns None if the crop has zero or negative size.
    """
    h, w = image.shape[:2]

    x1_int = int(max(0, min(w - 1, x1)))
    y1_int = int(max(0, min(h - 1, y1)))
    x2_int = int(max(0, min(w, x2)))
    y2_int = int(max(0, min(h, y2)))

    if x2_int <= x1_int or y2_int <= y1_int:
        return None

    return image[y1_int:y2_int, x1_int:x2_int]


def appearance_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return cosine distance in [0, 2].

    0   → identical
    1   → orthogonal
    2   → opposite

    (Small values mean more similar.)
    """
    if a.shape != b.shape:
        raise ValueError(f"appearance_distance: shape mismatch {a.shape} vs {b.shape}")

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 1.0

    cos_sim = float(np.dot(a, b) / (na * nb))
    return 1.0 - cos_sim



class AppearanceExtractor:
    """
    Lightweight, classic CV appearance descriptor for person boxes.

    Output vectors are L2-normalized float32 arrays. They are designed
    to work well with cosine similarity (like in OCSortTracker).
    """

    def __init__(self, config: Optional[AppearanceConfig] = None) -> None:
        self.config = config or AppearanceConfig()

        if self.config.grid_rows <= 0 or self.config.grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive integers")

    def compute_feature(self, image: np.ndarray, box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Compute a single appearance feature for a given bounding box.

        Parameters
        ----------
        image : np.ndarray
            BGR image (HxWx3).
        box : (x1, y1, x2, y2)
            Bounding box in absolute pixel coordinates.

        Returns
        -------
        Optional[np.ndarray]
            L2-normalized feature vector of shape (dim,), or None if
            the crop is invalid.
        """
        if image is None or image.size == 0:
            logger.warning("AppearanceExtractor.compute_feature: empty image")
            return None

        if image.ndim != 3 or image.shape[2] != 3:
            logger.warning("AppearanceExtractor.compute_feature: expected HxWx3 BGR image, got shape %s", image.shape)
            return None

        x1, y1, x2, y2 = box
        crop = _safe_crop(image, x1, y1, x2, y2)
        if crop is None:
            return None

        crop = cv2.resize(
            crop,
            (self.config.resize_width, self.config.resize_height),
            interpolation=cv2.INTER_LINEAR,
        )

        if self.config.blur_kernel is not None:
            kx, ky = self.config.blur_kernel
            kx = max(1, kx | 1)
            ky = max(1, ky | 1)
            crop = cv2.GaussianBlur(crop, (kx, ky), 0)

        if self.config.use_hsv:
            feat_img = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        else:
            feat_img = crop

        h, w = feat_img.shape[:2]
        gh = h // self.config.grid_rows
        gw = w // self.config.grid_cols

        cells: List[float] = []
        for gy in range(self.config.grid_rows):
            for gx in range(self.config.grid_cols):
                y_start = gy * gh
                y_end = h if gy == self.config.grid_rows - 1 else (gy + 1) * gh
                x_start = gx * gw
                x_end = w if gx == self.config.grid_cols - 1 else (gx + 1) * gw

                cell = feat_img[y_start:y_end, x_start:x_end]
                if cell.size == 0:
                    cells.extend([0.0, 0.0, 0.0])
                    continue

                mean_vals = cell.reshape(-1, 3).mean(axis=0)
                cells.extend(mean_vals.tolist())

        feature = np.asarray(cells, dtype=np.float32)

        norm = float(np.linalg.norm(feature))
        if norm > 1e-6:
            feature /= norm

        return feature

    def compute_features_for_detections(
        self,
        frame_or_image: Frame | np.ndarray,
        detections: Sequence[Detection],
    ) -> List[Optional[np.ndarray]]:
        """
        Compute appearance features for a list of detections, aligned
        by index.

        Parameters
        ----------
        frame_or_image : Frame | np.ndarray
            Either a Frame object (from schemas.Frame) or a raw BGR image.
        detections : Sequence[Detection]
            Detections whose boxes we will use to crop.

        Returns
        -------
        List[Optional[np.ndarray]]
            For each detection, either a feature vector or None
            (if crop/processing failed).
        """
        if isinstance(frame_or_image, Frame):
            image = frame_or_image.image
        else:
            image = frame_or_image

        features: List[Optional[np.ndarray]] = []
        for det in detections:
            box = (det.x1, det.y1, det.x2, det.y2)
            feat = self.compute_feature(image, box)
            features.append(feat)

        return features
