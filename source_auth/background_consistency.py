
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .types import BBoxTuple

logger = logging.getLogger(__name__)




def compute_background_inconsistency_score(
    frames_window: List[Tuple[np.ndarray, Optional[BBoxTuple]]],
    cfg: object,
) -> Tuple[float, bool, Dict[str, float]]:
    """
    Compute a scalar background inconsistency score using a small temporal
    window of frames around a track.

    Parameters
    ----------
    frames_window :
        List of (image, bbox) pairs:
          - image: HxWx3 BGR (uint8 or convertible)
          - bbox : (x1, y1, x2, y2) in frame coordinates, or None.
    cfg :
        SourceAuthConfig-like object. Only a few fields are used; we always
        access them via getattr(..., default) so that older configs do not
        break.

    Returns
    -------
    score : float
        In [0, 1]. 0 means "background inside vs outside is very similar",
        1 means "strongly different background statistics".
    reliable : bool
        True only if we had enough usable frames and regions.
    debug : Dict[str, float]
        Diagnostic metrics: color/texture distances, valid frame counts, etc.
    """
    neutral = float(getattr(cfg, "neutral_score", 0.5))

    if not frames_window:
        return neutral, False, {}

    inner_margin_ratio = float(
        getattr(cfg, "background_inner_margin_ratio", 0.15)
    )
    outer_margin_ratio = float(
        getattr(cfg, "background_outer_margin_ratio", 0.40)
    )
    min_region_pixels = int(
        getattr(cfg, "background_min_region_pixels", 500)
    )
    hist_bins = int(getattr(cfg, "background_hist_bins", 32))
    w_color = float(getattr(cfg, "background_hist_color_weight", 0.7))
    w_texture = float(getattr(cfg, "background_hist_texture_weight", 0.3))
    dist_low = float(getattr(cfg, "background_dist_low", 0.10))
    dist_high = float(getattr(cfg, "background_dist_high", 0.35))
    min_valid_frames = int(getattr(cfg, "background_min_valid_frames", 3))

    per_frame_color_dists: List[float] = []
    per_frame_texture_dists: List[float] = []
    per_frame_combined_dists: List[float] = []

    total_frames = len(frames_window)
    valid_frames = 0

    for idx, (img, bbox) in enumerate(frames_window):
        img_bgr = _ensure_bgr_uint8(img)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        x1i = max(0, int(np.floor(min(x1, x2))))
        x2i = min(w, int(np.ceil(max(x1, x2))))
        y1i = max(0, int(np.floor(min(y1, y2))))
        y2i = min(h, int(np.ceil(max(y1, y2))))

        if x2i <= x1i or y2i <= y1i:
            continue

        box_w = x2i - x1i
        box_h = y2i - y1i
        if box_w * box_h < min_region_pixels:
            continue

        inner_mask, outer_mask = _compute_inner_outer_masks(
            img_bgr.shape[:2],
            (x1i, y1i, x2i, y2i),
            inner_margin_ratio=inner_margin_ratio,
            outer_margin_ratio=outer_margin_ratio,
        )

        inner_pixels = int(np.count_nonzero(inner_mask))
        outer_pixels = int(np.count_nonzero(outer_mask))
        if inner_pixels < min_region_pixels or outer_pixels < min_region_pixels:
            continue

        color_in, texture_in = _compute_region_stats(
            img_bgr, inner_mask, hist_bins
        )
        color_out, texture_out = _compute_region_stats(
            img_bgr, outer_mask, hist_bins
        )

        color_dist = float(np.linalg.norm(color_in - color_out))

        texture_dist = float(
            np.abs(texture_in["log_var"] - texture_out["log_var"])
            + np.abs(texture_in["edge_mean"] - texture_out["edge_mean"])
        )

        combined_dist = w_color * color_dist + w_texture * texture_dist

        per_frame_color_dists.append(color_dist)
        per_frame_texture_dists.append(texture_dist)
        per_frame_combined_dists.append(combined_dist)
        valid_frames += 1

    if valid_frames < min_valid_frames:
        debug = {
            "total_frames": float(total_frames),
            "valid_frames": float(valid_frames),
            "color_dist_median": 0.0,
            "texture_dist_median": 0.0,
            "combined_dist_median": 0.0,
        }
        return neutral, False, debug

    color_median = float(np.median(per_frame_color_dists)) if per_frame_color_dists else 0.0
    texture_median = float(np.median(per_frame_texture_dists)) if per_frame_texture_dists else 0.0
    combined_median = float(np.median(per_frame_combined_dists)) if per_frame_combined_dists else 0.0

    score = _map_dist_to_score(
        combined_median,
        low=dist_low,
        high=dist_high,
    )

    reliable = True

    debug = {
        "total_frames": float(total_frames),
        "valid_frames": float(valid_frames),
        "color_dist_median": color_median,
        "texture_dist_median": texture_median,
        "combined_dist_median": combined_median,
    }

    return score, reliable, debug




def _ensure_bgr_uint8(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Ensure we have an HxWx3 uint8 BGR image, similar to the checks in
    FaceDetectorAligner.

    - Rejects empty / invalid arrays.
    - Converts grayscale to BGR.
    - Clips non-uint8 to [0,255] and casts to uint8.
    """
    if image is None:
        return None

    img = np.asarray(image)
    if img.size == 0:
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        logger.debug(
            "background_consistency: unexpected image shape %s", img.shape
        )
        return None

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _compute_inner_outer_masks(
    shape_hw: Tuple[int, int],
    bbox_xyxy: Tuple[int, int, int, int],
    inner_margin_ratio: float,
    outer_margin_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build boolean masks for inner and outer background regions.

    Inner background:
      - "Ring" inside the bbox:
          bbox_inner  = bbox shrunk by inner_margin_ratio on each side
          inner_mask  = bbox_mask AND NOT central_mask
        where central_mask is an even more shrunk rectangle (half of inner).

    Outer background:
      - "Ring" outside the bbox:
          bbox_outer  = bbox expanded by outer_margin_ratio on each side
          outer_ring  = bbox_outer_mask AND NOT bbox_mask

    All operations are clipped to image bounds, and masks are boolean (uint8 0/1).
    """
    h, w = shape_hw
    x1, y1, x2, y2 = bbox_xyxy

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    bw = x2 - x1
    bh = y2 - y1

    dx_in = int(round(bw * inner_margin_ratio))
    dy_in = int(round(bh * inner_margin_ratio))

    inner_x1 = min(max(x1 + dx_in, 0), w)
    inner_y1 = min(max(y1 + dy_in, 0), h)
    inner_x2 = min(max(x2 - dx_in, 0), w)
    inner_y2 = min(max(y2 - dy_in, 0), h)

    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    dx_center = int(round((inner_x2 - inner_x1) * 0.5 * inner_margin_ratio))
    dy_center = int(round((inner_y2 - inner_y1) * 0.5 * inner_margin_ratio))

    center_x1 = min(max(inner_x1 + dx_center, 0), w)
    center_y1 = min(max(inner_y1 + dy_center, 0), h)
    center_x2 = min(max(inner_x2 - dx_center, 0), w)
    center_y2 = min(max(inner_y2 - dy_center, 0), h)

    inner_mask = np.zeros((h, w), dtype=np.uint8)
    central_mask = np.zeros((h, w), dtype=np.uint8)

    inner_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 1
    if center_x2 > center_x1 and center_y2 > center_y1:
        central_mask[center_y1:center_y2, center_x1:center_x2] = 1

    inner_ring = np.logical_and(inner_mask == 1, central_mask == 0).astype(
        np.uint8
    )

    dx_out = int(round(bw * outer_margin_ratio))
    dy_out = int(round(bh * outer_margin_ratio))

    outer_x1 = max(0, x1 - dx_out)
    outer_y1 = max(0, y1 - dy_out)
    outer_x2 = min(w, x2 + dx_out)
    outer_y2 = min(h, y2 + dy_out)

    outer_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_mask = np.zeros((h, w), dtype=np.uint8)

    outer_mask[outer_y1:outer_y2, outer_x1:outer_x2] = 1
    bbox_mask[y1:y2, x1:x2] = 1

    outer_ring = np.logical_and(outer_mask == 1, bbox_mask == 0).astype(
        np.uint8
    )

    return inner_ring, outer_ring


def _compute_region_stats(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    hist_bins: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute color and texture statistics for a masked region.

    Color:
      - Convert to HSV.
      - Compute normalized histogram over H and S (2D histogram flattened).

    Texture:
      - Compute Laplacian magnitude inside mask (edge strength).
      - Return:
          * log_var   : log(variance + epsilon)
          * edge_mean : mean edge magnitude

    Returns
    -------
    color_feat : 1D np.ndarray (normalized histogram)
    texture_feat : dict with keys {"log_var", "edge_mean"}
    """
    if mask.shape[:2] != img_bgr.shape[:2]:
        raise ValueError("Mask and image shapes do not match")

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        hist = np.zeros(hist_bins * hist_bins, dtype=np.float32)
        texture = {"log_var": 0.0, "edge_mean": 0.0}
        return hist, texture

    h_channel = img_hsv[:, :, 0][mask_bool]
    s_channel = img_hsv[:, :, 1][mask_bool]

    hist_2d, _, _ = np.histogram2d(
        h_channel.astype(np.float32),
        s_channel.astype(np.float32),
        bins=hist_bins,
        range=[[0, 180], [0, 256]],
    )
    hist_flat = hist_2d.astype(np.float32).reshape(-1)
    hist_sum = float(hist_flat.sum())
    if hist_sum > 1e-6:
        hist_flat /= hist_sum
    else:
        pass

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)

    region_edges = lap_abs[mask_bool]
    if region_edges.size == 0:
        var = 0.0
        mean_edge = 0.0
    else:
        var = float(np.var(region_edges))
        mean_edge = float(np.mean(region_edges))

    log_var = float(np.log(var + 1e-6))

    texture = {
        "log_var": log_var,
        "edge_mean": mean_edge,
    }

    return hist_flat, texture


def _map_dist_to_score(dist: float, low: float, high: float) -> float:
    """
    Map a non-negative distance to [0,1] inconsistency score using:
      - dist <= low  → 0
      - dist >= high → 1
      - else         → linear interpolation between.

    If high <= low, fall back to a simple saturating mapping.
    """
    d = max(0.0, float(dist))

    if high <= low:
        return 0.0 if d <= low else 1.0

    if d <= low:
        return 0.0
    if d >= high:
        return 1.0

    return (d - low) / (high - low)
