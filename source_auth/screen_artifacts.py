
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]
FrameEntry = Tuple[np.ndarray, Optional[BBox]]




def compute_screen_artifact_score(
    frames_window: Sequence[FrameEntry],
    cfg,
) -> Tuple[float, bool, Dict[str, float]]:
    """
    Compute screen-artifact cue for a single track over a short time window.

    Parameters
    ----------
    frames_window : sequence of (image_bgr, bbox_in_frame)
        - image_bgr: np.ndarray HxWx3 (BGR)
        - bbox_in_frame: (x1, y1, x2, y2) or None

    cfg : SourceAuthConfig-like
        Expected attributes (with safe defaults if missing):
          - neutral_score: float (default 0.5)
          - screen_min_frames: int (default 4)
          - screen_min_face_px: int (default 40)
          - screen_border_margin_frac: float (default 0.15)
          - screen_weight_border: float (default 0.60)
          - screen_weight_flicker: float (default 0.20)
          - screen_weight_moire: float (default 0.20)
          - screen_flicker_var_norm: float (default 0.01)
          - screen_moire_norm: float (default 300.0)
          - screen_rectangularity_angle_tol: float (default 15.0)
          - screen_bezel_uniformity_norm: float (default 40.0)
          - screen_border_reliable_min: float (default 0.50)
          - screen_border_reliable_std_max: float (default 0.15)

    Returns
    -------
    score : float
        Screen artifact score in [0,1] (0 = no screen evidence, 1 = strong screen).

    reliable : bool
        True if the estimate is based on enough frames / signal.

    debug : dict
        Diagnostic values:
          - "border_strength"
          - "border_strength_raw_mean"
          - "border_strength_median"
          - "border_strength_std"
          - "border_edge_density"
          - "border_rectangularity"
          - "border_bezel_uniformity"
          - "flicker_strength"
          - "moire_strength"
          - "mean_moire_value"
          - "valid_frames"
          - "raw_brightness_var"
          - "brightness_rel_var"
    """
    neutral = float(getattr(cfg, "neutral_score", 0.5))

    if not frames_window:
        return neutral, False, {}

    min_frames = int(getattr(cfg, "screen_min_frames", 4))
    min_face_px = int(getattr(cfg, "screen_min_face_px", 40))
    border_margin_frac = float(getattr(cfg, "screen_border_margin_frac", 0.15))

    w_border = float(getattr(cfg, "screen_weight_border", 0.60))
    w_flicker = float(getattr(cfg, "screen_weight_flicker", 0.20))
    w_moire = float(getattr(cfg, "screen_weight_moire", 0.20))

    flicker_var_norm = float(getattr(cfg, "screen_flicker_var_norm", 0.01))
    moire_norm = float(getattr(cfg, "screen_moire_norm", 300.0))

    angle_tol_deg = float(getattr(cfg, "screen_rectangularity_angle_tol", 15.0))
    bezel_uniformity_norm = float(
        getattr(cfg, "screen_bezel_uniformity_norm", 40.0)
    )

    border_reliable_min = float(
        getattr(cfg, "screen_border_reliable_min", 0.50)
    )
    border_reliable_std_max = float(
        getattr(cfg, "screen_border_reliable_std_max", 0.15)
    )

    border_scores: List[float] = []
    border_edge_densities: List[float] = []
    border_rect_scores: List[float] = []
    border_bezel_scores: List[float] = []

    brightness_series: List[float] = []
    moire_values: List[float] = []

    valid_frames = 0

    for img_bgr, bbox in frames_window:
        img = _to_valid_bgr(img_bgr)
        if img is None:
            continue

        h, w = img.shape[:2]

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        ix1 = max(0, min(w - 1, int(np.floor(x1))))
        iy1 = max(0, min(h - 1, int(np.floor(y1))))
        ix2 = max(0, min(w, int(np.ceil(x2))))
        iy2 = max(0, min(h, int(np.ceil(y2))))

        if ix2 <= ix1 + 1 or iy2 <= iy1 + 1:
            continue

        face_w = ix2 - ix1
        face_h = iy2 - iy1

        if face_w < min_face_px or face_h < min_face_px:
            continue

        face_region = img[iy1:iy2, ix1:ix2]
        if face_region.size == 0:
            continue

        outer_region = _extract_outer_ring(
            img, (ix1, iy1, ix2, iy2), margin_frac=border_margin_frac
        )

        if outer_region is None or outer_region.size == 0:
            outer_region = None

        try:
            if outer_region is not None:
                (
                    border_strength_frame,
                    edge_density_frame,
                    rect_score_frame,
                    bezel_score_frame,
                ) = _border_strength(
                    outer_region,
                    angle_tol_deg=angle_tol_deg,
                    bezel_uniformity_norm=bezel_uniformity_norm,
                )
            else:
                border_strength_frame = 0.0
                edge_density_frame = 0.0
                rect_score_frame = 0.0
                bezel_score_frame = 0.0

            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray_face))
            brightness_series.append(brightness)

            moire_val = _laplacian_energy(gray_face)
            moire_values.append(moire_val)

            border_scores.append(border_strength_frame)
            border_edge_densities.append(edge_density_frame)
            border_rect_scores.append(rect_score_frame)
            border_bezel_scores.append(bezel_score_frame)

            valid_frames += 1
        except Exception:
            logger.exception("screen_artifacts: per-frame processing failed")
            continue

    if valid_frames < min_frames:
        return neutral, False, {
            "border_strength": 0.0,
            "border_strength_raw_mean": 0.0,
            "border_strength_median": 0.0,
            "border_strength_std": 0.0,
            "border_edge_density": 0.0,
            "border_rectangularity": 0.0,
            "border_bezel_uniformity": 0.0,
            "flicker_strength": 0.0,
            "moire_strength": 0.0,
            "mean_moire_value": 0.0,
            "valid_frames": float(valid_frames),
            "raw_brightness_var": 0.0,
            "brightness_rel_var": 0.0,
        }

    if border_scores:
        border_scores_arr = np.asarray(border_scores, dtype=np.float32)
        border_mean = float(np.mean(border_scores_arr))
        border_median = float(np.median(border_scores_arr))
        border_std = float(np.std(border_scores_arr))
        border_strength = 0.5 * border_mean + 0.5 * border_median
    else:
        border_mean = 0.0
        border_median = 0.0
        border_std = 0.0
        border_strength = 0.0

    border_strength = _clamp01(border_strength)

    def _mean_or_zero(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    border_edge_density_mean = _clamp01(_mean_or_zero(border_edge_densities))
    border_rectangularity_mean = _clamp01(_mean_or_zero(border_rect_scores))
    border_bezel_uniformity_mean = _clamp01(_mean_or_zero(border_bezel_scores))

    flicker_strength, raw_var, rel_var = _flicker_strength(
        brightness_series, flicker_var_norm=flicker_var_norm
    )

    moire_strength = _moire_strength(moire_values, moire_norm=moire_norm)
    mean_moire_value = float(np.mean(moire_values)) if moire_values else 0.0

    weight_sum = w_border + w_flicker + w_moire
    if weight_sum <= 0.0:
        return neutral, False, {
            "border_strength": border_strength,
            "border_strength_raw_mean": border_mean,
            "border_strength_median": border_median,
            "border_strength_std": border_std,
            "border_edge_density": border_edge_density_mean,
            "border_rectangularity": border_rectangularity_mean,
            "border_bezel_uniformity": border_bezel_uniformity_mean,
            "flicker_strength": flicker_strength,
            "moire_strength": moire_strength,
            "mean_moire_value": mean_moire_value,
            "valid_frames": float(valid_frames),
            "raw_brightness_var": raw_var,
            "brightness_rel_var": rel_var,
        }

    score_raw = (
        w_border * border_strength
        + w_flicker * flicker_strength
        + w_moire * moire_strength
    ) / weight_sum

    score = _clamp01(score_raw)

    border_stable = (
        border_strength >= border_reliable_min
        and border_std <= border_reliable_std_max
    )

    reliable = True
    if (not border_stable) and (rel_var < 0.01) and (mean_moire_value < 1e-3):
        reliable = False
        score = neutral

    debug = {
        "border_strength": float(border_strength),
        "border_strength_raw_mean": float(border_mean),
        "border_strength_median": float(border_median),
        "border_strength_std": float(border_std),
        "border_edge_density": float(border_edge_density_mean),
        "border_rectangularity": float(border_rectangularity_mean),
        "border_bezel_uniformity": float(border_bezel_uniformity_mean),
        "flicker_strength": float(flicker_strength),
        "moire_strength": float(moire_strength),
        "mean_moire_value": float(mean_moire_value),
        "valid_frames": float(valid_frames),
        "raw_brightness_var": float(raw_var),
        "brightness_rel_var": float(rel_var),
    }

    return score, reliable, debug




def _to_valid_bgr(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Ensure the input is a valid HxWx3 uint8 BGR image.

    - Rejects empty or invalid arrays.
    - Converts grayscale to BGR.
    - Converts non-uint8 types to uint8 with clipping.
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
        logger.debug("screen_artifacts: unexpected image shape %s", img.shape)
        return None

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _extract_outer_ring(
    image: np.ndarray,
    bbox_int: Tuple[int, int, int, int],
    margin_frac: float = 0.15,
) -> Optional[np.ndarray]:
    """
    Extract a rectangular region slightly larger than the face bbox,
    to capture potential phone / screen borders.

    We keep this simple: a single expanded rectangle around the face,
    which is sufficient for edge + uniformity statistics.

    Parameters
    ----------
    image : np.ndarray
        Full frame HxWx3 (BGR).

    bbox_int : (x1, y1, x2, y2)
        Integer coordinates of face / head region.

    margin_frac : float
        Expansion margin proportional to bbox size (e.g. 0.15 = 15%).

    Returns
    -------
    outer_region : np.ndarray or None
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_int

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    margin_x = int(round(bw * margin_frac))
    margin_y = int(round(bh * margin_frac))

    ox1 = max(0, x1 - margin_x)
    oy1 = max(0, y1 - margin_y)
    ox2 = min(w, x2 + margin_x)
    oy2 = min(h, y2 + margin_y)

    if ox2 <= ox1 + 1 or oy2 <= oy1 + 1:
        return None

    return image[oy1:oy2, ox1:ox2]




def _border_strength(
    outer_region: np.ndarray,
    angle_tol_deg: float = 15.0,
    bezel_uniformity_norm: float = 40.0,
) -> Tuple[float, float, float, float]:
    """
    Estimate how strong rectangular borders are around the face region.

    Heuristic:
      1) Run Canny edge detector on the outer region.
      2) Measure edge density in bands near top/bottom/left/right.
      3) Run HoughLinesP on the edge map and count:
           - horizontal lines near top/bottom,
           - vertical lines near left/right.
         → rectangularity_score ∈ [0,1].
      4) Compute bezel uniformity:
           - intensity std-dev along the same border bands.
           - lower std-dev → higher bezel_uniformity_score.
      5) Combine:
           edge_density = average border edge density.
           rect_score   = rectangularity_score.
           bezel_score  = bezel_uniformity_score.

           border_edge_rect = 0.5 * edge_density + 0.5 * rect_score
           final_strength    = 0.5 * border_edge_rect + 0.5 * bezel_score

    Returns
    -------
    (strength, edge_density, rectangularity_score, bezel_uniformity_score)
        All values clamped to [0,1].
    """
    if outer_region is None or outer_region.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    gray = cv2.cvtColor(outer_region, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 200)

    h, w = edges.shape[:2]
    if h < 4 or w < 4:
        return 0.0, 0.0, 0.0, 0.0

    band = max(1, int(0.05 * min(h, w)))

    top_band_edges = edges[0:band, :]
    bottom_band_edges = edges[h - band : h, :]
    left_band_edges = edges[:, 0:band]
    right_band_edges = edges[:, w - band : w]

    top_band_gray = gray[0:band, :]
    bottom_band_gray = gray[h - band : h, :]
    left_band_gray = gray[:, 0:band]
    right_band_gray = gray[:, w - band : w]

    def density(mask: np.ndarray) -> float:
        total = mask.size
        if total == 0:
            return 0.0
        return float(np.count_nonzero(mask)) / float(total)

    d_top = density(top_band_edges)
    d_bottom = density(bottom_band_edges)
    d_left = density(left_band_edges)
    d_right = density(right_band_edges)

    edge_density = 0.5 * (0.5 * (d_top + d_bottom) + 0.5 * (d_left + d_right))
    edge_density = _clamp01(edge_density)

    rectangularity_score = _rectangularity_from_edges(
        edges, band=band, angle_tol_deg=angle_tol_deg
    )

    bezel_uniformity_score = _bezel_uniformity_from_bands(
        [top_band_gray, bottom_band_gray, left_band_gray, right_band_gray],
        bezel_uniformity_norm=bezel_uniformity_norm,
    )

    border_edge_rect = 0.5 * edge_density + 0.5 * rectangularity_score
    border_edge_rect = _clamp01(border_edge_rect)

    strength = 0.5 * border_edge_rect + 0.5 * bezel_uniformity_score
    strength = _clamp01(strength)

    return strength, edge_density, rectangularity_score, bezel_uniformity_score


def _rectangularity_from_edges(
    edges: np.ndarray,
    band: int,
    angle_tol_deg: float,
) -> float:
    """
    Estimate how well edges form a rectangular frame.

    - Use HoughLinesP to detect line segments.
    - Classify segments as horizontal / vertical based on angle tolerance.
    - Require them to lie near the image borders (within 'band').
    - Score = total length of such border-aligned segments
             / (approximate perimeter length), clamped to [0,1].
    """
    if edges is None or edges.size == 0:
        return 0.0

    h, w = edges.shape[:2]
    min_dim = min(h, w)
    if min_dim < 4:
        return 0.0

    try:
        lines = cv2.HoughLinesP(
            edges,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=max(20, int(0.05 * min_dim)),
            minLineLength=max(10, int(0.3 * min_dim)),
            maxLineGap=5,
        )
    except Exception:
        return 0.0

    if lines is None:
        return 0.0

    angle_tol = max(1.0, float(angle_tol_deg))
    border_band = band

    total_border_len = 0.0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length <= 1.0:
            continue

        angle_rad = math.atan2(dy, dx)
        angle_deg = abs(angle_rad * 180.0 / math.pi)
        if angle_deg >= 180.0:
            angle_deg = angle_deg % 180.0

        is_horizontal = min(angle_deg, 180.0 - angle_deg) <= angle_tol
        is_vertical = abs(angle_deg - 90.0) <= angle_tol

        if not (is_horizontal or is_vertical):
            continue

        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)

        near_top = my <= border_band
        near_bottom = my >= (h - border_band)
        near_left = mx <= border_band
        near_right = mx >= (w - border_band)

        if is_horizontal and (near_top or near_bottom):
            total_border_len += length
        elif is_vertical and (near_left or near_right):
            total_border_len += length

    if total_border_len <= 0.0:
        return 0.0

    perimeter = 2.0 * (float(w) + float(h))
    if perimeter <= 0.0:
        return 0.0

    score = total_border_len / perimeter
    return _clamp01(score)


def _bezel_uniformity_from_bands(
    bands: List[np.ndarray],
    bezel_uniformity_norm: float,
) -> float:
    """
    Compute bezel uniformity score from a list of gray bands
    (top, bottom, left, right).

    - Concatenate all bands into a single 1D array of intensities.
    - Compute std-dev; lower std-dev → more uniform bezel.
    - Map to [0,1] as:
          score = 1 - clamp(std / norm)
    """
    vals: List[float] = []
    for band in bands:
        if band is None or band.size == 0:
            continue
        flat = band.astype(np.float32).ravel()
        if flat.size == 0:
            continue
        vals.append(flat)

    if not vals:
        return 0.0

    all_vals = np.concatenate(vals, axis=0)
    if all_vals.size == 0:
        return 0.0

    std_val = float(np.std(all_vals))
    norm = max(1e-3, float(bezel_uniformity_norm))

    raw = 1.0 - (std_val / norm)
    return _clamp01(raw)




def _flicker_strength(
    brightness_series: List[float],
    flicker_var_norm: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Compute flicker strength from a brightness time series.

    Heuristic:
      - Compute raw variance and relative variance var / (mean^2).
      - Compute fraction of alternating sign changes in first differences.
      - Combine both into a strength ∈ [0,1].

    Returns
    -------
    strength : float in [0,1]
    raw_var : float
    rel_var : float
    """
    n = len(brightness_series)
    if n < 3:
        return 0.0, 0.0, 0.0

    arr = np.asarray(brightness_series, dtype=np.float32)
    mean = float(np.mean(arr))
    var = float(np.var(arr))

    if mean <= 1e-3:
        rel_var = 0.0
    else:
        rel_var = float(var / (mean * mean + 1e-6))

    if flicker_var_norm <= 1e-6:
        var_score = 0.0
    else:
        var_score = rel_var / (rel_var + flicker_var_norm)
    var_score = _clamp01(var_score)

    diffs = np.diff(arr)
    if diffs.size < 2:
        osc_score = 0.0
    else:
        signs = np.sign(diffs)
        sign_changes = 0
        total_pairs = 0
        for i in range(1, len(signs)):
            if signs[i - 1] == 0 or signs[i] == 0:
                continue
            total_pairs += 1
            if signs[i - 1] * signs[i] < 0:
                sign_changes += 1
        osc_score = (
            float(sign_changes) / float(total_pairs) if total_pairs > 0 else 0.0
        )

    osc_score = _clamp01(osc_score)

    strength = 0.6 * var_score + 0.4 * osc_score
    strength = _clamp01(strength)

    return strength, var, rel_var




def _laplacian_energy(gray: np.ndarray) -> float:
    """
    Compute Laplacian variance as a proxy for high-frequency energy.

    A high, very structured value over time can correspond to screen patterns.
    """
    if gray is None or gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def _moire_strength(
    moire_values: List[float],
    moire_norm: float = 300.0,
) -> float:
    """
    Map Laplacian-variance statistics to a [0,1] moiré strength.

    Heuristic:
      - Use mean Laplacian variance over the window.
      - Normalise with a soft knee: score = mean / (mean + moire_norm).
    """
    if not moire_values:
        return 0.0

    mean_val = float(np.mean(moire_values))

    if mean_val <= 0.0 or moire_norm <= 1e-6:
        return 0.0

    score = mean_val / (mean_val + moire_norm)
    return _clamp01(score)




def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
