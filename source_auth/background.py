
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]
FrameEntry = Tuple[np.ndarray, Optional[BBox]]




def compute_background_consistency(
    frames_window: Sequence[FrameEntry],
    cfg,
) -> Tuple[float, bool, Dict[str, float]]:
    """
    Compute background consistency cue for a single track.

    Parameters
    ----------
    frames_window : sequence of (image_bgr, bbox_in_frame)
        - image_bgr: np.ndarray HxWx3 (BGR)
        - bbox_in_frame: (x1, y1, x2, y2) or None

    cfg : SourceAuthConfig-like
        Can be:
          - SourceAuthConfig (with a .background sub-config), or
          - SourceAuthBackgroundConfig directly.

    Returns
    -------
    score : float
        Background consistency in [0,1].
        0   → strong mismatch (inner vs outer differ a lot, likely phone)
        1   → strong consistency (inner vs outer similar, likely real head)

    reliable : bool
        True if we had enough frames / pixels to trust the cue.

    debug : dict
        Diagnostic values:
          - "color_delta"
          - "texture_delta"
          - "valid_frames"
          - "inner_pixels"
          - "outer_pixels"
    """
    neutral = float(getattr(cfg, "neutral_score", 0.5))

    if not frames_window:
        return neutral, False, {}

    bg_cfg = getattr(cfg, "background", cfg)

    min_face_px = int(getattr(cfg, "background_min_face_px", 40))
    color_norm = float(getattr(cfg, "background_color_norm", 0.3))
    texture_norm = float(getattr(cfg, "background_texture_norm", 20.0))
    min_frames = int(getattr(cfg, "background_min_frames", 3))

    inner_margin_frac = float(getattr(bg_cfg, "inner_margin_frac", 0.15))
    outer_margin_frac = float(getattr(bg_cfg, "outer_margin_frac", 0.25))
    min_region_pixels = int(getattr(bg_cfg, "min_region_pixels", 500))

    color_hist_bins = int(getattr(bg_cfg, "color_hist_bins", 32))

    blur_sigma = float(getattr(bg_cfg, "blur_sigma_for_noise", 0.5))
    enable_noise_hint = bool(
        getattr(bg_cfg, "enable_noise_sharpness_hint", True)
    )

    w_color = float(getattr(bg_cfg, "w_color", 0.7))
    w_texture = float(getattr(bg_cfg, "w_texture", 0.3))

    color_deltas: List[float] = []
    texture_deltas: List[float] = []
    scores: List[float] = []

    total_inner_pixels = 0
    total_outer_pixels = 0
    valid_frames = 0

    for img_bgr, bbox in frames_window:
        img = _to_valid_bgr(img_bgr)
        if img is None or bbox is None:
            continue

        h, w = img.shape[:2]

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
        face_min = min(face_w, face_h)
        if face_min < min_face_px:
            continue

        inner = _extract_inner_region(
            img, (ix1, iy1, ix2, iy2), margin_frac=inner_margin_frac
        )
        if inner is None or inner.size == 0:
            continue

        outer = _extract_outer_ring(
            img, (ix1, iy1, ix2, iy2), margin_frac=outer_margin_frac
        )
        if outer is None or outer.size == 0:
            continue

        inner_pixels = int(inner.shape[0] * inner.shape[1])
        outer_pixels = int(outer.shape[0] * outer.shape[1])
        if inner_pixels < min_region_pixels or outer_pixels < min_region_pixels:
            continue

        try:
            hist_inner = _color_hist_hsv(inner, color_hist_bins)
            hist_outer = _color_hist_hsv(outer, color_hist_bins)

            color_delta = _hist_l1_distance(hist_inner, hist_outer)

            if enable_noise_hint:
                tex_inner = _laplacian_variance(inner, blur_sigma)
                tex_outer = _laplacian_variance(outer, blur_sigma)
                texture_delta = abs(tex_inner - tex_outer)
            else:
                tex_inner = tex_outer = 0.0
                texture_delta = 0.0

            color_mismatch = _soft_norm(color_delta, color_norm)
            texture_mismatch = _soft_norm(texture_delta, texture_norm)

            mismatch = w_color * color_mismatch + w_texture * texture_mismatch
            mismatch = max(0.0, min(1.0, mismatch))
            bg_score = 1.0 - mismatch
            bg_score = max(0.0, min(1.0, bg_score))

            color_deltas.append(float(color_delta))
            texture_deltas.append(float(texture_delta))
            scores.append(float(bg_score))

            total_inner_pixels += inner_pixels
            total_outer_pixels += outer_pixels
            valid_frames += 1

        except Exception:
            logger.exception("background: per-frame processing failed")
            continue

    if valid_frames == 0:
        return neutral, False, {
            "color_delta": 0.0,
            "texture_delta": 0.0,
            "valid_frames": 0.0,
            "inner_pixels": float(total_inner_pixels),
            "outer_pixels": float(total_outer_pixels),
        }

    score = float(np.mean(scores))

    avg_color_delta = float(np.mean(color_deltas)) if color_deltas else 0.0
    avg_texture_delta = float(np.mean(texture_deltas)) if texture_deltas else 0.0

    reliable = valid_frames >= min_frames

    debug = {
        "color_delta": avg_color_delta,
        "texture_delta": avg_texture_delta,
        "valid_frames": float(valid_frames),
        "inner_pixels": float(total_inner_pixels),
        "outer_pixels": float(total_outer_pixels),
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
        logger.debug("background: unexpected image shape %s", img.shape)
        return None

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _extract_inner_region(
    image: np.ndarray,
    bbox_int: Tuple[int, int, int, int],
    margin_frac: float = 0.15,
) -> Optional[np.ndarray]:
    """
    Extract a shrunken inner rectangle within the face bbox.

    The idea is to avoid the very border (where hair/edges dominate) and
    take a core region that still includes some background + head context.

    Parameters
    ----------
    image : np.ndarray
        Full frame HxWx3 (BGR).
    bbox_int : (x1, y1, x2, y2)
        Integer coordinates of face / head region.
    margin_frac : float
        Fraction of bbox width/height to shrink from each side.

    Returns
    -------
    inner_region : np.ndarray or None
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_int

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    mx = int(round(bw * margin_frac))
    my = int(round(bh * margin_frac))

    ix1 = max(0, x1 + mx)
    iy1 = max(0, y1 + my)
    ix2 = min(w, x2 - mx)
    iy2 = min(h, y2 - my)

    if ix2 <= ix1 + 1 or iy2 <= iy1 + 1:
        return None

    return image[iy1:iy2, ix1:ix2]


def _extract_outer_ring(
    image: np.ndarray,
    bbox_int: Tuple[int, int, int, int],
    margin_frac: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Extract a ring-like region around the face bbox.

    Implementation detail:
      - Expand the bbox by margin_frac.
      - Build four bands (top, bottom, left, right) outside the original bbox.
      - Concatenate them into a single region.

    This approximates the "environment" around the head/phone.

    Parameters
    ----------
    image : np.ndarray
        Full frame HxWx3 (BGR).
    bbox_int : (x1, y1, x2, y2)
        Integer coordinates of face / head region.
    margin_frac : float
        Expansion margin proportional to bbox size.

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

    mx = int(round(bw * margin_frac))
    my = int(round(bh * margin_frac))

    ox1 = max(0, x1 - mx)
    oy1 = max(0, y1 - my)
    ox2 = min(w, x2 + mx)
    oy2 = min(h, y2 + my)

    if ox2 <= ox1 + 1 or oy2 <= oy1 + 1:
        return None

    patches: List[np.ndarray] = []

    if oy1 < y1:
        top = image[oy1:y1, ox1:ox2]
        if top.size > 0:
            patches.append(top)

    if y2 < oy2:
        bottom = image[y2:oy2, ox1:ox2]
        if bottom.size > 0:
            patches.append(bottom)

    if ox1 < x1:
        left = image[y1:y2, ox1:x1]
        if left.size > 0:
            patches.append(left)

    if x2 < ox2:
        right = image[y1:y2, x2:ox2]
        if right.size > 0:
            patches.append(right)

    if not patches:
        return None

    outer_region = np.concatenate(
        [p.reshape(-1, 3) for p in patches],
        axis=0,
    ).reshape(-1, 1, 3)

    return outer_region




def _color_hist_hsv(
    patch: np.ndarray,
    bins: int = 32,
) -> np.ndarray:
    """
    Compute a concatenated HSV histogram (H, S, V) normalised to sum 1.

    Returns
    -------
    hist : np.ndarray of shape (bins*3,), dtype=float32
    """
    if patch is None or patch.size == 0:
        return np.zeros((bins * 3,), dtype=np.float32)

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    h_range = [0, 180]
    sv_range = [0, 256]

    h_hist = cv2.calcHist([hsv], [0], None, [bins], h_range)
    s_hist = cv2.calcHist([hsv], [1], None, [bins], sv_range)
    v_hist = cv2.calcHist([hsv], [2], None, [bins], sv_range)

    hist = np.concatenate([h_hist, s_hist, v_hist], axis=0).astype(np.float32)
    total = float(hist.sum())
    if total > 0.0:
        hist /= total

    return hist.reshape(-1)


def _hist_l1_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Simple L1 distance between two normalised histograms.

    Range:
      - If both hist are valid probability distributions,
        L1 ∈ [0, 2]. We then re-normalise with a soft factor.
    """
    if h1 is None or h2 is None:
        return 0.0
    h1 = np.asarray(h1, dtype=np.float32).reshape(-1)
    h2 = np.asarray(h2, dtype=np.float32).reshape(-1)
    if h1.shape != h2.shape:
        n = min(h1.shape[0], h2.shape[0])
        h1 = h1[:n]
        h2 = h2[:n]
    return float(np.sum(np.abs(h1 - h2)))


def _laplacian_variance(
    patch_bgr: np.ndarray,
    blur_sigma: float = 0.5,
) -> float:
    """
    Laplacian variance as a proxy for texture / sharpness.

    Parameters
    ----------
    patch_bgr : np.ndarray
        BGR patch (any shape).
    blur_sigma : float
        Optional Gaussian blur sigma before Laplacian to reduce noise.

    Returns
    -------
    var : float
        Variance of Laplacian response.
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    if blur_sigma > 0.0:
        ksize = int(max(3, 2 * int(blur_sigma * 3) + 1))
        gray = cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))




def _soft_norm(value: float, norm: float) -> float:
    """
    Map a non-negative value to [0,1] with a "soft-knee" normalisation:

        score = value / (value + norm)

    If norm is very small or value is non-positive, returns 0.
    """
    if value <= 0.0 or norm <= 1e-6:
        return 0.0
    return float(value / (value + norm))
