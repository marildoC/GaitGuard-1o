

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from source_auth.config import SourceAuthConfig
from source_auth.types import LandmarkFrame

logger = logging.getLogger(__name__)


def _fit_affine_2d(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Fit a 2D affine transform A (2x3) such that:

        [x'; y']^T ≈ A @ [x, y, 1]^T

    using least squares over all landmark pairs.

    Parameters
    ----------
    src : (N, 2) float32
        Source 2D points (reference frame landmarks).
    dst : (N, 2) float32
        Destination 2D points (current frame landmarks).

    Returns
    -------
    (A, ok) : (np.ndarray, bool)
        A  : (2, 3) affine matrix (if ok is True; undefined otherwise)
        ok : False if there were too few points or the system was singular.

    Notes
    -----
    - We require N >= 3 to fit a full affine model.
    - If fitting fails, caller should ignore this pair.
    """
    try:
        src = np.asarray(src, dtype=np.float32).reshape(-1, 2)
        dst = np.asarray(dst, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return np.zeros((2, 3), dtype=np.float32), False

    n = src.shape[0]
    if n < 3 or dst.shape[0] != n:
        return np.zeros((2, 3), dtype=np.float32), False

    A = np.zeros((2 * n, 6), dtype=np.float32)
    b = np.zeros((2 * n,), dtype=np.float32)

    x = src[:, 0]
    y = src[:, 1]
    xp = dst[:, 0]
    yp = dst[:, 1]

    A[0::2, 0] = x
    A[0::2, 1] = y
    A[0::2, 2] = 1.0
    A[1::2, 3] = x
    A[1::2, 4] = y
    A[1::2, 5] = 1.0

    b[0::2] = xp
    b[1::2] = yp

    try:
        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return np.zeros((2, 3), dtype=np.float32), False

    if rank < 4:
        return np.zeros((2, 3), dtype=np.float32), False

    A_mat = np.array(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
        ],
        dtype=np.float32,
    )
    return A_mat, True


def _apply_affine_2d(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply 2D affine transform A (2x3) to points pts (N, 2).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([pts, ones], axis=1)
    out = (A @ hom.T).T
    return out


def _estimate_face_size_px(history: List[LandmarkFrame]) -> float:
    """
    Estimate a characteristic face size in pixels from LandmarkFrame.bbox
    over the window. We use mean(min(width, height)).
    """
    sizes: List[float] = []
    for frm in history:
        bbox = getattr(frm, "bbox", None)
        if bbox is None:
            continue
        try:
            x1, y1, x2, y2 = bbox
            w = float(max(0.0, x2 - x1))
            h = float(max(0.0, y2 - y1))
            s = min(w, h)
            if s > 0.0:
                sizes.append(s)
        except Exception:
            continue

    if not sizes:
        return 0.0
    return float(np.mean(sizes))


def _compute_yaw_pitch_span(history: List[LandmarkFrame]) -> Tuple[float, float]:
    """
    Compute yaw and pitch span (max - min) over the window, in degrees.
    If no values are available, returns (0.0, 0.0).
    """
    yaws: List[float] = []
    pitches: List[float] = []

    for frm in history:
        y = getattr(frm, "yaw_deg", None)
        p = getattr(frm, "pitch_deg", None)
        if y is not None:
            try:
                yaws.append(float(y))
            except Exception:
                pass
        if p is not None:
            try:
                pitches.append(float(p))
            except Exception:
                pass

    yaw_span = max(yaws) - min(yaws) if len(yaws) >= 2 else 0.0
    pitch_span = max(pitches) - min(pitches) if len(pitches) >= 2 else 0.0
    return float(yaw_span), float(pitch_span)


def _accumulate_motion_stats_for_ref(
    history: List[LandmarkFrame],
    ref_index: int,
) -> Tuple[List[float], List[float]]:
    """
    For a given reference frame index, accumulate:
      - global_motion_list : per-pair mean ||p_t - p_ref||
      - residual_list      : per-pair mean ||p_t - T(p_ref)||, T = affine

    Only pairs with matching landmark counts and valid affine fit
    contribute to residual_list; pairs with mismatched landmark count
    only contribute to global_motion_list.
    """
    ref = history[ref_index]
    ref_pts = np.asarray(ref.landmarks_2d, dtype=np.float32).reshape(-1, 2)
    n_landmarks = ref_pts.shape[0]

    global_motion_list: List[float] = []
    residual_list: List[float] = []

    if n_landmarks == 0:
        return global_motion_list, residual_list

    for frm in history[ref_index + 1 :]:
        cur_pts = np.asarray(frm.landmarks_2d, dtype=np.float32).reshape(-1, 2)
        if cur_pts.shape[0] != n_landmarks:
            continue

        disp = cur_pts - ref_pts
        disp_norm = np.linalg.norm(disp, axis=1)
        global_motion_mag = float(disp_norm.mean())
        global_motion_list.append(global_motion_mag)

        A, ok = _fit_affine_2d(ref_pts, cur_pts)
        if not ok:
            continue

        pred = _apply_affine_2d(A, ref_pts)
        residual = cur_pts - pred
        residual_norm = np.linalg.norm(residual, axis=1)
        residual_energy = float(residual_norm.mean())
        residual_list.append(residual_energy)

    return global_motion_list, residual_list


def compute_3d_motion_score(
    history: List[LandmarkFrame],
    cfg: SourceAuthConfig,
) -> Tuple[float, bool, Dict[str, float]]:
    """
    Compute a 3D-vs-planar motion score based on a short history of landmarks.

    Parameters
    ----------
    history : List[LandmarkFrame]
        Time-ordered list (oldest first) of LandmarkFrame samples over
        ~motion_window_sec. Each contains ts, bbox, landmarks_2d, quality.
    cfg : SourceAuthConfig
        Configuration with motion thresholds and defaults.

    Returns
    -------
    score : float
        0   → motion is very planar / card-like
        0.5 → neutral / insufficient evidence
        1   → strong 3D parallax (real head)
    reliable : bool
        True  → enough motion & samples to trust this cue.
        False → treat as neutral; other cues should dominate.
    debug : dict
        Diagnostic stats:
            - parallax_ratio
            - global_motion_mag
            - residual_energy
            - n_frames
            - span_sec
            - n_landmarks
            - yaw_span_deg
            - pitch_span_deg
            - face_size_px
            - min_motion_pixels_used
            - parallax_ratio_ref0
            - parallax_ratio_ref1
    """
    neutral = float(getattr(cfg, "neutral_score", 0.5))

    n_frames = len(history)
    debug: Dict[str, float] = {
        "parallax_ratio": 0.0,
        "global_motion_mag": 0.0,
        "residual_energy": 0.0,
        "n_frames": float(n_frames),
        "span_sec": 0.0,
        "n_landmarks": 0.0,
        "yaw_span_deg": 0.0,
        "pitch_span_deg": 0.0,
        "face_size_px": 0.0,
        "min_motion_pixels_used": 0.0,
        "parallax_ratio_ref0": 0.0,
        "parallax_ratio_ref1": 0.0,
    }

    if n_frames < 2:
        return neutral, False, debug

    motion_window_sec = float(
        getattr(cfg, "motion_window_sec", getattr(cfg, "window_sec", 1.5))
    )
    min_span_factor = float(getattr(cfg, "motion_min_span_factor", 0.5))
    min_span_sec = motion_window_sec * min_span_factor

    t0 = float(history[0].ts)
    t1 = float(history[-1].ts)
    span_sec = max(0.0, t1 - t0)
    debug["span_sec"] = span_sec

    if span_sec < min_span_sec:
        return neutral, False, debug

    ref0 = history[0]
    ref0_pts = np.asarray(ref0.landmarks_2d, dtype=np.float32).reshape(-1, 2)
    n_landmarks = ref0_pts.shape[0]
    debug["n_landmarks"] = float(n_landmarks)

    min_landmarks = int(getattr(cfg, "motion_min_landmarks", 3))
    if n_landmarks < min_landmarks:
        return neutral, False, debug

    yaw_span_deg, pitch_span_deg = _compute_yaw_pitch_span(history)
    debug["yaw_span_deg"] = yaw_span_deg
    debug["pitch_span_deg"] = pitch_span_deg

    face_size_px = _estimate_face_size_px(history)
    debug["face_size_px"] = face_size_px

    base_pixels = float(
        getattr(
            cfg,
            "motion_min_motion_pixels_base",
            getattr(cfg, "min_motion_pixels", 2.0),
        )
    )
    frac = float(getattr(cfg, "motion_min_motion_frac", 0.01))
    adaptive_pixels = face_size_px * frac if face_size_px > 0.0 else 0.0
    min_motion_pixels = max(base_pixels, adaptive_pixels)
    debug["min_motion_pixels_used"] = min_motion_pixels

    all_global_motion: List[float] = []
    all_residuals: List[float] = []
    parallax_by_ref: List[float] = []

    ref_indices: List[int] = [0]

    if n_frames >= 3:
        mid_index = n_frames // 2
        if mid_index not in ref_indices:
            ref_indices.append(mid_index)

    parallax_ref_debug: List[float] = []

    for ref_idx in ref_indices:
        global_motion_list, residual_list = _accumulate_motion_stats_for_ref(
            history, ref_idx
        )

        if not global_motion_list:
            parallax_ref_debug.append(0.0)
            continue

        global_motion_mag_ref = float(np.mean(global_motion_list))
        residual_energy_ref = float(np.mean(residual_list)) if residual_list else 0.0

        all_global_motion.extend(global_motion_list)
        all_residuals.extend(residual_list)

        if global_motion_mag_ref > 0.0 and residual_energy_ref > 0.0:
            parallax_ratio_ref = float(
                residual_energy_ref / (global_motion_mag_ref + 1e-6)
            )
        else:
            parallax_ratio_ref = 0.0

        parallax_by_ref.append(parallax_ratio_ref)
        parallax_ref_debug.append(parallax_ratio_ref)

    if parallax_ref_debug:
        debug["parallax_ratio_ref0"] = float(parallax_ref_debug[0])
        if len(parallax_ref_debug) > 1:
            debug["parallax_ratio_ref1"] = float(parallax_ref_debug[1])

    if not all_global_motion:
        return neutral, False, debug

    global_motion_mag = float(np.mean(all_global_motion))
    residual_energy = float(np.mean(all_residuals)) if all_residuals else 0.0

    debug["global_motion_mag"] = global_motion_mag
    debug["residual_energy"] = residual_energy

    if residual_energy <= 0.0:
        return neutral, False, debug

    if global_motion_mag < min_motion_pixels:
        return neutral, False, debug

    eps = 1e-6
    if parallax_by_ref:
        parallax_ratio = float(max(parallax_by_ref))
    else:
        parallax_ratio = float(residual_energy / (global_motion_mag + eps))

    debug["parallax_ratio"] = parallax_ratio

    parallax_low = float(getattr(cfg, "parallax_low", 0.05))
    parallax_high = float(getattr(cfg, "parallax_high", 0.25))

    if parallax_high <= parallax_low:
        parallax_high = parallax_low + 1e-3

    if parallax_ratio <= parallax_low:
        score = 0.0
    elif parallax_ratio >= parallax_high:
        score = 1.0
    else:
        alpha = (parallax_ratio - parallax_low) / (parallax_high - parallax_low)
        score = float(alpha)

    yaw_thresh = float(getattr(cfg, "motion_yaw_deg_threshold", 8.0))
    yaw_pitch_span = max(yaw_span_deg, pitch_span_deg)
    yaw_boost = float(getattr(cfg, "motion_yaw_boost", 0.15))

    if yaw_pitch_span >= yaw_thresh and residual_energy > 0.0:
        score = min(1.0, max(0.0, score + yaw_boost))

    score = max(0.0, min(1.0, score))

    reliable = True

    return score, reliable, debug
