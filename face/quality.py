
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import FaceConfig, FaceThresholdConfig, default_face_config

logger = logging.getLogger(__name__)

_W_DET    = 0.35
_W_SHARP  = 0.25
_W_POSE   = 0.20
_W_SIZE   = 0.15
_W_BRIGHT = 0.05


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _to_gray(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert image to single-channel uint8 grayscale, or return None if invalid.
    """
    if image is None or image.size == 0:
        return None

    img = np.asarray(image)
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        logger.warning("Unexpected image shape in quality module: %s", img.shape)
        return None

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def estimate_brightness_score(image: np.ndarray) -> float:
    """
    Simple normalized brightness check.
    Returns 0 (too dark/too bright) → 1 (ideal brightness).
    """
    gray = _to_gray(image)
    if gray is None:
        return 0.0

    mean = float(np.mean(gray))

    ideal_low, ideal_high = 80.0, 170.0

    if mean <= ideal_low:
        return _clamp01(mean / max(ideal_low, 1e-6))

    if mean >= ideal_high:
        return _clamp01((255.0 - mean) / max(255.0 - ideal_high, 1e-6))

    return 1.0


def estimate_sharpness(image: np.ndarray) -> float:
    """
    Laplacian-of-variance → mapped softly into [0,1].

    High-variance edges → close to 1.
    Flat / blurred images → close to 0.
    """
    gray = _to_gray(image)
    if gray is None:
        return 0.0

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())

    k = 250.0
    sharp = var / (var + k)
    return _clamp01(sharp)


def compute_size_score(
    bbox: Tuple[float, float, float, float],
    img_shape: Tuple[int, int, int],
    th: FaceThresholdConfig,
) -> float:
    """
    Score based on face height relative to image height.

    - Faces below min_face_height_px → 0.
    - Around 25% of the image height → ideal.
    - Very large faces up to ~60% of image height → still good (slightly boosted).
    """
    x1, y1, x2, y2 = bbox
    face_h = float(max(0.0, y2 - y1))
    img_h = float(img_shape[0])

    if face_h <= 0.0 or img_h <= 0.0:
        return 0.0
    if face_h < th.min_face_height_px:
        return 0.0

    ratio = face_h / img_h
    ideal = 0.25
    max_ratio = 0.60

    if ratio >= max_ratio:
        return 1.0

    if ratio <= ideal:
        return _clamp01(ratio / max(ideal, 1e-6))

    extra = (ratio - ideal) / max(max_ratio - ideal, 1e-6)
    return _clamp01(0.90 + 0.10 * extra)


def compute_pose_score(
    yaw: Optional[float],
    pitch: Optional[float],
    th: FaceThresholdConfig,
) -> float:
    """
    Score for head pose based on yaw and pitch.

    - 1.0 for perfectly frontal faces.
    - Gradual decay as we approach max_yaw_deg / max_pitch_deg.
    - 0.0 for poses beyond those maximums.

    This is used both:
      - for runtime gating (q_runtime),
      - and as a signal for 3D: we still keep side views if they are within
        configured limits; we just score them lower quality.
    """
    if yaw is None or pitch is None:
        return 0.55

    ay = abs(float(yaw))
    ap = abs(float(pitch))

    if ay >= th.max_yaw_deg:
        sy = 0.0
    else:
        sy = 1.0 - ay / max(th.max_yaw_deg, 1e-6)

    if ap >= th.max_pitch_deg:
        sp = 0.0
    else:
        sp = 1.0 - ap / max(th.max_pitch_deg, 1e-6)

    score = 0.6 * sy + 0.4 * sp
    return _clamp01(score)


def combine_face_quality(
    det_score: float,
    sharpness: float,
    pose_score: float,
    size_score: float,
    brightness: float,
    th: Optional[FaceThresholdConfig] = None,
) -> float:
    """
    Fuse all quality factors into a single q ∈ [0,1].

    This is the central gate used by:
      - enrollment (q_enroll, q_embed),
      - runtime identity (q_runtime).
    """
    th = th or default_face_config().thresholds

    ds = _clamp01(det_score)
    sh = _clamp01(sharpness)
    ps = _clamp01(pose_score)
    sz = _clamp01(size_score)
    br = _clamp01(brightness)

    if ds < th.min_det_score:
        return 0.0

    q = (
        _W_DET    * ds +
        _W_SHARP  * sh +
        _W_POSE   * ps +
        _W_SIZE   * sz +
        _W_BRIGHT * br
    )

    return _clamp01(q)


def compute_full_quality(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    det_score: float,
    yaw: Optional[float],
    pitch: Optional[float],
    cfg: Optional[FaceConfig] = None,
) -> float:
    """
    Full face-quality pipeline:
      sharpness + pose + size + brightness fused with detector score.

    Returns
    -------
    q : float
        Quality in [0,1].

    This function is used both at enrollment and at runtime, so it defines the
    common "language" for quality in the whole system (2D + 3D).
    """
    cfg = cfg or default_face_config()
    th = cfg.thresholds

    sharp = estimate_sharpness(image)
    size = compute_size_score(bbox, image.shape, th)
    pose = compute_pose_score(yaw, pitch, th)
    bright = estimate_brightness_score(image)

    return combine_face_quality(
        det_score=det_score,
        sharpness=sharp,
        pose_score=pose,
        size_score=size,
        brightness=bright,
        th=th,
    )


def compute_quality_from_candidate(
    full_frame_bgr: np.ndarray,
    bbox: Tuple[float, float, float, float],
    det_score: float,
    yaw: Optional[float],
    pitch: Optional[float],
    cfg: Optional[FaceConfig] = None,
) -> float:
    """
    Convenience wrapper to compute quality from the same ingredients that a
    FaceCandidate carries.

    This keeps enrollment and runtime usage consistent:
      - both use the same q_face logic,
      - 3D logic can later reuse q_face for deciding which samples are allowed
        to update multi-view head models.
    """
    return compute_full_quality(
        image=full_frame_bgr,
        bbox=bbox,
        det_score=det_score,
        yaw=yaw,
        pitch=pitch,
        cfg=cfg,
    )
