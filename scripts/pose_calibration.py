"""
scripts/pose_calibration.py

Interactive tool to MEASURE yaw, pitch and a pose-based quality score
from the buffalo_l stack on your own camera.

Goal:
  - See what yaw/pitch/quality the model actually produces when you look
    FRONT / LEFT / RIGHT / UP / DOWN.
  - Use these stats to tune FaceMultiViewConfig and quality thresholds
    so guided multi-view enrollment stops rejecting good views.

Usage:
    python -m scripts.pose_calibration

Controls while window is open:
    f  -> record current face sample as FRONT
    l  -> record current face sample as LEFT
    r  -> record current face sample as RIGHT
    u  -> record current face sample as UP
    d  -> record current face sample as DOWN
    c  -> clear all recorded samples
    q  -> quit and show summary
"""

from __future__ import annotations

import collections
import statistics
from typing import Dict, List, Tuple

import cv2
import numpy as np

from insightface.app import FaceAnalysis

from face.config import FaceConfig, default_face_config

PoseSample = Tuple[float, float, float]




def _draw_text(
    img,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.6,
    thickness: int = 1,
):
    """Simple OpenCV text helper."""
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def _summarise(samples: List[PoseSample]) -> str:
    """Return a compact summary string for a list of pose samples."""
    if not samples:
        return "n=0"

    yaws = [s[0] for s in samples]
    pitches = [s[1] for s in samples]
    quals = [s[2] for s in samples]

    def stats(arr):
        return (
            min(arr),
            statistics.mean(arr),
            max(arr),
        )

    yaw_min, yaw_mean, yaw_max = stats(yaws)
    p_min, p_mean, p_max = stats(pitches)
    q_min, q_mean, q_max = stats(quals)

    return (
        f"n={len(samples)} | "
        f"yaw: [{yaw_min:.1f}, {yaw_mean:.1f}, {yaw_max:.1f}] | "
        f"pitch: [{p_min:.1f}, {p_mean:.1f}, {p_max:.1f}] | "
        f"q: [{q_min:.3f}, {q_mean:.3f}, {q_max:.3f}]"
    )


def _build_face_config() -> FaceConfig:
    """
    Load FaceConfig in the same way as the rest of the system.

    If FaceConfig.from_env() exists, use it (reads YAML etc.),
    otherwise fall back to default_face_config().
    """
    if hasattr(FaceConfig, "from_env"):
        cfg = FaceConfig.from_env()
    else:
        cfg = default_face_config(prefer_gpu=True)

    print(
        "[pose_calibration] FaceConfig loaded | "
        f"device={cfg.device.device} half={cfg.device.use_half} | "
        f"q_enroll={cfg.thresholds.min_quality_enroll:.2f} | "
        f"mv(yaw_front={cfg.multiview.yaw_front_deg}, "
        f"yaw_side_max={cfg.multiview.yaw_side_max_deg}, "
        f"pitch_up={cfg.multiview.pitch_up_deg}, "
        f"pitch_down={cfg.multiview.pitch_down_deg})"
    )
    return cfg


def _build_insight_app(cfg: FaceConfig) -> FaceAnalysis:
    """
    Build an InsightFace FaceAnalysis app that mirrors your buffalo_l stack.
    """
    pack_name = cfg.models.retinaface_name or "buffalo_l"

    device_str = str(cfg.device.device).lower()
    if "cuda" in device_str or device_str in ("0", "0,0", "cuda:0"):
        ctx_id = 0
    else:
        ctx_id = -1

    app = FaceAnalysis(name=pack_name)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    return app


def _compute_quality(
    yaw_deg: float,
    pitch_deg: float,
    det_score: float,
    cfg: FaceConfig,
) -> float:
    """
    Approximate a face quality score using:
      - detection score from InsightFace
      - pose penalty based on your FaceThresholdConfig

    This does NOT have to be identical to face/quality.py; for calibration
    we mainly care that q is higher for good, frontal-ish faces and lower
    for extreme poses.
    """
    thr = cfg.thresholds

    q_det = float(det_score)
    if q_det < 0.0:
        q_det = 0.0
    if q_det > 1.0:
        q_det = 1.0

    max_yaw = max(1e-6, abs(thr.max_yaw_deg))
    max_pitch = max(1e-6, abs(thr.max_pitch_deg))

    yaw_penalty = max(0.0, 1.0 - abs(yaw_deg) / max_yaw)
    pitch_penalty = max(0.0, 1.0 - abs(pitch_deg) / max_pitch)

    q_pose = 0.5 * (yaw_penalty + pitch_penalty)

    q = q_det * q_pose
    return float(max(0.0, min(1.0, q)))




def main():
    cfg = _build_face_config()
    app = _build_insight_app(cfg)

    print("[pose_calibration] FaceAnalysis (buffalo_l) initialised.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open default camera (id=0).")

    print("[pose_calibration] Camera opened. Press 'q' to quit.")
    print(
        "[pose_calibration] Controls: "
        "f=FRONT, l=LEFT, r=RIGHT, u=UP, d=DOWN, c=CLEAR, q=QUIT"
    )

    bins: Dict[str, List[PoseSample]] = collections.defaultdict(list)
    key_help = (
        "Controls: f=FRONT, l=LEFT, r=RIGHT, u=UP, d=DOWN, c=CLEAR, q=QUIT"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[pose_calibration] Failed to read frame from camera.")
            break

        img = frame.copy()
        h, w = img.shape[:2]

        faces = app.get(img)

        largest = None
        largest_area = 0.0

        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > largest_area:
                largest_area = area
                largest = f

        if largest is not None:
            x1, y1, x2, y2 = map(int, largest.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            yaw_deg = 0.0
            pitch_deg = 0.0

            if hasattr(largest, "pose") and largest.pose is not None:
                pose = largest.pose
                if len(pose) >= 2:
                    pitch_deg = float(pose[0])
                    yaw_deg = float(pose[1])

            det_score = float(getattr(largest, "det_score", 1.0))
            q = _compute_quality(yaw_deg, pitch_deg, det_score, cfg)

            _draw_text(
                img,
                f"yaw: {yaw_deg:.1f}",
                (x1, max(0, y1 - 40)),
                scale=0.6,
            )
            _draw_text(
                img,
                f"pitch: {pitch_deg:.1f}",
                (x1, max(0, y1 - 20)),
                scale=0.6,
            )
            _draw_text(
                img,
                f"q: {q:.3f}",
                (x1, y1 + 15),
                scale=0.6,
            )

        counts = (
            f"FRONT={len(bins['front'])} | "
            f"LEFT={len(bins['left'])} | "
            f"RIGHT={len(bins['right'])} | "
            f"UP={len(bins['up'])} | "
            f"DOWN={len(bins['down'])}"
        )
        _draw_text(img, counts, (10, 30), scale=0.6)
        _draw_text(img, key_help, (10, h - 20), scale=0.5)

        cv2.imshow("GaitGuard Pose Calibration (buffalo_l)", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[pose_calibration] Quitting.")
            break

        if largest is None:
            if key == ord("c"):
                print("[pose_calibration] Clearing all recorded samples.")
                bins.clear()
            continue

        pose = getattr(largest, "pose", None)
        yaw_deg = 0.0
        pitch_deg = 0.0
        if pose is not None and len(pose) >= 2:
            pitch_deg = float(pose[0])
            yaw_deg = float(pose[1])
        det_score = float(getattr(largest, "det_score", 1.0))
        q = _compute_quality(yaw_deg, pitch_deg, det_score, cfg)

        if key in (ord("f"), ord("l"), ord("r"), ord("u"), ord("d")):
            if key == ord("f"):
                label = "front"
            elif key == ord("l"):
                label = "left"
            elif key == ord("r"):
                label = "right"
            elif key == ord("u"):
                label = "up"
            else:
                label = "down"

            bins[label].append((yaw_deg, pitch_deg, q))
            print(
                f"[pose_calibration] Recorded {label.upper()} sample: "
                f"yaw={yaw_deg:.1f}, pitch={pitch_deg:.1f}, q={q:.3f} "
                f"(total {len(bins[label])})"
            )

        elif key == ord("c"):
            print("[pose_calibration] Clearing all recorded samples.")
            bins.clear()

    cap.release()
    cv2.destroyAllWindows()

    print("\n================ POSE CALIBRATION SUMMARY ================")
    for label in ["front", "left", "right", "up", "down"]:
        samples = bins.get(label, [])
        print(f"{label.upper():5s}: {_summarise(samples)}")
    print("==========================================================")


if __name__ == "__main__":
    main()
