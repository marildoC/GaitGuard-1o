"""
Core types and helpers for the pseudo-3D / multi-view face representation.

This module is intentionally self-contained:

- No coupling to FaceConfig / FaceGallery / IdentityEngine.
- Just data structures and small helpers that other modules can import.

High-level idea:

For each real person we keep a compact "multi-view head":
  - several pose bins (front / left / right / up / down / ...),
  - a small set of high-quality samples per bin,
  - a centroid embedding per bin.

Later modules (builder, gallery view, matcher, mv_report) will use these types.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)




class PoseBin(str, Enum):
    """
    Discrete view bins for the pseudo-3D face model.

    Values are strings so they are easy to log, store in JSON, and
    attach to template metadata.
    """

    FRONT = "front"       # looking roughly at the camera
    LEFT = "left"         # turned to the subject's left (camera sees right profile)
    RIGHT = "right"       # turned to the subject's right (camera sees left profile)
    UP = "up"             # head tilted up
    DOWN = "down"         # head tilted down

    OCCLUDED = "occluded"  # strong occlusion (mask, hands, heavy shadow)
    UNKNOWN = "unknown"    # pose outside configured bins / unreliable angles


ALL_PRIMARY_BINS: List[PoseBin] = [
    PoseBin.FRONT,
    PoseBin.LEFT,
    PoseBin.RIGHT,
    PoseBin.UP,
    PoseBin.DOWN,
]


def posebin_from_hint(hint: Any) -> PoseBin:
    """
    Robust helper to map arbitrary metadata values to a PoseBin.

    Accepts:
      - PoseBin values directly;
      - strings like "front", "FRONT", "Front";
      - strings equal to enum .name ("FRONT") or .value ("front").

    Falls back to PoseBin.UNKNOWN if no match.
    """
    if isinstance(hint, PoseBin):
        return hint

    if not isinstance(hint, str):
        return PoseBin.UNKNOWN

    s = hint.strip()
    if not s:
        return PoseBin.UNKNOWN

    low = s.lower()

    for b in PoseBin:
        if low == b.value:
            return b

    up = s.upper()
    for b in PoseBin:
        if up == b.name:
            return b

    return PoseBin.UNKNOWN




@dataclass(frozen=True)
class MultiViewConfig:
    """
    Configuration knobs for mapping head pose to bins and managing samples.

    All angles are in degrees, using the usual convention from face pose
    estimators:
      - yaw:   left/right rotation (0 = looking at camera)
      - pitch: up/down rotation   (0 = level)
    """


    front_yaw_max_deg: float = 15.0
    front_pitch_max_deg: float = 12.0

    side_yaw_min_deg: float = 20.0
    side_yaw_max_deg: float = 70.0

    up_pitch_min_deg: float = 17.0
    down_pitch_min_deg: float = 25.0


    max_samples_per_bin: int = 16

    min_quality_for_model: float = 0.40

    ema_alpha: float = 0.10


    def validate(self) -> None:
        """
        Basic sanity checks. Called by MultiViewBuilder on startup.
        """
        if self.front_yaw_max_deg <= 0:
            raise ValueError("front_yaw_max_deg must be positive")

        if self.front_pitch_max_deg <= 0:
            raise ValueError("front_pitch_max_deg must be positive")

        if self.side_yaw_min_deg <= 0 or self.side_yaw_max_deg <= 0:
            raise ValueError("side_yaw_* thresholds must be positive")

        if self.side_yaw_min_deg >= self.side_yaw_max_deg:
            raise ValueError("side_yaw_min_deg must be < side_yaw_max_deg")

        if self.up_pitch_min_deg <= 0 or self.down_pitch_min_deg <= 0:
            raise ValueError("up/down pitch thresholds must be positive")

        if self.max_samples_per_bin <= 0:
            raise ValueError("max_samples_per_bin must be positive")

        if not (0.0 <= self.min_quality_for_model <= 1.0):
            raise ValueError("min_quality_for_model must be in [0, 1]")

        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1].")





@dataclass
class MultiViewSample:
    """
    One face observation used for multi-view modelling.

    This is *not* a new storage format; higher-level code will typically
    derive these from existing gallery templates or fresh detections.

    Canonical fields (aligned with Step 1 & Step 2):

      embedding   : 512-D float32, L2-normalised.
      yaw_deg     : yaw angle in degrees (optional).
      pitch_deg   : pitch angle in degrees (optional).
      roll_deg    : roll angle in degrees (optional).
      quality     : scalar in [0, 1] (same metric as face.quality).
      pose_bin    : PoseBin assigned using classify_pose_bin or metadata hint.
      source      : short tag for provenance ("guided_enroll", "runtime", ...).
      ts          : timestamp (seconds since epoch).
      metadata    : free-form dict (extra diagnostics).
    """

    embedding: np.ndarray

    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None

    quality: float = 0.0

    pose_bin: PoseBin = PoseBin.UNKNOWN

    source: str = "unknown"
    ts: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone_with_bin(self, pose_bin: PoseBin) -> "MultiViewSample":
        """
        Return a shallow copy with a different pose_bin.
        Useful when re-binning samples after config changes.
        """
        return MultiViewSample(
            embedding=self.embedding,
            yaw_deg=self.yaw_deg,
            pitch_deg=self.pitch_deg,
            roll_deg=self.roll_deg,
            quality=self.quality,
            pose_bin=pose_bin,
            source=self.source,
            ts=self.ts,
            metadata=dict(self.metadata),
        )

    def as_report_dict(self) -> Dict[str, Any]:
        """
        Lightweight dict useful for logging / mv_report.
        Does NOT include the raw embedding.
        """
        return {
            "pose_bin": self.pose_bin.value,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "quality": float(self.quality),
            "source": self.source,
            "ts": float(self.ts),
        }


@dataclass
class MultiViewBin:
    """
    Container for all samples and centroid for one pose bin of a person.
    """

    pose_bin: PoseBin
    samples: List[MultiViewSample] = field(default_factory=list)

    centroid: Optional[np.ndarray] = None

    avg_quality: float = 0.0
    last_updated_ts: float = field(default_factory=lambda: time.time())

    def is_populated(self) -> bool:
        """
        True if this bin has at least one sample and a valid centroid.
        """
        return bool(self.samples) and self.centroid is not None

    def num_samples(self) -> int:
        return len(self.samples)

    def to_report_dict(self) -> Dict[str, Any]:
        """
        Small summary for reporting / mv_report.
        """
        return {
            "pose_bin": self.pose_bin.value,
            "num_samples": self.num_samples(),
            "avg_quality": float(self.avg_quality),
            "has_centroid": self.centroid is not None,
            "last_updated_ts": float(self.last_updated_ts),
        }


@dataclass
class MultiViewPersonModel:
    """
    Pseudo-3D head model for a single person.

    Higher-level code is responsible for building this model from gallery
    templates (or runtime observations) using MultiViewBuilder.
    """

    person_id: str
    bins: Dict[PoseBin, MultiViewBin] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    notes: Dict[str, Any] = field(default_factory=dict)

    def primary_bins(self) -> List[PoseBin]:
        """
        Return the primary bins that are actually populated for this person.
        """
        return [
            b for b in ALL_PRIMARY_BINS
            if b in self.bins and self.bins[b].is_populated()
        ]

    def num_populated_bins(self) -> int:
        """
        Number of primary bins with at least one valid centroid.
        """
        return len(self.primary_bins())

    def coverage_score(self, expected_bins: Iterable[PoseBin] = ALL_PRIMARY_BINS) -> float:
        """
        Return a simple 0–1 score of how many primary bins are populated.

        This is not a strict biometric score, just a heuristic to quickly
        understand how "complete" the multi-view model is for reporting
        and diagnostics.
        """
        expected = list(expected_bins)
        if not expected:
            return 0.0

        filled = sum(
            1 for b in expected
            if b in self.bins and self.bins[b].is_populated()
        )
        return filled / float(len(expected))

    def bin_or_none(self, pose_bin: PoseBin) -> Optional[MultiViewBin]:
        """
        Safe accessor: returns the bin if present, else None.
        """
        return self.bins.get(pose_bin)

    def to_report_dict(self) -> Dict[str, Any]:
        """
        Compact representation used by mv_report.py:
          - per-bin summary,
          - coverage,
          - global stats.
        """
        bin_reports = {
            b.value: self.bins[b].to_report_dict()
            for b in self.bins.keys()
        }
        return {
            "person_id": self.person_id,
            "bins": bin_reports,
            "coverage": self.coverage_score(),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "notes": dict(self.notes),
        }




def _abs_or_none(x: Optional[float]) -> Optional[float]:
    return None if x is None else abs(float(x))


def classify_pose_bin(
    yaw_deg: Optional[float],
    pitch_deg: Optional[float],
    *,
    cfg: MultiViewConfig,
    is_occluded: bool = False,
) -> PoseBin:
    """
    Map (yaw, pitch) to a PoseBin using the provided MultiViewConfig.

    Behaviour is intentionally simple and deterministic:

    - If `is_occluded` is True → OCCLUDED, regardless of angles.
    - Otherwise:
        FRONT  if |yaw| <= front_yaw_max_deg and |pitch| <= front_pitch_max_deg
        LEFT   if yaw is negative and side_yaw_min_deg <= |yaw| <= side_yaw_max_deg
        RIGHT  if yaw is positive and side_yaw_min_deg <= |yaw| <= side_yaw_max_deg
        UP     if pitch >= up_pitch_min_deg
        DOWN   if pitch <= -down_pitch_min_deg
        UNKNOWN otherwise

    This function does *not* try to be super-smart; builder / matcher
    can always add extra logic on top (e.g. prefer FRONT over UP when
    both roughly match).
    """
    if yaw_deg is None or pitch_deg is None:
       return PoseBin.UNKNOWN

    if is_occluded:
        return PoseBin.OCCLUDED

    y = _abs_or_none(yaw_deg)
    p = pitch_deg

    if y is None or p is None:
        return PoseBin.UNKNOWN

    ay = float(y)
    ap = abs(float(p))

    if ay <= cfg.front_yaw_max_deg and ap <= cfg.front_pitch_max_deg:
        return PoseBin.FRONT

    if cfg.side_yaw_min_deg <= ay <= cfg.side_yaw_max_deg:
        if yaw_deg is not None and yaw_deg < 0:
            return PoseBin.LEFT
        elif yaw_deg is not None and yaw_deg > 0:
            return PoseBin.RIGHT

    if p >= cfg.up_pitch_min_deg:
        return PoseBin.UP
    if p <= -cfg.down_pitch_min_deg:
        return PoseBin.DOWN

    return PoseBin.UNKNOWN


def compute_coverage_score(
    models: Mapping[str, MultiViewPersonModel],
    *,
    expected_bins: Iterable[PoseBin] = ALL_PRIMARY_BINS,
) -> float:
    """
    Aggregate coverage score across many persons for quick diagnostics.

    Returns
    -------
    float
        Average coverage score in [0, 1] over all provided models.
        Returns 0.0 if `models` is empty.
    """
    if not models:
        return 0.0

    expected = list(expected_bins)
    if not expected:
        return 0.0

    scores = [m.coverage_score(expected) for m in models.values()]
    return float(sum(scores) / len(scores))
