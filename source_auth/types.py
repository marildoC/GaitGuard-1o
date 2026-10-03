
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np



SourceAuthState = Literal[
    "REAL",
    "LIKELY_REAL",
    "UNCERTAIN",
    "LIKELY_SPOOF",
    "SPOOF",
]



BBoxTuple = Tuple[float, float, float, float]



SourceAuthDebug = Dict[str, Union[float, int, bool, str]]




@dataclass
class SourceAuthComponentScores:
    """
    Per-cue scores for a single track at a given time.

    All values are in [0,1] and have *cue* semantics:

      - planar_3d:
            1.0 → motion looks strongly like a real 3D head
            0.0 → motion looks planar / rigid-card-like

      - screen_artifacts:
            1.0 → strong evidence of screen / phone / display
            0.0 → no visible screen-like behaviour

      - background_consistency:
            1.0 → background inside face patch matches outer scene
            0.0 → backgrounds look like two different worlds
    """

    planar_3d: float = 0.5
    screen_artifacts: float = 0.5
    background_consistency: float = 0.5


@dataclass
class SourceAuthReliabilityFlags:
    """
    Reliability flags describing how trustworthy the cues are for
    this track at this time.
    """

    enough_motion: bool = False

    enough_landmarks: bool = False

    enough_background: bool = False




@dataclass
class SourceAuthScores:
    """
    Fused view of SourceAuth for a single track at a given time.

    Core fields (used by engine + diagnostics):

      - track_id:
            Track identifier (matches Tracklet.track_id).

      - source_auth_score:
            Smoothed REAL-HEAD likelihood in [0,1]:
                1.0 → very likely REAL head in the live scene
                0.0 → very likely 2D spoof (phone / screen / printed photo)

      - state:
            Discrete label based on source_auth_score + persistence:
                REAL / LIKELY_REAL / UNCERTAIN / LIKELY_SPOOF / SPOOF

      - components:
            Raw cue scores (planar_3d, screen_artifacts, background_consistency).

      - reliability:
            Reliability flags (enough_motion, enough_landmarks, enough_background).

      - debug:
            Rich diagnostic payload (small scalar metrics, phase tags, etc.).
            Convention for keys:
                "motion_*"      → motion / parallax / landmark stats
                "screen_*"      → screen / phone artifact stats
                "background_*"  → background consistency stats
                "fusion_*"      → fused evidence / thresholds / modes

    Legacy-compatibility helpers:

      - score_3d_motion                ↔ components.planar_3d
      - score_screen_artifacts         ↔ components.screen_artifacts
      - score_background_consistency   ↔ components.background_consistency
      - motion_reliable                ↔ reliability.enough_motion
      - screen_reliable                ↔ reliability.enough_background
      - background_reliable            ↔ reliability.enough_background

      and an as_dict() method that flattens to the old key names.
    """

    track_id: int

    source_auth_score: float = 0.5

    state: SourceAuthState = "UNCERTAIN"

    components: SourceAuthComponentScores = field(
        default_factory=SourceAuthComponentScores
    )
    reliability: SourceAuthReliabilityFlags = field(
        default_factory=SourceAuthReliabilityFlags
    )

    debug: SourceAuthDebug = field(default_factory=dict)



    @property
    def score_3d_motion(self) -> float:
        return float(self.components.planar_3d)

    @score_3d_motion.setter
    def score_3d_motion(self, value: float) -> None:
        self.components.planar_3d = float(value)


    @property
    def score_screen_artifacts(self) -> float:
        return float(self.components.screen_artifacts)

    @score_screen_artifacts.setter
    def score_screen_artifacts(self, value: float) -> None:
        self.components.screen_artifacts = float(value)


    @property
    def score_background_consistency(self) -> float:
        return float(self.components.background_consistency)

    @score_background_consistency.setter
    def score_background_consistency(self, value: float) -> None:
        self.components.background_consistency = float(value)


    @property
    def motion_reliable(self) -> bool:
        return bool(self.reliability.enough_motion)

    @motion_reliable.setter
    def motion_reliable(self, value: bool) -> None:
        self.reliability.enough_motion = bool(value)
        if value:
            self.reliability.enough_landmarks = True


    @property
    def screen_reliable(self) -> bool:
        return bool(self.reliability.enough_background)

    @screen_reliable.setter
    def screen_reliable(self, value: bool) -> None:
        self.reliability.enough_background = bool(value)

    @property
    def background_reliable(self) -> bool:
        return bool(self.reliability.enough_background)

    @background_reliable.setter
    def background_reliable(self, value: bool) -> None:
        self.reliability.enough_background = bool(value)


    def update_debug(self, prefix: str, metrics: SourceAuthDebug) -> None:
        """
        Merge a dictionary of metrics into this track's debug payload,
        applying a string prefix to each key.

        Example:
            scores.update_debug("motion_", {"parallax_ratio": 0.23})
            → debug["motion_parallax_ratio"] = 0.23
        """
        if not metrics:
            return
        for key, value in metrics.items():
            self.debug[f"{prefix}{str(key)}"] = value


    def as_dict(self) -> Dict[str, Union[int, float, str, bool]]:
        """
        Lightweight serialisation for logging / overlays / network telemetry.

        Exposes both:
          - the new structured view (components / reliability) via scalar aliases,
          - and the legacy flattened keys (score_3d_motion, motion_reliable, ...).

        NOTE:
            The 'debug' payload is intentionally NOT included here to keep
            this method lightweight. Debug can be logged separately when needed.
        """
        data: Dict[str, Union[int, float, str, bool]] = {
            "track_id": int(self.track_id),
            "source_auth_score": float(self.source_auth_score),
            "state": str(self.state),
            "score_3d_motion": float(self.score_3d_motion),
            "score_screen_artifacts": float(self.score_screen_artifacts),
            "score_background_consistency": float(
                self.score_background_consistency
            ),
            "motion_reliable": bool(self.motion_reliable),
            "screen_reliable": bool(self.screen_reliable),
            "background_reliable": bool(self.background_reliable),
        }
        return data




@dataclass
class LandmarkFrame:
    """
    Minimal per-frame snapshot of a face track, for SourceAuth.

    This is the common unit stored in SourceAuth internal buffers.

    Fields:
      - ts:
            Stream timestamp in seconds. Must be monotonically
            non-decreasing within a track.

      - bbox:
            Optional face bounding box in frame coordinates, (x1, y1, x2, y2).
            We use floats to match detector / tracker outputs and allow
            sub-pixel motion analysis.

      - landmarks_2d:
            2D facial landmarks as a float32 array of shape (K, 2).

      - quality:
            Overall face quality for this frame (0–1).

      - yaw / pitch / roll:
            Optional pose angles (in degrees), if available.

      - det_score:
            Optional detector confidence score for the face candidate.
    """

    ts: float

    bbox: Optional[BBoxTuple]

    landmarks_2d: np.ndarray

    quality: float = 0.0

    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None

    det_score: Optional[float] = None

    def num_landmarks(self) -> int:
        """
        Return the number of landmarks (K). Safe even if landmarks_2d is
        malformed – we fall back to 0.
        """
        try:
            return int(self.landmarks_2d.shape[0])
        except Exception:
            return 0

    def as_float_array(self) -> np.ndarray:
        """
        Return landmarks as a float32 array of shape (K, 2).

        This helper ensures consistent dtype/shape for downstream
        motion computation (parallax, affine fitting, etc.).
        """
        arr = np.asarray(self.landmarks_2d, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            size = arr.size
            if size % 2 != 0:
                return np.zeros((0, 2), dtype=np.float32)
            arr = arr.reshape(-1, 2)
        return arr
