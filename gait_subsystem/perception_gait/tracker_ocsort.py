from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)



def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area / union)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity in [-1, 1]. If any vector is near-zero, return 0.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))



@dataclass
class OCSortConfig:
    """
    Configuration for OCSortTracker.

    - max_age:       how many frames a track can be unseen before being dropped
    - min_hits:      how many hits are required before a track is considered confirmed
    - iou_threshold: minimum IoU for matching detections to existing tracks
    - appearance_lambda: weight in [0,1] to combine IoU with appearance similarity
                         0.0 → IoU only; 0.5 → equal weight
    - ema_alpha:     EMA coefficient for updating appearance embeddings
    """
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    appearance_lambda: float = 0.3
    ema_alpha: float = 0.7


@dataclass
class Track:
    """
    Public track representation returned by OCSortTracker.update().

    bbox: current [x1, y1, x2, y2]
    score: detection score of the last associated detection
    """
    track_id: int
    bbox: np.ndarray
    score: float
    class_id: int
    class_name: str
    age: int
    time_since_update: int
    hits: int
    confirmed: bool



@dataclass
class _InternalTrack:
    """
    Internal tracking state (similar spirit to OC-SORT / SORT).

    - bbox: last updated box [x1, y1, x2, y2]
    - velocity: estimate of box displacement between frames
    - appearance: EMA of appearance feature (if used)
    """
    track_id: int
    bbox: np.ndarray
    score: float
    class_id: int
    class_name: str

    age: int = 0
    time_since_update: int = 0
    hits: int = 0
    confirmed: bool = False

    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    appearance: Optional[np.ndarray] = None

    def predict(self) -> None:
        """
        Simple linear prediction: bbox += velocity.

        This is not a full Kalman filter but gives motion continuity
        similar to OC-SORT's motion bias.
        """
        self.age += 1
        self.time_since_update += 1

        self.velocity *= 0.9
        self.bbox = self.bbox + self.velocity

    def update(
        self,
        new_bbox: np.ndarray,
        new_score: float,
        new_class_id: int,
        new_class_name: str,
        new_appearance: Optional[np.ndarray],
        ema_alpha: float,
    ) -> None:
        """
        Update track with a new associated detection.
        """
        self.velocity = new_bbox - self.bbox

        self.bbox = new_bbox

        self.score = float(new_score)
        self.class_id = int(new_class_id)
        self.class_name = new_class_name

        self.time_since_update = 0
        self.hits += 1

        if new_appearance is not None:
            if self.appearance is None:
                self.appearance = new_appearance.astype(np.float32)
            else:
                self.appearance = (
                    ema_alpha * new_appearance.astype(np.float32)
                    + (1.0 - ema_alpha) * self.appearance
                )



class OCSortTracker:
    """
    Lightweight OC-SORT–style tracker.

    Usage:
        tracker = OCSortTracker()
        tracks = tracker.update(detections, appearance_features)

    - detections: List[Detection]
    - appearance_features: Optional[List[np.ndarray]] aligned with detections
      (index i of features corresponds to detections[i]).
    """

    def __init__(self, config: Optional[OCSortConfig] = None) -> None:
        self.config = config or OCSortConfig()
        self.tracks: List[_InternalTrack] = []
        self._next_id: int = 1

    def reset(self) -> None:
        """
        Clear all tracks (e.g., when restarting a video).
        """
        self.tracks.clear()
        self._next_id = 1

    def update(
        self,
        detections: List[Detection],
        appearance_features: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> List[Track]:
        """
        Main tracking step.

        Parameters
        ----------
        detections : List[Detection]
            Detections for the current frame.
        appearance_features : Optional[Sequence[Optional[np.ndarray]]]
            Optional appearance embeddings aligned with detections.
            If provided, len(appearance_features) must equal len(detections).

        Returns
        -------
        List[Track]
            Confirmed tracks that are updated in this frame.
        """
        if appearance_features is not None and len(appearance_features) != len(detections):
            raise ValueError("appearance_features must match detections length")

        for trk in self.tracks:
            trk.predict()

        matches, unmatched_tracks, unmatched_dets = self._associate(
            detections, appearance_features
        )

        for track_idx, det_idx in matches:
            trk = self.tracks[track_idx]
            det = detections[det_idx]
            feat = appearance_features[det_idx] if appearance_features is not None else None

            new_bbox = np.array(
                [det.x1, det.y1, det.x2, det.y2], dtype=np.float32
            )
            trk.update(
                new_bbox=new_bbox,
                new_score=det.score,
                new_class_id=det.class_id,
                new_class_name=det.class_name,
                new_appearance=feat,
                ema_alpha=self.config.ema_alpha,
            )

            if not trk.confirmed and trk.hits >= self.config.min_hits:
                trk.confirmed = True

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            feat = appearance_features[det_idx] if appearance_features is not None else None

            bbox = np.array(
                [det.x1, det.y1, det.x2, det.y2], dtype=np.float32
            )
            new_trk = _InternalTrack(
                track_id=self._next_id,
                bbox=bbox,
                score=det.score,
                class_id=det.class_id,
                class_name=det.class_name,
            )
            new_trk.update(
                new_bbox=bbox,
                new_score=det.score,
                new_class_id=det.class_id,
                new_class_name=det.class_name,
                new_appearance=feat,
                ema_alpha=self.config.ema_alpha,
            )
            self._next_id += 1
            self.tracks.append(new_trk)

        alive_tracks: List[_InternalTrack] = []
        for trk in self.tracks:
            if trk.time_since_update <= self.config.max_age:
                alive_tracks.append(trk)
        self.tracks = alive_tracks

        output_tracks: List[Track] = []
        for trk in self.tracks:
            if not trk.confirmed:
                continue
            if trk.time_since_update != 0:
                continue

            output_tracks.append(
                Track(
                    track_id=trk.track_id,
                    bbox=trk.bbox.copy(),
                    score=trk.score,
                    class_id=trk.class_id,
                    class_name=trk.class_name,
                    age=trk.age,
                    time_since_update=trk.time_since_update,
                    hits=trk.hits,
                    confirmed=trk.confirmed,
                )
            )

        return output_tracks

    def _associate(
        self,
        detections: List[Detection],
        appearance_features: Optional[Sequence[Optional[np.ndarray]]],
    ):
        """
        Greedy association using IoU + optional appearance similarity.

        Returns
        -------
        matches: List[Tuple[track_idx, det_idx]]
        unmatched_tracks: List[int]
        unmatched_dets: List[int]
        """
        num_tracks = len(self.tracks)
        num_dets = len(detections)

        if num_tracks == 0 or num_dets == 0:
            return [], list(range(num_tracks)), list(range(num_dets))

        scores = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for t_idx, trk in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                det_box = np.array([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
                iou = iou_xyxy(trk.bbox, det_box)

                if iou <= 0.0:
                    scores[t_idx, d_idx] = 0.0
                    continue

                if (
                    appearance_features is not None
                    and appearance_features[d_idx] is not None
                    and trk.appearance is not None
                    and self.config.appearance_lambda > 0.0
                ):
                    cos_sim = cosine_similarity(
                        trk.appearance, appearance_features[d_idx]
                    )
                    cos_sim = (cos_sim + 1.0) * 0.5
                    combined = (
                        (1.0 - self.config.appearance_lambda) * iou
                        + self.config.appearance_lambda * cos_sim
                    )
                    scores[t_idx, d_idx] = combined
                else:
                    scores[t_idx, d_idx] = iou

        matches: List[tuple[int, int]] = []
        unmatched_tracks = list(range(num_tracks))
        unmatched_dets = list(range(num_dets))

        while True:
            if len(unmatched_tracks) == 0 or len(unmatched_dets) == 0:
                break

            best_score = 0.0
            best_t = -1
            best_d = -1
            for t_idx in unmatched_tracks:
                row = scores[t_idx]
                for d_idx in unmatched_dets:
                    s = row[d_idx]
                    if s > best_score:
                        best_score = s
                        best_t = t_idx
                        best_d = d_idx

            if best_t == -1 or best_d == -1:
                break

            if best_score < self.config.iou_threshold:
                break

            matches.append((best_t, best_d))
            unmatched_tracks.remove(best_t)
            unmatched_dets.remove(best_d)

        return matches, unmatched_tracks, unmatched_dets
