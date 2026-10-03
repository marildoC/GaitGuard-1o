
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from .face_sample import FaceSample


@dataclass
class IdSignals:
    """
    All identity-related features for a given track at a given moment.

    This structure is what the IdentityEngine receives from FaceRoute
    (and later, gait + appearance routes). It can be extended safely as
    long as existing fields are not removed.

    Primary modern view (for 2D + 3D):

      - track_id       : int
      - best_face      : FaceSample or None
      - recent_faces   : optional small history of FaceSample
      - gait_embedding : temporal gait features
      - gait_quality   : 0–1
      - appearance_embedding : clothing / color features
      - appearance_quality   : 0–1

    Legacy/compatibility fields (still supported):

      - face_embedding : kept for older code; mirrors best_face.embedding
      - face_quality   : kept; mirrors best_face.quality
      - extra          : free-form dict, kept but best_face.extra is preferred
      - raw_face_box   : kept; mirrors best_face.bbox
      - pose_bin_hint  : kept; mirrors best_face.pose_bin

    Multiview / 3D extensions should prefer to read:

      - best_face.yaw / pitch / roll
      - best_face.pose_bin
      - best_face.quality
      - best_face.det_score (if stored in FaceSample.extra)
    """

    track_id: int

    best_face: Optional[FaceSample] = None

    recent_faces: List[FaceSample] = field(default_factory=list)

    face_embedding: Optional[np.ndarray] = None
    face_quality: float = 0.0

    gait_embedding: Optional[np.ndarray] = None
    gait_quality: float = 0.0

    appearance_embedding: Optional[np.ndarray] = None
    appearance_quality: float = 0.0

    extra: Optional[Dict[str, Any]] = None

    raw_face_box: Optional[Tuple[float, float, float, float]] = None

    pose_bin_hint: Optional[str] = None

    
    landmarks_2d: Optional[np.ndarray] = None
    
    face_bbox_in_frame: Optional[Tuple[float, float, float, float]] = None


    @property
    def has_face(self) -> bool:
        """
        Quick check: do we have any face evidence (canonical or legacy)?
        """
        if self.best_face is not None and self.best_face.embedding is not None:
            return True
        if self.face_embedding is not None:
            return True
        return False

    def sync_from_best_face(self) -> None:
        """
        Mirror best_face into legacy fields.

        Call this after best_face is set/updated so that older parts
        of the system that still read legacy fields stay consistent.
        """
        bf = self.best_face
        if bf is None:
            self.face_embedding = None
            self.face_quality = 0.0
            self.raw_face_box = None
            self.pose_bin_hint = None
            return

        if bf.embedding is not None:
            self.face_embedding = np.asarray(bf.embedding, dtype=np.float32).reshape(-1)
        else:
            self.face_embedding = None

        q = float(bf.quality)
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        self.face_quality = q

        self.raw_face_box = bf.bbox
        self.pose_bin_hint = bf.pose_bin

        if bf.extra:
            if self.extra:
                merged = dict(self.extra)
                merged.update(bf.extra)
                self.extra = merged
            else:
                self.extra = dict(bf.extra)

    def set_best_face(self, face: FaceSample) -> None:
        """
        Convenience setter: assign best_face and update legacy mirrors.
        """
        self.best_face = face
        self.sync_from_best_face()

    def add_recent_face(self, face: FaceSample, max_len: int = 5) -> None:
        """
        Append a face sample to recent_faces with a fixed maximum length.

        This is useful for temporal fusion logic (e.g. multiview + gait).
        """
        self.recent_faces.append(face)
        if len(self.recent_faces) > max_len:
            self.recent_faces.pop(0)


    def ensure_best_face_from_legacy(self, ts: Optional[float] = None) -> None:
        """
        If best_face is missing but legacy face_embedding / raw_face_box /
        extra / pose_bin_hint exist, create a minimal FaceSample so that
        3D / multiview code can operate on canonical structure.

        This is *non-breaking* and is only used when best_face is None.
        """
        if self.best_face is not None:
            return

        if self.face_embedding is None:
            return

        emb = np.asarray(self.face_embedding, dtype=np.float32).reshape(-1)
        extra = self.extra or {}

        self.best_face = FaceSample(
            bbox=self.raw_face_box,
            embedding=emb,
            det_score=float(extra.get("det_score", 0.0)),
            quality=float(self.face_quality),
            yaw=float(extra.get("yaw", 0.0)) if "yaw" in extra else None,
            pitch=float(extra.get("pitch", 0.0)) if "pitch" in extra else None,
            roll=float(extra.get("roll", 0.0)) if "roll" in extra else None,
            ts=ts,
            pose_bin=self.pose_bin_hint,
            source="legacy",
            extra=extra if extra else None,
        )
        self.sync_from_best_face()



@dataclass
class IdSignal:
    """
    Represents a single identity signal from one recognition modality.

    This is used by the gait engine and future appearance routes to produce
    per-track identity suggestions that are later fused with face evidence.

    Attributes
    ----------
    track_id : int
        The ID of the track this signal pertains to.
    identity_id : Optional[str]
        The suggested person ID. None if unknown/below threshold.
    confidence : float
        Confidence score (0-1) for this identity suggestion.
    method : str
        The modality that produced this signal: "face", "gait", "appearance"
    """
    track_id: int
    identity_id: Optional[str]
    confidence: float
    method: str
    extra: Optional[Dict[str, Any]] = None
