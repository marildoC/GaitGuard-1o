
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

Category = Literal["resident", "visitor", "watchlist", "unknown"]


@dataclass
class IdentityDecision:
    """
    Final identity / category decision for a track.

    Mandatory:
      - track_id    : link to Tracklet.track_id
      - identity_id : stable ID (user id or None)
      - category    : resident / visitor / watchlist / unknown
      - confidence  : 0–1 confidence score
      - reason      : short explanation for logs / debugging

    Telemetry (optional, used e.g. by FaceMetrics):
      - quality     : face quality (0–1) behind this decision, if available

    Multiview / 3D extensions (optional):
      - pose_bin    : which 3D view matched (FRONT/LEFT/RIGHT/UP/DOWN/...)
      - engine      : "classic" or "multiview" or other engine id
      - distance    : gallery distance (for explainability)
      - score       : raw similarity score (e.g. cosine similarity)
      - extra       : dict for future custom diagnostics (fusion, gait, etc.)

    Source authenticity (“SourceAuth”) extensions (optional):
      - source_auth_score  : 0–1 "realness" score
                             (1.0 → very likely real 3D head in scene,
                              0.0 → very likely screen/photo spoof)
      - source_auth_state  : discrete label
                             e.g. "REAL", "LIKELY_REAL", "UNCERTAIN",
                                  "LIKELY_SPOOF", "SPOOF"
      - source_auth_reason : short explanation / debug string for the
                             SourceAuth decision (can be separate from
                             the main `reason` or combined later).
    """

    track_id: int
    identity_id: Optional[str] = None
    category: Category = "unknown"
    confidence: float = 0.0
    reason: Optional[str] = None

    canonical_id: Optional[int] = None

    binding_state: Optional[str] = None

    pose_bin: Optional[str] = None
    engine: Optional[str] = None
    distance: Optional[float] = None
    score: Optional[float] = None

    extra: Optional[Dict[str, Any]] = None

    quality: float = 0.0

    source_auth_score: Optional[float] = None 
    source_auth_state: Optional[str] = None
    source_auth_reason: Optional[str] = None
