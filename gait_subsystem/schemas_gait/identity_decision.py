from dataclasses import dataclass
from typing import Optional, Literal


Category = Literal["resident", "visitor", "watchlist", "unknown"]


@dataclass
class IdentityDecision:
    """
    Final identity / category decision for a track.

    - track_id    : link to Tracklet.track_id
    - identity_id : stable ID (e.g. user id in DB) or None if unknown
    - category    : resident / visitor / watchlist / unknown
    - confidence  : 0–1 confidence in this decision
    - reason      : optional short explanation for debugging / logs
    """
    track_id: int
    identity_id: Optional[str] = None
    category: Category = "unknown"
    confidence: float = 0.0
    reason: Optional[str] = None
