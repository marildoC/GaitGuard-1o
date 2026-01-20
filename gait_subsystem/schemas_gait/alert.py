from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Alert:
    """
    Alert raised by the system that the operator / log should see.

    - alert_id          : unique ID (can be incremental integer)
    - created_at        : timestamp in seconds (time.time())
    - camera_id         : camera where this happened
    - track_ids         : tracks involved in the alert
    - type              : e.g. "weapon", "fallen", "suspicious_identity"
    - severity          : 1–5 (5 = critical)
    - message           : short human-readable text
    - evidence_clip_path: optional path to saved video clip
    - snapshot_path     : optional path to snapshot image
    - resolved          : whether operator has handled this alert
    """
    alert_id: int
    created_at: float
    camera_id: str
    track_ids: List[int] = field(default_factory=list)

    type: str = "generic"
    severity: int = 1
    message: str = ""

    evidence_clip_path: Optional[str] = None
    snapshot_path: Optional[str] = None

    resolved: bool = False
