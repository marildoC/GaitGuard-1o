
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Alert:
    """
    Alert raised by the system that the operator / log should see.

    Attributes
    ----------
    alert_id          : int
        Unique ID (can be incremental integer).
    created_at        : float
        Timestamp in seconds (time.time()) when the alert was created.
    camera_id         : str
        Camera where this happened.
    track_ids         : list[int]
        IDs of tracks involved in the alert.

    type              : str
        Short code for the alert type (e.g. "weapon", "fallen",
        "suspicious_identity", "system").
    severity          : int
        1–5 (5 = critical).
    message           : str
        Short human-readable message for logs/UI.

    evidence_clip_path: Optional[str]
        Optional path to a saved video clip related to this alert.
    snapshot_path     : Optional[str]
        Optional path to a snapshot image.

    resolved          : bool
        Whether an operator has handled / acknowledged this alert.
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
