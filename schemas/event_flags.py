
from dataclasses import dataclass


@dataclass
class EventFlags:
    """
    Event scores for one track at a given time window.

    Attributes
    ----------
    track_id     : int
        Link to Tracklet.track_id.
    weapon_score : float
        0–1 score (higher → more likely weapon present).
    fight_score  : float
        0–1 score (higher → more likely fight).
    fallen_score : float
        0–1 score (higher → more likely person fallen).

    weapon       : bool
        True if weapon_score exceeds configured threshold.
    fight        : bool
        True if fight_score exceeds configured threshold.
    fallen       : bool
        True if fallen_score exceeds configured threshold.
    """

    track_id: int

    weapon_score: float = 0.0
    fight_score: float = 0.0
    fallen_score: float = 0.0

    weapon: bool = False
    fight: bool = False
    fallen: bool = False
