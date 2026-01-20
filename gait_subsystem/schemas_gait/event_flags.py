from dataclasses import dataclass


@dataclass
class EventFlags:
    """
    Event scores for one track at a given time window.

    - track_id     : link to Tracklet.track_id
    - weapon_score : 0–1 score (higher → more likely weapon)
    - fight_score  : 0–1 score (higher → more likely fight)
    - fallen_score : 0–1 score (higher → more likely person fallen)
    - weapon       : boolean flag if weapon_score exceeds threshold
    - fight        : boolean flag if fight_score exceeds threshold
    - fallen       : boolean flag if fallen_score exceeds threshold
    """
    track_id: int

    weapon_score: float = 0.0
    fight_score: float = 0.0
    fallen_score: float = 0.0

    weapon: bool = False
    fight: bool = False
    fallen: bool = False
