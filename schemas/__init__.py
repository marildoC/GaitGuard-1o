"""
schemas/__init__.py
Central exports for lightweight data structures used across GaitGuard.

We keep each schema in its own module (frame, tracklet, id_signals, ...)
and re-export them here for convenience:

    from schemas import Frame, Tracklet, IdSignals, IdentityDecision, ...

This file should remain VERY lightweight (no heavy imports or model code).
"""

from .frame import Frame
from .tracklet import Tracklet
from .id_signals import IdSignals
from .identity_decision import IdentityDecision
from .event_flags import EventFlags
from .alert import Alert
from .face_sample import FaceSample

__all__ = [
    "Frame",
    "Tracklet",
    "IdSignals",
    "IdentityDecision",
    "EventFlags",
    "Alert",
    "FaceSample",
]
