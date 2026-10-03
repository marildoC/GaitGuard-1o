"""
core/interfaces.py

Defines abstract interfaces (contracts) for the main engines in GaitGuard.

We don't put any heavy logic here, only method signatures and docstrings.
Real implementations will subclass these interfaces in later phases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from schemas import (
    Frame,
    Tracklet,
    IdSignals,
    IdentityDecision,
    EventFlags,
    Alert,
)


class PerceptionEngine(ABC):
    """
    Detector + tracker.

    Responsibility:
      - Take a single Frame.
      - Return the list of active Tracklets for that frame.
    """

    @abstractmethod
    def process_frame(self, frame: Frame) -> List[Tracklet]:
        """
        Process one frame and update internal tracker state.

        Returns a list of current active Tracklets.
        """
        raise NotImplementedError


class IdentityEngine(ABC):
    """
    Identity / category engine.

    Responsibility:
      - Extract/refresh IdSignals (face, gait, appearance) for tracks.
      - Fuse them into IdentityDecision objects.
    """

    @abstractmethod
    def update_signals(self, frame: Frame, tracks: List[Tracklet]) -> List[IdSignals]:
        """
        Given the current frame and its tracks,
        return updated IdSignals for those tracks.
        """
        raise NotImplementedError

    @abstractmethod
    def decide(self, signals: List[IdSignals]) -> List[IdentityDecision]:
        """
        Turn IdSignals into high-level IdentityDecision objects
        (who is this? resident / visitor / watchlist / unknown).
        """
        raise NotImplementedError


class EventsEngine(ABC):
    """
    Events engine.

    Responsibility:
      - Look at motion / posture / context
      - Produce EventFlags (weapon/fight/fallen scores) for tracks.
    """

    @abstractmethod
    def update(
        self,
        frame: Frame,
        tracks: List[Tracklet],
        decisions: List[IdentityDecision],
    ) -> List[EventFlags]:
        """
        Inspect current state and produce EventFlags
        (scores for weapon / fight / fallen, etc.).
        """
        raise NotImplementedError


class AlertEngine(ABC):
    """
    Alert engine.

    Responsibility:
      - Take EventFlags + IdentityDecision
      - Decide if we need to create one or more Alert objects.
    """

    @abstractmethod
    def update(
        self,
        frame: Frame,
        events: List[EventFlags],
        decisions: List[IdentityDecision],
    ) -> List[Alert]:
        """
        Based on events & identities, decide if new alerts must be created.

        Returns a list of Alert objects (possibly empty).
        """
        raise NotImplementedError
