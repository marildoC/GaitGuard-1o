
from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

from schemas import IdentityDecision, Tracklet

logger = logging.getLogger(__name__)


class FaceMetrics:
    """
    Wave-3 rolling metrics for identity pipeline.

    Sliding window = last N seconds (default 5s)
    Controlled by config.runtime.log_face_metrics (True/False)

    IMPORTANT: classification is now *semantic*:
      - We first look at engine / reason / extra["strength"] to decide
        whether a decision is strong / weak / unknown.
      - Only if we cannot infer the band from semantics we fall back to
        confidence thresholds for backwards compatibility.

    SourceAuth (Phase-5.4):
      - We additionally track per-frame SourceAuth telemetry:
          * how many decisions had source_auth_score/state
          * rolling average of source_auth_score
          * rolling average of q_face when SourceAuth is present
          * average counts per state (REAL / LIKELY_REAL / LIKELY_SPOOF /
            SPOOF / UNCERTAIN / MISSING)
      - We also track JOINT behaviour with identity strength:
          * Identity STRONG & SA REAL-ish (REAL or LIKELY_REAL)
          * Identity STRONG & SA SPOOF-ish (SPOOF or LIKELY_SPOOF)
          * Identity WEAK   & SA REAL-ish
          * Identity WEAK   & SA SPOOF-ish
      - These are logged on a separate "FaceMetrics-SA" line and exposed via
            get_source_auth_summary()
        without changing any existing identity metrics behaviour.
    """

    __slots__ = (
        "_window_sec",
        "_last_log_ts",
        "_log_every_sec",
        "_records",     # list of (ts, n_tracks, n_strong, n_weak, n_unknown, avg_q, avg_conf)
        "_sa_records",  # list of (
    )

    def __init__(
        self,
        window_sec: float = 5.0,
        log_every_sec: float = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        window_sec : float
            Sliding window size for keeping metrics.
        log_every_sec : float
            How often we auto-emit a summary log line.
        """
        self._window_sec = float(window_sec)
        self._log_every_sec = float(log_every_sec)

        self._records: List[
            Tuple[float, int, int, int, int, float, float]
        ] = []

        self._sa_records: List[
            Tuple[
                float,
                int,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
            ]
        ] = []

        self._last_log_ts = time.perf_counter()

        logger.info(
            "FaceMetrics initialised | window=%.1fs log_every=%.1fs",
            self._window_sec,
            self._log_every_sec,
        )


    def update(
        self,
        decisions: List[IdentityDecision],
        tracks: List[Tracklet],
        ts_now: float,
    ) -> None:
        """
        Collect metrics for the current frame.

        We extract:
          - n_tracks:            number of active tracks
          - n_strong / n_weak / n_unknown decisions
          - avg_q_face:          mean face quality from decisions
          - avg_confidence:      mean identity confidence (0..1)

        Classification strategy:
          1) If identity_id is None -> unknown.
          2) Else, try to infer a band from the decision:
             - multiview engine:
                   use decision.extra["strength"] if present ("strong"/"weak"/"none")
             - any engine:
                   parse decision.reason for:
                     * "face_match_strong:"      -> strong
                     * "face_match_weak:"        -> weak
                     * "face_candidate_weak"     -> weak (candidate)
                     * "face_candidate_strong"   -> weak (strong-candidate not yet locked)
                     * "mv_match_strong"/"mv_match_weak" (future multiview tags)
             - if still unknown, fall back to confidence thresholds:
                   confidence >= 0.8       -> strong
                   0.5 <= confidence < 0.8 -> weak
                   else                     -> unknown

        SourceAuth additions (Phase-5.4):
          - For each decision, we *optionally* read:
                decision.source_auth_score : float [0,1]
                decision.source_auth_state : str in {REAL, LIKELY_REAL,
                                                     LIKELY_SPOOF, SPOOF,
                                                     UNCERTAIN, ...}
          - We accumulate:
                * number of decisions that had a usable source_auth_score
                * average source_auth_score over those
                * average q_face over those
                * counts per normalised state
          - And JOINT metrics:
                * identity STRONG & SA REAL-ish
                * identity STRONG & SA SPOOF-ish
                * identity WEAK   & SA REAL-ish
                * identity WEAK   & SA SPOOF-ish
        """
        n_tracks = len(tracks)

        strong = 0
        weak = 0
        unknown = 0
        qualities: List[float] = []
        confidences: List[float] = []

        sa_present = 0
        sa_score_sum = 0.0
        sa_q_sum = 0.0
        sa_q_n = 0

        sa_real = 0.0
        sa_lreal = 0.0
        sa_lspoof = 0.0
        sa_spoof = 0.0
        sa_uncertain = 0.0
        sa_missing = 0.0

        joint_strong_real = 0.0
        joint_strong_spoof = 0.0
        joint_weak_real = 0.0
        joint_weak_spoof = 0.0

        for d in decisions:
            q = _safe_float(getattr(d, "quality", None))
            if q >= 0.0:
                qualities.append(q)

            c = _safe_float(getattr(d, "confidence", None))
            if c >= 0.0:
                confidences.append(c)

            sa_score_raw = getattr(d, "source_auth_score", None)
            sa_score = _safe_float(sa_score_raw) if sa_score_raw is not None else -1.0

            sa_state_raw = getattr(d, "source_auth_state", None)
            sa_state = _normalise_source_auth_state(sa_state_raw)

            if sa_score >= 0.0:
                sa_present += 1
                sa_score_sum += sa_score

                if q >= 0.0:
                    sa_q_sum += q
                    sa_q_n += 1

            if sa_state is None:
                sa_missing += 1.0
            elif sa_state == "REAL":
                sa_real += 1.0
            elif sa_state == "LIKELY_REAL":
                sa_lreal += 1.0
            elif sa_state == "LIKELY_SPOOF":
                sa_lspoof += 1.0
            elif sa_state == "SPOOF":
                sa_spoof += 1.0
            elif sa_state == "UNCERTAIN":
                sa_uncertain += 1.0
            else:
                sa_uncertain += 1.0

            cat = (getattr(d, "category", "unknown") or "unknown").lower()
            if cat not in ("resident", "visitor", "watchlist"):
                unknown += 1
                continue

            band = "unknown"

            identity_id = getattr(d, "identity_id", None)
            if identity_id is None:
                band = "unknown"
            else:
                engine = (getattr(d, "engine", "classic") or "classic").lower()
                extra = getattr(d, "extra", None)
                reason = str(getattr(d, "reason", "") or "")

                if engine == "multiview" and isinstance(extra, dict):
                    strength_label = str(extra.get("strength", "")).lower()
                    if strength_label == "strong":
                        band = "strong"
                    elif strength_label == "weak":
                        band = "weak"
                    elif strength_label == "none":
                        band = "unknown"

                if band == "unknown":
                    if "face_match_strong:" in reason or "mv_match_strong" in reason:
                        band = "strong"
                    elif (
                        "face_match_weak:" in reason
                        or "face_candidate_weak" in reason
                        or "mv_match_weak" in reason
                    ):
                        band = "weak"
                    elif "face_candidate_strong" in reason:
                        band = "weak"

                if band == "unknown" and c >= 0.0:
                    if c >= 0.8:
                        band = "strong"
                    elif c >= 0.5:
                        band = "weak"

            if band == "strong":
                strong += 1
            elif band == "weak":
                weak += 1
            else:
                unknown += 1

            if sa_state is not None:
                is_realish = sa_state in ("REAL", "LIKELY_REAL")
                is_spoofish = sa_state in ("SPOOF", "LIKELY_SPOOF")

                if band == "strong":
                    if is_realish:
                        joint_strong_real += 1.0
                    elif is_spoofish:
                        joint_strong_spoof += 1.0
                elif band == "weak":
                    if is_realish:
                        joint_weak_real += 1.0
                    elif is_spoofish:
                        joint_weak_spoof += 1.0

        avg_q = sum(qualities) / len(qualities) if qualities else 0.0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        self._records.append(
            (
                ts_now,
                n_tracks,
                strong,
                weak,
                unknown,
                float(avg_q),
                float(avg_conf),
            )
        )

        avg_sa_score = sa_score_sum / sa_present if sa_present > 0 else 0.0
        avg_q_with_sa = sa_q_sum / sa_q_n if sa_q_n > 0 else 0.0

        self._sa_records.append(
            (
                ts_now,
                int(sa_present),
                float(avg_sa_score),
                float(sa_real),
                float(sa_lreal),
                float(sa_lspoof),
                float(sa_spoof),
                float(sa_uncertain),
                float(sa_missing),
                float(avg_q_with_sa),
                float(joint_strong_real),
                float(joint_strong_spoof),
                float(joint_weak_real),
                float(joint_weak_spoof),
            )
        )

        self._prune(ts_now)


    def maybe_log(self, ts_now: float) -> None:
        """
        Emit a rolling summary every log_every_sec seconds.

        We keep the original FaceMetrics line intact and add a second
        line for SourceAuth when SA data is present.
        """
        if (ts_now - self._last_log_ts) < self._log_every_sec:
            return

        self._last_log_ts = ts_now

        if not self._records:
            logger.info("FaceMetrics: no data yet")
            return

        (
            n_tracks_avg,
            strong_avg,
            weak_avg,
            unknown_avg,
            avg_q,
            avg_conf,
        ) = self._aggregate()

        logger.info(
            "FaceMetrics | tracks=%.1f strong=%.1f weak=%.1f unknown=%.1f "
            "| q_face=%.3f | conf=%.3f",
            n_tracks_avg,
            strong_avg,
            weak_avg,
            unknown_avg,
            avg_q,
            avg_conf,
        )

        if not self._sa_records:
            return

        has_any_sa = any(rec[1] > 0 for rec in self._sa_records)
        if not has_any_sa:
            return

        (
            sa_present_avg,
            sa_score_avg,
            sa_real_avg,
            sa_lreal_avg,
            sa_lspoof_avg,
            sa_spoof_avg,
            sa_uncertain_avg,
            sa_missing_avg,
            sa_q_avg,
            joint_strong_real_avg,
            joint_strong_spoof_avg,
            joint_weak_real_avg,
            joint_weak_spoof_avg,
        ) = self._aggregate_source_auth()

        logger.info(
            "FaceMetrics-SA | sa_present=%.1f sa_score=%.3f sa_q=%.3f "
            "| sa_states REAL=%.1f L_REAL=%.1f L_SPOOF=%.1f SPOOF=%.1f UNC=%.1f MISS=%.1f "
            "| joint SR=%.1f SS=%.1f WR=%.1f WS=%.1f",
            sa_present_avg,
            sa_score_avg,
            sa_q_avg,
            sa_real_avg,
            sa_lreal_avg,
            sa_lspoof_avg,
            sa_spoof_avg,
            sa_uncertain_avg,
            sa_missing_avg,
            joint_strong_real_avg,
            joint_strong_spoof_avg,
            joint_weak_real_avg,
            joint_weak_spoof_avg,
        )


    def _prune(self, ts_now: float) -> None:
        """
        Remove records older than _window_sec from both identity and
        SourceAuth rolling buffers.
        """
        cutoff = ts_now - self._window_sec

        while self._records and self._records[0][0] < cutoff:
            self._records.pop(0)

        while self._sa_records and self._sa_records[0][0] < cutoff:
            self._sa_records.pop(0)

    def _aggregate(self) -> Tuple[float, float, float, float, float, float]:
        """
        Aggregate sliding-window metrics for identity pipeline.
        Returns:
            (avg_tracks, avg_strong, avg_weak, avg_unknown,
             avg_quality, avg_confidence)
        """
        if not self._records:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        total = len(self._records)
        sum_tracks = 0.0
        sum_strong = 0.0
        sum_weak = 0.0
        sum_unknown = 0.0
        sum_q = 0.0
        sum_conf = 0.0

        for (_, nt, st, wk, un, q, c) in self._records:
            sum_tracks += nt
            sum_strong += st
            sum_weak += wk
            sum_unknown += un
            sum_q += q
            sum_conf += c

        return (
            sum_tracks / total,
            sum_strong / total,
            sum_weak / total,
            sum_unknown / total,
            sum_q / total,
            sum_conf / total,
        )

    def _aggregate_source_auth(
        self,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
        """
        Aggregate sliding-window metrics for SourceAuth.

        Returns:
            (avg_sa_present,
             avg_sa_score,
             avg_sa_real,
             avg_sa_lreal,
             avg_sa_lspoof,
             avg_sa_spoof,
             avg_sa_uncertain,
             avg_sa_missing,
             avg_q_with_sa,
             avg_joint_strong_real,
             avg_joint_strong_spoof,
             avg_joint_weak_real,
             avg_joint_weak_spoof)
        """
        if not self._sa_records:
            return (0.0,) * 13

        total = len(self._sa_records)
        sum_present = 0.0
        sum_score = 0.0
        sum_real = 0.0
        sum_lreal = 0.0
        sum_lspoof = 0.0
        sum_spoof = 0.0
        sum_unc = 0.0
        sum_missing = 0.0
        sum_q_sa = 0.0
        sum_joint_strong_real = 0.0
        sum_joint_strong_spoof = 0.0
        sum_joint_weak_real = 0.0
        sum_joint_weak_spoof = 0.0

        for (
            _ts,
            n_present,
            avg_score,
            n_real,
            n_lreal,
            n_lspoof,
            n_spoof,
            n_unc,
            n_miss,
            avg_q_sa,
            j_sr,
            j_ss,
            j_wr,
            j_ws,
        ) in self._sa_records:
            sum_present += n_present
            sum_score += avg_score
            sum_real += n_real
            sum_lreal += n_lreal
            sum_lspoof += n_lspoof
            sum_spoof += n_spoof
            sum_unc += n_unc
            sum_missing += n_miss
            sum_q_sa += avg_q_sa
            sum_joint_strong_real += j_sr
            sum_joint_strong_spoof += j_ss
            sum_joint_weak_real += j_wr
            sum_joint_weak_spoof += j_ws

        return (
            sum_present / total,
            sum_score / total,
            sum_real / total,
            sum_lreal / total,
            sum_lspoof / total,
            sum_spoof / total,
            sum_unc / total,
            sum_missing / total,
            sum_q_sa / total,
            sum_joint_strong_real / total,
            sum_joint_strong_spoof / total,
            sum_joint_weak_real / total,
            sum_joint_weak_spoof / total,
        )


    def get_source_auth_summary(self) -> Dict[str, float]:
        """
        Return a dictionary with the current sliding-window SourceAuth
        aggregates.

        This is purely additive and does not affect logging behaviour.
        Keys:
          - sa_present_avg
          - sa_score_avg
          - sa_real_avg
          - sa_lreal_avg
          - sa_lspoof_avg
          - sa_spoof_avg
          - sa_uncertain_avg
          - sa_missing_avg
          - sa_q_avg
          - joint_strong_real_avg
          - joint_strong_spoof_avg
          - joint_weak_real_avg
          - joint_weak_spoof_avg
        """
        (
            sa_present_avg,
            sa_score_avg,
            sa_real_avg,
            sa_lreal_avg,
            sa_lspoof_avg,
            sa_spoof_avg,
            sa_uncertain_avg,
            sa_missing_avg,
            sa_q_avg,
            joint_strong_real_avg,
            joint_strong_spoof_avg,
            joint_weak_real_avg,
            joint_weak_spoof_avg,
        ) = self._aggregate_source_auth()

        return {
            "sa_present_avg": sa_present_avg,
            "sa_score_avg": sa_score_avg,
            "sa_real_avg": sa_real_avg,
            "sa_lreal_avg": sa_lreal_avg,
            "sa_lspoof_avg": sa_lspoof_avg,
            "sa_spoof_avg": sa_spoof_avg,
            "sa_uncertain_avg": sa_uncertain_avg,
            "sa_missing_avg": sa_missing_avg,
            "sa_q_avg": sa_q_avg,
            "joint_strong_real_avg": joint_strong_real_avg,
            "joint_strong_spoof_avg": joint_strong_spoof_avg,
            "joint_weak_real_avg": joint_weak_real_avg,
            "joint_weak_spoof_avg": joint_weak_spoof_avg,
        }



def _safe_float(x) -> float:
    """
    Defensive conversion:
      - If x is None → -1.0
      - If cast fails → -1.0
      - Otherwise returns float(x)
    """
    try:
        return float(x)
    except Exception:
        return -1.0


def _normalise_source_auth_state(raw) -> str | None:
    """
    Normalise SourceAuth state into a canonical upper-case label, or None
    if we cannot interpret it.

    Expected inputs (case-insensitive, tolerant to extra whitespace):
      - "REAL"
      - "LIKELY_REAL"
      - "LIKELY-SPOOF" / "LIKELY_SPOOF"
      - "SPOOF"
      - "UNCERTAIN"
      - None / "" → None
    """
    if raw is None:
        return None

    try:
        s = str(raw).strip().upper()
    except Exception:
        return None

    if not s:
        return None

    s = s.replace("-", "_")

    if s in ("REAL", "LIKELY_REAL", "LIKELY_SPOOF", "SPOOF", "UNCERTAIN"):
        return s

    return s
