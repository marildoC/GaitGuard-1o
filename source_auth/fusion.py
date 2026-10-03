
from __future__ import annotations

from typing import Any, Dict, Optional

from source_auth.types import (
    SourceAuthComponentScores,
    SourceAuthReliabilityFlags,
    SourceAuthScores,
    SourceAuthState,
)


def fuse_source_auth(
    track_id: int,
    components: SourceAuthComponentScores,
    reliability: SourceAuthReliabilityFlags,
    cfg: Any,
    prev_scores: Optional[SourceAuthScores] = None,
) -> SourceAuthScores:
    """
    Fuse motion / screen / background cues into SourceAuthScores.

    Parameters
    ----------
    track_id:
        Track identifier (as used by the perception / identity stack).

    components:
        SourceAuthComponentScores with:
          - planar_3d           → motion cue (1 = strong 3D, 0 = planar)
          - screen_artifacts    → screen cue (1 = strong screen evidence)
          - background_consistency
                               → background cue (1 = same-world, 0 = mismatch)

    reliability:
        SourceAuthReliabilityFlags with coarse reliability flags:
          - enough_motion      → motion cue is usable
          - enough_landmarks   → landmarks were good (informational)
          - enough_background  → at least one of screen/background cues usable

    cfg:
        SourceAuthConfig or duck-typed object providing:
          - neutral_score
          - unreliable_cue_weight_factor
          - fusion_weight_motion / screen / background
          - ema_alpha
          - real_min_score / likely_real_min_score
          - spoof_max_score / likely_spoof_max_score
          - fusion_mode (optional: "legacy" | "v2")

    prev_scores:
        Previously fused SourceAuthScores for this track (if any). Used for:
          - EMA smoothing (based on previous source_auth_score)
          - light hysteresis on discrete states.

    Returns
    -------
    SourceAuthScores
        New fused scores and discrete state for this track.
    """
    neutral = float(getattr(cfg, "neutral_score", 0.5))

    M = float(components.planar_3d)
    S = float(components.screen_artifacts)
    B = float(components.background_consistency)

    has_motion = bool(getattr(reliability, "enough_motion", False))
    has_background = bool(getattr(reliability, "enough_background", False))

    if not (has_motion or has_background):
        scores = SourceAuthScores(
            track_id=track_id,
            source_auth_score=neutral,
            state="UNCERTAIN",
            components=components,
            reliability=reliability,
        )
        scores.update_debug(
            "",
            {
                "fusion_phase": "source_auth_fusion",
                "fusion_mode": str(getattr(cfg, "fusion_mode", "v2")),
                "fusion_reason": "no_reliable_cues",
            },
        )
        return scores

    base_w_motion = float(getattr(cfg, "fusion_weight_motion", 0.5))
    base_w_screen = float(getattr(cfg, "fusion_weight_screen", 0.3))
    base_w_bg = float(getattr(cfg, "fusion_weight_background", 0.2))

    unreliable_shrink = float(
        getattr(cfg, "unreliable_cue_weight_factor", 0.3)
    )
    unreliable_shrink = max(0.0, min(1.0, unreliable_shrink))

    def _soften_and_weight(
        value: float, reliable_flag: bool, base_w: float
    ) -> tuple[float, float]:
        """
        Push value towards neutral and reduce weight when unreliable.

        Returns:
          (value_eff, weight_eff)
        """
        if reliable_flag:
            return value, base_w

        r = unreliable_shrink
        value_eff = neutral + r * (value - neutral)
        weight_eff = base_w * r
        return value_eff, weight_eff

    M_eff, wM_eff = _soften_and_weight(M, has_motion, base_w_motion)
    S_eff, wS_eff = _soften_and_weight(S, has_background, base_w_screen)
    B_eff, wB_eff = _soften_and_weight(B, has_background, base_w_bg)

    fusion_mode = str(getattr(cfg, "fusion_mode", "v2")).lower()

    if fusion_mode == "legacy":
        L_motion = M_eff
        L_screen = 1.0 - S_eff
        L_bg = B_eff

        weighted_sum = wM_eff * L_motion + wS_eff * L_screen + wB_eff * L_bg
        weight_total = wM_eff + wS_eff + wB_eff

        if weight_total <= 0.0:
            sa_linear = neutral
            fusion_reason = "weights_zero_legacy"
        else:
            sa_linear = weighted_sum / weight_total
            fusion_reason = "legacy_weighted_average"

        sa_linear = max(0.0, min(1.0, float(sa_linear)))

        real_evidence = weighted_sum
        spoof_evidence = 0.0

    else:
        real_evidence = wM_eff * M_eff + wB_eff * B_eff
        spoof_evidence = wS_eff * S_eff + wB_eff * (1.0 - B_eff)

        raw = real_evidence - spoof_evidence

        denom = wM_eff + wS_eff + wB_eff
        if denom <= 0.0:
            sa_linear = neutral
            fusion_reason = "weights_zero_v2"
        else:
            x = raw / denom
            x = max(-1.0, min(1.0, float(x)))
            sa_linear = 0.5 + 0.5 * x
            fusion_reason = "v2_real_minus_spoof"

        sa_linear = max(0.0, min(1.0, float(sa_linear)))

    alpha = float(getattr(cfg, "ema_alpha", 0.35))  # CHANGED: 0.6 → 0.35
    alpha = max(0.0, min(1.0, alpha))

    if prev_scores is not None:
        prev_score = float(getattr(prev_scores, "source_auth_score", neutral))
    else:
        prev_score = neutral

    sa_smoothed = alpha * sa_linear + (1.0 - alpha) * prev_score
    sa_smoothed = max(0.0, min(1.0, float(sa_smoothed)))

    real_min = float(getattr(cfg, "real_min_score", 0.75))  # CHANGED: 0.90 → 0.75
    likely_real_min = float(getattr(cfg, "likely_real_min_score", 0.55))  # CHANGED: 0.70 → 0.55
    spoof_max = float(getattr(cfg, "spoof_max_score", 0.15))  # CHANGED: 0.10 → 0.15
    likely_spoof_max = float(getattr(cfg, "likely_spoof_max_score", 0.35))  # CHANGED: 0.30 → 0.35

    if sa_smoothed >= real_min:
        new_state: SourceAuthState = "REAL"
    elif sa_smoothed >= likely_real_min:
        new_state = "LIKELY_REAL"
    elif sa_smoothed <= spoof_max:
        new_state = "SPOOF"
    elif sa_smoothed <= likely_spoof_max:
        new_state = "LIKELY_SPOOF"
    else:
        new_state = "UNCERTAIN"

    if prev_scores is not None:
        prev_state: SourceAuthState = prev_scores.state

        if (
            prev_state == "REAL"
            and sa_smoothed < real_min
            and sa_smoothed >= likely_real_min
        ):
            new_state = "REAL"

        if (
            prev_state == "SPOOF"
            and sa_smoothed > spoof_max
            and sa_smoothed <= likely_spoof_max
        ):
            new_state = "SPOOF"

    scores = SourceAuthScores(
        track_id=track_id,
        source_auth_score=sa_smoothed,
        state=new_state,
        components=components,
        reliability=reliability,
    )

    debug: Dict[str, Any] = {
        "fusion_phase": "source_auth_fusion",
        "fusion_mode": fusion_mode,
        "fusion_reason": fusion_reason,
        "fusion_M_planar_3d": M,
        "fusion_S_screen_artifacts": S,
        "fusion_B_background_consistency": B,
        "fusion_M_eff": M_eff,
        "fusion_S_eff": S_eff,
        "fusion_B_eff": B_eff,
        "fusion_wM_eff": wM_eff,
        "fusion_wS_eff": wS_eff,
        "fusion_wB_eff": wB_eff,
        "fusion_real_evidence": float(real_evidence),
        "fusion_spoof_evidence": float(spoof_evidence),
        "fusion_linear_score": sa_linear,
        "fusion_smoothed_score": sa_smoothed,
        "fusion_has_motion": has_motion,
        "fusion_has_background": has_background,
    }

    scores.update_debug("", debug)
    return scores
