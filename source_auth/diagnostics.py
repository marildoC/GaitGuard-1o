
from __future__ import annotations

from typing import Optional, List, Iterable

from source_auth.types import (
    SourceAuthScores,
    SourceAuthDebug,
)




def _format_reliability_suffix(scores: SourceAuthScores) -> Optional[str]:
    """
    Turn reliability flags into a compact suffix string, or None if
    there is nothing interesting to report yet.

    Example formats:
      - "rel=none"                      → no flag is satisfied (rarely emitted)
      - "rel=motion"                    → only motion evidence good
      - "rel=motion,landmarks,bg"       → all main cues are reliable
    """
    rel = scores.reliability
    if rel is None:
        return None

    flags: List[str] = []
    if getattr(rel, "enough_motion", False):
        flags.append("motion")
    if getattr(rel, "enough_landmarks", False):
        flags.append("landmarks")
    if getattr(rel, "enough_background", False):
        flags.append("bg")

    if not flags:
        return None

    return "rel=" + ",".join(flags)


def _format_phase_suffix(debug: Optional[SourceAuthDebug]) -> Optional[str]:
    """
    Extract a lightweight 'phase' tag from debug, if provided.

    Example:
      - "phase=source_auth_phase5_fusion_state_machine"
    """
    if not debug or not isinstance(debug, dict):
        return None

    phase = debug.get("phase")
    if not phase:
        return None

    return f"phase={phase}"


def _safe_float(value: object) -> Optional[float]:
    """
    Best-effort conversion to float; returns None if not convertible.
    """
    try:
        return float(value)
    except Exception:
        return None


def _append_scalar(
    parts: List[str],
    label: str,
    value: object,
) -> None:
    """
    Append 'label=xx.xx' to parts if value can be converted to float.
    """
    v = _safe_float(value)
    if v is None:
        return
    parts.append(f"{label}={v:.2f}")


def _first_debug_scalar(
    debug: Optional[SourceAuthDebug],
    candidate_keys: Iterable[str],
) -> Optional[float]:
    """
    Look up the first present key in debug and return it as float
    (or None if no usable value is found).
    """
    if not debug or not isinstance(debug, dict):
        return None

    for key in candidate_keys:
        if key in debug:
            v = _safe_float(debug[key])
            if v is not None:
                return v
    return None




def format_source_auth_reason(
    scores: SourceAuthScores,
    debug: Optional[SourceAuthDebug] = None,
) -> str:
    """
    Produce a compact, human-readable fragment describing SourceAuth state.

    Contract (prefix is stable):
      - ALWAYS starts with:
            "src_auth:score={:.2f}:state={STATE}"

        where STATE is one of "REAL", "LIKELY_REAL", "SPOOF", "LIKELY_SPOOF",
        or "UNCERTAIN" (and possible future extensions).

    Phase-5.2:
      - When component scores and debug statistics are available, we enrich
        the fragment with colon-separated key=value pairs:

          "src_auth:score=0.18:state=SPOOF"
          ":3d=0.12:screen=0.85:bg=0.62"
          ":parallax=0.08:border=0.81:flicker=0.74"
          ":bg_color=0.67:bg_tex=0.59:rel=motion,bg:phase=source_auth_phase5..."

      - All extra fields are *optional* and added only if data is present.
      - Callers must treat the returned string as opaque; no consumer should
        depend on a fixed number or order of fields.
    """
    raw_score = float(getattr(scores, "source_auth_score", 0.0))
    s = max(0.0, min(1.0, raw_score))

    state = getattr(scores, "state", None) or "UNCERTAIN"

    parts: List[str] = [
        f"src_auth:score={s:.2f}",
        f"state={state}",
    ]

    comps = getattr(scores, "components", None)
    if comps is not None:
        _append_scalar(parts, "3d", getattr(comps, "planar_3d", None))
        _append_scalar(parts, "screen", getattr(comps, "screen_artifacts", None))
        _append_scalar(
            parts,
            "bg",
            getattr(comps, "background_consistency", None),
        )

    parallax_val = _first_debug_scalar(
        debug,
        candidate_keys=[
            "motion_parallax_ratio",
            "motion_parallax",
            "motion_depth_cue",
        ],
    )
    if parallax_val is not None:
        _append_scalar(parts, "parallax", parallax_val)

    border_val = _first_debug_scalar(
        debug,
        candidate_keys=[
            "screen_border_strength",
            "screen_border_score",
            "screen_border_ratio",
        ],
    )
    if border_val is not None:
        _append_scalar(parts, "border", border_val)

    flicker_val = _first_debug_scalar(
        debug,
        candidate_keys=[
            "screen_flicker_score",
            "screen_temporal_inconsistency",
        ],
    )
    if flicker_val is not None:
        _append_scalar(parts, "flicker", flicker_val)

    bg_color_val = _first_debug_scalar(
        debug,
        candidate_keys=[
            "background_color_delta_norm",
            "background_color_mismatch",
        ],
    )
    if bg_color_val is not None:
        _append_scalar(parts, "bg_color", bg_color_val)

    bg_tex_val = _first_debug_scalar(
        debug,
        candidate_keys=[
            "background_texture_delta_norm",
            "background_texture_mismatch",
        ],
    )
    if bg_tex_val is not None:
        _append_scalar(parts, "bg_tex", bg_tex_val)

    rel_suffix = _format_reliability_suffix(scores)
    if rel_suffix:
        parts.append(rel_suffix)

    phase_suffix = _format_phase_suffix(debug)
    if phase_suffix:
        parts.append(phase_suffix)

    return ":".join(parts)
