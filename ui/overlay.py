"""
ui/overlay.py

Phase-1/2A overlay (Wave-3).

Responsibilities:
  - draw per-track bounding boxes
  - color-code boxes by identity category (resident/visitor/watchlist/unknown)
  - render identity labels (name/person_id + confidence) when enabled
  - optional debug line with face-reason (distance/quality)
  - optional compact tags for engine / pose_bin (3D) when enabled
  - optional SourceAuth tag (REAL / SPOOF / SAR / SAS / UNC) when enabled
  - optional SourceAuth border tint (e.g. red for SPOOF) when enabled
  - status HUD with #tracks, #alerts, and optional FPS

Config flags in cfg.ui (all optional):
  - show_identity_labels      : bool (default True)
  - show_debug_face_hud       : bool (default False)
  - show_fps                  : bool (default True)
  - show_engine_tag           : bool (default False)  # Wave-3 3D
  - show_pose_tag             : bool (default False)  # Wave-3 3D
  - show_source_auth_tag      : bool (default True)   # SourceAuth badge
  - show_source_auth_border   : bool (default True)   # SourceAuth border hint
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np

from schemas import Frame, Tracklet, IdentityDecision, EventFlags, Alert




def _get_ui_flag(ui_cfg: Any, name: str, default: bool) -> bool:
    """
    Safe helper to read boolean flags from cfg.ui.

    Works if ui_cfg is:
      - a dataclass (attributes)
      - a simple object with attributes
      - a dict-like object (with .get)
    """
    if ui_cfg is None:
        return default

    if isinstance(ui_cfg, dict):
        val = ui_cfg.get(name, default)
    else:
        val = getattr(ui_cfg, name, default)

    try:
        return bool(val)
    except Exception:
        return default


def _norm_xyxy(arr: Any) -> Optional[Tuple[int, int, int, int]]:
    """
    Normalise any 4-number bbox to (x1, y1, x2, y2).

    We DON'T trust the order. We just take:
      x1 = min(x-like coords)
      y1 = min(y-like coords)
      x2 = max(x-like coords)
      y2 = max(y-like coords)

    This makes us robust to:
      - xyxy
      - tlbr  (top,left,bottom,right)
      - ltrb  (left,top,right,bottom)
    as long as there are 4 numbers.
    """
    if arr is None:
        return None

    a = np.asarray(arr).reshape(-1)
    if a.size != 4:
        return None

    x_candidates = [a[0], a[2]]
    y_candidates = [a[1], a[3]]

    x1 = int(min(x_candidates))
    y1 = int(min(y_candidates))
    x2 = int(max(x_candidates))
    y2 = int(max(y_candidates))

    return x1, y1, x2, y2


def _extract_bbox(trk: Tracklet) -> Optional[Tuple[int, int, int, int]]:
    """
    Try to extract a (x1, y1, x2, y2) bounding box from a Tracklet.

    We support multiple attribute names to stay robust against schema changes:
      - tlbr       (OC-SORT / BYTETrack style)
      - bbox_xyxy
      - bbox
      - xyxy
      - xywh  (converted to xyxy)

    As a last resort, we scan all attributes and pick the first 4-element
    array-like as the bbox.
    """
    for attr in ("tlbr", "bbox_xyxy", "bbox", "xyxy"):
        if hasattr(trk, attr):
            return _norm_xyxy(getattr(trk, attr))

    if hasattr(trk, "xywh"):
        x, y, w, h = np.asarray(getattr(trk, "xywh")).reshape(-1).astype(float)
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return _norm_xyxy([x1, y1, x2, y2])

    try:
        for name, value in vars(trk).items():
            if isinstance(value, (list, tuple, np.ndarray)):
                a = np.asarray(value).reshape(-1)
                if a.size == 4:
                    return _norm_xyxy(a)
    except Exception:
        pass

    return None


def _category_color(category: str) -> Tuple[int, int, int]:
    """
    Map identity category -> BGR color for drawing.

    resident  -> green
    visitor   -> blue/orange-ish
    watchlist -> red
    unknown   -> grey/white
    """
    c = (category or "").lower()
    if c == "resident":
        return (0, 255, 0)
    if c == "visitor":
        return (255, 200, 0)
    if c == "watchlist":
        return (0, 0, 255)
    return (200, 200, 200)


def _build_decision_map(
    decisions: List[IdentityDecision],
) -> Dict[int, IdentityDecision]:
    """
    Build a quick lookup: track_id -> IdentityDecision.
    """
    by_track: Dict[int, IdentityDecision] = {}
    for d in decisions:
        try:
            tid = int(d.track_id)
        except Exception:
            continue
        by_track[tid] = d
    return by_track



def _compute_identity_consensus(
    decisions: List[IdentityDecision],
) -> Optional[str]:
    """
    LAYER 4A: Determine consensus person_id across all tracks.
    
    When OC-SORT creates multiple tracks for same physical person,
    compute consensus identity to avoid showing "unknown" next to known identity.
    
    Algorithm:
      1. Count how many tracks are bound to each person_id
      2. If person_id has ≥50% of tracks → use as consensus
      3. If multiple people tied → use person with most evidence
      4. If no majority → return None (no consensus)
    
    Benefits (LAYER 4A):
      - Eliminates visual oscillation ("unknown" ↔ "marildo")
      - Shows coherent identity even when OC-SORT creates multiple tracks
      - Improves perceived system responsiveness
      - Zero impact on binding logic or accuracy
    
    Args:
        decisions: List of IdentityDecision objects
    
    Returns:
        consensus_person_id (str) if found, else None
    """
    if not decisions:
        return None
    
    person_counts: Dict[Optional[str], int] = {}
    for decision in decisions:
        try:
            pid = decision.identity_id
            if pid and pid != "unknown":  # Count only known identities
                person_counts[pid] = person_counts.get(pid, 0) + 1
        except Exception:
            pass
    
    if not person_counts:
        return None
    
    total_tracks = len(decisions)
    majority_threshold = total_tracks * 0.5
    
    for person_id, count in person_counts.items():
        if count >= majority_threshold:
            return person_id
    
    return max(person_counts.items(), key=lambda x: x[1])[0]


def _identity_label(
    decision: Optional[IdentityDecision],
    ui_cfg: Any = None,
    consensus_person: Optional[str] = None,
) -> Tuple[str, str, Tuple[int, int, int]]:
    """
    Compute (main_label, debug_line, color) for a given decision.

    main_label: what is written above the box (e.g. 'p_0005 ✓ (0.92)' or 'unknown').
    debug_line: optional extra line with distance/quality encoded inside 'reason'.
                When enabled via cfg.ui.show_engine_tag / show_pose_tag, we append
                a compact tag like '[multiview,front]'.
    color:      BGR for box & text (base color by identity category).
    
    FIX 5 Enhancement:
      - Display binding state: ✓ (CONFIRMED), ⧕ (PENDING), ◯ (UNKNOWN), ✗ (ERROR)
      - Show binding confidence with identity confidence
      - Color-code by binding state (green=confirmed, orange=pending, gray=unknown)
    
    LAYER 4A Enhancement:
      If decision is None but consensus_person provided,
      display consensus_person instead of "unknown".
      This eliminates visual oscillation from multiple tracks.
    """
    if decision is None:
        if consensus_person:
            name = consensus_person
            color = _category_color("resident")  # Assume resident for consensus
            return f"{name} (consensus)", "", color
        else:
            return "unknown", "", _category_color("unknown")

    if hasattr(decision, "display_name") and getattr(decision, "display_name"):
        name = str(getattr(decision, "display_name"))
    elif hasattr(decision, "name") and getattr(decision, "name"):
        name = str(getattr(decision, "name"))
    elif getattr(decision, "identity_id", None):
        name = str(decision.identity_id)
    else:
        if consensus_person:
            name = consensus_person
        else:
            name = "unknown"

    binding_state = getattr(decision, "binding_state", "UNKNOWN") or "UNKNOWN"  # Handle None
    if not isinstance(binding_state, str):
        binding_state = str(binding_state)
    binding_confidence = getattr(decision, "binding_confidence", 0.0)
    try:
        binding_confidence = float(binding_confidence)
    except (ValueError, TypeError):
        binding_confidence = 0.0
    
    binding_emoji = ""
    try:
        if binding_state == "CONFIRMED_STRONG" or binding_state == "CONFIRMED_WEAK":
            binding_emoji = "✓"  # Checkmark for confirmed
        elif binding_state == "PENDING" or binding_state == "SWITCH_PENDING":
            binding_emoji = "⧕"  # Hourglass for pending
        elif binding_state == "UNKNOWN":
            binding_emoji = "◯"  # Circle for unknown
        else:
            binding_emoji = "?"  # Unknown state
    except Exception:
        if binding_state == "CONFIRMED_STRONG" or binding_state == "CONFIRMED_WEAK":
            binding_emoji = "[Y]"
        elif binding_state == "PENDING" or binding_state == "SWITCH_PENDING":
            binding_emoji = "[~]"
        elif binding_state == "UNKNOWN":
            binding_emoji = "[?]"
        else:
            binding_emoji = "[!]"

    conf = getattr(decision, "confidence", None)
    if conf is None:
        main_label = name
        if binding_emoji:
            main_label = f"{binding_emoji} {name}"
    else:
        try:
            c = max(0.0, min(float(conf), 1.0))
        except Exception:
            c = 0.0
        if binding_emoji:
            main_label = f"{binding_emoji} {name} ({c:.2f})"
        else:
            main_label = f"{name} ({c:.2f})"

    reason = getattr(decision, "reason", "") or ""
    if len(reason) > 48:
        reason = reason[:45] + "..."

    dbg = reason

    show_engine_tag = _get_ui_flag(ui_cfg, "show_engine_tag", False)
    show_pose_tag = _get_ui_flag(ui_cfg, "show_pose_tag", False)
    show_binding_state = _get_ui_flag(ui_cfg, "show_binding_state", True)

    tags: List[str] = []
    if show_engine_tag and hasattr(decision, "engine"):
        try:
            eng = str(getattr(decision, "engine") or "").strip()
        except Exception:
            eng = ""
        if eng:
            tags.append(eng)

    if show_pose_tag and hasattr(decision, "pose_bin"):
        try:
            pb = str(getattr(decision, "pose_bin") or "").strip()
        except Exception:
            pb = ""
        if pb and pb.lower() != "none":
            tags.append(pb)
    
    if show_binding_state and binding_state:
        try:
            bs = str(binding_state).upper()
            if bs != "BYPASS":  # Don't show bypass state
                tags.append(f"binding:{bs}")
        except Exception:
            pass

    if tags:
        tag_str = "[" + ",".join(tags) + "]"
        dbg = (dbg + " " + tag_str).strip() if dbg else tag_str

    base_color = _category_color(getattr(decision, "category", "unknown") or "unknown")
    
    if binding_state in ["CONFIRMED_STRONG", "CONFIRMED_WEAK"]:
        color = (0, 255, 0)
    elif binding_state in ["PENDING", "SWITCH_PENDING"]:
        color = (0, 165, 255)
    elif binding_state == "UNKNOWN":
        color = (128, 128, 128)
    else:
        color = base_color
    
    return main_label, dbg, color




def _get_source_auth_state_and_score(
    decision: Optional[IdentityDecision],
) -> Tuple[Optional[str], Optional[float]]:
    """
    Extract SourceAuth state/score from IdentityDecision, if present.

    Expected fields (optional on IdentityDecision):
      - decision.source_auth_state : "REAL" / "LIKELY_REAL" / "LIKELY_SPOOF" /
                                     "SPOOF" / "UNCERTAIN" / None
      - decision.source_auth_score : float in [0,1]
    """
    if decision is None:
        return None, None

    state = getattr(decision, "source_auth_state", None)
    if state is not None:
        try:
            state = str(state).upper()
        except Exception:
            state = None

    score_raw = getattr(decision, "source_auth_score", None)
    score: Optional[float]
    try:
        score = float(score_raw) if score_raw is not None else None
    except Exception:
        score = None

    return state, score


def _apply_source_auth_color(
    decision: Optional[IdentityDecision],
    base_color: Tuple[int, int, int],
    ui_cfg: Any = None,
) -> Tuple[int, int, int]:
    """
    Optionally adjust box color based on SourceAuth state.

    - REAL / LIKELY_REAL: keep base color (identity category dominates).
    - LIKELY_SPOOF / SPOOF: optionally tilt to strong red, to highlight risk.

    Controlled by cfg.ui.show_source_auth_border (default True).
    """
    show_border_hint = _get_ui_flag(ui_cfg, "show_source_auth_border", True)
    if not show_border_hint or decision is None:
        return base_color

    state, _ = _get_source_auth_state_and_score(decision)
    if not state:
        return base_color

    if state in ("SPOOF", "LIKELY_SPOOF"):
        return (0, 0, 255)

    return base_color


def _source_auth_badge(
    decision: Optional[IdentityDecision],
    ui_cfg: Any = None,
) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    """
    Decide whether to show a small SourceAuth badge next to the box, and
    with which text/color.

    Wave-3 policy (UI-level, conservative at engine level):

      1) If the engine explicitly flags:
         - SPOOF or LIKELY_SPOOF  → "SPOOF" (red)
         - REAL or LIKELY_REAL with good score (>= 0.55) → "REAL" (green)

      2) For all other states (UNCERTAIN, None, low-score REAL-ish),
         we derive a continuous, score-based view around a neutral point 0.5:

         - score <  0.5 → "SAS" (SourceAuth Spoof-leaning, red)
         - score >  0.5 → "SAR" (SourceAuth Real-leaning, green)
         - score == 0.5 (± epsilon) → "UNC" (uncertain, orange)

      3) If both state and score are absent, no badge is shown.

    This makes the badge expressive for operators:
      - HARD states from the engine → REAL / SPOOF
      - SOFT, score-only regime     → SAR / SAS / UNC
    """
    if not _get_ui_flag(ui_cfg, "show_source_auth_tag", True):
        return None

    if decision is None:
        return None

    state, score = _get_source_auth_state_and_score(decision)
    if state is None and score is None:
        return None

    s = (state or "").upper()

    if s in ("SPOOF", "LIKELY_SPOOF"):
        return "SPOOF", (0, 0, 255)  # red (BGR)

    ok_threshold = 0.55
    if s in ("REAL", "LIKELY_REAL") and score is not None and score >= ok_threshold:
        return "REAL", (0, 200, 0)  # green (BGR)

    neutral = 0.5

    if score is None:
        return "UNC", (0, 165, 255)

    eps = 1e-3

    if abs(score - neutral) <= eps:
        return "UNC", (0, 165, 255)  # orange (BGR)

    if score < neutral:
        return "SAS", (0, 0, 255)  # red (BGR)

    return "SAR", (0, 200, 0)  # green (BGR)


def _draw_source_auth_badge(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    badge: Tuple[str, Tuple[int, int, int]],
) -> None:
    """
    Render a small filled rectangle with SourceAuth text near the bbox
    (top-right corner), without going outside image borders.
    """
    if img is None or badge is None:
        return

    h, w = img.shape[:2]
    text, color = badge
    if not text:
        return

    x1, y1, x2, y2 = bbox

    margin_x = 4
    margin_y = 4

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    br_x2 = min(x2 - margin_x, w - 5)
    br_x1 = max(br_x2 - tw - 6, 5)
    br_y1 = max(y1 + margin_y, th + baseline + 5)
    br_y2 = br_y1 + th + baseline

    cv2.rectangle(
        img,
        (br_x1 - 1, br_y1 - 1),
        (br_x2 + 1, br_y2 + 1),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.rectangle(
        img,
        (br_x1, br_y1),
        (br_x2, br_y2),
        color,
        thickness=-1,
    )

    cv2.putText(
        img,
        text,
        (br_x1 + 3, br_y2 - baseline),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )




def draw_overlay(
    frame: Frame,
    tracks: List[Tracklet],
    decisions: List[IdentityDecision],
    events: List[EventFlags],
    alerts: List[Alert],
    ui_cfg: Any = None,
    fps: Optional[float] = None,
) -> np.ndarray:
    """
    Return an image (BGR) to display with identity-aware overlay.

    For each track:
      - draw a bounding box
      - (optionally) draw an identity label (name/person_id + confidence)
      - (optionally) draw a debug line with face-reason
      - (optionally) append compact [engine,pose_bin] tags to the debug line
      - (optionally) draw a SourceAuth badge (REAL / SPOOF / SAR / SAS / UNC)
      - color-code by category, optionally tilted by SourceAuth spoof state

    Also keep a small status line at the top with #tracks and #alerts,
    and optionally FPS (controlled by ui.show_fps).
    """
    if frame.image is None:
        raise ValueError("Frame.image is None inside draw_overlay")

    img = frame.image.copy()
    h, w = img.shape[:2]

    show_identity_labels = _get_ui_flag(ui_cfg, "show_identity_labels", True)
    show_debug_face_hud = _get_ui_flag(ui_cfg, "show_debug_face_hud", False)
    show_fps = _get_ui_flag(ui_cfg, "show_fps", True)

    if show_fps and fps is not None and fps > 0.0:
        status = f"FPS: {fps:4.1f} | tracks: {len(tracks)} | alerts: {len(alerts)}"
    else:
        status = f"tracks: {len(tracks)} | alerts: {len(alerts)}"

    cv2.putText(
        img,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    dec_by_track = _build_decision_map(decisions)

    consensus_person = _compute_identity_consensus(decisions)
    
    for trk in tracks:
        bbox = _extract_bbox(trk)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        tid = int(getattr(trk, "track_id", -1))

        decision = dec_by_track.get(tid)
        main_label, dbg_line, color = _identity_label(
            decision, ui_cfg=ui_cfg, consensus_person=consensus_person
        )

        color = _apply_source_auth_color(decision, color, ui_cfg=ui_cfg)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        sa_badge = _source_auth_badge(decision, ui_cfg=ui_cfg)
        if sa_badge is not None:
            _draw_source_auth_badge(img, bbox, sa_badge)

        if not show_identity_labels:
            continue

        label = f"ID {tid}: {main_label}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_x = x1
        label_y = max(y1 - 10, th + 5)

        cv2.rectangle(
            img,
            (label_x - 2, label_y - th - baseline),
            (label_x + tw + 2, label_y + baseline),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        if show_debug_face_hud and dbg_line:
            dbg_y = label_y + th + 6
            if dbg_y < h - 5:
                (dw, dh), db = cv2.getTextSize(
                    dbg_line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                )
                cv2.rectangle(
                    img,
                    (label_x - 2, dbg_y - dh - db),
                    (label_x + dw + 2, dbg_y + db),
                    (0, 0, 0),
                    thickness=-1,
                )
                cv2.putText(
                    img,
                    dbg_line,
                    (label_x, dbg_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                )

    return img
