
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any




@dataclass
class SourceAuthMotionConfig:
    """
    Configuration for the 3D-vs-planar motion cue.

    Intuition:
      Over a short time window (e.g. 1–2 seconds), a real 3D head produces
      *parallax* between landmarks (nose vs ears vs cheeks), while a phone/photo
      behaves like a flat card (all landmarks move in sync under a single 2D
      transform).

    The engine will:
      - keep a short history of landmark positions per track,
      - compare multiple reference frames (oldest + mid) to later frames,
      - fit a global 2D affine model (planar card motion),
      - measure residual motion of each landmark vs that model,
      - aggregate residuals into a motion score ∈ [0,1].

    This config controls:
      - how long the window is,
      - how much temporal span we require,
      - how many landmarks we need,
      - how big motion must be (adaptive to face size),
      - how we map parallax → [0,1] score,
      - optional boost from yaw/pitch span.
    """

    window_sec: float = 1.5

    min_span_factor: float = 0.5

    min_landmarks: int = 5

    min_motion_pixels_base: float = 2.0

    min_motion_frac: float = 0.01

    yaw_deg_threshold: float = 8.0
    yaw_boost: float = 0.15

    parallax_low: float = 0.08
    parallax_high: float = 0.22




@dataclass
class SourceAuthScreenConfig:
    """
    Configuration for the "screen / phone" cue.

    Intuition:
      A face shown on a phone or monitor tends to exhibit:
        - rectangular borders / bezels around the image,
        - high-contrast straight edges and right angles near the face,
        - subtle flicker / refresh patterns across frames,
        - moiré textures and pixel grid patterns.

      This cue focuses on visual evidence that the *object* around the face
      is a display, not a real head.

    The engine will:
      - look around the face bounding box for long straight edges and corners,
      - analyse temporal brightness variations for flicker-like behaviour,
      - inspect high-frequency textures for moiré patterns,
      - aggregate these into a screen_score ∈ [0,1].
    """

    enabled: bool = True

    border_margin_frac: float = 0.15

    min_edge_length_frac: float = 0.35

    corner_angle_tolerance_deg: float = 15.0

    border_votes_for_strong: int = 3

    flicker_window_sec: float = 1.0

    min_brightness_variance: float = 0.002

    enable_flicker_frequency_hint: bool = True

    enable_moire_hint: bool = True

    w_border: float = 0.6
    w_flicker: float = 0.2
    w_moire: float = 0.2

    evidence_low: float = 0.2
    evidence_high: float = 0.7


    min_frames: int = 4

    min_face_px: int = 40

    flicker_var_norm: float = 0.01

    moire_norm: float = 300.0

    rectangularity_angle_tol: float = 15.0

    bezel_uniformity_norm: float = 40.0

    border_reliable_min: float = 0.50
    border_reliable_std_max: float = 0.15




@dataclass
class SourceAuthBackgroundConfig:
    """
    Configuration for the background consistency cue.

    Intuition:
      For a real head:
        - the background directly behind the head (inside the bbox, non-face
          areas) belongs to the *same physical environment* as the strip
          just outside the face (walls, room, etc.).

      For a phone / screen:
        - inside the face patch you see a *different world* (other room),
        - outside you see your current room (desk, walls, etc.),
        - colour distributions, noise patterns, sharpness differ.

    The engine will:
      - define two regions:
          * inner_bg: parts of the face bbox likely to be "background",
          * outer_bg: a ring around the bbox (same scene).
      - compute simple statistics / histograms for both,
      - measure their difference,
      - map difference → background_consistency ∈ [0,1]
            1 = consistent same-world background
            0 = strong mismatch / different world.
    """

    enabled: bool = True

    inner_margin_frac: float = 0.15

    outer_margin_frac: float = 0.25

    min_region_pixels: int = 500

    min_face_px: int = 40

    color_hist_bins: int = 32

    blur_sigma_for_noise: float = 0.5

    diff_low: float = 0.15
    diff_high: float = 0.6

    enable_noise_sharpness_hint: bool = True

    w_color: float = 0.7
    w_texture: float = 0.3

    color_norm: float = 0.5
    texture_norm: float = 50.0




@dataclass
class SourceAuthFusionConfig:
    """
    Configuration for fusing motion, screen, and background cues into:

        - source_auth_score ∈ [0,1]
        - source_auth_state ∈ {REAL, LIKELY_REAL, UNCERTAIN, LIKELY_SPOOF, SPOOF}

    Core idea:
      - Each cue produces a score in [0,1] with semantics:
          * motion_score      → 1 = strong 3D, 0 = planar
          * screen_score      → 1 = screen-like, 0 = non-screen
          * background_score  → 1 = same-world, 0 = different-world
      - The engine then maps them to a *realness* score and runs a
        temporal state machine to obtain a discrete label.
    """

    w_motion: float = 0.5
    w_screen: float = 0.3
    w_background: float = 0.2

    w_bg_inverse: float = 0.2

    alpha: float = 4.0

    fused_window_sec: float = 1.0

    min_fused_samples: int = 5

    real_strong: float = 0.70
    real_weak: float = 0.50
    spoof_weak: float = 0.40
    spoof_strong: float = 0.20

    enable_spoof_persistence: bool = True
    min_spoof_persistence_sec: float = 0.5

    enable_real_persistence: bool = True
    min_real_persistence_sec: float = 0.5

    allow_uncertain_state: bool = True

    bias_towards_real_threshold: float = 0.55




@dataclass
class SourceAuthConfig:
    """
    Top-level configuration for the Source Authenticity module.

    This is the single object that SourceAuthEngine receives.

    It contains:
      - global toggles and timing parameters,
      - sub-configs for each cue (motion/screen/background),
      - fusion parameters for combining cues into final score/state.
    """

    enabled: bool = True

    min_track_age_sec_for_strong_state: float = 0.8

    max_idle_sec: float = 3.0

    use_face_route_pose: bool = True

    motion: SourceAuthMotionConfig = field(default_factory=SourceAuthMotionConfig)
    screen: SourceAuthScreenConfig = field(default_factory=SourceAuthScreenConfig)
    background: SourceAuthBackgroundConfig = field(
        default_factory=SourceAuthBackgroundConfig
    )
    fusion: SourceAuthFusionConfig = field(default_factory=SourceAuthFusionConfig)

    fusion_mode: str = "v2"

    preset_name: Optional[str] = None




def default_source_auth_config(
    face_cfg: Optional[Any] = None,
    **kwargs: Any,
) -> SourceAuthConfig:
    """
    Construct a SourceAuthConfig with tuned but conservative defaults.

    Parameters
    ----------
    face_cfg:
        Optional face configuration object (e.g. FaceConfig). If provided,
        we MAY derive a few SourceAuth thresholds from it (duck-typed via
        getattr; we never import heavy face modules here).

    **kwargs:
        Reserved for future overrides. Any extra keyword will be applied
        as a top-level attribute on the returned config, so that call sites
        can pass extra tuning knobs without breaking.

    Notes
    -----
    This helper mirrors the pattern used elsewhere in the project
    (e.g. face.config.default_face_config) and allows other modules
    to obtain a ready-to-use configuration without touching dataclass
    details.

    The defaults are intended to:
      - make real heads tend to "REAL / LIKELY_REAL",
      - make obvious phones/screens tend to "SPOOF / LIKELY_SPOOF",
      - avoid hard decisions for ambiguous cases (stay in UNCERTAIN).
    """
    cfg = SourceAuthConfig()


    cfg.neutral_score = 0.5

    cfg.window_sec = float(cfg.motion.window_sec)

    cfg.motion_window_sec = float(cfg.motion.window_sec)
    cfg.screen_window_sec = float(
        getattr(cfg.screen, "flicker_window_sec", cfg.motion.window_sec)
    )
    cfg.background_window_sec = float(cfg.motion.window_sec)

    cfg.unreliable_cue_weight_factor = 0.3

    cfg.fusion_weight_motion = float(cfg.fusion.w_motion)
    cfg.fusion_weight_screen = float(cfg.fusion.w_screen)
    cfg.fusion_weight_background = float(cfg.fusion.w_background)

    cfg.fusion_w_motion = float(cfg.fusion.w_motion)
    cfg.fusion_w_background = float(cfg.fusion.w_background)
    cfg.fusion_w_screen = float(cfg.fusion.w_screen)
    cfg.fusion_w_bg_inverse = float(
        getattr(cfg.fusion, "w_bg_inverse", cfg.fusion.w_background)
    )

    cfg.fusion_alpha = float(getattr(cfg.fusion, "alpha", 4.0))

    cfg.fusion_min_track_age_sec = float(
        getattr(cfg, "min_track_age_sec_for_strong_state", 0.8)
    )

    cfg.fusion_mode = getattr(cfg, "fusion_mode", "v2")

    cfg.ema_alpha = 0.35

    cfg.real_min_score = float(cfg.fusion.real_strong)
    cfg.likely_real_min_score = float(cfg.fusion.real_weak)
    cfg.spoof_max_score = float(cfg.fusion.spoof_strong)
    cfg.likely_spoof_max_score = float(cfg.fusion.spoof_weak)

    cfg.fusion_real_high = float(cfg.fusion.real_strong)
    cfg.fusion_real_low = float(cfg.fusion.real_weak)
    cfg.fusion_spoof_low = float(cfg.fusion.spoof_strong)
    cfg.fusion_spoof_high = float(cfg.fusion.spoof_weak)

    cfg.frames_to_confirm_real = 5
    cfg.frames_to_confirm_spoof = 5

    cfg.motion_min_span_factor = float(
        getattr(cfg.motion, "min_span_factor", 0.5)
    )
    cfg.motion_min_landmarks = int(
        getattr(cfg.motion, "min_landmarks", 3)
    )
    cfg.motion_min_motion_pixels_base = float(
        getattr(cfg.motion, "min_motion_pixels_base", 2.0)
    )
    cfg.motion_min_motion_frac = float(
        getattr(cfg.motion, "min_motion_frac", 0.01)
    )
    cfg.motion_yaw_deg_threshold = float(
        getattr(cfg.motion, "yaw_deg_threshold", 8.0)
    )
    cfg.motion_yaw_boost = float(
        getattr(cfg.motion, "yaw_boost", 0.15)
    )

    cfg.min_motion_pixels = float(cfg.motion_min_motion_pixels_base)

    cfg.parallax_low = float(
        getattr(cfg.motion, "parallax_low", 0.08)
    )
    cfg.parallax_high = float(
        getattr(cfg.motion, "parallax_high", 0.22)
    )

    if face_cfg is not None:
        q_runtime = getattr(face_cfg, "q_runtime", None)
        if q_runtime is not None:
            try:
                cfg.motion_min_quality = float(q_runtime)
            except Exception:
                cfg.motion_min_quality = 0.55
        else:
            cfg.motion_min_quality = 0.55
    else:
        cfg.motion_min_quality = 0.55

    cfg.screen_min_frames = int(getattr(cfg.screen, "min_frames", 4))
    cfg.screen_min_face_px = int(getattr(cfg.screen, "min_face_px", 40))

    cfg.screen_border_margin_frac = float(
        getattr(cfg.screen, "border_margin_frac", 0.15)
    )

    cfg.screen_weight_border = float(getattr(cfg.screen, "w_border", 0.6))
    cfg.screen_weight_flicker = float(getattr(cfg.screen, "w_flicker", 0.2))
    cfg.screen_weight_moire = float(getattr(cfg.screen, "w_moire", 0.2))

    cfg.screen_flicker_var_norm = float(
        getattr(cfg.screen, "flicker_var_norm", 0.01)
    )
    cfg.screen_moire_norm = float(
        getattr(cfg.screen, "moire_norm", 300.0)
    )

    cfg.screen_rectangularity_angle_tol = float(
        getattr(cfg.screen, "rectangularity_angle_tol", 15.0)
    )
    cfg.screen_bezel_uniformity_norm = float(
        getattr(cfg.screen, "bezel_uniformity_norm", 40.0)
    )

    cfg.screen_border_reliable_min = float(
        getattr(cfg.screen, "border_reliable_min", 0.50)
    )
    cfg.screen_border_reliable_std_max = float(
        getattr(cfg.screen, "border_reliable_std_max", 0.15)
    )

    cfg.background_min_face_px = int(
        getattr(cfg.background, "min_face_px", 40)
    )
    cfg.background_min_region_pixels = int(
        getattr(cfg.background, "min_region_pixels", 500)
    )
    cfg.background_color_hist_bins = int(
        getattr(cfg.background, "color_hist_bins", 32)
    )
    cfg.background_blur_sigma_for_noise = float(
        getattr(cfg.background, "blur_sigma_for_noise", 0.5)
    )

    cfg.background_diff_low = float(
        getattr(cfg.background, "diff_low", 0.15)
    )
    cfg.background_diff_high = float(
        getattr(cfg.background, "diff_high", 0.6)
    )

    cfg.background_enable_noise_sharpness_hint = bool(
        getattr(cfg.background, "enable_noise_sharpness_hint", True)
    )

    cfg.background_weight_color = float(
        getattr(cfg.background, "w_color", 0.7)
    )
    cfg.background_weight_texture = float(
        getattr(cfg.background, "w_texture", 0.3)
    )

    cfg.background_color_norm = float(
        getattr(cfg.background, "color_norm", 0.5)
    )
    cfg.background_texture_norm = float(
        getattr(cfg.background, "texture_norm", 50.0)
    )

    for key, value in kwargs.items():
        setattr(cfg, key, value)

    return cfg
