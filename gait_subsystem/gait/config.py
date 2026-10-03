"""
Central configuration for the Gait Recognition system.
Optimized for Angle-Invariant CNN-GRU models and OpenVINO execution.

ROBUST DESIGN:
- Uses pathlib for cross-platform path handling
- Auto-detects gait_subsystem location from __file__
- Falls back gracefully to CPU if GPU unavailable
- Resolves paths relative to gait_subsystem folder
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import torch

from core.device import select_device

logger = logging.getLogger(__name__)

GAIT_SUBSYSTEM_DIR = Path(__file__).parent.parent.resolve()


@dataclass
class GaitDeviceConfig:
    """Hardware acceleration settings."""
    device: str = "cpu" 
    use_half: bool = False 


@dataclass
class GaitModelConfig:
    """Paths for pose estimation and gait embedding models."""
    pose_model_name: str = "yolov8n-pose_openvino_model/"
    gait_embedding_model_path: str = "models/gait_temporal_encoder.pth"


@dataclass
class GaitThresholdConfig:
    """
    Numeric thresholds for filtering and identity matching.
    
    DEEP ROBUST DESIGN:
    - max_match_distance: Cosine distance threshold for STRONG match (lower = stricter)
    - max_weak_match_distance: Threshold for WEAK match
    - min_match_margin: Minimum gap between top-2 candidates to avoid ambiguity
    """
    min_visibility: float = 0.4
    min_valid_joints: int = 10
    max_match_distance: float = 0.30
    max_weak_match_distance: float = 0.40
    min_gait_quality: float = 0.5
    min_match_margin: float = 0.10


@dataclass
class GaitRouteConfig:
    """
    Operational parameters for the real-time processing pipeline.
    
    DEEP ROBUST DESIGN:
    - process_every_n_frames: Skip frames to reduce GPU load
    - min_sequence_length: Minimum frames for reliable gait analysis
    - keypoint_ema_alpha: Smoothing factor to reduce jitter
    """
    process_every_n_frames: int = 3
    max_seconds_lookback: float = 3.0
    max_entries_per_track: int = 60 
    min_sequence_length: int = 30
    keypoint_ema_alpha: float = 0.65
    keypoint_history_length: int = 45 
    img_size: int = 640


@dataclass
class GaitGalleryConfig:
    """
    Settings for the identity database and vector search.
    
    DEEP ROBUST DESIGN:
    - dim: 256-dim embeddings (compact yet discriminative)
    - metric: Cosine similarity (normalized for fair comparison)
    - ema_alpha: Template update rate (0.5 = equal weight old/new)
    """
    dim: int = 256
    metric: str = "cosine"
    gallery_path: Path = Path("data/gait_gallery.pkl")
    encryption_key_env: str = "GAITGUARD_GAIT_KEY"
    ema_alpha: float = 0.5


@dataclass
class GaitRobustConfig:
    """
    Gold Spec Robustness Parameters.
    Controls the state machine, scheduling, and decision policy.
    """
    min_seq_len: int = 30
    eval_period: float = 0.7
    
    quality_min: float = 0.55
    quality_confirm: float = 0.65
    
    threshold_candidate: float = 0.60
    threshold_confirm: float = 0.70
    margin_confirm: float = 0.05
    confirm_streak: int = 2
    
    min_motion: float = 0.05

    anthro_threshold: float = 0.15
    anthro_penalty_weight: float = 0.5



@dataclass
class GaitConfig:
    """Aggregate configuration object."""
    device: GaitDeviceConfig
    models: GaitModelConfig
    thresholds: GaitThresholdConfig
    route: GaitRouteConfig
    gallery: GaitGalleryConfig
    robust: GaitRobustConfig = field(default_factory=GaitRobustConfig)


def default_gait_config(
    prefer_gpu: bool = True,
    base_dir: Optional[Path] = None,
) -> GaitConfig:
    """
    Factory function to initialize the configuration with optimized defaults.
    
    DEEP ROBUST DESIGN:
    1. Auto-detects gait_subsystem directory from __file__
    2. Resolves all model paths relative to gait_subsystem
    3. Falls back gracefully if OpenVINO model not found
    4. Uses main system's device selection for consistency
    """
    device_str, use_half = select_device(prefer_gpu=prefer_gpu)
    device_cfg = GaitDeviceConfig(device=device_str, use_half=use_half)

    base_dir = base_dir or GAIT_SUBSYSTEM_DIR
    
    gait_model_path = (base_dir / "models" / "gait_temporal_encoder.pth").resolve()
    openvino_path = (base_dir / "yolov8n-pose_openvino_model").resolve()
    
    main_data_dir = (base_dir.parent / "data").resolve()
    
    if not gait_model_path.exists():
        logger.warning(f"⚠️ Gait model not found: {gait_model_path}")
    else:
        logger.info(f"✅ Gait model found: {gait_model_path}")
    
    if openvino_path.exists():
        pose_model = str(openvino_path)
        logger.info(f"✅ Using OpenVINO pose model: {openvino_path}")
    else:
        pose_model = "yolov8n-pose.pt"
        logger.warning(f"⚠️ OpenVINO not found at {openvino_path}. Falling back to PyTorch.")

    return GaitConfig(
        device=device_cfg,
        models=GaitModelConfig(
            pose_model_name=pose_model, 
            gait_embedding_model_path=str(gait_model_path)
        ),
        thresholds=GaitThresholdConfig(
            max_match_distance=0.30,
            max_weak_match_distance=0.40,
            min_match_margin=0.10
        ),
        route=GaitRouteConfig(
            min_sequence_length=30, 
            keypoint_history_length=45
        ),
        gallery=GaitGalleryConfig(
            dim=256, 
            gallery_path=(main_data_dir / "gait_gallery.pkl")
        )
    )