"""
gait/gait_extractor.py - POWER NORM VERSION
"""
import numpy as np
from typing import List, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from gait_subsystem.gait.config import GaitConfig

logger = logging.getLogger(__name__)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Unlike standard Max or Average pooling, GeM uses a learnable parameter 'p'.
    - If p -> infinity, it acts like Max Pooling.
    - If p -> 1, it acts like Average Pooling.
    This allows the network to learn which temporal features are most significant
    for identification automatically.
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), (x.size(-1))).pow(1./self.p)


class BasicBlock(nn.Module):
    """
    Standard Residual Block adapted for Gait Recognition.
    Crucially, it uses Instance Normalization (InstanceNorm2d) instead of Batch Normalization.
    
    Why InstanceNorm?
    Gait videos have high variance in 'style' (clothing, camera angle, contrast).
    InstanceNorm normalizes each sample independently, removing style information
    while preserving the structural biometric content.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.bn1 = nn.InstanceNorm2d(out_c, affine=True) 
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.InstanceNorm2d(out_c, affine=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_c, affine=True)
            )

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class GaitResNet(nn.Module):
    """
    The main Backbone Network.
    Structure:
    1. Input: 4 Channels (Pos X, Pos Y, Vel X, Vel Y).
    2. ResNet Layers: Extract spatial-temporal features.
    3. GeM Pooling: Aggregates temporal frames into a single vector.
    4. FC Head: Projects features into the 256-dim embedding space.
    """
    def __init__(self, in_channels=4, embedding_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn1 = nn.InstanceNorm2d(64, affine=True)
        self.layer1 = BasicBlock(64, 64)
        self.layer2 = BasicBlock(64, 128)
        self.layer3 = BasicBlock(128, 256)
        
        self.gem = GeM()
        
        self.bn_head = nn.BatchNorm1d(256 * 17)
        self.dropout = nn.Dropout(0.5) 
        self.fc = nn.Linear(256 * 17, embedding_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        B, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C*V, T) 
        
        x = self.gem(x).view(B, -1)
        
        x = self.bn_head(x)
        feat = self.fc(x)
        return self.bn_out(feat)



class GaitExtractor:
    """
    Wrapper class responsible for:
    1. Loading the model weights.
    2. Preprocessing raw keypoints (Cleaning, Interpolation, Normalization).
    3. Performing inference with TTA (Test Time Augmentation).
    """
    def __init__(self, config: GaitConfig):
        self.config = config
        self.device = torch.device(config.device.device)
        self.target_len = 60
        self.embedding_dim = 256
        self.model = GaitResNet(in_channels=4, embedding_dim=self.embedding_dim).to(self.device)
        
        try:
            model_path = Path(config.models.gait_embedding_model_path)
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"✅ Gait Model Loaded: {model_path}")
            else:
                logger.warning(f"⚠️ Model not found: {model_path}")
            
            self.model.eval()

        except Exception as e: 
            logger.error(f"❌ Error loading model: {e}")

    def _preprocess_sequence(self, raw_sequence: List[np.ndarray]) -> torch.Tensor:
        """
        Converts a list of raw YOLO keypoints into a normalized tensor.
        
        Key Steps:
        1. Zero Cleaning: Fills missing detection gaps.
        2. Interpolation: Resamples to fixed 60 frames.
        3. Hybrid Normalization: Solves the 'Fixed Camera' vs 'Tracking Camera' domain shift.
        """
        kp_all = np.array(raw_sequence).astype(np.float32)
        
        for t in range(1, kp_all.shape[0]):
            mask_zeros = (kp_all[t, :, 0] == 0) & (kp_all[t, :, 1] == 0)
            if np.any(mask_zeros):
                kp_all[t, mask_zeros] = kp_all[t-1, mask_zeros]

        T_orig = kp_all.shape[0]
        idxs = np.linspace(0, T_orig-1, self.target_len)
        kp_interp = np.zeros((self.target_len, 17, 3), dtype=np.float32)
        for k in range(17):
            for c in range(3):
                kp_interp[:, k, c] = np.interp(idxs, np.arange(T_orig), kp_all[:, k, c])
        
        kp_final = kp_interp[:, :, :2]

        hips = (kp_final[:, 11, :2] + kp_final[:, 12, :2]) / 2.0
        ankles = (kp_final[:, 15, :2] + kp_final[:, 16, :2]) / 2.0
        
        heights = np.linalg.norm(kp_final[:, 0, :2] - ankles, axis=1)
        valid_heights = heights[heights > 10]
        if len(valid_heights) > 0:
            avg_height = np.median(valid_heights)
        else:
            avg_height = 1.0

        center_x_per_frame = hips[:, 0]
        
        center_y_global = np.mean(hips[:, 1])
        
        centers = np.stack([center_x_per_frame, np.full(self.target_len, center_y_global)], axis=1)
        
        kp_xy = (kp_final - centers[:, None, :]) / (avg_height + 1e-6)
        
        vel = np.diff(kp_xy, axis=0, prepend=kp_xy[:1])

        features = np.concatenate([kp_xy, vel], axis=2).transpose(2, 0, 1)
        return torch.from_numpy(features).float()

    def extract_gait_embedding_and_quality(self, pose_sequence: List[np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
        """
        Runs the full inference pipeline.
        
        Includes:
        - Input validation.
        - TTA (Test Time Augmentation) via Mirroring.
        - Power Normalization (Signed Square Root).
        - L2 Normalization.
        """
        if not pose_sequence or len(pose_sequence) < 20: 
            return None, 0.0
            
        try:
            tensor = self._preprocess_sequence(pose_sequence).unsqueeze(0).to(self.device)
            
            tensor_flipped = tensor.clone()
            tensor_flipped[:, 0, :, :] *= -1
            tensor_flipped[:, 2, :, :] *= -1
            
            with torch.no_grad(): 
                emb_1 = self.model(tensor)
                emb_2 = self.model(tensor_flipped)
                
                emb_avg = (emb_1 + emb_2) / 2.0
                
                emb_pow = torch.sign(emb_avg) * torch.sqrt(torch.abs(emb_avg) + 1e-12)
                
                emb_final = F.normalize(emb_pow, p=2, dim=1)
            
            return emb_final.cpu().numpy().flatten(), 1.0
            
        except Exception as e:
            logger.error(f"Gait Inference Error: {e}")
            return None, 0.0

    def extract_anthropometry(self, pose_sequence: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        [DEEP ROBUST] Computes Scale-Invariant Anthropometric Ratios.
        
        Using Bone Length Ratios (instead of raw pixels) ensures this works 
        regardless of camera distance (zoom invariance).
        
        Returns:
            np.ndarray: [leg_ratio, width_ratio]
        """
        if not pose_sequence or len(pose_sequence) < 10:
            return None
            
        kp_all = np.array(pose_sequence)
        
        
        shoulders_y = kp_all[:, [5,6], 1].mean(axis=1)
        hips_y = kp_all[:, [11,12], 1].mean(axis=1)
        torso_len = np.abs(hips_y - shoulders_y)
        
        ankles_y = kp_all[:, [15,16], 1].mean(axis=1)
        leg_len = np.abs(ankles_y - hips_y)
        
        shoulder_width = np.abs(kp_all[:, 5, 0] - kp_all[:, 6, 0])
        
        hip_width = np.abs(kp_all[:, 11, 0] - kp_all[:, 12, 0])
        
        valid_torso = torso_len > 10
        if not np.any(valid_torso): return None
        
        r_leg_torso = leg_len[valid_torso] / torso_len[valid_torso]
        
        valid_hips = hip_width > 5
        mask = valid_torso & valid_hips
        if not np.any(mask): 
             return np.array([np.median(r_leg_torso), 0.0])

        r_width = shoulder_width[mask] / hip_width[mask]
        
        median_leg = np.median(r_leg_torso)
        median_width = np.median(r_width)
        
        
        if median_leg < 0.60 or median_leg > 1.65:
            logger.warning(f"⚠️ Anomalous Body Shape (Leg Ratio {median_leg:.2f}). Learning this as Identity Feature.")
            
        return np.array([median_leg, median_width])
            
        return np.array([median_leg, median_width])