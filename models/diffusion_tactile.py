"""
models/diffusion_tactile.py

Tactile-augmented Diffusion Policy.

Strategy:
  Diffusion Policy uses a flat global_cond vector (state + image features) to
  condition a 1D UNet via FiLM. We inject tactile features by concatenating
  the 512-dim tactile embedding to the global_cond vector.

  global_cond = [state | img_features | tactile_features]

Architecture:
  DiffusionPolicy (pretrained)
    └─ DiffusionModel
         ├─ rgb_encoder (ResNet18 + SpatialSoftmax, frozen)
         ├─ unet (1D UNet, trainable)
         └─ noise_scheduler (DDPM/DDIM)
  + DualTactileGridEncoder (from Stage 3, frozen/finetune)
  + tactile_proj: Linear(512 → tactile_cond_dim)

Training:
  python train/5_diffusion_policy.py \\
    --data_path ./data/inboxpicking-01 \\
    --stage3_ckpt outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt \\
    --epochs 40 --batch_size 8
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionModel,
    DiffusionPolicy,
)
from lerobot.utils.constants import OBS_STATE, OBS_IMAGES, OBS_ENV_STATE


# ============================================================================
# TactileDiffusionModel
# ============================================================================

class TactileDiffusionModel(DiffusionModel):
    """
    Extends DiffusionModel by concatenating tactile features to global_cond.

    The UNet must be re-initialized with the expanded global_cond_dim.
    To avoid that, we use a projection layer to map tactile features to a
    fixed dimension that we add post-hoc via a learnable FiLM-style modulation,
    or more simply, we rebuild the UNet with the correct dimension.

    Simpler approach: project tactile to match existing cond dim, then add.
    Even simpler: rebuild UNet with expanded cond dim.

    We use the rebuild approach for correctness.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        tactile_feat_dim: int = 512,
        n_obs_steps: int = 1,
    ):
        # Temporarily init parent (creates unet with original cond_dim)
        super().__init__(config)

        from models.tactile_encoder import DualTactileGridEncoder
        from lerobot.policies.diffusion.modeling_diffusion import (
            DiffusionConditionalUnet1d,
        )

        self.tactile_feat_dim = tactile_feat_dim

        # Tactile encoder
        self.tactile_encoder = DualTactileGridEncoder(proj_dim=tactile_feat_dim)

        # Tactile projection: 512 → small cond dim
        self.tactile_cond_dim = 128
        self.tactile_proj = nn.Sequential(
            nn.Linear(tactile_feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.tactile_cond_dim),
        )

        # Recompute global_cond_dim with tactile
        base_global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if hasattr(self, "rgb_encoder"):
                if isinstance(self.rgb_encoder, nn.ModuleList):
                    rgb_feat_dim = self.rgb_encoder[0].feature_dim * num_images
                else:
                    rgb_feat_dim = self.rgb_encoder.feature_dim * num_images
            else:
                rgb_feat_dim = 0
            base_global_cond_dim += rgb_feat_dim
        if config.env_state_feature:
            base_global_cond_dim += config.env_state_feature.shape[0]

        new_global_cond_dim = (base_global_cond_dim + self.tactile_cond_dim) * config.n_obs_steps

        # Rebuild UNet with expanded conditioning dimension
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=new_global_cond_dim)

        # Placeholder for tactile data injection
        self._current_tactile: Optional[Tensor] = None

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Override to include tactile features in global conditioning."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        # Image features (same as parent)
        if self.config.image_features:
            import einops
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat([
                    encoder(images)
                    for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ])
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Tactile features
        if self._current_tactile is not None:
            tactile_input = self._current_tactile
            device = batch[OBS_STATE].device
            dtype = batch[OBS_STATE].dtype

            z_global, _ = self.tactile_encoder(tactile_input.to(device=device, dtype=dtype))
            tactile_cond = self.tactile_proj(z_global)  # [B, tactile_cond_dim]

            # Expand to match n_obs_steps: [B, n_obs_steps, tactile_cond_dim]
            tactile_cond = tactile_cond.unsqueeze(1).expand(-1, n_obs_steps, -1)
            global_cond_feats.append(tactile_cond)

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)


# ============================================================================
# Build function
# ============================================================================

def build_tactile_diffusion(
    config: DiffusionConfig,
    stage3_ckpt: Optional[str] = None,
    freeze_tactile: bool = True,
    device: str = "cpu",
) -> DiffusionPolicy:
    """
    Build a Diffusion Policy with tactile conditioning.

    1. Create DiffusionPolicy with custom TactileDiffusionModel
    2. Load Stage 3 tactile encoder weights
    3. Freeze/unfreeze tactile encoder
    """
    # Create policy
    policy = DiffusionPolicy(config)

    # Replace diffusion model with tactile-augmented version
    tactile_model = TactileDiffusionModel(
        config,
        tactile_feat_dim=512,
    )

    # Transfer RGB encoder weights from original
    if hasattr(policy.diffusion, "rgb_encoder") and hasattr(tactile_model, "rgb_encoder"):
        tactile_model.rgb_encoder.load_state_dict(
            policy.diffusion.rgb_encoder.state_dict()
        )

    # Transfer noise scheduler
    tactile_model.noise_scheduler = policy.diffusion.noise_scheduler

    policy.diffusion = tactile_model

    # Load Stage 3 tactile weights
    if stage3_ckpt:
        from models.smolvla_tactile import load_stage3_tactile_weights
        load_stage3_tactile_weights(
            tactile_model.tactile_encoder, stage3_ckpt, device=device
        )

    # Freeze tactile encoder
    if freeze_tactile:
        for p in tactile_model.tactile_encoder.parameters():
            p.requires_grad_(False)

    policy.to(device)
    return policy
