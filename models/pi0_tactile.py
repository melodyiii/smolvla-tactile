"""
models/pi0_tactile.py

Tactile-augmented Pi0 (Flow Matching VLA) and Pi0-FAST (Autoregressive VLA).

Strategy (same pattern as SmolVLA):
  - Override embed_prefix() to inject tactile tokens between language and state
  - Tactile tokens use att_mask=0 (bidirectional, same as image/language)
  - Tactile data flows via _current_tactile instance variable

Architecture:
  Pi0 (PaliGemma 2B VLM + Gemma 300M Action Expert):
    ├─ SigLIP 视觉编码器 (frozen)
    ├─ Gemma 2B 语言模型 (frozen/LoRA)
    ├─ Gemma 300M Action Expert (trainable)
    ├─ DualTactileGridEncoder (from Stage 3, frozen/finetune)
    ├─ TactileMLPProjector: 512 → n_tokens × paligemma_hidden
    ├─ state_proj (trainable)
    └─ action_in/out_proj + action_time_mlp (trainable)

  Pi0-FAST (PaliGemma 2B VLM, autoregressive):
    ├─ SigLIP + Gemma 2B (same as Pi0, no separate expert)
    ├─ DualTactileGridEncoder (from Stage 3)
    ├─ TactileMLPProjector: 512 → n_tokens × gemma_hidden
    └─ FAST action tokenizer (discrete tokens)

Usage:
  # Pi0 + Tactile
  from models.pi0_tactile import build_tactile_pi0
  policy = build_tactile_pi0(
      pretrained_path="lerobot/pi0_base",
      stage3_ckpt="outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt",
  )

  # Pi0-FAST + Tactile
  from models.pi0_tactile import build_tactile_pi0fast
  policy = build_tactile_pi0fast(
      pretrained_path="lerobot/pi0fast_base",
      stage3_ckpt="outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt",
  )
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Shared: TactileMLPProjector (same architecture as SmolVLA version)
# ============================================================================

class TactileMLPProjector(nn.Module):
    """[B, tactile_dim] → [B, n_tokens, hidden_size]"""

    def __init__(self, tactile_dim: int, hidden_size: int, n_tactile_tokens: int = 8):
        super().__init__()
        self.n_tactile_tokens = n_tactile_tokens
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(tactile_dim, tactile_dim * 2),
            nn.GELU(),
            nn.Linear(tactile_dim * 2, n_tactile_tokens * hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        y = self.mlp(x)
        return y.view(b, self.n_tactile_tokens, self.hidden_size)


# ============================================================================
# TactilePi0FlowMatching — Pi0 with tactile injection
# ============================================================================

class TactilePi0FlowMatching(nn.Module):
    """
    Wraps PI0Pytorch to inject tactile tokens into embed_prefix().

    Pi0 uses PaliGemma (SigLIP + Gemma 2B) for prefix processing and
    a separate Gemma 300M expert for action generation.

    Tactile tokens are injected into the prefix (PaliGemma stream)
    between language and state embeddings, using bidirectional attention.

    New modules:
      - tactile_encoder: DualTactileGridEncoder [B,T,2,16,16] → [B,512]
      - tactile_proj: TactileMLPProjector [B,512] → [B, n_tokens, paligemma_hidden]
    """

    def __init__(
        self,
        pi0_model,  # PI0Pytorch instance
        n_tactile_tokens: int = 8,
        tactile_feat_dim: int = 512,
    ):
        super().__init__()
        self.pi0 = pi0_model
        self.config = pi0_model.config
        self.n_tactile_tokens = n_tactile_tokens
        self.tactile_feat_dim = tactile_feat_dim

        from models.tactile_encoder import DualTactileGridEncoder

        # PaliGemma hidden size (Gemma 2B = 2048)
        paligemma_config = pi0_model.paligemma_with_expert.paligemma.config
        self.vlm_hidden_size = paligemma_config.text_config.hidden_size

        self.tactile_encoder = DualTactileGridEncoder(proj_dim=tactile_feat_dim)
        self.tactile_proj = TactileMLPProjector(
            tactile_dim=tactile_feat_dim,
            hidden_size=self.vlm_hidden_size,
            n_tactile_tokens=n_tactile_tokens,
        )

        self._current_tactile: Optional[Tensor] = None

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Override Pi0's embed_prefix to inject tactile tokens.

        Prefix token order:
          [image_tokens] × N_cameras    (att_mask=0, bidirectional)
          [language_tokens]              (att_mask=0, bidirectional)
          [tactile_tokens]  ← NEW       (att_mask=0, bidirectional)
        """
        # Get base prefix from Pi0
        embs, pad_masks, att_masks = self.pi0.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # Inject tactile tokens
        if self._current_tactile is not None:
            tactile_input = self._current_tactile
            device = embs.device
            dtype = embs.dtype
            bsize = embs.shape[0]

            z_global, _ = self.tactile_encoder(
                tactile_input.to(device=device, dtype=dtype)
            )
            tactile_emb = self.tactile_proj(z_global)

            # Normalize like Pi0 does for other embeddings
            tac_dim = tactile_emb.shape[-1]
            tactile_emb = tactile_emb * math.sqrt(tac_dim)

            tactile_mask = torch.ones(
                bsize, self.n_tactile_tokens, dtype=torch.bool, device=device
            )

            # Concatenate tactile tokens to prefix
            embs = torch.cat([embs, tactile_emb.to(dtype=dtype)], dim=1)
            pad_masks = torch.cat([pad_masks, tactile_mask], dim=1)

            # Expand att_masks: tactile tokens are bidirectional (0)
            tactile_att = torch.zeros(
                bsize, self.n_tactile_tokens, dtype=att_masks.dtype, device=device
            )
            att_masks = torch.cat([att_masks, tactile_att], dim=1)

        return embs, pad_masks, att_masks

    # Delegate everything else to the wrapped pi0 model
    def embed_suffix(self, state, noisy_actions, timestep):
        return self.pi0.embed_suffix(state, noisy_actions, timestep)

    def sample_noise(self, shape, device):
        return self.pi0.sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return self.pi0.sample_time(bsize, device)

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions,
        noise=None, time=None
    ) -> Tensor:
        """Flow matching forward with tactile prefix injection."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Use our overridden embed_prefix (with tactile)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time
        )

        # Ensure matching dtypes
        if prefix_embs.dtype != suffix_embs.dtype:
            prefix_embs = prefix_embs.to(dtype=suffix_embs.dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self.pi0._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = self.pi0.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.pi0.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       noise=None, num_steps=None):
        """Denoising inference with tactile-augmented prefix."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device

        # Precompute prefix (with tactile) and cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        noise_out = self.sample_noise(
            (bsize, self.config.chunk_size, self.config.max_action_dim), device
        )
        if noise is not None:
            noise_out = noise

        # Iterative denoising
        dt = 1.0 / num_steps
        for i in range(num_steps):
            time_val = torch.full((bsize,), 1.0 - i * dt, device=device)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
                self.embed_suffix(state, noise_out, time_val)
            )

            if prefix_embs.dtype != suffix_embs.dtype:
                prefix_embs = prefix_embs.to(dtype=suffix_embs.dtype)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self.pi0._prepare_attention_masks_4d(att_2d_masks)

            (_, suffix_out), _ = self.pi0.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.pi0.action_out_proj(suffix_out)

            noise_out = noise_out - dt * v_t

        return noise_out

    # Properties to pass through
    @property
    def paligemma_with_expert(self):
        return self.pi0.paligemma_with_expert

    @property
    def state_proj(self):
        return self.pi0.state_proj

    @property
    def action_in_proj(self):
        return self.pi0.action_in_proj

    @property
    def action_out_proj(self):
        return self.pi0.action_out_proj


# ============================================================================
# TactilePi0Fast — Pi0-FAST with tactile injection
# ============================================================================

class TactilePi0Fast(nn.Module):
    """
    Wraps PI0FastPytorch to inject tactile tokens into embed_prefix_fast().

    Pi0-FAST uses PaliGemma only (no separate expert) and predicts actions
    as discrete tokens via FAST tokenizer + autoregressive decoding.

    Tactile tokens are injected between language tokens and FAST action tokens,
    using bidirectional attention (same as image/language).
    """

    def __init__(
        self,
        pi0fast_model,  # PI0FastPytorch instance
        n_tactile_tokens: int = 8,
        tactile_feat_dim: int = 512,
    ):
        super().__init__()
        self.pi0fast = pi0fast_model
        self.config = pi0fast_model.config
        self.n_tactile_tokens = n_tactile_tokens
        self.tactile_feat_dim = tactile_feat_dim

        from models.tactile_encoder import DualTactileGridEncoder

        paligemma_config = pi0fast_model.paligemma_with_expert.paligemma.config
        self.vlm_hidden_size = paligemma_config.text_config.hidden_size

        self.tactile_encoder = DualTactileGridEncoder(proj_dim=tactile_feat_dim)
        self.tactile_proj = TactileMLPProjector(
            tactile_dim=tactile_feat_dim,
            hidden_size=self.vlm_hidden_size,
            n_tactile_tokens=n_tactile_tokens,
        )

        self._current_tactile: Optional[Tensor] = None

    def embed_prefix_fast(
        self, images, img_masks, tokens, masks,
        fast_action_tokens=None, fast_action_masks=None,
    ):
        """
        Override Pi0Fast's embed_prefix_fast to inject tactile tokens.

        Token order:
          [image_tokens]      (bidirectional)
          [language_tokens]   (bidirectional)
          [tactile_tokens]    ← NEW (bidirectional)
          [FAST action_tokens] (causal among themselves, attend to all above)
        """
        embs = []
        pad_masks = []
        att_mask_segments = []
        total_t_images = 0
        num_fast_embs = 0

        bsize = images[0].shape[0]
        device = images[0].device

        # ---- Image embeddings (same as parent) ----
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.pi0fast.paligemma_with_expert.embed_image(img)
            b, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(b, num_img_embs))
            att_mask_segments.append(("image", num_img_embs))
            total_t_images += num_img_embs

        # ---- Language embeddings (same as parent) ----
        lang_emb = self.pi0fast.paligemma_with_expert.embed_language_tokens(tokens)
        lang_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_dim)
        embs.append(lang_emb)
        pad_masks.append(masks)
        num_lang_embs = lang_emb.shape[1]
        att_mask_segments.append(("language", num_lang_embs))

        # ---- Tactile embeddings (NEW) ----
        if self._current_tactile is not None:
            tactile_input = self._current_tactile
            dtype = lang_emb.dtype

            z_global, _ = self.tactile_encoder(
                tactile_input.to(device=device, dtype=dtype)
            )
            tactile_emb = self.tactile_proj(z_global)
            tac_dim = tactile_emb.shape[-1]
            tactile_emb = tactile_emb * math.sqrt(tac_dim)

            embs.append(tactile_emb.to(dtype=dtype))
            tactile_mask = torch.ones(
                bsize, self.n_tactile_tokens, dtype=torch.bool, device=device
            )
            pad_masks.append(tactile_mask)
            # Bidirectional — treated like image/language
            att_mask_segments.append(("language", self.n_tactile_tokens))

        # ---- FAST action tokens (same as parent) ----
        if fast_action_tokens is not None:
            fast_emb = self.pi0fast.paligemma_with_expert.embed_language_tokens(
                fast_action_tokens
            )
            fast_dim = fast_emb.shape[-1]
            fast_emb = fast_emb * math.sqrt(fast_dim)
            embs.append(fast_emb)
            num_fast_embs = fast_action_tokens.shape[1]
            pad_masks.append(fast_action_masks)
            att_mask_segments.append(("fast", num_fast_embs))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        att_masks = self.pi0fast._create_custom_attention_mask_fast(
            att_mask_segments, pad_masks, bsize
        )

        return embs, pad_masks, att_masks, total_t_images, num_fast_embs

    def forward(
        self, images, img_masks, tokens, masks,
        fast_action_tokens, fast_action_masks,
    ) -> dict:
        """Forward pass with tactile-augmented prefix."""
        if fast_action_tokens is None or fast_action_masks is None:
            raise ValueError("fast_action_tokens and fast_action_masks required")

        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, num_fast_embs = (
            self.embed_prefix_fast(
                images, img_masks, tokens, masks,
                fast_action_tokens=fast_action_tokens,
                fast_action_masks=fast_action_masks,
            )
        )

        # Match dtype
        q_proj_weight = (
            self.pi0fast.paligemma_with_expert.paligemma
            .language_model.layers[0].self_attn.q_proj.weight
        )
        if q_proj_weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_2d_4d = self.pi0fast._prepare_attention_masks_4d(
            prefix_att_masks, dtype=prefix_embs.dtype
        )

        (prefix_out, _), _ = self.pi0fast.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )

        # Extract FAST logits (last num_fast_embs tokens)
        fast_logits = prefix_out[:, -num_fast_embs:]
        lm_head = self.pi0fast.paligemma_with_expert.paligemma.language_model.lm_head
        fast_logits = lm_head(fast_logits)

        # Next-token prediction: shift by 1
        pred_logits = fast_logits[:, :-1]  # [B, T-1, vocab]
        targets = fast_action_tokens[:, 1:]  # [B, T-1]
        target_masks = fast_action_masks[:, 1:]  # [B, T-1]

        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        loss = loss.view(targets.shape)
        loss = (loss * target_masks).sum() / target_masks.sum().clamp(min=1)

        return {"loss": loss, "ce_loss": loss}

    @property
    def paligemma_with_expert(self):
        return self.pi0fast.paligemma_with_expert


# ============================================================================
# Build functions
# ============================================================================

def build_tactile_pi0(
    pretrained_path: str = "lerobot/pi0_base",
    n_tactile_tokens: int = 8,
    tactile_feat_dim: int = 512,
    stage3_ckpt: Optional[str] = None,
    freeze_tactile: bool = True,
    device: str = "cpu",
    image_keys: Optional[list[str]] = None,
) -> "PI0Policy":
    """
    Build Pi0 policy with tactile token injection.

    1. Load pretrained PI0Policy
    2. Wrap policy.model with TactilePi0FlowMatching
    3. Load Stage 3 tactile encoder weights
    4. Remap image keys to dataset camera names
    """
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    print(f"[Pi0-Tactile] Loading pretrained Pi0: {pretrained_path}")
    policy = PI0Policy.from_pretrained(pretrained_path)

    # Wrap the inner model
    tactile_model = TactilePi0FlowMatching(
        pi0_model=policy.model,
        n_tactile_tokens=n_tactile_tokens,
        tactile_feat_dim=tactile_feat_dim,
    )

    # Load Stage 3 tactile weights
    if stage3_ckpt:
        from models.smolvla_tactile import load_stage3_tactile_weights
        load_stage3_tactile_weights(
            tactile_model.tactile_encoder, stage3_ckpt, device=device
        )

    if freeze_tactile:
        for p in tactile_model.tactile_encoder.parameters():
            p.requires_grad_(False)

    # Replace model with tactile-augmented version
    policy.model = tactile_model

    # Remap image keys if needed
    if image_keys:
        config = policy.config
        original_keys = list(config.image_features.keys())
        if len(original_keys) >= len(image_keys):
            new_features = {}
            for old_key, new_key in zip(original_keys, image_keys):
                new_features[new_key] = config.image_features[old_key]
            config.image_features = new_features

    policy.to(device)
    return policy


def build_tactile_pi0fast(
    pretrained_path: str = "lerobot/pi0fast_base",
    n_tactile_tokens: int = 8,
    tactile_feat_dim: int = 512,
    stage3_ckpt: Optional[str] = None,
    freeze_tactile: bool = True,
    device: str = "cpu",
    image_keys: Optional[list[str]] = None,
) -> "PI0FastPolicy":
    """
    Build Pi0-FAST policy with tactile token injection.

    1. Load pretrained PI0FastPolicy
    2. Wrap policy.model with TactilePi0Fast
    3. Load Stage 3 tactile encoder weights
    """
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

    print(f"[Pi0Fast-Tactile] Loading pretrained Pi0-FAST: {pretrained_path}")
    policy = PI0FastPolicy.from_pretrained(pretrained_path)

    tactile_model = TactilePi0Fast(
        pi0fast_model=policy.model,
        n_tactile_tokens=n_tactile_tokens,
        tactile_feat_dim=tactile_feat_dim,
    )

    if stage3_ckpt:
        from models.smolvla_tactile import load_stage3_tactile_weights
        load_stage3_tactile_weights(
            tactile_model.tactile_encoder, stage3_ckpt, device=device
        )

    if freeze_tactile:
        for p in tactile_model.tactile_encoder.parameters():
            p.requires_grad_(False)

    policy.model = tactile_model

    if image_keys:
        config = policy.config
        original_keys = list(config.image_features.keys())
        if len(original_keys) >= len(image_keys):
            new_features = {}
            for old_key, new_key in zip(original_keys, image_keys):
                new_features[new_key] = config.image_features[old_key]
            config.image_features = new_features

    policy.to(device)
    return policy
