"""
models/smolvla_tactile.py

TactileVLAFlowMatching: 在 SmolVLA 的 VLAFlowMatching 基础上注入触觉 token。

设计思路:
  1. 继承 VLAFlowMatching，仅重写 embed_prefix()
  2. 在 language embeddings 和 state embedding 之间插入触觉 token
  3. 触觉 token 使用 att_mask=0（与图像/语言相同：双向注意力）
  4. 不改变 forward() 签名——通过 _current_tactile 实例变量传递触觉数据

触觉数据流:
  训练循环 → model._current_tactile = [B, T, 2, 16, 16]
  embed_prefix() → tactile_encoder([B,T,2,16,16]) → [B,512]
                  → tactile_proj([B,512]) → [B, n_tokens, 768]
                  → 拼入 prefix embeddings

用法:
  policy = build_tactile_smolvla(
      pretrained_path="lerobot/smolvla_base",
      n_tactile_tokens=8,
      stage3_ckpt="outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt",
  )
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    make_att_2d_masks,
    pad_tensor,
)


# ============================================================================
# TactileMLPProjector (复用 overfit/models.py 的设计)
# ============================================================================

class TactileMLPProjector(nn.Module):
    """
    [B, tactile_dim] → [B, n_tactile_tokens, hidden_size]
    """

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
# TactileVLAFlowMatching
# ============================================================================

class TactileVLAFlowMatching(VLAFlowMatching):
    """
    在 SmolVLA 的 embed_prefix() 中注入触觉 token。

    新增模块:
      - tactile_encoder: DualTactileGridEncoder [B,T,2,16,16] → [B,512]
      - tactile_proj:    TactileMLPProjector     [B,512] → [B, n_tokens, 768]

    触觉数据通过 _current_tactile 实例变量传入:
      model._current_tactile = batch["observation.tactile"]  # 训练循环中设置
    """

    def __init__(
        self,
        config: SmolVLAConfig,
        n_tactile_tokens: int = 8,
        tactile_feat_dim: int = 512,
        rtc_processor=None,
    ):
        super().__init__(config, rtc_processor=rtc_processor)

        from models.tactile_encoder import DualTactileGridEncoder

        self.n_tactile_tokens = n_tactile_tokens
        self.tactile_feat_dim = tactile_feat_dim

        # 获取 VLM 的 hidden_size（embed_prefix 输出维度）
        self.vlm_hidden_size = self.vlm_with_expert.config.text_config.hidden_size

        self.tactile_encoder = DualTactileGridEncoder(proj_dim=tactile_feat_dim)
        self.tactile_proj = TactileMLPProjector(
            tactile_dim=tactile_feat_dim,
            hidden_size=self.vlm_hidden_size,
            n_tactile_tokens=n_tactile_tokens,
        )

        # 触觉数据通过此实例变量传入（在训练循环中由外部设置）
        self._current_tactile: Optional[Tensor] = None

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: Tensor = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        重写 VLAFlowMatching.embed_prefix()，在 language tokens 和 state token 之间
        插入触觉 tokens。

        prefix 顺序:
          [image_special_tokens] [image_embs] [image_end_tokens]  × N_cameras
          [language_embs]
          [tactile_embs]   ← 新增
          [state_emb]
        """
        embs = []
        pad_masks = []
        att_masks = []

        # ---- 图像 embeddings（与父类完全一致）----
        for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * image_end_mask.shape[1]

        # ---- 语言 embeddings（与父类一致）----
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # ---- 触觉 embeddings（新增）----
        if self._current_tactile is not None:
            tactile_input = self._current_tactile
            device = lang_emb.device
            dtype = lang_emb.dtype

            # tactile_encoder: [B, T, 2, 16, 16] → (z_global, logit_scale)
            z_global, _ = self.tactile_encoder(tactile_input.to(device=device, dtype=dtype))
            # z_global: [B, 512]

            # tactile_proj: [B, 512] → [B, n_tokens, hidden_size]
            tactile_emb = self.tactile_proj(z_global)
            # [B, n_tactile_tokens, hidden_size]

            # 归一化（与图像/语言一致：乘以 √dim）
            tac_emb_dim = tactile_emb.shape[-1]
            tactile_emb = tactile_emb * math.sqrt(tac_emb_dim)

            bsize = tactile_emb.shape[0]
            tactile_mask = torch.ones(
                bsize, self.n_tactile_tokens, dtype=torch.bool, device=device
            )

            embs.append(tactile_emb)
            pad_masks.append(tactile_mask)
            # att_mask=0: 触觉与图像/语言双向注意力
            att_masks += [0] * self.n_tactile_tokens

        # ---- State embedding（与父类一致）----
        state = state.to(
            device=self.state_proj.weight.device,
            dtype=self.state_proj.weight.dtype,
        )
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # att_mask=1: 图像/语言/触觉不会 attend 到 state（但 state 可以 attend 到它们）
        att_masks += [1] * states_seq_len

        # ---- 拼接与 padding ----
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        noisy_actions = noisy_actions.to(
            device=self.action_in_proj.weight.device,
            dtype=self.action_in_proj.weight.dtype,
        )
        return super().embed_suffix(noisy_actions, timestep)

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(
            device=self.action_out_proj.weight.device,
            dtype=self.action_out_proj.weight.dtype,
        )
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t.float(), v_t.float(), reduction="none")
        return losses


# ============================================================================
# Stage 3 权重加载
# ============================================================================

def load_stage3_tactile_weights(
    tactile_encoder: nn.Module,
    ckpt_path: str,
    device: str = "cpu",
) -> None:
    """
    从 Stage 3 checkpoint 加载 DualTactileGridEncoder 权重。

    支持两种 checkpoint 格式:
      A) full_final:  扁平 state_dict {"tactile_encoder.cnn.0.weight": ...}
      B) depth_final: 嵌套 dict {"tactile_encoder": OrderedDict({...})}
    """
    if not os.path.exists(ckpt_path):
        print(f"[SmolVLA-Tactile] Stage 3 权重不存在: {ckpt_path}，触觉编码器随机初始化。")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    s3_dict = ckpt.get("model", ckpt)

    tac_weights = {}
    if "tactile_encoder" in s3_dict and isinstance(s3_dict["tactile_encoder"], dict):
        for k, v in s3_dict["tactile_encoder"].items():
            tac_weights[f"encoder.{k}"] = v
    else:
        for k, v in s3_dict.items():
            if k.startswith("tactile_encoder."):
                new_k = k.replace("tactile_encoder.", "encoder.", 1)
                tac_weights[new_k] = v

    if not tac_weights:
        print("[SmolVLA-Tactile] checkpoint 中无 tactile_encoder 权重，跳过。")
        return

    msg = tactile_encoder.load_state_dict(tac_weights, strict=False)
    print(f"[SmolVLA-Tactile] 加载 {len(tac_weights)} 个触觉编码器参数层。")
    if msg.missing_keys:
        print(f"  missing keys: {msg.missing_keys[:5]}...")
    if msg.unexpected_keys:
        print(f"  unexpected keys: {msg.unexpected_keys[:5]}...")


# ============================================================================
# 构建 SmolVLA + 触觉编码器
# ============================================================================

def build_tactile_smolvla(
    pretrained_path: str = "lerobot/smolvla_base",
    n_tactile_tokens: int = 8,
    tactile_feat_dim: int = 512,
    stage3_ckpt: Optional[str] = None,
    device: str = "cpu",
    image_keys: Optional[list[str]] = None,
) -> "SmolVLAPolicy":
    """
    构建带触觉注入的 SmolVLA 策略:
      1. 加载预训练 SmolVLAPolicy
      2. 重映射 image_features 到本数据集的相机名称
      3. 替换 policy.model 为 TactileVLAFlowMatching
      4. 迁移预训练权重（strict=False）
      5. 加载 Stage 3 触觉编码器权重

    Parameters
    ----------
    image_keys : list[str] | None
        数据集里的相机 key（如 ["observation.images.side", "observation.images.realsense_rgb"]）。
        若不传则默认 SmolVLA 原始 camera1/camera2/camera3。
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"[SmolVLA-Tactile] 加载预训练 SmolVLA: {pretrained_path}")
    policy = SmolVLAPolicy.from_pretrained(
        pretrained_path,
        local_files_only=True,
    )
    config = policy.config

    # 重映射 image_features 到我们的数据集相机名
    if image_keys is not None:
        from lerobot.policies.smolvla.configuration_smolvla import PolicyFeature, FeatureType
        # 保留非图像特征（state等），替换图像特征
        new_input_features = {}
        for k, v in config.input_features.items():
            if v.type != FeatureType.VISUAL:
                new_input_features[k] = v
        for key in image_keys:
            new_input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 512, 512),
            )
        config.input_features = new_input_features
        print(f"[SmolVLA-Tactile] image_features 重映射: {list(config.image_features.keys())}")

    # 保存原始 VLAFlowMatching 权重
    original_state = policy.model.state_dict()

    # 创建 TactileVLAFlowMatching（带触觉模块）
    print(f"[SmolVLA-Tactile] 创建 TactileVLAFlowMatching "
          f"(n_tokens={n_tactile_tokens}, feat_dim={tactile_feat_dim})")
    tactile_model = TactileVLAFlowMatching(
        config=config,
        n_tactile_tokens=n_tactile_tokens,
        tactile_feat_dim=tactile_feat_dim,
        rtc_processor=policy.rtc_processor if hasattr(policy, "rtc_processor") else None,
    )

    # 迁移预训练权重（strict=False 因为新增了 tactile_encoder 和 tactile_proj）
    msg = tactile_model.load_state_dict(original_state, strict=False)
    print(f"[SmolVLA-Tactile] 权重迁移完成。")
    if msg.missing_keys:
        print(f"  新增模块（随机初始化）: {len(msg.missing_keys)} keys")
    if msg.unexpected_keys:
        print(f"  WARNING unexpected keys: {msg.unexpected_keys[:5]}...")

    # 加载 Stage 3 触觉编码器预训练权重
    if stage3_ckpt:
        load_stage3_tactile_weights(tactile_model.tactile_encoder, stage3_ckpt, device)

    # 替换 policy 的 model
    policy.model = tactile_model

    if device != "cpu":
        policy = policy.to(device)

    print(f"[SmolVLA-Tactile] 构建完成。")
    return policy
