"""
tactile_modules.py — 触觉编码器和投影层，从 checkpoint 权重形状逆向重建。

无需复制训练代码。此模块自包含触觉编码器（CNN+GRU）和 MLP 投影层。
架构已从 ckpt_stage4_smolvla_final.pt 权重张量形状精确还原。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TactileGridEncoder(nn.Module):
    """
    单通道 16×16 触觉网格编码器: CNN → GRU → 线性投影。

    架构（从 checkpoint 权重还原）:
        CNN: Conv2d(1,64,3,pad=1) → BN(64) → ReLU
             → Conv2d(64,128,3,pad=1) → BN(128) → ReLU
             → Conv2d(128,128,3,pad=1) → BN(128)
        Flatten: 128×16×16 = 32768
        GRU:  input=32768, hidden=256, 1-layer
        Proj: LayerNorm(256) → Linear(256, 512)
    """

    def __init__(self, cnn_out_channels: int = 128, gru_hidden: int = 256, out_dim: int = 512) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),           # 0
            nn.BatchNorm2d(64),                         # 1
            nn.ReLU(inplace=True),                      # 2
            nn.Identity(),                              # 3 (placeholder to match index 4)
            nn.Conv2d(64, cnn_out_channels, 3, padding=1),   # 4
            nn.BatchNorm2d(cnn_out_channels),           # 5
            nn.ReLU(inplace=True),                      # 6
            nn.Conv2d(cnn_out_channels, cnn_out_channels, 3, padding=1),  # 7
            nn.BatchNorm2d(cnn_out_channels),           # 8
        )

        cnn_flat_dim = cnn_out_channels * 16 * 16  # 32768
        self.gru = nn.GRU(input_size=cnn_flat_dim, hidden_size=gru_hidden, batch_first=True)

        self.proj = nn.Sequential(
            nn.LayerNorm(gru_hidden),    # 0
            nn.Linear(gru_hidden, out_dim),  # 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W) 单通道触觉序列
        Returns:
            (B, out_dim) 聚合后的触觉特征
        """
        B, T, H, W = x.shape
        # 逐帧过 CNN
        x = x.reshape(B * T, 1, H, W)
        x = self.cnn(x)                          # (B*T, C, H, W)
        x = x.reshape(B, T, -1)                  # (B, T, C*H*W)
        # GRU 时序聚合
        _, h_n = self.gru(x)                      # h_n: (1, B, hidden)
        h = h_n.squeeze(0)                         # (B, hidden)
        # 投影
        return self.proj(h)                        # (B, out_dim)


class DualTactileGridEncoder(nn.Module):
    """
    双触觉编码器：左右各一个 TactileGridEncoder，输出拼接后取均值。

    输入: (B, T, 2, H, W) — 2 代表左右通道
    输出: (B, out_dim)
    """

    def __init__(self, out_dim: int = 512) -> None:
        super().__init__()
        self.encoder = TactileGridEncoder(out_dim=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 2, H, W) 双通道触觉序列
        Returns:
            (B, out_dim) — 左右平均
        """
        left = x[:, :, 0, :, :]   # (B, T, H, W)
        right = x[:, :, 1, :, :]  # (B, T, H, W)
        feat_l = self.encoder(left)
        feat_r = self.encoder(right)
        return (feat_l + feat_r) / 2.0


class TactileMLPProjector(nn.Module):
    """
    触觉特征 → VLM 嵌入空间投影。

    架构（从 checkpoint 权重还原）:
        MLP: Linear(512, 1024) → GELU → Linear(1024, vlm_hidden)
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 1024, out_dim: int = 7680) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim)
        Returns:
            (B, out_dim) — 可 reshape 成 (B, n_tokens, token_dim) 注入 VLM
        """
        return self.mlp(x)


def load_tactile_modules(
    ckpt_path: str,
    device: torch.device | str = "cpu",
) -> tuple[DualTactileGridEncoder, TactileMLPProjector]:
    """
    从 checkpoint 加载触觉编码器和投影层。

    checkpoint 结构:
        ckpt["tactile_encoder"] — DualTactileGridEncoder.encoder 的 state_dict
        ckpt["tactile_proj"]    — TactileMLPProjector 的 state_dict

    Returns:
        (encoder, projector) 已加载权重，eval 模式
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    encoder = DualTactileGridEncoder(out_dim=512)
    proj = TactileMLPProjector(in_dim=512, hidden_dim=1024, out_dim=7680)

    # 加载权重 — ckpt 保存的是 DualTactileGridEncoder 级别的 state_dict
    encoder.load_state_dict(ckpt["tactile_encoder"], strict=True)
    proj.load_state_dict(ckpt["tactile_proj"], strict=True)

    encoder.to(device).eval()
    proj.to(device).eval()

    return encoder, proj
