"""
models/tactile_encoder.py

双触觉 Grid CNN：复用已有 TactileEncoder，处理左右两路 (16,16) 压敏矩阵。

输入: [B, T, 2, 16, 16]  （channel 0=left, 1=right）
       或 [B, T, 1, 16, 16]（单触觉兼容模式）
输出: z_global [B, proj_dim]，可选 z_seq [B, T, proj_dim]

设计原则：
  - 不重写 CNN/GRU，直接 import TactileEncoder
  - 左右各过一个共享权重的 encoder（参数共享，减少参数量）
  - 最终 mean-pool 左右全局特征
  - 保持与原始 TactileEncoder 相同的输出接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tlv_student import TactileEncoder


class DualTactileGridEncoder(nn.Module):
    """
    双路触觉编码器（权重共享）。

    Args:
        proj_dim:  输出特征维度（默认 512，对齐 CLIP 空间）
        hid:       CNN 中间通道数
        d_model:   GRU hidden size
        tau:       对比温度初始值
        share_weights: True=左右共享同一个 TactileEncoder（推荐，参数少）
    """

    def __init__(
        self,
        proj_dim: int = 512,
        hid: int = 128,
        d_model: int = 256,
        tau: float = 0.07,
        share_weights: bool = True,
    ):
        super().__init__()
        self.share_weights = share_weights
        self.proj_dim = proj_dim

        # 主编码器（左路 / 或共享）
        self.encoder = TactileEncoder(
            proj_dim=proj_dim, hid=hid, d_model=d_model, tau=tau
        )

        if not share_weights:
            # 右路独立编码器（不推荐，参数翻倍）
            self.encoder_right = TactileEncoder(
                proj_dim=proj_dim, hid=hid, d_model=d_model, tau=tau
            )

    def forward(
        self,
        x: torch.Tensor,
        return_seq: bool = False,
        return_feat_map: bool = False,
    ):
        """
        x: [B, T, C, 16, 16]  C=2（双触觉）或 C=1（单触觉兼容）
           或 [B, T, 16, 16]（无通道维，视为单触觉）

        返回格式与 TactileEncoder 保持一致：
          - return_seq=False: (z_global, logit_scale)
          - return_seq=True:  (z_seq, z_global, logit_scale)
          - return_feat_map:  末尾追加 feat_map
        """
        # --- 维度标准化 ---
        if x.dim() == 4:
            # [B, T, 16, 16] -> [B, T, 1, 16, 16]
            x = x.unsqueeze(2)
        B, T, C, H, W = x.shape

        if C == 1:
            # 单触觉：直接走原始 encoder，squeeze 通道维
            return self.encoder(
                x.squeeze(2),  # [B, T, 16, 16]
                return_seq=return_seq,
                return_feat_map=return_feat_map,
            )

        # --- 双触觉（C=2）：分别编码左右 ---
        x_left = x[:, :, 0, :, :]    # [B, T, 16, 16]
        x_right = x[:, :, 1, :, :]   # [B, T, 16, 16]

        enc_r = self.encoder if self.share_weights else self.encoder_right

        if return_seq and return_feat_map:
            seq_l, g_l, s_l, fm_l = self.encoder(x_left, return_seq=True, return_feat_map=True)
            seq_r, g_r, s_r, fm_r = enc_r(x_right, return_seq=True, return_feat_map=True)
            z_seq = (seq_l + seq_r) / 2       # [B, T, proj_dim]
            z_global = F.normalize((g_l + g_r) / 2, dim=-1)  # [B, proj_dim]
            feat_map = (fm_l + fm_r) / 2       # [B, hid, 16, 16]
            return z_seq, z_global, s_l, feat_map

        elif return_seq:
            seq_l, g_l, s_l = self.encoder(x_left, return_seq=True)
            seq_r, g_r, s_r = enc_r(x_right, return_seq=True)
            z_seq = (seq_l + seq_r) / 2
            z_global = F.normalize((g_l + g_r) / 2, dim=-1)
            return z_seq, z_global, s_l

        elif return_feat_map:
            g_l, s_l, fm_l = self.encoder(x_left, return_feat_map=True)
            g_r, s_r, fm_r = enc_r(x_right, return_feat_map=True)
            z_global = F.normalize((g_l + g_r) / 2, dim=-1)
            feat_map = (fm_l + fm_r) / 2
            return z_global, s_l, feat_map

        else:
            g_l, s_l = self.encoder(x_left)
            g_r, s_r = enc_r(x_right)
            z_global = F.normalize((g_l + g_r) / 2, dim=-1)
            return z_global, s_l


# 向后兼容别名：overfit/models.py 里 import 这个名字
MyGridTactileEncoder = DualTactileGridEncoder
