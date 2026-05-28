"""
models/dense_contrastive.py - 密集/局部对比学习 + 重构分支

1. PatchContrastiveHead
   触觉局部 Patch <-> Wrist RGB 局部 Patch 对比学习
   正例：同一样本、同一空间位置的 (tac_patch, rgb_patch)
   负例：batch 内其他样本的同位置 patch

2. TactileReconHead
   重构分支：从 z_t[B,512] 重构触觉图的时间均值帧
   损失 = MSE + lambda_grad * Sobel梯度误差
   迫使编码器保留高频物理纹理信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Patch 投影头（共用工具模块）
# ============================================================================

class PatchProjector(nn.Module):
    """
    将 CNN 特征图每个空间位置投影到对比学习空间。
    输入：[B, C, H, W]
    输出：[B, H*W, proj_dim]  L2 归一化
    """
    def __init__(self, in_channels: int, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, kernel_size=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(),
            nn.Conv2d(proj_dim, proj_dim, kernel_size=1),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_map.shape
        proj = self.proj(feat_map)                      # [B, proj_dim, H, W]
        proj = proj.flatten(2).transpose(1, 2)          # [B, H*W, proj_dim]
        return F.normalize(proj, dim=-1)


# ============================================================================
# 2. Dense Patch Contrastive Head
# ============================================================================

class PatchContrastiveHead(nn.Module):
    """
    触觉 Patch <-> RGB Patch 密集对比学习。

    设计原则：
    - 触觉图 16x16，经 CNN 后特征图仍为 16x16
    - RGB 图 480x640，经 ResNet18 layer1 后约 120x160，需 AdaptivePool 对齐
    - 同 batch 同位置的 (tac_patch, rgb_patch) 为正例
    - batch 内其他样本的对应位置 patch 为负例

    Args:
        tac_channels: 触觉 CNN 输出通道数（TactileEncoder.cnn 最后层输出 hid=128）
        rgb_channels: RGB CNN 中间层通道数（ResNet18 layer1 输出 64）
        proj_dim:     投影维度
        temperature:  对比温度（patch级别用较小值 0.1）
        n_patches:    随机采样的 patch 数（减少计算量，16 足够）
    """
    def __init__(
        self,
        tac_channels: int = 128,
        rgb_channels: int = 64,
        proj_dim: int = 128,
        temperature: float = 0.1,
        n_patches: int = 16,
    ):
        super().__init__()
        self.tac_proj    = PatchProjector(tac_channels, proj_dim)
        self.rgb_proj    = PatchProjector(rgb_channels, proj_dim)
        self.temperature = temperature
        self.n_patches   = n_patches

    def forward(
        self,
        tac_feat_map: torch.Tensor,   # [B, C_t, H_t, W_t]  触觉 CNN 特征图
        rgb_feat_map: torch.Tensor,   # [B, C_v, H_v, W_v]  RGB CNN 特征图
    ) -> torch.Tensor:
        """
        returns: patch-level InfoNCE loss（标量）
        """
        _, _, H_t, W_t = tac_feat_map.shape

        # 将 RGB 特征图空间分辨率对齐到触觉特征图
        rgb_aligned = F.adaptive_avg_pool2d(rgb_feat_map, (H_t, W_t))

        z_tac = self.tac_proj(tac_feat_map)   # [B, H_t*W_t, D]
        z_rgb = self.rgb_proj(rgb_aligned)    # [B, H_t*W_t, D]

        N = z_tac.shape[1]
        if self.n_patches < N:
            idx = torch.randperm(N, device=z_tac.device)[:self.n_patches]
            z_tac = z_tac[:, idx, :]
            z_rgb = z_rgb[:, idx, :]

        return self._patch_infonce(z_tac, z_rgb)

    def _patch_infonce(
        self, z_t: torch.Tensor, z_v: torch.Tensor
    ) -> torch.Tensor:
        """
        z_t, z_v: [B, N, D]
        对每个 patch 位置 n，在 batch 维度上计算 InfoNCE。
        """
        B, N, D = z_t.shape
        z_t = z_t.permute(1, 0, 2)  # [N, B, D]
        z_v = z_v.permute(1, 0, 2)  # [N, B, D]

        # [N, B, B]  logits[n, i, j] = sim(tac[n,i], rgb[n,j])
        logits = torch.bmm(z_t, z_v.transpose(1, 2)) / self.temperature
        labels = torch.arange(B, device=z_t.device).unsqueeze(0).expand(N, -1)  # [N, B]

        loss_t2v = F.cross_entropy(
            logits.reshape(N * B, B), labels.reshape(N * B)
        )
        loss_v2t = F.cross_entropy(
            logits.transpose(1, 2).contiguous().reshape(N * B, B),
            labels.reshape(N * B),
        )
        return (loss_t2v + loss_v2t) / 2


# ============================================================================
# 3. Tactile Reconstruction Head
# ============================================================================

class TactileReconHead(nn.Module):
    """
    触觉纹理重构分支。

    从全局触觉特征 z_t [B, 512] 重构触觉图的时间均值帧 [B, 1, 16, 16]。
    重构损失 = MSE（像素级）+ lambda_grad * Sobel梯度误差（高频纹理）

    为什么有效：
    - 纯对比学习可能忽略绝对的物理纹理细节（只学排序关系）
    - 重构损失强制 z_t 保留可以重建原始信号的信息量
    - Sobel 梯度损失专门惩罚高频纹理（凸起、粗糙度）的丢失

    Args:
        z_dim:       输入特征维度（默认 512）
        lambda_grad: 梯度损失权重（推荐 0.1~0.5）
    """
    def __init__(self, z_dim: int = 512, lambda_grad: float = 0.1):
        super().__init__()
        self.lambda_grad = lambda_grad

        # 解码器：512 -> Linear -> reshape [B,256,4,4] -> TransposeConv -> [B,1,16,16]
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256 * 4 * 4),
            nn.ReLU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),  # 8->16
            nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),   # 输出 [0,1]，与归一化触觉图一致
        )

        # Sobel 滤波器（固定，不参与训练），用于提取高频梯度信息
        sobel_x = torch.tensor(
            [[-1.,  0.,  1.],
             [-2.,  0.,  2.],
             [-1.,  0.,  1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient_map(self, x: torch.Tensor) -> torch.Tensor:
        """计算 Sobel 梯度幅度图。x: [B, 1, 16, 16]"""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(
        self,
        z_t:   torch.Tensor,   # [B, 512]
        x_tac: torch.Tensor,   # [B, T, 16, 16]
    ):
        """
        returns:
          loss:    标量重构总损失
          x_recon: [B, 1, 16, 16]  重构触觉图（可视化用）
        """
        B = z_t.shape[0]

        # 目标：触觉序列的时间均值帧（代表平均纹理状态）
        x_target = x_tac.mean(dim=1, keepdim=True)  # [B, 1, 16, 16]
        # clamp 到 [0,1] 保证与 Sigmoid 输出一致
        x_target = x_target.clamp(0., 1.)

        # 解码
        feat = self.fc(z_t)                      # [B, 256*4*4]
        feat = feat.view(B, 256, 4, 4)           # [B, 256, 4, 4]
        x_recon = self.upsample(feat)            # [B, 1, 16, 16]

        # 损失 1：像素级 MSE
        l_mse = F.mse_loss(x_recon, x_target)

        # 损失 2：Sobel 梯度误差（专门惩罚高频纹理差异）
        grad_pred   = self._gradient_map(x_recon)   # [B, 1, 16, 16]
        grad_target = self._gradient_map(x_target)  # [B, 1, 16, 16]
        l_grad = F.mse_loss(grad_pred, grad_target)

        loss = l_mse + self.lambda_grad * l_grad
        return loss, x_recon
