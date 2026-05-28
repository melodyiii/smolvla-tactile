"""
train_dummy.py - 多视角融合的 Dummy Data 测试脚本（选项 B + 验证集）

功能：
1. 生成符合 LeRobot Schema 的假数据（手眼相机、全局RGB、深度、触觉、文本）
2. 实现三个视觉分支的独立编码
3. 使用注意力融合（CrossAttention）让触觉序列同时关注三个视觉视角
4. 计算多模态对比损失
5. 运行训练/验证循环，验证梯度反向传播和性能

运行方式：
    conda run -n tlv python train_dummy.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from typing import Dict, List, Tuple
import numpy as np
import sys
import os

# 将 PyTorch 模型缓存目录重定向到项目内，避免系统权限问题
_cache_dir = os.path.join(os.path.dirname(__file__), ".torch_cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ["TORCH_HOME"] = _cache_dir

# 导入现有模型
sys.path.insert(0, os.path.dirname(__file__))
from models.tlv_student import TactileEncoder, improved_multi_pos_infonce
from models.dense_contrastive import PatchContrastiveHead, TactileReconHead


# ============================================================================
# 1. DummyDataset & DataLoader (数据伪造层)
# ============================================================================

class DummyMultimodalDataset(Dataset):
    """
    生成符合 LeRobot Schema 的单样本数据（三个视觉分支）。
    """
    
    def __init__(self, num_samples: int = 16, seq_len: int = 16):
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        self.instructions = [
            "抓取红色方块",
            "推杯子到左边",
            "放置物体在架子上",
            "旋转手腕 90 度",
            "轻轻接触表面",
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        # 触觉数据：[T, 16, 16]
        tactile = torch.randn(self.seq_len, 16, 16, dtype=torch.float32).clamp(0, 1)
        
        # 三个视觉分支
        wrist_rgb = torch.randn(self.seq_len, 3, 480, 640, dtype=torch.float32).clamp(0, 1)
        realsense_rgb = torch.randn(self.seq_len, 3, 480, 640, dtype=torch.float32).clamp(0, 1)
        realsense_depth = torch.randn(self.seq_len, 1, 480, 640, dtype=torch.float32).clamp(0, 1)
        
        # 文本嵌入
        num_texts = np.random.randint(2, 5)
        text_embeddings = [
            torch.randn(1, 512, dtype=torch.float32) 
            for _ in range(num_texts)
        ]
        
        # 视觉嵌入
        vision_embedding = torch.randn(512, dtype=torch.float32)
        
        sample = {
            "tactile": tactile,
            "wrist_rgb": wrist_rgb,
            "realsense_rgb": realsense_rgb,
            "realsense_depth": realsense_depth,
            "text_embeddings": text_embeddings,
            "vision_embedding": vision_embedding,
            "instruction": self.instructions[idx % len(self.instructions)],
        }
        return sample


def collate_dummy(batch):
    """自定义 collate 函数"""
    tactiles = []
    wrist_rgbs = []
    realsense_rgbs = []
    realsense_depths = []
    text_lists = []
    vision_embeddings = []
    
    for sample in batch:
        tactiles.append(sample["tactile"])
        wrist_rgbs.append(sample["wrist_rgb"])
        realsense_rgbs.append(sample["realsense_rgb"])
        realsense_depths.append(sample["realsense_depth"])
        text_lists.append(sample["text_embeddings"])
        vision_embeddings.append(sample["vision_embedding"])
    
    tactile_batch = torch.stack(tactiles, dim=0)
    wrist_rgb_batch = torch.stack(wrist_rgbs, dim=0)
    realsense_rgb_batch = torch.stack(realsense_rgbs, dim=0)
    realsense_depth_batch = torch.stack(realsense_depths, dim=0)
    vision_batch = torch.stack(vision_embeddings, dim=0)
    
    return {
        "tactile": tactile_batch,
        "wrist_rgb": wrist_rgb_batch,
        "realsense_rgb": realsense_rgb_batch,
        "realsense_depth": realsense_depth_batch,
        "text_embeddings": text_lists,
        "vision_embedding": vision_batch,
    }


# ============================================================================
# 2. Vision Encoders (工业级预训练 ResNet18 视觉编码器)
# ============================================================================

import torchvision.models as tv_models

class PretrainedVisionEncoder(nn.Module):
    """
    基于预训练 ResNet18 的视觉编码器，同时支持 RGB（3通道）和 Depth（1通道）。

    处理时序输入 [B, T, C, H, W]：
      1. 折叠时间维度 -> [B*T, C, H, W]
      2. 通过 ResNet18 backbone 提取特征 -> [B*T, 512]
      3. 经 Projection 投影到目标维度 -> [B*T, feature_dim]
      4. 恢复时间维度 -> [B, T, feature_dim]
      5. 时间维度上求平均 -> [B, feature_dim]

    Args:
        is_depth: 是否为深度图分支（True 时执行单通道权重魔改）
        feature_dim: 输出特征维度
        freeze_backbone: True 时冻结 ResNet backbone，只训练 Projection 层
    """

    def __init__(
        self,
        is_depth: bool = False,
        feature_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.is_depth = is_depth

        # ---- 加载 ResNet18 ----
        # 注意：当前使用随机初始化（weights=None）以支持离线环境。
        # 有网络时改为 weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 即可加载 ImageNet 预训练权重。
        resnet = tv_models.resnet18(weights=None)

        if is_depth:
            # ---- 深度图单通道权重魔改 ----
            # 原始 conv1 权重形状: [64, 3, 7, 7]
            # 策略: 在 in_channels 维度求平均，得到 [64, 1, 7, 7]
            # 这样保留了预训练的边缘/纹理检测能力，同时适配单通道输入
            orig_weight = resnet.conv1.weight.data          # [64, 3, 7, 7]
            new_weight = orig_weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]

            # 用新权重替换 conv1（in_channels=1）
            resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            resnet.conv1.weight.data = new_weight           # 赋值魔改后的权重

        # 移除 ResNet 最后的全连接分类层（fc），保留到 avgpool
        # 拆成两段：early_layers 到 layer1（用于暴露局部特征图），late_layers 到 avgpool
        self.early_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,   # 输出 [B*T, 64, H/4, W/4]，用于 Dense 对比学习
        )
        self.late_layers = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,  # -> [B*T, 512, 1, 1]
        )

        # ---- 冻结策略 ----
        if freeze_backbone:
            self.early_layers.requires_grad_(False)
            self.late_layers.requires_grad_(False)

        # Projection 层始终可训练
        self.projection = nn.Linear(512, feature_dim)

    def forward(self, x: torch.Tensor, return_feat_map: bool = False):
        """
        x: [B, T, C, H, W]
        return_feat_map: 若为 True，同时返回 layer1 特征图 [B, 64, H', W']
        output: [B, feature_dim]  或  ([B, feature_dim], [B, 64, H', W'])
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # early: 到 layer1，特征图用于 Dense 对比学习
        feat_map_bt = self.early_layers(x_flat)   # [B*T, 64, H', W']
        # late: 到 avgpool
        feat = self.late_layers(feat_map_bt)       # [B*T, 512, 1, 1]
        feat = feat.view(B * T, -1)               # [B*T, 512]

        # 投影到目标特征维度
        feat = self.projection(feat)          # [B*T, feature_dim]

        # 恢复时间维度并取均值: [B*T, D] -> [B, T, D] -> [B, D]
        out = feat.view(B, T, self.feature_dim).mean(dim=1)  # [B, feature_dim]

        if return_feat_map:
            # 取最后一帧的 layer1 特征图作为局部特征代表
            # [B*T, 64, H', W'] -> [B, T, 64, H', W'] -> [B, 64, H', W']
            _, C2, H2, W2 = feat_map_bt.shape
            feat_map = feat_map_bt.view(B, T, C2, H2, W2)[:, -1]  # [B, 64, H', W']
            return out, feat_map
        return out               # [B, feature_dim]


# ============================================================================
# 3. Multi-View Attention Fusion (多视角注意力融合)
# ============================================================================

class CrossModalAttentionPool(nn.Module):
    """跨模态注意力池化"""
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, tactile_seq, anchor_emb):
        query = anchor_emb.unsqueeze(1)
        attn_out, _ = self.multihead_attn(query, tactile_seq, tactile_seq)
        x = self.norm(query + attn_out)
        x = self.norm_ffn(x + self.ffn(x))
        return x.squeeze(1)


class MultiViewFusionModel(nn.Module):
    """多视角融合模型（选项 B）- 使用预训练 ResNet18"""
    def __init__(self, tau_init=0.07, feature_dim=512, freeze_backbone=True):
        super().__init__()
        
        self.tactile_encoder = TactileEncoder(proj_dim=feature_dim, tau=tau_init)
        
        # 三个视觉编码器（均使用预训练 ResNet18）
        # wrist_encoder: 手眼 RGB 相机，3 通道
        self.wrist_encoder = PretrainedVisionEncoder(
            is_depth=False, feature_dim=feature_dim, freeze_backbone=freeze_backbone
        )
        # rgb_encoder: RealSense 全局 RGB，3 通道
        self.rgb_encoder = PretrainedVisionEncoder(
            is_depth=False, feature_dim=feature_dim, freeze_backbone=freeze_backbone
        )
        # depth_encoder: RealSense 深度图，1 通道（权重魔改版 ResNet18）
        self.depth_encoder = PretrainedVisionEncoder(
            is_depth=True, feature_dim=feature_dim, freeze_backbone=freeze_backbone
        )
        
        # 三个独立的交叉注意力模块
        self.cross_attn_wrist = CrossModalAttentionPool(dim=feature_dim)
        self.cross_attn_rgb = CrossModalAttentionPool(dim=feature_dim)
        self.cross_attn_depth = CrossModalAttentionPool(dim=feature_dim)
        
        # 融合权重（可学习）
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Vision-Tactile 对齐的温度系数
        self.logit_scale_tv = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ---- Dense 对比学习 + 重构分支 ----
        # tac_channels=128: TactileEncoder.cnn 最后层输出 (hid=128)
        # rgb_channels=64:  ResNet18 layer1 输出通道数
        self.patch_contrast = PatchContrastiveHead(
            tac_channels=128, rgb_channels=64,
            proj_dim=128, temperature=0.1, n_patches=16,
        )
        self.recon_head = TactileReconHead(z_dim=feature_dim, lambda_grad=0.1)

    def forward(self, x_tactile, wrist_rgb, realsense_rgb, realsense_depth):
        """
        x_tactile:        [B, T, 16, 16]
        wrist_rgb:        [B, T, 3, H, W]
        realsense_rgb:    [B, T, 3, H, W]
        realsense_depth:  [B, T, 1, H, W]
        """
        # ========== 触觉编码 ==========
        # return_feat_map=True 同时返回 CNN 中间特征图，用于 Dense 对比学习
        z_t_seq, z_t_global, s_text, tac_feat_map = self.tactile_encoder(
            x_tactile, return_seq=True, return_feat_map=True
        )
        # z_t_seq:     [B, T, 512]
        # z_t_global:  [B, 512]
        # tac_feat_map:[B, 128, 16, 16]  CNN 最后层特征图（hid=128）
        
        # ========== 三个视觉分支编码 ==========
        # wrist_encoder 同时返回全局特征和 layer1 特征图（用于 Dense 对比学习）
        z_wrist, wrist_feat_map = self.wrist_encoder(wrist_rgb, return_feat_map=True)
        # wrist_feat_map: [B, 64, H', W']
        z_rgb   = self.rgb_encoder(realsense_rgb)        # [B, 512]
        z_depth = self.depth_encoder(realsense_depth)    # [B, 512]
        
        # ========== 注意力融合（选项 B）==========
        # 让触觉序列分别通过 CrossAttention 关注三个视觉视角
        z_t_aligned_wrist = self.cross_attn_wrist(z_t_seq, z_wrist)   # [B, 512]
        z_t_aligned_rgb   = self.cross_attn_rgb(z_t_seq, z_rgb)       # [B, 512]
        z_t_aligned_depth = self.cross_attn_depth(z_t_seq, z_depth)   # [B, 512]
        
        # 可学习权重加权融合三个对齐特征
        weights = F.softmax(self.fusion_weights, dim=0)
        z_t_aligned_fused = (
            weights[0] * z_t_aligned_wrist +
            weights[1] * z_t_aligned_rgb +
            weights[2] * z_t_aligned_depth
        )  # [B, 512]
        
        return {
            "z_t_seq": z_t_seq,
            "z_t_global": z_t_global,
            "z_t_aligned_wrist": z_t_aligned_wrist,
            "z_t_aligned_rgb": z_t_aligned_rgb,
            "z_t_aligned_depth": z_t_aligned_depth,
            "z_t_aligned_fused": z_t_aligned_fused,
            "z_wrist": z_wrist,
            "z_rgb": z_rgb,
            "z_depth": z_depth,
            "logit_scale_text": s_text,
            "logit_scale_tv": self.logit_scale_tv,
            # Dense 对比学习所需的中间特征图
            "tac_feat_map":   tac_feat_map,    # [B, 128, 16, 16]
            "wrist_feat_map": wrist_feat_map,  # [B, 64, H', W']
        }


# ============================================================================
# 4. Loss Functions (损失函数)
# ============================================================================

def contrastive_loss_tv(feat_t, feat_v, logit_scale):
    """CLIP-style Contrastive Loss"""
    feat_t = F.normalize(feat_t, dim=-1)
    feat_v = F.normalize(feat_v, dim=-1)
    
    logit_scale = logit_scale.exp()
    logits = logit_scale * feat_t @ feat_v.t()
    
    labels = torch.arange(len(logits), device=logits.device)
    loss_t2v = F.cross_entropy(logits, labels)
    loss_v2t = F.cross_entropy(logits.t(), labels)
    
    return (loss_t2v + loss_v2t) / 2


# ============================================================================
# 5. 训练和验证循环
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, w_text=1.0, w_vision=1.0):
    """训练一个 Epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_loss_text = 0.0
    epoch_loss_vision = 0.0
    epoch_loss_patch = 0.0
    epoch_loss_recon = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        tactile = batch["tactile"].to(device).float()
        wrist_rgb = batch["wrist_rgb"].to(device).float()
        realsense_rgb = batch["realsense_rgb"].to(device).float()
        realsense_depth = batch["realsense_depth"].to(device).float()
        text_embeddings_list = batch["text_embeddings"]
        
        # 前向传播
        output = model(
            x_tactile=tactile,
            wrist_rgb=wrist_rgb,
            realsense_rgb=realsense_rgb,
            realsense_depth=realsense_depth,
        )
        
        z_t_global = output["z_t_global"]
        z_t_aligned_fused = output["z_t_aligned_fused"]
        s_text = output["logit_scale_text"]
        s_tv = output["logit_scale_tv"]
        
        # 仅在第一个 Batch 打印各分支特征维度，验证整条数据流正确
        if batch_idx == 0:
            print("\n【第一个 Batch 特征维度验证】")
            print(f"  输入触觉             : {tactile.shape}")
            print(f"  输入手眼 RGB         : {wrist_rgb.shape}")
            print(f"  输入全局 RGB         : {realsense_rgb.shape}")
            print(f"  输入深度图           : {realsense_depth.shape}")
            print(f"  触觉序列 z_t_seq     : {output['z_t_seq'].shape}")
            print(f"  触觉全局 z_t_global  : {output['z_t_global'].shape}")
            print(f"  手眼视觉 z_wrist     : {output['z_wrist'].shape}")
            print(f"  全局RGB  z_rgb       : {output['z_rgb'].shape}")
            print(f"  深度     z_depth     : {output['z_depth'].shape}")
            print(f"  触觉对齐(手眼)       : {output['z_t_aligned_wrist'].shape}")
            print(f"  触觉对齐(RGB)        : {output['z_t_aligned_rgb'].shape}")
            print(f"  触觉对齐(深度)       : {output['z_t_aligned_depth'].shape}")
            print(f"  融合触觉特征         : {output['z_t_aligned_fused'].shape}")
            w = F.softmax(model.fusion_weights, dim=0).detach()
            print(f"  可学习融合权重       : wrist={w[0]:.3f} rgb={w[1]:.3f} depth={w[2]:.3f}\n")
        
        # 计算损失
        
        # Loss 1: 触觉-文本对齐
        all_text_embeddings = []
        text_pos_list = []
        
        for text_list in text_embeddings_list:
            for text_emb in text_list:
                all_text_embeddings.append(text_emb)
            text_pos_list.append([t.to(device) for t in text_list])
        
        if all_text_embeddings:
            all_text = torch.cat(all_text_embeddings, dim=0)
            all_text = F.normalize(all_text, dim=-1)
            
            pos_lists_normalized = []
            for pos_list in text_pos_list:
                pos_lists_normalized.append([
                    F.normalize(p, dim=-1) for p in pos_list
                ])
            
            loss_text = improved_multi_pos_infonce(
                z_t_global, 
                pos_lists_normalized[0],
                all_text, 
                s_text, 
                method="simple"
            )
        else:
            loss_text = torch.tensor(0.0, device=device)
        
        # Loss 2: 触觉-融合视觉对齐
        z_vision_fused = (
            output["z_wrist"] + output["z_rgb"] + output["z_depth"]
        ) / 3
        loss_vision = contrastive_loss_tv(z_t_aligned_fused, z_vision_fused, s_tv)
        
        # Loss 3: Dense Patch 对比学习
        # 触觉 CNN 局部特征图 [B, 128, 16, 16] vs Wrist RGB layer1 特征图 [B, 64, H', W']
        loss_patch = model.patch_contrast(
            tac_feat_map=output["tac_feat_map"],
            rgb_feat_map=output["wrist_feat_map"],
        )

        # Loss 4: 触觉重构（高频纹理保留）
        # 从全局特征 z_t_global 重构触觉图的时间均值帧
        loss_recon, _ = model.recon_head(
            z_t=output["z_t_global"],
            x_tac=tactile,
        )
        
        # 总损失（新增 patch 对比权重 0.5，重构权重 0.3）
        loss_total = (w_text * loss_text + w_vision * loss_vision
                      + 0.5 * loss_patch + 0.3 * loss_recon)
        
        # 反向传播
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss_total.item()
        epoch_loss_text += loss_text.item()
        epoch_loss_vision += loss_vision.item()
        epoch_loss_patch += loss_patch.item()
        epoch_loss_recon += loss_recon.item()
        num_batches += 1
    
    return {
        "loss":       epoch_loss / num_batches,
        "loss_text":  epoch_loss_text / num_batches,
        "loss_vision": epoch_loss_vision / num_batches,
        "loss_patch": epoch_loss_patch / num_batches,
        "loss_recon": epoch_loss_recon / num_batches,
    }


def validate(model, dataloader, device, w_text=1.0, w_vision=1.0):
    """验证集评估"""
    model.eval()
    epoch_loss = 0.0
    epoch_loss_text = 0.0
    epoch_loss_vision = 0.0
    epoch_loss_patch = 0.0
    epoch_loss_recon = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            tactile = batch["tactile"].to(device).float()
            wrist_rgb = batch["wrist_rgb"].to(device).float()
            realsense_rgb = batch["realsense_rgb"].to(device).float()
            realsense_depth = batch["realsense_depth"].to(device).float()
            text_embeddings_list = batch["text_embeddings"]
            
            output = model(
                x_tactile=tactile,
                wrist_rgb=wrist_rgb,
                realsense_rgb=realsense_rgb,
                realsense_depth=realsense_depth,
            )
            
            z_t_global = output["z_t_global"]
            z_t_aligned_fused = output["z_t_aligned_fused"]
            s_text = output["logit_scale_text"]
            s_tv = output["logit_scale_tv"]
            
            # Loss 1: 触觉-文本对齐
            all_text_embeddings = []
            text_pos_list = []
            for text_list in text_embeddings_list:
                for text_emb in text_list:
                    all_text_embeddings.append(text_emb)
                text_pos_list.append([t.to(device) for t in text_list])
            
            if all_text_embeddings:
                all_text = torch.cat(all_text_embeddings, dim=0)
                all_text = F.normalize(all_text, dim=-1)
                pos_lists_normalized = [
                    [F.normalize(p, dim=-1) for p in pos_list]
                    for pos_list in text_pos_list
                ]
                loss_text = improved_multi_pos_infonce(
                    z_t_global, pos_lists_normalized[0],
                    all_text, s_text, method="simple"
                )
            else:
                loss_text = torch.tensor(0.0, device=device)
            
            # Loss 2: 触觉-融合视觉对齐
            z_vision_fused = (
                output["z_wrist"] + output["z_rgb"] + output["z_depth"]
            ) / 3
            loss_vision = contrastive_loss_tv(z_t_aligned_fused, z_vision_fused, s_tv)
            
            # Loss 3: Dense Patch 对比
            loss_patch = model.patch_contrast(
                tac_feat_map=output["tac_feat_map"],
                rgb_feat_map=output["wrist_feat_map"],
            )

            # Loss 4: 触觉重构
            loss_recon, _ = model.recon_head(
                z_t=output["z_t_global"],
                x_tac=tactile,
            )

            loss_total = (w_text * loss_text + w_vision * loss_vision
                          + 0.5 * loss_patch + 0.3 * loss_recon)
            
            epoch_loss       += loss_total.item()
            epoch_loss_text  += loss_text.item()
            epoch_loss_vision += loss_vision.item()
            epoch_loss_patch += loss_patch.item()
            epoch_loss_recon += loss_recon.item()
            num_batches += 1
    
    return {
        "loss":        epoch_loss / num_batches,
        "loss_text":   epoch_loss_text / num_batches,
        "loss_vision": epoch_loss_vision / num_batches,
        "loss_patch":  epoch_loss_patch / num_batches,
        "loss_recon":  epoch_loss_recon / num_batches,
    }


# ============================================================================
# 6. 主训练循环
# ============================================================================

def train_dummy():
    """主训练循环：使用 Dummy Data 验证多视角融合模型"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    
    # ========== 初始化数据加载器 ==========
    print("=" * 80)
    print("1. 初始化 DummyDataset 和 DataLoader")
    print("=" * 80)
    
    train_dataset = DummyMultimodalDataset(num_samples=12, seq_len=16)
    val_dataset = DummyMultimodalDataset(num_samples=4, seq_len=16)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=collate_dummy
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=collate_dummy
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"Batch 大小: 4")
    print(f"序列长度 (T): 16")
    print(f"特征维度: 512\n")
    
    # ========== 初始化模型 ==========
    print("=" * 80)
    print("2. 初始化多视角融合模型（选项 B）")
    print("=" * 80)
    
    model = MultiViewFusionModel(tau_init=0.07, feature_dim=512, freeze_backbone=True).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量:    {total_params:,}")
    print(f"可训练参数量:    {trainable_params:,}")
    print(f"冻结参数量:      {total_params - trainable_params:,}  (ResNet backbone)\n")
    
    # ========== 初始化优化器 ==========
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    w_text = 1.0
    w_vision = 1.0
    
    # ========== 训练循环 ==========
    print("=" * 80)
    print("3. 开始训练循环（2 个 Epoch + 验证）")
    print("=" * 80 + "\n")
    
    num_epochs = 2
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, device, w_text, w_vision
        )
        
        # 验证
        val_metrics = validate(model, val_dataloader, device, w_text, w_vision)
        
        # 打印结果
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  [Train] Loss: {train_metrics['loss']:.4f} | "
              f"Text: {train_metrics['loss_text']:.4f} | "
              f"Vision: {train_metrics['loss_vision']:.4f} | "
              f"Patch: {train_metrics['loss_patch']:.4f} | "
              f"Recon: {train_metrics['loss_recon']:.4f}")
        print(f"  [Val]   Loss: {val_metrics['loss']:.4f} | "
              f"Text: {val_metrics['loss_text']:.4f} | "
              f"Vision: {val_metrics['loss_vision']:.4f} | "
              f"Patch: {val_metrics['loss_patch']:.4f} | "
              f"Recon: {val_metrics['loss_recon']:.4f}")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            print(f"  ✓ 最佳验证损失更新: {best_val_loss:.6f}")
        
        print()
    
    print("=" * 80)
    print("训练完成！")
    print("  - 多视角融合（选项 B）验证成功")
    print("  - 三个视觉分支（手眼、全局RGB、深度）独立编码")
    print("  - 注意力融合让触觉序列同时关注三个视觉视角")
    print("  - Dense Patch 对比学习：触觉局部特征 <-> Wrist RGB 局部特征")
    print("  - 重构分支（MSE + Sobel 梯度损失）保留高频物理纹理信息")
    print("  - 前向传播、Loss 计算和反向传播均验证成功")
    print("=" * 80)


if __name__ == "__main__":
    train_dummy()
