"""
train/3_3D_align.py - Stage 3: 深度图几何特征在线对齐训练

改动说明：
- 废弃 PointNetPPEncoder 和本地 depth.npy / visual_embed.pt 读取
- 接入 LeRobotDataset，提取触觉序列 + RealSense 深度图序列
- 使用在线 DepthEncoder（单通道 ResNet18 魔改版）实时编码 realsense_depth
- 新增 OnlineRGBEncoder（手眼 side 视角）用于 dense patch 对齐
- 继承 Stage 2 的 tactile_encoder + cross_attn(->vision_guide_attn) 权重
- 三路 Loss 算法保持不变

运行方式：
  # 单卡
  python train/3_3D_align.py --data_path ./data/inboxpicking-01 --epochs 50
  # 多卡 (accelerate)
  accelerate launch --config_file multigpu_config.yaml train/3_3D_align.py \
    --data_path ./data/inboxpicking-01 --batch_size 32 --epochs 50
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.models as tv_models
from tqdm import tqdm
from accelerate import Accelerator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.tlv_student import TactileEncoder
from overfit.dataset import LeRobotTactileDataset

# 兼容 lerobot 新旧版本路径
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        LeRobotDataset = None


# ============================================================================
# 1. 在线深度图编码器（单通道 ResNet18 魔改）
# ============================================================================

class OnlineDepthEncoder(nn.Module):
    """
    深度图在线编码器：把 ResNet18 第一层 conv1 改为单通道输入。

    输入：[B, T, 1, H, W]  单通道深度图序列
    输出：
      - 全局特征 [B, 512]
      - 可选空间特征序列 [B, T, 64, H', W']（来自 layer1，做空间 patch 对齐）
    """
    def __init__(self, feature_dim=512, freeze_backbone=False):
        super().__init__()
        resnet = tv_models.resnet18(weights=None)

        # 单通道权重魔改
        orig_w = resnet.conv1.weight.data
        new_w  = orig_w.mean(dim=1, keepdim=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = new_w

        # early/late 拆分：early 暴露空间特征，late 产生全局向量
        self.early_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,   # [B*T, 64, H', W']
        )
        self.late_layers = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,  # [B*T, 512, 1, 1]
        )

        if freeze_backbone:
            self.early_layers.requires_grad_(False)
            self.late_layers.requires_grad_(False)

        self.projection = nn.Linear(512, feature_dim)

    def forward(self, x, return_spatial_seq=False):
        """
        x: [B, T, 1, H, W]
        return_spatial_seq=True 时额外返回 [B, T, 64, H', W']
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        feat_map_bt = self.early_layers(x_flat)               # [B*T, 64, H', W']
        feat = self.late_layers(feat_map_bt).view(B * T, -1)  # [B*T, 512]
        feat = self.projection(feat)                           # [B*T, D]
        z_global = F.normalize(feat.view(B, T, -1).mean(dim=1), dim=-1)

        if return_spatial_seq:
            _, C2, H2, W2 = feat_map_bt.shape
            spatial_seq = feat_map_bt.view(B, T, C2, H2, W2)  # [B,T,64,H',W']
            return z_global, spatial_seq

        return z_global


# ============================================================================
# 2. 在线手眼 RGB 编码器（用于 dense patch 对齐）
# ============================================================================

class OnlineRGBEncoder(nn.Module):
    """
    手眼 RGB 在线编码器（3通道 ResNet18）。

    输入：[B, T, 3, H, W]
    输出：
      - 全局特征 [B, 512]
      - 可选空间特征序列 [B, T, 64, H', W']（来自 layer1）
    """
    def __init__(self, feature_dim=512, freeze_backbone=False):
        super().__init__()
        resnet = tv_models.resnet18(weights=None)

        self.early_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,   # [B*T, 64, H', W']
        )
        self.late_layers = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,  # [B*T, 512, 1, 1]
        )

        if freeze_backbone:
            self.early_layers.requires_grad_(False)
            self.late_layers.requires_grad_(False)

        self.projection = nn.Linear(512, feature_dim)

    def forward(self, x, return_spatial_seq=False):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        feat_map_bt = self.early_layers(x_flat)               # [B*T, 64, H', W']
        feat = self.late_layers(feat_map_bt).view(B * T, -1)  # [B*T, 512]
        feat = self.projection(feat)                           # [B*T, D]
        z_global = F.normalize(feat.view(B, T, -1).mean(dim=1), dim=-1)

        if return_spatial_seq:
            _, C2, H2, W2 = feat_map_bt.shape
            spatial_seq = feat_map_bt.view(B, T, C2, H2, W2)  # [B,T,64,H',W']
            return z_global, spatial_seq

        return z_global


# ============================================================================
# 3. 跨模态注意力池化（与 Stage 2 完全一致）
# ============================================================================

class CrossModalAttentionPool(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm     = nn.LayerNorm(dim)
        self.ffn      = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        q = query.unsqueeze(1)
        attn_out, _ = self.multihead_attn(q, key_value, key_value)
        x = self.norm(q + attn_out)
        x = self.norm_ffn(x + self.ffn(x))
        return x.squeeze(1)


class GeometricTactilePredictor(nn.Module):
    def __init__(
        self,
        feature_dim=512,
        q_pos_dim=7,
        map_size=16,
        mode="mlp",
    ):
        super().__init__()
        self.map_size = map_size

        self.q_proj = nn.Sequential(
            nn.Linear(q_pos_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        if mode == "transformer":
            self.decoder = nn.TransformerDecoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 2,
                batch_first=True,
                dropout=0.1,
            )
        else:
            self.decoder = None

        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, map_size * map_size),
        )

    def align_tactile_target(self, tactile_gt):
        if tactile_gt.dim() == 5:
            tactile_gt = tactile_gt[:, 0, ...]
        if tactile_gt.dim() == 4:
            if tactile_gt.shape[1] == 1:
                tactile_gt = tactile_gt.squeeze(1)
            else:
                tactile_gt = tactile_gt[:, 0, ...]

        # 不强制 float()——保持与模型参数一致的 dtype（bf16 / fp32）
        if tactile_gt.shape[-1] != self.map_size or tactile_gt.shape[-2] != self.map_size:
            tactile_gt = F.adaptive_avg_pool2d(
                tactile_gt.unsqueeze(1),
                (self.map_size, self.map_size),
            ).squeeze(1)
        return tactile_gt

    def forward(self, vision_tokens, q_pos):
        if vision_tokens.dim() == 2:
            vision_tokens = vision_tokens.unsqueeze(1)

        q_tok = self.q_proj(q_pos).unsqueeze(1)
        if self.decoder is not None:
            fused = self.decoder(tgt=q_tok, memory=vision_tokens).squeeze(1)
        else:
            fused = vision_tokens.mean(dim=1)

        x = torch.cat([fused, self.q_proj(q_pos)], dim=-1)
        pred = self.head(x)
        return pred.view(-1, self.map_size, self.map_size)

    def get_loss(self, pred, tactile_gt):
        tactile_gt = self.align_tactile_target(tactile_gt)
        return F.mse_loss(pred, tactile_gt)


class TokenPatchContrastiveHead(nn.Module):
    """
    真·空间 Patch 级对齐：
      输入: depth_tokens  [B*T, P_d, C_d]
           tactile_tokens[B*T, P_t, C_t]
      处理:
        1) 通道投影到统一维度
        2) 若 P_d != P_t，用 1D 线性插值对齐到 min(P_d, P_t)
        3) 对每个 patch 位置，在 batch-time 维上做 InfoNCE
    """
    def __init__(self, depth_channels=64, tactile_channels=128, proj_dim=128, temperature=0.1):
        super().__init__()
        self.depth_proj = nn.Linear(depth_channels, proj_dim)
        self.tactile_proj = nn.Linear(tactile_channels, proj_dim)
        self.temperature = temperature

    def _align_patch_length(self, z_d, z_t):
        # z_d: [BT, Pd, D], z_t: [BT, Pt, D]
        pd = z_d.shape[1]
        pt = z_t.shape[1]
        if pd == pt:
            return z_d, z_t

        target_p = min(pd, pt)
        z_d = F.interpolate(z_d.transpose(1, 2), size=target_p, mode="linear", align_corners=False).transpose(1, 2)
        z_t = F.interpolate(z_t.transpose(1, 2), size=target_p, mode="linear", align_corners=False).transpose(1, 2)
        return z_d, z_t

    def forward(self, depth_tokens, tactile_tokens):
        """
        depth_tokens:   [BT, Pd, Cd]
        tactile_tokens: [BT, Pt, Ct]
        """
        z_d = F.normalize(self.depth_proj(depth_tokens), dim=-1)      # [BT, Pd, D]
        z_t = F.normalize(self.tactile_proj(tactile_tokens), dim=-1)  # [BT, Pt, D]

        z_d, z_t = self._align_patch_length(z_d, z_t)                 # -> [BT, P, D]

        bt, p, _ = z_d.shape
        z_d = z_d.permute(1, 0, 2)  # [P, BT, D]
        z_t = z_t.permute(1, 0, 2)  # [P, BT, D]

        logits = torch.bmm(z_d, z_t.transpose(1, 2)) / self.temperature  # [P, BT, BT]
        labels = torch.arange(bt, device=logits.device).unsqueeze(0).expand(p, -1)

        loss_d2t = F.cross_entropy(logits.reshape(p * bt, bt), labels.reshape(p * bt))
        loss_t2d = F.cross_entropy(logits.transpose(1, 2).contiguous().reshape(p * bt, bt), labels.reshape(p * bt))
        return (loss_d2t + loss_t2d) / 2


# ============================================================================
# 3. Stage 3 融合模型
# ============================================================================

class DepthGuidedTLVModel(nn.Module):
    """
    数据流：
      x_tac  [B,T,16,16] -> TactileEncoder -> z_t_seq[B,T,512], z_t_global[B,512]
      x_dep  [B,T,1,H,W] -> OnlineDepthEncoder -> z_depth[B,512]
      z_v    [B,512] (外部传入的在线 RGB 编码，或占位零向量)

      CrossModalAttention(query=z_v/z_depth, kv=z_t_seq) -> z_t_aligned[B,512]
    """
    def __init__(
        self,
        feature_dim=512,
        tau_init=0.07,
        use_gtr_side_head=True,
        gtr_mode="mlp",
        gtr_q_pos_dim=7,
        gtr_map_size=16,
        patch_proj_dim=128,
        patch_temperature=0.1,
    ):
        super().__init__()
        # A. 触觉编码器（继承 Stage 2 权重，小学习率微调）
        self.tactile_encoder  = TactileEncoder(proj_dim=feature_dim, tau=tau_init)
        # B. 深度图在线编码器（Stage 3 新增，主力训练）
        self.depth_encoder    = OnlineDepthEncoder(feature_dim=feature_dim)
        # C. 手眼 RGB 编码器（dense patch 对齐使用）
        self.side_rgb_encoder = OnlineRGBEncoder(feature_dim=feature_dim)
        # D. 视觉引导注意力（继承 Stage 2 的 cross_attn 权重）
        self.vision_guide_attn = CrossModalAttentionPool(dim=feature_dim)

        # E. 真·空间 patch 稠密对齐头（深度 patch <-> 触觉 patch）
        self.patch_contrast_head = TokenPatchContrastiveHead(
            depth_channels=64,
            tactile_channels=128,
            proj_dim=patch_proj_dim,
            temperature=patch_temperature,
        )

        self.use_gtr_side_head = use_gtr_side_head
        self.gtr_predictor = GeometricTactilePredictor(
            feature_dim=feature_dim,
            q_pos_dim=gtr_q_pos_dim,
            map_size=gtr_map_size,
            mode=gtr_mode,
        ) if use_gtr_side_head else None

    def forward(self, x_tac, x_dep, x_side_rgb=None, z_vision_frozen=None, q_pos=None):
        """
        x_tac:           [B, T, 16, 16]
        x_dep:           [B, T, 1, H, W]  在线深度图
        x_side_rgb:      [B, T, 3, H, W]  手眼 RGB（可选，仅用于全局视觉引导）
        z_vision_frozen: [B, 512] 可选（Stage 2 遗留的视觉锚点）
        """
        # 触觉编码 + 空间特征序列
        z_t_seq, z_t_global, _, tactile_spatial_seq = self.tactile_encoder(
            x_tac, return_seq=True, return_spatial_seq=True
        )

        # 深度图在线编码（全局几何主干 + 空间特征，用于 dense 对齐）
        z_depth, depth_spatial_seq = self.depth_encoder(x_dep, return_spatial_seq=True)

        # 手眼 RGB 在线编码（可选，仅用于全局视觉引导 anchor，不参与 dense 对齐）
        if x_side_rgb is not None:
            z_side, _ = self.side_rgb_encoder(x_side_rgb, return_spatial_seq=True)
        else:
            z_side = z_depth

        # 暴露并拼接 Vision+Depth tokens（用于 cross-attn anchor）
        depth_tokens = z_depth.unsqueeze(1)
        if z_vision_frozen is not None:
            vision_tokens = z_vision_frozen.unsqueeze(1) if z_vision_frozen.dim() == 2 else z_vision_frozen
            vision_depth_tokens = torch.cat([vision_tokens, depth_tokens], dim=1)
            anchor = vision_depth_tokens.mean(dim=1)
        else:
            vision_depth_tokens = depth_tokens
            anchor = z_depth

        # 视觉引导注意力
        z_t_aligned = self.vision_guide_attn(
            query=anchor, key_value=z_t_seq
        )  # [B, 512]

        # 触觉空间 patch tokens: [B,T,128,16,16] -> [B,T,256,128]
        b, t, c_t, h_t, w_t = tactile_spatial_seq.shape
        tactile_tokens = tactile_spatial_seq.flatten(-2).permute(0, 1, 3, 2)

        # 深度空间 patch tokens: [B,T,64,Hd,Wd] -> [B,T,Hd*Wd,64]
        # 这里是 dense 对齐的主角：深度 patch <-> 触觉 patch
        depth_tokens_spatial = depth_spatial_seq.flatten(-2).permute(0, 1, 3, 2)

        # side_tokens 也暴露（供 fallback / 扩展使用）
        side_tokens = depth_tokens_spatial   # 当前与 depth 一致（deep geometric anchor）

        out = {
            "z_t_aligned":        z_t_aligned,
            "z_depth":            z_depth,
            "z_side":             z_side,
            "z_t_raw":            z_t_global,
            "vision_depth_tokens": vision_depth_tokens,
            "tactile_tokens":     tactile_tokens,        # [B,T,P_t,128]
            "depth_tokens":       depth_tokens_spatial,  # [B,T,P_d,64]
            "side_tokens":        side_tokens,           # [B,T,P_d,64] alias for depth
        }

        # 仅训练阶段启用 GTR 监督分支（推理零开销）
        if self.training and self.use_gtr_side_head and q_pos is not None:
            out["pred_tactile"] = self.gtr_predictor(vision_depth_tokens, q_pos)

        return out


# ============================================================================
# 4. 损失函数
# ============================================================================

def contrastive_loss(x, y, temp=0.07):
    """标准 InfoNCE 对比损失"""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    logits = (x @ y.t()) / temp
    labels = torch.arange(len(x), device=x.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


# ============================================================================
# 5. 权重继承（Stage 2 -> Stage 3）
# ============================================================================

def load_stage2_weights(model, stage2_path, device):
    """
    从 Stage 2 检查点继承权重：
      - tactile_encoder.*  直接复制
      - cross_attn.*  ->  vision_guide_attn.*  键名映射
    depth_encoder 为 Stage 3 新增，随机初始化。
    """
    print(f"[Init] 加载 Stage 2 权重: {stage2_path}")
    if not os.path.exists(stage2_path):
        print("[Warning] Stage 2 权重不存在，从头训练。")
        return model

    ckpt = torch.load(stage2_path, map_location=device)
    s2_dict = ckpt["model"]
    m_dict = model.state_dict()
    loaded = []

    for k, v in s2_dict.items():
        if k.startswith("tactile_encoder.") and k in m_dict:
            m_dict[k] = v
            loaded.append(k)
        elif k.startswith("cross_attn."):
            new_k = k.replace("cross_attn.", "vision_guide_attn.")
            if new_k in m_dict:
                m_dict[new_k] = v
                loaded.append(new_k)

    model.load_state_dict(m_dict, strict=False)
    print(f"[Init] 成功继承 {len(loaded)} 个参数层（depth_encoder 随机初始化）。")
    return model


def load_resume_checkpoint(model, ckpt_path, device):
    """
    从 Stage 3 checkpoint 续跑。

    当前 checkpoint 只保存模型权重和 epoch，不包含优化器状态，
    因此续跑时会重新初始化 AdamW，但仍可避免从头训练。
    """
    print(f"[Resume] 加载 checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"resume checkpoint 不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)

    start_epoch = int(ckpt.get("epoch", 0))
    print(f"[Resume] 将从 epoch {start_epoch + 1} 开始继续训练（优化器状态重新初始化）")
    return model, start_epoch


# ============================================================================
# 6. 配置
# ============================================================================

DEFAULT_CFG = """
seed: 0
window_T: 16
target_fps: 20
batch_size: 16
accum_steps: 4
lr: 1e-4
weight_decay: 1e-4
tau_init: 0.07
epochs: 50
amp: true
lambda_dv: 0.5
lambda_td: 0.5
lambda_patch: 0.3
patch_proj_dim: 128
patch_temperature: 0.1

# GTR auxiliary task
use_gtr_side_head: true
gtr_mode: mlp
gtr_q_pos_dim: 6
gtr_map_size: 16
lambda_gtr: 1.0
freeze_depth_backbone: false
"""

def parse_args():
    p = argparse.ArgumentParser(description="Stage 3: 深度-触觉 3D 对齐训练")
    p.add_argument("--repo_id",      type=str, default="local/inboxpicking",
                   help="LeRobot 数据集 ID（本地回退时可任意填写）")
    p.add_argument("--data_path",    type=str, default=None,
                   help="本地数据根目录（同时用作 root 和 sidecar_root）")
    p.add_argument("--sidecar_root", type=str, default=None,
                   help="Sidecar 根目录，默认与 data_path 相同")
    p.add_argument("--config",       type=str, default="configs/stage3_depth.yaml")
    p.add_argument("--stage2_ckpt",  type=str, default="runs/ckpt_stage2_vision_attn.pt")
    p.add_argument("--batch_size",   type=int, default=None,
                   help="覆盖每卡 batch size")
    p.add_argument("--epochs",       type=int, default=None,
                   help="覆盖总 epoch 数")
    p.add_argument("--output_dir",   type=str, default="outputs/stage3_align",
                   help="权重输出目录")
    p.add_argument("--save_every",   type=int, default=10,
                   help="每 N 个 epoch 保存")
    p.add_argument("--resume_ckpt",  type=str, default=None,
                   help="从已有 Stage 3 checkpoint 继续训练")
    p.add_argument("--root",         type=str, default=None,
                   help="向后兼容旧参数")
    return p.parse_args()


def load_cfg(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(DEFAULT_CFG)
    cfg = yaml.safe_load(open(path))
    cfg["lr"]                    = float(cfg["lr"])
    cfg["weight_decay"]          = float(cfg["weight_decay"])
    cfg["tau_init"]              = float(cfg["tau_init"])
    cfg["batch_size"]            = int(cfg["batch_size"])
    cfg["accum_steps"]           = int(cfg["accum_steps"])
    cfg["epochs"]                = int(cfg["epochs"])
    cfg["amp"]                   = str(cfg.get("amp","false")).lower() in ["1","true","yes","y"]
    cfg["lambda_dv"]             = float(cfg.get("lambda_dv", 0.5))
    cfg["lambda_td"]             = float(cfg.get("lambda_td", 0.5))
    cfg["lambda_patch"]          = float(cfg.get("lambda_patch", 0.3))
    cfg["patch_proj_dim"]        = int(cfg.get("patch_proj_dim", 128))
    cfg["patch_temperature"]     = float(cfg.get("patch_temperature", 0.1))
    cfg["use_gtr_side_head"]     = bool(cfg.get("use_gtr_side_head", True))
    cfg["gtr_mode"]              = str(cfg.get("gtr_mode", "mlp"))
    cfg["gtr_q_pos_dim"]         = int(cfg.get("gtr_q_pos_dim", 7))
    cfg["gtr_map_size"]          = int(cfg.get("gtr_map_size", 16))
    cfg["lambda_gtr"]            = float(cfg.get("lambda_gtr", 1.0))
    cfg["freeze_depth_backbone"] = bool(cfg.get("freeze_depth_backbone", False))
    cfg["freeze_depth_backbone"] = bool(cfg.get("freeze_depth_backbone", False))
    cfg["target_fps"]            = int(cfg.get("target_fps", 20))
    return cfg


# ============================================================================
# 7. 主训练函数
# ============================================================================

def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    # CLI 参数覆盖配置
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    data_root    = args.data_path or args.root
    sidecar_root = args.sidecar_root or data_root

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ================================================================== #
    # Accelerator（自动管理分布式、混合精度、梯度累积）
    # ================================================================== #
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.get("amp") else "no",
        gradient_accumulation_steps=cfg["accum_steps"],
    )
    device  = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"[Stage3] device={device}  mixed_precision={accelerator.mixed_precision}  "
              f"num_processes={accelerator.num_processes}")

    # ================================================================== #
    # 1. 数据集（统一使用 LeRobotTactileDataset，自动回退本地读取）
    # ================================================================== #
    T = cfg["window_T"]
    target_fps = cfg["target_fps"]

    # 支持逗号分隔的多数据集路径
    data_paths = [p.strip() for p in data_root.split(",")] if data_root else []
    if not data_paths:
        raise ValueError("请通过 --data_path 指定数据路径（支持逗号分隔多路径）")

    datasets = []
    for dp in data_paths:
        sr = sidecar_root if (sidecar_root and len(data_paths) == 1) else dp
        sub_ds = LeRobotTactileDataset(
            repo_id=args.repo_id,
            root=dp,
            sidecar_root=sr,
            target_fps=target_fps,
            T=T,
        )
        datasets.append(sub_ds)
        if is_main:
            print(f"  ↳ {dp}: {len(sub_ds)} samples")

    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    if is_main:
        print(f"[Stage3] 数据集: {len(ds)} 样本，{len(dl)} batches/epoch")

    # ================================================================== #
    # 2. 模型初始化 + Stage 2 权重继承
    # ================================================================== #
    model = DepthGuidedTLVModel(
        feature_dim=512,
        tau_init=cfg["tau_init"],
        use_gtr_side_head=cfg["use_gtr_side_head"],
        gtr_mode=cfg["gtr_mode"],
        gtr_q_pos_dim=cfg["gtr_q_pos_dim"],
        gtr_map_size=cfg["gtr_map_size"],
        patch_proj_dim=cfg["patch_proj_dim"],
        patch_temperature=cfg["patch_temperature"],
    )
    if args.resume_ckpt:
        model, start_epoch = load_resume_checkpoint(model, args.resume_ckpt, device)
    else:
        model = load_stage2_weights(model, args.stage2_ckpt, device)
        start_epoch = 0

    # ================================================================== #
    # 3. 差异化学习率
    # ================================================================== #
    param_groups = [
        {"params": model.tactile_encoder.parameters(),     "lr": cfg["lr"] * 0.1},
        {"params": model.vision_guide_attn.parameters(),   "lr": cfg["lr"] * 0.1},
        {"params": model.depth_encoder.parameters(),       "lr": cfg["lr"]},
        {"params": model.side_rgb_encoder.parameters(),    "lr": cfg["lr"]},
        {"params": model.patch_contrast_head.parameters(), "lr": cfg["lr"]},
    ]
    if model.gtr_predictor is not None:
        param_groups.append({"params": model.gtr_predictor.parameters(), "lr": cfg["lr"]})

    opt = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])

    # ================================================================== #
    # 4. Accelerate prepare
    # ================================================================== #
    model, opt, dl = accelerator.prepare(model, opt, dl)

    # ================================================================== #
    # 5. 训练循环
    # ================================================================== #
    output_dir = args.output_dir
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[Stage3] 开始训练: epochs={cfg['epochs']}  batch_size={cfg['batch_size']}  "
              f"l_tv=1.0  l_dv={cfg['lambda_dv']}  l_td={cfg['lambda_td']}  "
              f"l_patch={cfg['lambda_patch']}  l_gtr={cfg['lambda_gtr']}")

    for ep in range(start_epoch, cfg["epochs"]):
        model.train()
        stats = {"loss": 0.0, "l_tv": 0.0, "l_dv": 0.0, "l_td": 0.0, "l_patch": 0.0, "l_gtr": 0.0}
        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{cfg['epochs']}", disable=not is_main)

        for it, batch in enumerate(pbar):
            with accelerator.accumulate(model), accelerator.autocast():
                # 数据类型与模型一致（DeepSpeed bf16 时需要）
                dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32

                # --- 触觉: [B,T,2,16,16] -> 左路 -> [B,T,16,16] ---
                tac = batch["tactile_grid"].to(device, dtype=dtype)
                if tac.dim() == 5 and tac.shape[2] >= 1:
                    tac = tac[:, :, 0, :, :]  # left channel

                # --- 深度: [B,T,1,H,W] ---
                dep = batch["depth"].to(device, dtype=dtype)
                if dep.dim() == 4:
                    dep = dep.unsqueeze(2)  # 保证 [B,T,1,H,W]

                # --- 手眼 RGB: [B,T,3,H,W] ---
                side_rgb = batch["side_rgb"].to(device, dtype=dtype)

                # --- GTR: 用 action 近似 q_pos ---
                q_pos = batch["action"].to(device, dtype=dtype)
                if q_pos.dim() == 1:
                    q_pos = q_pos.unsqueeze(0)
                q_pos = q_pos[:, :cfg["gtr_q_pos_dim"]]

                # --- GTR 触觉监督目标: 最后帧左路 [B,16,16] ---
                tactile_gt_now = batch["tactile_grid"][:, -1, 0, :, :].to(device, dtype=dtype)

                # --- Forward ---
                out = model(tac, dep, x_side_rgb=side_rgb, z_vision_frozen=None, q_pos=q_pos)

                z_t_aligned = out["z_t_aligned"]
                z_depth     = out["z_depth"]
                z_t_raw     = out["z_t_raw"]

                # 主干 Loss
                l_tv = contrastive_loss(z_t_aligned, z_depth)
                l_dv = contrastive_loss(z_depth, z_t_aligned)
                l_td = contrastive_loss(z_t_raw, z_depth)
                loss_main = l_tv + cfg["lambda_dv"] * l_dv + cfg["lambda_td"] * l_td

                # Patch dense alignment Loss
                b, t, p_t, c_t = out["tactile_tokens"].shape
                _, _, p_d, c_d = out["depth_tokens"].shape
                tactile_tokens_bt = out["tactile_tokens"].reshape(b * t, p_t, c_t)
                depth_tokens_bt   = out["depth_tokens"].reshape(b * t, p_d, c_d)
                unwrapped = accelerator.unwrap_model(model)
                l_patch = unwrapped.patch_contrast_head(depth_tokens_bt, tactile_tokens_bt)

                # GTR 辅助任务 Loss
                if cfg["use_gtr_side_head"] and "pred_tactile" in out:
                    l_gtr = unwrapped.gtr_predictor.get_loss(out["pred_tactile"], tactile_gt_now)
                else:
                    l_gtr = torch.tensor(0.0, device=device, dtype=dtype)

                loss_total = loss_main + cfg["lambda_patch"] * l_patch + cfg["lambda_gtr"] * l_gtr

                accelerator.backward(loss_total)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            stats["loss"]    += loss_total.item()
            stats["l_tv"]    += l_tv.item()
            stats["l_dv"]    += l_dv.item()
            stats["l_td"]    += l_td.item()
            stats["l_patch"] += l_patch.item()
            stats["l_gtr"]   += l_gtr.item()

        n = max(len(dl), 1)
        if is_main:
            print(f"[Epoch {ep+1}/{cfg['epochs']}] "
                  f"Loss={stats['loss']/n:.4f} | "
                  f"l_tv={stats['l_tv']/n:.4f} | "
                  f"l_dv={stats['l_dv']/n:.4f} | "
                  f"l_td={stats['l_td']/n:.4f} | "
                  f"l_patch={stats['l_patch']/n:.4f} | "
                  f"l_gtr={stats['l_gtr']/n:.4f}")

        # --- 周期保存 ---
        if is_main and ((ep + 1) % args.save_every == 0):
            uw = accelerator.unwrap_model(model)
            ckpt_path = os.path.join(output_dir, f"ckpt_stage3_ep{ep+1}.pt")
            torch.save({"epoch": ep + 1, "model": uw.state_dict()}, ckpt_path)
            print(f"  → Checkpoint: {ckpt_path}")

    # ================================================================== #
    # 6. 最终保存
    # ================================================================== #
    if is_main:
        uw = accelerator.unwrap_model(model)
        os.makedirs(output_dir, exist_ok=True)

        # 精简权重（仅用于下游 Stage 4）
        final_path = os.path.join(output_dir, "ckpt_stage3_depth_final.pt")
        torch.save({
            "model": {
                "tactile_encoder":   uw.tactile_encoder.state_dict(),
                "vision_guide_attn": uw.vision_guide_attn.state_dict(),
            }
        }, final_path)

        # 完整模型（可恢复训练）
        full_path = os.path.join(output_dir, "ckpt_stage3_full_final.pt")
        torch.save({"epoch": cfg["epochs"], "model": uw.state_dict()}, full_path)

        print(f"\n[Stage3] 训练完成！")
        print(f"  触觉+注意力权重: {final_path}")
        print(f"  完整模型权重:   {full_path}")


if __name__ == "__main__":
    main()
