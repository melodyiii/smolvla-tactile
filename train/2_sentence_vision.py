"""
train/2_sentence_vision.py - Stage 2: 触觉-视觉-文本在线融合训练

改动说明：
- 废弃本地 TactileSetS2 / visual_embed.pt / text_embed.pt
- 接入 LeRobotDataset，提取触觉序列 + 手眼 RGB 序列
- 使用在线 ResNet18 (OnlineRGBEncoder) 实时编码 wrist_rgb
- 使用在线 CLIPTextModel 实时编码 language_instruction
- 继承 Stage 1 的 tactile_encoder 权重
- Loss 算法保持不变

运行方式：
  python train/2_sentence_vision.py \
    --repo_id <your_dataset> --stage1_ckpt runs/ckpt_stage1.pt
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tv_models

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.tlv_student import TactileEncoder, improved_multi_pos_infonce

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from transformers import CLIPTokenizer, CLIPTextModel


# ============================================================================
# 1. 在线视觉编码器（ResNet18）
# ============================================================================

class OnlineRGBEncoder(nn.Module):
    """
    输入：[B, T, 3, H, W]
    处理：折叠 B*T -> ResNet18 -> 恢复 [B, T, D] -> 时间均值
    输出：[B, 512]  L2 归一化
    """
    def __init__(self, feature_dim=512, freeze_backbone=False):
        super().__init__()
        resnet = tv_models.resnet18(weights=None)  # 有网络时改为 IMAGENET1K_V1
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 512, 1, 1]
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.projection = nn.Linear(512, feature_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # [B, T, 3, H, W] -> [B*T, 3, H, W]
        x_flat = x.view(B * T, C, H, W)
        feat = self.backbone(x_flat).view(B * T, -1)  # [B*T, 512]
        feat = self.projection(feat)                  # [B*T, D]
        # [B*T, D] -> [B, T, D] -> [B, D]
        return F.normalize(feat.view(B, T, -1).mean(dim=1), dim=-1)


# ============================================================================
# 2. 在线文本编码器（CLIP，冻结 Teacher）
# ============================================================================

class OnlineTextEncoder(nn.Module):
    CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, device="cpu"):
        super().__init__()
        self.tokenizer  = CLIPTokenizer.from_pretrained(self.CLIP_MODEL)
        self.text_model = CLIPTextModel.from_pretrained(self.CLIP_MODEL)
        self.text_model.requires_grad_(False)
        self._device = device

    def forward(self, texts):
        tokens = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self.text_model(**tokens)
        return F.normalize(out.pooler_output, dim=-1)  # [B, 512]


# ============================================================================
# 3. 跨模态注意力池化（与原版完全一致）
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

    def forward(self, tactile_seq, anchor_emb):
        query = anchor_emb.unsqueeze(1)
        attn_out, _ = self.multihead_attn(query, tactile_seq, tactile_seq)
        x = self.norm(query + attn_out)
        x = self.norm_ffn(x + self.ffn(x))
        return x.squeeze(1)


class GeometricTactilePredictor(nn.Module):
    """
    GTR Side-head:
    输入: vision/depth token + q_pos
    输出: 预测触觉压力图 [B, N, N]（或触觉隐藏向量）

    - mode="mlp": 轻量 MLP
    - mode="transformer": 单层 TransformerDecoder
    """
    def __init__(
        self,
        feature_dim=512,
        q_pos_dim=7,
        map_size=16,
        mode="mlp",
        predict_hidden=False,
        hidden_dim=512,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.q_pos_dim = q_pos_dim
        self.map_size = map_size
        self.mode = mode
        self.predict_hidden = predict_hidden

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

        out_dim = hidden_dim if predict_hidden else map_size * map_size
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, out_dim),
        )

    def align_tactile_target(self, tactile_gt):
        # tactile_gt 可为 [B,N,N] / [B,1,N,N] / [B,T,1,N,N]
        if tactile_gt.dim() == 5:
            tactile_gt = tactile_gt[:, 0, ...]
        if tactile_gt.dim() == 4:
            if tactile_gt.shape[1] == 1:
                tactile_gt = tactile_gt.squeeze(1)  # [B,N,N]
            else:
                tactile_gt = tactile_gt[:, 0, ...]

        tactile_gt = tactile_gt.float()
        if tactile_gt.shape[-1] != self.map_size or tactile_gt.shape[-2] != self.map_size:
            tactile_gt = F.adaptive_avg_pool2d(tactile_gt.unsqueeze(1), (self.map_size, self.map_size)).squeeze(1)
        return tactile_gt

    def forward(self, vision_tokens, q_pos):
        """
        vision_tokens: [B, D] 或 [B, T_v, D]
        q_pos:         [B, q_pos_dim]
        """
        if vision_tokens.dim() == 2:
            vision_tokens = vision_tokens.unsqueeze(1)  # [B,1,D]

        q_tok = self.q_proj(q_pos).unsqueeze(1)  # [B,1,D]

        if self.decoder is not None:
            fused = self.decoder(tgt=q_tok, memory=vision_tokens).squeeze(1)  # [B,D]
        else:
            fused = vision_tokens.mean(dim=1)  # [B,D]

        x = torch.cat([fused, self.q_proj(q_pos)], dim=-1)  # [B,2D]
        pred = self.head(x)

        if self.predict_hidden:
            return pred  # [B, hidden_dim]
        return pred.view(-1, self.map_size, self.map_size)  # [B,N,N]

    def get_loss(self, pred, tactile_gt, target_hidden=None):
        """
        pred:       [B,N,N] 或 [B,H]
        tactile_gt: 来自 LeRobotDataset 的 observation.tactile
        """
        if self.predict_hidden:
            if target_hidden is None:
                raise ValueError("predict_hidden=True 时需要提供 target_hidden")
            return F.mse_loss(pred, target_hidden)

        tactile_gt = self.align_tactile_target(tactile_gt)
        return F.mse_loss(pred, tactile_gt)



# ============================================================================
# 4. Stage 2 整合模型
# ============================================================================

class VisionGuidedTactileModel(nn.Module):
    def __init__(
        self,
        tau_init=0.07,
        feature_dim=512,
        use_gtr_side_head=True,
        gtr_mode="mlp",
        gtr_q_pos_dim=7,
        gtr_map_size=16,
        gtr_predict_hidden=False,
    ):
        super().__init__()
        self.tactile_encoder = TactileEncoder(tau=tau_init)
        self.cross_attn      = CrossModalAttentionPool(dim=feature_dim)
        self.logit_scale_tv  = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.use_gtr_side_head = use_gtr_side_head
        self.gtr_predictor = GeometricTactilePredictor(
            feature_dim=feature_dim,
            q_pos_dim=gtr_q_pos_dim,
            map_size=gtr_map_size,
            mode=gtr_mode,
            predict_hidden=gtr_predict_hidden,
            hidden_dim=feature_dim,
        ) if use_gtr_side_head else None

    def forward(self, x_tactile, z_vision_anchor, q_pos=None):
        z_t_seq, z_t_global, s_text = self.tactile_encoder(x_tactile, return_seq=True)

        # 暴露并融合视觉 token（Stage2 仅 RGB，可直接视作 vision_depth_tokens）
        rgb_tokens = z_vision_anchor.unsqueeze(1) if z_vision_anchor.dim() == 2 else z_vision_anchor
        vision_depth_tokens = rgb_tokens

        # 主干对齐：保持原有 cross-attn 逻辑
        z_t_aligned = self.cross_attn(z_t_seq, vision_depth_tokens.mean(dim=1))

        out = {
            "z_t_seq":             z_t_seq,
            "z_t_global":          z_t_global,
            "z_t_aligned":         z_t_aligned,
            "vision_depth_tokens": vision_depth_tokens,
            "logit_scale_text":    s_text,
            "logit_scale_tv":      self.logit_scale_tv,
        }

        # 仅训练阶段启用 GTR 监督分支（推理零开销）
        if self.training and self.use_gtr_side_head and q_pos is not None:
            out["pred_tactile"] = self.gtr_predictor(vision_depth_tokens, q_pos)

        return out


# ============================================================================
# 5. InfoNCE Loss for Vision-Tactile
# ============================================================================

def contrastive_loss_tv(feat_t, feat_v, logit_scale):
    feat_t = F.normalize(feat_t, dim=-1)
    feat_v = F.normalize(feat_v, dim=-1)
    scale  = logit_scale.exp()
    logits = scale * feat_t @ feat_v.t()
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


# ============================================================================
# 6. 配置
# ============================================================================

DEFAULT_CFG = """
seed: 42
window_T: 16
batch_size: 16
accum_steps: 4
lr: 3e-4
weight_decay: 1e-4
tau_init: 0.07
epochs: 10
amp: false
lambda_tv: 1.0
freeze_rgb_backbone: false

# GTR side-head
use_gtr_side_head: true
gtr_mode: mlp
gtr_q_pos_key: observation.state
gtr_q_pos_dim: 7
gtr_map_size: 16
gtr_predict_hidden: false
lambda_gtr: 1.0
gtr_target_key: observation.tactile
"""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id",     type=str, required=True)
    p.add_argument("--config",      type=str, default="configs/stage2_vision_attn.yaml")
    p.add_argument("--stage1_ckpt", type=str, default="runs/ckpt_stage1.pt")
    p.add_argument("--root",        type=str, default=None)
    return p.parse_args()


def load_cfg(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(DEFAULT_CFG)
    cfg = yaml.safe_load(open(path))
    cfg["lr"]                  = float(cfg["lr"])
    cfg["weight_decay"]        = float(cfg["weight_decay"])
    cfg["tau_init"]            = float(cfg["tau_init"])
    cfg["batch_size"]          = int(cfg["batch_size"])
    cfg["accum_steps"]         = int(cfg["accum_steps"])
    cfg["epochs"]              = int(cfg["epochs"])
    cfg["amp"]                 = str(cfg.get("amp","false")).lower() in ["1","true","yes","y"]
    cfg["lambda_tv"]           = float(cfg.get("lambda_tv", 1.0))
    cfg["freeze_rgb_backbone"] = bool(cfg.get("freeze_rgb_backbone", False))

    cfg["use_gtr_side_head"]   = bool(cfg.get("use_gtr_side_head", True))
    cfg["gtr_mode"]            = str(cfg.get("gtr_mode", "mlp"))
    cfg["gtr_q_pos_key"]       = str(cfg.get("gtr_q_pos_key", "observation.state"))
    cfg["gtr_q_pos_dim"]       = int(cfg.get("gtr_q_pos_dim", 7))
    cfg["gtr_map_size"]        = int(cfg.get("gtr_map_size", 16))
    cfg["gtr_predict_hidden"]  = bool(cfg.get("gtr_predict_hidden", False))
    cfg["lambda_gtr"]          = float(cfg.get("lambda_gtr", 1.0))
    cfg["gtr_target_key"]      = str(cfg.get("gtr_target_key", "observation.tactile"))
    return cfg


# ============================================================================
# 7. 主训练函数
# ============================================================================

def main():
    args = parse_args()
    cfg  = load_cfg(args.config)
    torch.manual_seed(cfg["seed"])

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg["amp"] and torch.cuda.is_available()
    print(f"[Stage2] 设备: {device}  AMP: {use_amp}")

    # ------------------------------------------------------------------ #
    # 1. LeRobotDataset：同时提取触觉序列 + 手眼 RGB
    # ------------------------------------------------------------------ #
    T  = cfg["window_T"]
    ts = [-0.05 * i for i in range(T - 1, -1, -1)]  # [-0.75, ..., 0.0]

    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        delta_timestamps={
            "observation.images.tactile_left": ts,
            "observation.images.wrist_rgb":    ts,
            cfg["gtr_target_key"]: [0.0],
            cfg["gtr_q_pos_key"]: [0.0],
        },
    )
    dl = DataLoader(
        ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=4,
        drop_last=True, pin_memory=(device == "cuda"),
    )
    print(f"[Stage2] 数据集: {len(ds)} 样本，{len(dl)} batches")

    # ------------------------------------------------------------------ #
    # 2. 模型初始化
    # ------------------------------------------------------------------ #
    model       = VisionGuidedTactileModel(
        tau_init=cfg["tau_init"],
        feature_dim=512,
        use_gtr_side_head=cfg["use_gtr_side_head"],
        gtr_mode=cfg["gtr_mode"],
        gtr_q_pos_dim=cfg["gtr_q_pos_dim"],
        gtr_map_size=cfg["gtr_map_size"],
        gtr_predict_hidden=cfg["gtr_predict_hidden"],
    ).to(device)
    rgb_encoder = OnlineRGBEncoder(
        feature_dim=512,
        freeze_backbone=cfg["freeze_rgb_backbone"],
    ).to(device)
    text_encoder = OnlineTextEncoder(device=device).to(device)

    # ------------------------------------------------------------------ #
    # 3. 加载 Stage 1 权重（触觉编码器）
    # ------------------------------------------------------------------ #
    if os.path.exists(args.stage1_ckpt):
        print(f"[Stage2] 加载 Stage 1 权重: {args.stage1_ckpt}")
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        msg  = model.tactile_encoder.load_state_dict(ckpt["model"], strict=False)
        print(f"[Stage2] 继承完成，missing keys: {len(msg.missing_keys)}")
    else:
        print("[Stage2] 未找到 Stage 1 权重，从头训练。")

    # ------------------------------------------------------------------ #
    # 4. 优化器
    # ------------------------------------------------------------------ #
    # 主干 + GTR 头放同一个 optimizer，确保 loss_gtr 同时更新视觉主干与预测头。
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(rgb_encoder.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ------------------------------------------------------------------ #
    # 5. 训练循环
    # ------------------------------------------------------------------ #
    for ep in range(cfg["epochs"]):
        model.train()
        rgb_encoder.train()
        text_encoder.eval()  # Teacher 始终 eval
        stats = {"loss": 0.0, "loss_text": 0.0, "loss_tv": 0.0, "loss_gtr": 0.0}

        for it, batch in enumerate(dl):
            # 触觉: [B, T, 1, 16, 16] -> squeeze -> [B, T, 16, 16]
            tac = batch["observation.images.tactile_left"].to(device).float()
            tac = tac.squeeze(2)  # [B, T, 16, 16]

            # 手眼 RGB: [B, T, 3, H, W]
            wrist_rgb = batch["observation.images.wrist_rgb"].to(device).float()

            # 文本指令: List[str]
            instructions = batch["language_instruction"]

            # 机械臂状态 q_pos（LeRobot 标准键: observation.state）
            q_raw = batch[cfg["gtr_q_pos_key"]].to(device).float()
            if q_raw.dim() == 3:
                q_pos = q_raw[:, 0, :cfg["gtr_q_pos_dim"]]
            else:
                q_pos = q_raw[:, :cfg["gtr_q_pos_dim"]]

            # GTR 监督目标（LeRobot 标准键: observation.tactile）
            tactile_gt = batch[cfg["gtr_target_key"]].to(device).float()
            if tactile_gt.dim() >= 4:
                tactile_gt_now = tactile_gt[:, 0, ...]
            else:
                tactile_gt_now = tactile_gt

            with torch.cuda.amp.autocast(enabled=use_amp):
                # 在线视觉编码: [B, T, 3, H, W] -> [B, 512]
                z_v = rgb_encoder(wrist_rgb)

                # 触觉前向 + GTR side-head
                out         = model(tac, z_v, q_pos=q_pos)
                z_t_global  = out["z_t_global"]    # [B, 512]
                z_t_aligned = out["z_t_aligned"]   # [B, 512]
                s_text      = out["logit_scale_text"]
                s_tv        = out["logit_scale_tv"]

                # 在线文本编码 (frozen Teacher): List[str] -> [B, 512]
                z_text = text_encoder(instructions)

                # Loss 1: 触觉-文本对齐（保持 Stage 1 能力）
                loss_text = improved_multi_pos_infonce(
                    z_t_global, [z_text], z_text, s_text, method="weighted"
                )

                # Loss 2: 触觉-视觉对齐（Stage 2 核心）
                loss_tv = contrastive_loss_tv(z_t_aligned, z_v, s_tv)

                # 主干 Loss（Stage2 主链路）
                loss_main = loss_text + cfg["lambda_tv"] * loss_tv

                # GTR Loss 融合
                if cfg["use_gtr_side_head"] and ("pred_tactile" in out):
                    loss_gtr = model.gtr_predictor.get_loss(
                        out["pred_tactile"],
                        tactile_gt_now,
                    )
                else:
                    loss_gtr = torch.tensor(0.0, device=device)

                loss_total = loss_main + cfg["lambda_gtr"] * loss_gtr
                total_loss = loss_total / cfg["accum_steps"]

            scaler.scale(total_loss).backward()

            if (it + 1) % cfg["accum_steps"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(rgb_encoder.parameters()), 1.0
                )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            stats["loss"]      += total_loss.item() * cfg["accum_steps"]
            stats["loss_text"] += loss_text.item()
            stats["loss_tv"]   += loss_tv.item()
            stats["loss_gtr"]  += loss_gtr.item()

        n = len(dl)
        print(f"[Epoch {ep+1}/{cfg['epochs']}] "
              f"Loss={stats['loss']/n:.4f} | "
              f"Text={stats['loss_text']/n:.4f} | "
              f"Vision={stats['loss_tv']/n:.4f} | "
              f"GTR={stats['loss_gtr']/n:.4f}")

    # ------------------------------------------------------------------ #
    # 6. 保存
    # 仅保存 tactile 相关权重（rgb_encoder 是辅助在线编码器，不需要继承）
    # ------------------------------------------------------------------ #
    os.makedirs("runs", exist_ok=True)
    save_path = "runs/ckpt_stage2_vision_attn.pt"
    torch.save({"model": model.state_dict()}, save_path)
    print(f"[Stage2] Model saved to {save_path}")


if __name__ == "__main__":
    main()
