"""
train/1_phrase.py - Stage 1: 触觉-文本在线对齐训练

改动说明：
- 废弃本地 TactileSet / text_embed.pt 离线读取
- 接入 LeRobotDataset，使用 delta_timestamps 提取 T=16 帧历史触觉数据
- 使用在线 CLIPTextModel 实时编码 language_instruction
- Loss 算法（improved_multi_pos_infonce）保持不变

运行方式：
  python train/1_phrase.py --repo_id <your_lerobot_dataset_id>
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.tlv_student import TactileEncoder, improved_multi_pos_infonce

# LeRobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# CLIP 文本编码器
from transformers import CLIPTokenizer, CLIPTextModel


# ============================================================================
# 1. 在线文本编码器
# ============================================================================

class OnlineTextEncoder(nn.Module):
    """
    使用 CLIP ViT-B/32 的文本编码器实时编码 language_instruction。

    输入：List[str]，长度为 B
    输出：[B, 512]（已 L2 归一化，维度与 TactileEncoder 的 proj_dim=512 对齐）

    注意：CLIPTextModel 的 pooler_output 维度为 512，与我们的特征空间一致，
    无需额外投影层。冻结 CLIP 权重，仅作为 Teacher 使用。
    """
    CLIP_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, device="cpu", freeze: bool = True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(self.CLIP_MODEL)
        self.text_model = CLIPTextModel.from_pretrained(self.CLIP_MODEL)
        if freeze:
            # 作为 Teacher 使用，冻结所有参数
            self.text_model.requires_grad_(False)
        self._device = device

    def forward(self, texts):
        """
        texts: List[str]，长度为 B
        output: [B, 512]  L2 归一化后的文本特征
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}
        with torch.no_grad() if not self.training else torch.enable_grad():
            out = self.text_model(**tokens)
        # pooler_output: [B, 512]（[CLS] token 经投影后的语义特征）
        return F.normalize(out.pooler_output, dim=-1)  # [B, 512]


# ============================================================================
# 2. 配置解析
# ============================================================================

DEFAULT_CFG = """
seed: 0
window_T: 16
batch_size: 32
accum_steps: 4
lr: 3e-4
weight_decay: 1e-4
tau_init: 0.07
epochs: 3
amp: false
"""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id",  type=str, required=True,
                   help="LeRobot 数据集 ID，例如 'your_org/your_dataset'")
    p.add_argument("--config",   type=str, default="configs/stage1_phrase.yaml")
    p.add_argument("--root",     type=str, default=None,
                   help="数据集本地根目录（可选，留空则从 HuggingFace Hub 拉取）")
    return p.parse_args()


def load_cfg(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(DEFAULT_CFG)
    cfg = yaml.safe_load(open(path))
    cfg["lr"]           = float(cfg["lr"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["tau_init"]     = float(cfg["tau_init"])
    cfg["batch_size"]   = int(cfg["batch_size"])
    cfg["accum_steps"]  = int(cfg["accum_steps"])
    cfg["epochs"]       = int(cfg["epochs"])
    cfg["amp"]          = str(cfg.get("amp", "false")).lower() in ["1", "true", "yes", "y"]
    return cfg


# ============================================================================
# 3. 主训练函数
# ============================================================================

def main():
    args = parse_args()
    cfg  = load_cfg(args.config)
    torch.manual_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg["amp"] and torch.cuda.is_available()
    print(f"[Stage1] 设备: {device}  AMP: {use_amp}")

    # ------------------------------------------------------------------ #
    # 1. LeRobotDataset：提取过去 T=16 帧的触觉和语言数据
    # ------------------------------------------------------------------ #
    T = cfg["window_T"]  # 16
    delta_timestamps = {
        # 采样过去 T 帧的触觉图像（-0.75s ~ 0s，假设 20Hz，间隔 0.05s）
        "observation.images.tactile_left": [
            -0.05 * i for i in range(T - 1, -1, -1)
        ],  # [-0.75, -0.70, ..., 0.0]
    }

    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        delta_timestamps=delta_timestamps,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    print(f"[Stage1] 数据集: {len(ds)} 样本，{len(dl)} batches")

    # ------------------------------------------------------------------ #
    # 2. 模型初始化
    # ------------------------------------------------------------------ #
    # Student：触觉编码器
    tactile_encoder = TactileEncoder(tau=cfg["tau_init"]).to(device)

    # Teacher：在线 CLIP 文本编码器（冻结）
    text_encoder = OnlineTextEncoder(device=device, freeze=True).to(device)

    # ------------------------------------------------------------------ #
    # 3. 优化器（只优化 tactile_encoder，text_encoder 已冻结）
    # ------------------------------------------------------------------ #
    opt = torch.optim.AdamW(
        tactile_encoder.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ------------------------------------------------------------------ #
    # 4. 训练循环
    # ------------------------------------------------------------------ #
    for ep in range(cfg["epochs"]):
        tactile_encoder.train()
        text_encoder.eval()  # Teacher 始终 eval 模式
        tot_loss = 0.0

        for it, batch in enumerate(dl):
            # ---- 触觉数据 ----
            # batch["observation.images.tactile_left"]: [B, T, 1, 16, 16]
            tac_raw = batch["observation.images.tactile_left"].to(device).float()
            B, T_, C, H, W = tac_raw.shape  # C=1, H=W=16
            # squeeze 通道维 -> [B, T, 16, 16]（TactileEncoder 期望此格式）
            tac = tac_raw.squeeze(2)  # [B, T, 16, 16]

            # ---- 文本数据（在线编码）----
            # batch["language_instruction"]: List[str]，长度为 B
            instructions = batch["language_instruction"]  # List[str]

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Student 前向：触觉 -> z_t [B, 512]
                z_t, s = tactile_encoder(tac)

                # Teacher 在线编码：language_instruction -> z_text [B, 512]
                # 注意：text_encoder 已冻结，梯度不流入
                z_text = text_encoder(instructions)  # [B, 512]

                # 构造正例列表（每个样本只有 1 个文本正例）
                # improved_multi_pos_infonce 期望 pos_list: List[Tensor[B, 512]]
                z_x_list = [z_text]   # 单正例，后续可扩展多短语
                all_text = z_text     # [B, 512]，作为负例库

                loss = improved_multi_pos_infonce(
                    z_t, z_x_list, all_text, s, method="simple"
                ) / cfg["accum_steps"]

            scaler.scale(loss).backward()

            if (it + 1) % cfg["accum_steps"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(tactile_encoder.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            tot_loss += loss.item() * cfg["accum_steps"]

        print(f"[Stage1] epoch {ep + 1}/{cfg['epochs']}: "
              f"loss={tot_loss / len(dl):.4f}")

    # ------------------------------------------------------------------ #
    # 5. 保存（仅保存 tactile_encoder，text_encoder 是冻结的 CLIP 不需要保存）
    # ------------------------------------------------------------------ #
    os.makedirs("runs", exist_ok=True)
    torch.save({"model": tactile_encoder.state_dict()}, "runs/ckpt_stage1.pt")
    print("[Stage1] Saved runs/ckpt_stage1.pt")


if __name__ == "__main__":
    main()
