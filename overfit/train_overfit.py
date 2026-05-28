"""
overfit/train_overfit.py

极端过拟合测试脚本（双视角 + 双触觉）：
- 50 dummy episodes 上过拟合
- 双视角输入：side(手眼) + realsense_rgb(全景) → SmolVLM
- 双触觉输入：tactile_grid [B,T,2,16,16] → DualTactileGridEncoder
- DummyVLA 占位（不下载大模型）
- AdamW 优化器，**严格只传入 requires_grad=True 的参数**
- MSE Loss 监督动作预测
- tqdm 进度条实时显示 loss

运行：
  cd <project_root>
  python -m overfit.train_overfit
"""

import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from overfit.dataset import OverfitDataset
from overfit.models import TactileVLAAdapter


# ============================================================================
# 辅助函数
# ============================================================================

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[3,H,W] float [0,1] -> PIL.Image"""
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(arr)


def batch_dual_view_to_pil(
    side_rgb: torch.Tensor,
    pano_rgb: torch.Tensor,
) -> List[List[Image.Image]]:
    """
    side_rgb:  [B, T, 3, H, W]  手眼视角
    pano_rgb:  [B, T, 3, H, W]  全景视角

    取最后一帧，返回 List[List[PIL.Image]]，每个样本两张图。
    """
    side_last = side_rgb[:, -1]  # [B, 3, H, W]
    pano_last = pano_rgb[:, -1]  # [B, 3, H, W]
    return [
        [tensor_to_pil(side_last[i]), tensor_to_pil(pano_last[i])]
        for i in range(side_last.shape[0])
    ]


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="过拟合测试：双视角 + 双触觉 -> VLA -> Action")
    p.add_argument("--episode_dir", type=str, default="episodes")
    p.add_argument("--repo_id", type=str, default="inboxpicking-01")
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--sidecar_root", type=str, default=None)
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="只在指定 episode 上过拟合，例如 --episodes 0")
    p.add_argument("--target_fps", type=int, default=10)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--use_dummy", action="store_true", default=False)
    p.add_argument("--max_batches", type=int, default=0,
                   help="每个 epoch 最多跑多少个 batch；0 表示完整遍历")
    p.add_argument("--n_dummy", type=int, default=50)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--use_dummy_vla", action="store_true", default=True)
    p.add_argument("--vla_model_id", type=str, default="HuggingFaceTB/SmolVLM-Instruct")

    p.add_argument("--train_tactile_encoder", action="store_true", default=True,
                   help="True=触觉编码器一起训练，False=冻结")
    return p.parse_args()


# ============================================================================
# 主训练循环
# ============================================================================

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. Dataset / DataLoader ----
    dataset = OverfitDataset(
        episode_dir=args.episode_dir,
        repo_id=args.repo_id,
        root=args.root,
        sidecar_root=args.sidecar_root,
        episodes=args.episodes,
        use_dummy=args.use_dummy,
        n_dummy=args.n_dummy,
        target_fps=args.target_fps,
        T=args.T,
    )
    sample0 = dataset[0]
    action_dim = int(sample0["action"].numel())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ---- 2. Model（DummyVLA，不下载大模型）----
    model = TactileVLAAdapter(
        vla_model_id=args.vla_model_id,
        use_dummy_vla=args.use_dummy_vla,
        tactile_feat_dim=512,
        n_tactile_tokens=8,
        action_dim=action_dim,
        device=device,
    ).to(device)

    # ---- 3. 冻结策略 ----
    # VLA 已在 TactileVLAAdapter.__init__ 中冻结。
    # 控制触觉编码器是否训练：
    for p in model.tactile_encoder.parameters():
        p.requires_grad_(args.train_tactile_encoder)

    # ====================================================================
    # ⚠️ 关键：只把 requires_grad=True 的参数交给优化器
    # 这样 VLA 的冻结参数绝不会被 optimizer 更新
    # ====================================================================
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    # ---- 4. 参数统计 ----
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable_params)
    n_frozen = n_total - n_train

    print(f"{'='*60}")
    print(f"  过拟合测试配置")
    print(f"{'='*60}")
    print(f"  Device              : {device}")
    print(f"  Dataset size        : {len(dataset)} episodes")
    print(f"  Batch size          : {args.batch_size}")
    print(f"  Epochs              : {args.epochs}")
    print(f"  Train tactile enc   : {args.train_tactile_encoder}")
    print(f"  总参数量            : {n_total:,}")
    print(f"  可训练参数          : {n_train:,}")
    print(f"  冻结参数(VLA)       : {n_frozen:,}")
    print(f"{'='*60}\n")

    # ---- 5. Training loop ----
    model.train()
    # DummyVLA 内部始终 eval（冻结 + no dropout）
    model.vla.eval()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{args.epochs}", leave=True)
        for batch in pbar:
            if args.max_batches and n_batches >= args.max_batches:
                break
            # --- 触觉 [B, T, 2, 16, 16] ---
            tactile = batch["tactile_grid"].to(device).float()

            # --- 双视角 -> List[List[PIL.Image]] ---
            images = batch_dual_view_to_pil(
                batch["side_rgb"],        # 手眼
                batch["realsense_rgb"],   # 全景
            )

            texts = list(batch["language_instruction"])
            action_gt = batch["action"].to(device).float()  # [B, 7]

            # Forward
            action_pred = model(
                images=images,
                texts=texts,
                tactile_grids=tactile,
            )  # [B, 7]

            loss = criterion(action_pred, action_gt)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = total_loss / max(n_batches, 1)
        tqdm.write(f"Epoch {epoch:03d}/{args.epochs}  avg_loss={avg_loss:.6f}")

    # ---- 6. 保存 ----
    os.makedirs("runs", exist_ok=True)
    save_path = "runs/overfit_adapter_ckpt.pt"
    torch.save({"model": model.state_dict()}, save_path)
    print(f"\nDone. Checkpoint saved: {save_path}")


if __name__ == "__main__":
    main()
