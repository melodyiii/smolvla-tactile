"""
overfit/test_forward.py

验证 TactileVLAAdapter 前向传播连通性：
  1. 实例化 TactileVLAAdapter（DummyVLA，不下载大模型）
  2. 用 OverfitDataset 构造一个 batch
  3. 跑一次 forward pass
  4. 打印 action_pred 的 shape，确认 == [B, action_dim]
  5. 验证梯度能反传到 tactile_encoder

运行：
  cd <project_root>
  python -m overfit.test_forward

⚠️ OOM 排查提示：
  - 若显存不足，优先降低 batch_size（当前 2）
  - 其次降低 T（时序窗口，当前 16 → 试 8）
  - rgb_h/rgb_w 可以从 224 降到 128（仅影响 PIL 图像大小，DummyVLA 不在意）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from overfit.dataset import OverfitDataset
from overfit.models import TactileVLAAdapter


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[3,H,W] float [0,1] → PIL"""
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(arr)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test_forward] device={device}\n")

    # ---- 1. Dataset / DataLoader ----
    # ⚠️ 若 OOM，降低 batch_size 或 T
    BATCH_SIZE = 2
    T = 16
    ACTION_DIM = 7

    ds = OverfitDataset(use_dummy=True, n_dummy=10, T=T, rgb_h=224, rgb_w=224, action_dim=ACTION_DIM)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    batch = next(iter(dl))

    # ---- 2. 模型实例化（DummyVLA，不下载大模型）----
    model = TactileVLAAdapter(
        use_dummy_vla=True,     # ← 关键：用本地 DummyVLA
        tactile_feat_dim=512,
        n_tactile_tokens=8,
        action_dim=ACTION_DIM,
        device=device,
    ).to(device)

    # ---- 3. 确认冻结/可训练参数 ----
    vla_frozen = sum(1 for p in model.vla.parameters() if not p.requires_grad)
    tac_train = sum(1 for p in model.tactile_encoder.parameters() if p.requires_grad)
    proj_train = sum(1 for p in model.projector.parameters() if p.requires_grad)
    head_train = sum(1 for p in model.action_head.parameters() if p.requires_grad)

    print(f"  VLA 冻结参数层: {vla_frozen}")
    print(f"  触觉编码器可训练层: {tac_train}")
    print(f"  Projector 可训练层: {proj_train}")
    print(f"  ActionHead 可训练层: {head_train}")
    print()

    # ---- 4. 准备输入 ----
    # 双视角：side(手眼) + realsense_rgb(全景)，取最后一帧转 PIL
    side_rgb = batch["side_rgb"][:, -1]           # [B, 3, H, W]
    pano_rgb = batch["realsense_rgb"][:, -1]      # [B, 3, H, W]
    images = [
        [tensor_to_pil(side_rgb[i]), tensor_to_pil(pano_rgb[i])]
        for i in range(BATCH_SIZE)
    ]

    texts = list(batch["text"])
    tactile = batch["tactile_grid"].to(device).float()  # [B, T, 2, 16, 16]
    action_gt = batch["action"].to(device).float()      # [B, 7]

    print(f"  输入 tactile_grid shape : {list(tactile.shape)}")
    print(f"  输入 images             : {len(images)} x {len(images[0])} PIL.Image")
    print(f"  输入 texts              : {texts}")
    print()

    # ---- 5. Forward pass ----
    model.train()
    action_pred = model(images=images, texts=texts, tactile_grids=tactile)

    print(f"  action_pred shape : {list(action_pred.shape)}")
    print(f"  期望 shape        : [{BATCH_SIZE}, {ACTION_DIM}]")
    shape_ok = list(action_pred.shape) == [BATCH_SIZE, ACTION_DIM]
    print(f"  Shape 校验: {'✓ 通过' if shape_ok else '✗ 失败'}")
    print()

    # ---- 6. 反向传播测试 ----
    loss = torch.nn.functional.mse_loss(action_pred, action_gt)
    loss.backward()

    # 检查梯度是否流入 tactile_encoder
    has_grad = False
    for name, p in model.tactile_encoder.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break

    print(f"  MSE Loss          : {loss.item():.6f}")
    print(f"  触觉编码器梯度    : {'✓ 有梯度流入' if has_grad else '✗ 无梯度'}")
    print()

    if shape_ok and has_grad:
        print("  ✅ 前向传播 + 反向传播验证通过！")
    else:
        print("  ❌ 存在问题，请检查上方输出。")
    print()


if __name__ == "__main__":
    main()
