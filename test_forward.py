"""
test_forward.py  -  验证双视角前向传播与显存占用

测试内容：
  1. TactileVLAAdapter 双视角前向（dummy VLA，无需下载大模型）
  2. 打印每一步的 tensor 形状
  3. 报告 GPU 显存占用（无 GPU 时跳过）

运行方式：
    python test_forward.py
    python test_forward.py --batch_size 4 --T 16
"""

import argparse
import gc
import numpy as np
from PIL import Image

import torch

from overfit.dataset import LeRobotTactileDataset
from overfit.models import TactileVLAAdapter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--T",          type=int, default=16)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--n_tac_tok",  type=int, default=8,
                   help="触觉 MLP projector token 数")
    return p.parse_args()


def mem_str(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 ** 2
        resv  = torch.cuda.memory_reserved()  / 1024 ** 2
        return f"GPU alloc={alloc:.1f} MB  reserved={resv:.1f} MB"
    return "(CPU)"


def make_batch(B: int, T: int, action_dim: int, device: str):
    """用 LeRobotTactileDataset dummy 模式生成一个 batch。"""
    ds = LeRobotTactileDataset(
        use_dummy=True,
        n_dummy=B,
        T=T,
        action_dim=action_dim,
    )
    samples = [ds[i] for i in range(B)]

    def stack(key):
        return torch.stack([s[key] for s in samples], dim=0).to(device)

    tactile  = stack("tactile_grid")     # [B,T,2,16,16]
    side_rgb = stack("side_rgb")         # [B,T,3,H,W]
    pano_rgb = stack("realsense_rgb")    # [B,T,3,H,W]
    action   = stack("action")           # [B, action_dim]
    texts    = [s["language_instruction"] for s in samples]   # List[str]

    # 转 PIL（双视角：手眼 + 全景）
    def to_pil(t):
        arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        return Image.fromarray(arr)

    # 取最后一帧，每个样本两张图 [hand_eye, panoramic]
    images_pil = [
        [to_pil(side_rgb[i, -1]), to_pil(pano_rgb[i, -1])]
        for i in range(B)
    ]

    return tactile, images_pil, texts, action


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T = args.batch_size, args.T

    print("=" * 60)
    print(f"device={device}  batch_size={B}  T={T}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. 构建模型（dummy VLA，跳过下载）
    # ------------------------------------------------------------------ #
    print("\n[1] 初始化 TactileVLAAdapter (dummy_vla=True) ...")
    model = TactileVLAAdapter(
        tactile_feat_dim=512,
        n_tactile_tokens=args.n_tac_tok,
        action_dim=args.action_dim,
        device=device,
        use_dummy_vla=True,
    ).to(device)
    print(f"    可训练参数: "
          f"tactile={sum(p.numel() for p in model.tactile_encoder.parameters()):,}  "
          f"proj={sum(p.numel() for p in model.projector.parameters()):,}  "
          f"head={sum(p.numel() for p in model.action_head.parameters()):,}")
    print(f"    {mem_str(device)}")

    # ------------------------------------------------------------------ #
    # 2. 生成 dummy batch
    # ------------------------------------------------------------------ #
    print(f"\n[2] 生成 dummy batch (B={B}, T={T}) ...")
    tactile, images_pil, texts, action_gt = make_batch(B, T, args.action_dim, device)
    print(f"    tactile_grid : {tuple(tactile.shape)}")
    print(f"    images_pil   : List[List[PIL]] len={len(images_pil)}, n_views={len(images_pil[0])}")
    print(f"    texts        : {texts}")
    print(f"    action_gt    : {tuple(action_gt.shape)}")

    # ------------------------------------------------------------------ #
    # 3. 前向传播
    # ------------------------------------------------------------------ #
    print(f"\n[3] 前向传播 ...")
    model.eval()
    with torch.no_grad():
        action_pred = model(
            images=images_pil,
            texts=texts,
            tactile_grids=tactile,
        )
    print(f"    action_pred  : {tuple(action_pred.shape)}  (期望 [{B}, {args.action_dim}])")
    assert action_pred.shape == (B, args.action_dim), \
        f"action_pred 形状错误: {tuple(action_pred.shape)}"
    print(f"    {mem_str(device)}")

    # ------------------------------------------------------------------ #
    # 4. 反向传播（验证梯度通路）
    # ------------------------------------------------------------------ #
    print(f"\n[4] 反向传播（验证梯度通路）...")
    model.train()
    model.vla.eval()   # VLA 始终冻结

    action_pred = model(
        images=images_pil,
        texts=texts,
        tactile_grids=tactile,
    )
    loss = torch.nn.functional.mse_loss(action_pred, action_gt)
    loss.backward()
    print(f"    loss={loss.item():.6f}")

    # 检查触觉编码器梯度是否存在
    grad_ok = any(
        p.grad is not None
        for p in model.tactile_encoder.parameters()
        if p.requires_grad
    )
    print(f"    tactile_encoder 梯度存在: {grad_ok}")
    assert grad_ok, "触觉编码器梯度不存在，检查计算图！"
    print(f"    {mem_str(device)}")

    # ------------------------------------------------------------------ #
    # 5. 清理
    # ------------------------------------------------------------------ #
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\n[PASS] 双视角前向 + 反向传播验证通过。")


if __name__ == "__main__":
    main()
