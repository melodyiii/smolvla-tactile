"""
overfit/test_dataloader.py

验证数据管道连通性：
  1. 实例化 OverfitDataset（dummy 模式，无需真实数据）
  2. 用 DataLoader 取一个 batch
  3. 漂亮打印每个键的 shape / dtype

运行：
  cd <project_root>
  python -m overfit.test_dataloader
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader
from overfit.dataset import OverfitDataset


def pretty_print_batch(batch: dict, title: str = "Batch 内容"):
    """格式化打印 batch 中每个键的类型、shape、dtype"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for key in sorted(batch.keys()):
        val = batch[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key:30s} | shape={str(list(val.shape)):20s} | dtype={val.dtype}")
        elif isinstance(val, (list, tuple)):
            # text 等字符串列表
            print(f"  {key:30s} | list[{type(val[0]).__name__}], len={len(val)}")
            if isinstance(val[0], str):
                print(f"  {'':30s}   示例: \"{val[0][:50]}\"")
        else:
            print(f"  {key:30s} | type={type(val).__name__}, value={val}")
    print(f"{'='*60}\n")


def main():
    print("[test_dataloader] 使用 OverfitDataset (dummy=True) 测试数据管道\n")

    # ---- 1. 实例化 Dataset ----
    ds = OverfitDataset(
        use_dummy=True,
        n_dummy=50,       # 模拟 50 个 episode
        T=16,             # 时序窗口
        tac_h=16, tac_w=16,
        rgb_h=224, rgb_w=224,
        action_dim=7,
    )
    print(f"  Dataset size: {len(ds)}")

    # ---- 2. DataLoader ----
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    # ---- 3. 取一个 Batch 并打印 ----
    batch = next(iter(dl))
    pretty_print_batch(batch, title="DataLoader Batch (B=2)")

    # ---- 4. 重点字段校验 ----
    print("[校验] 关键字段 shape 是否正确：")

    checks = {
        "tactile_grid":  (2, 16, 2, 16, 16),   # [B, T, 2, H, W]
        "side_rgb":      (2, 16, 3, 224, 224),  # [B, T, 3, H, W]
        "realsense_rgb": (2, 16, 3, 224, 224),  # [B, T, 3, H, W]
        "depth":         (2, 16, 1, 224, 224),  # [B, T, 1, H, W]
        "action":        (2, 7),                # [B, action_dim]
    }

    all_pass = True
    for key, expected in checks.items():
        actual = tuple(batch[key].shape)
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_pass = False
        print(f"  {status} {key:20s}  期望 {expected}  实际 {actual}")

    # 文本检查
    texts = batch["text"]
    t_ok = isinstance(texts, (list, tuple)) and len(texts) == 2 and isinstance(texts[0], str)
    print(f"  {'✓' if t_ok else '✗'} {'text':20s}  期望 list[str] len=2  实际 {type(texts).__name__} len={len(texts)}")
    if not t_ok:
        all_pass = False

    print()
    if all_pass:
        print("  ✅ 数据管道验证通过！所有字段 shape/dtype 正确。")
    else:
        print("  ❌ 数据管道存在问题，请检查上方标 ✗ 的字段。")
    print()


if __name__ == "__main__":
    main()
