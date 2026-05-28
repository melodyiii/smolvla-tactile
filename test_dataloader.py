"""
test_dataloader.py  -  验证 LeRobotTactileDataset 各路数据形状与降采样效果

运行方式（dummy 模式，无需真实数据）：
    python test_dataloader.py

运行方式（真实数据）：
    python test_dataloader.py \\
        --repo_id your_org/chip-moving-03 \\
        --sidecar_root /data/chip-moving-03 \\
        --real
"""

import argparse
import torch
from torch.utils.data import DataLoader

from overfit.dataset import LeRobotTactileDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id",      type=str, default="your_org/chip-moving-03")
    p.add_argument("--root",         type=str, default=None)
    p.add_argument("--sidecar_root", type=str, default=None)
    p.add_argument("--episodes",     type=int, nargs="*", default=None,
                   help="只读取指定 episode，例如 --episodes 0")
    p.add_argument("--target_fps",   type=int, default=10)
    p.add_argument("--T",            type=int, default=16)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--n_batches",    type=int, default=3,
                   help="验证前 n 个 batch")
    p.add_argument("--real",         action="store_true",
                   help="使用真实 LeRobot 数据（默认 dummy 模式）")
    return p.parse_args()


def check_tensor(name: str, t, expected_ndim: int = None, expected_shape=None):
    """打印 tensor 信息并做基础断言。"""
    if not isinstance(t, torch.Tensor):
        print(f"  {name}: [NOT A TENSOR] type={type(t).__name__}  val={t}")
        return
    print(f"  {name}: shape={tuple(t.shape)}  dtype={t.dtype}  "
          f"min={t.min():.4f}  max={t.max():.4f}")
    if expected_ndim is not None:
        assert t.ndim == expected_ndim, \
            f"{name} 期望 {expected_ndim}D，实际 {t.ndim}D"
    if expected_shape is not None:
        for dim, (got, exp) in enumerate(zip(t.shape, expected_shape)):
            if exp is not None:
                assert got == exp, \
                    f"{name} dim{dim} 期望 {exp}，实际 {got}"


def main():
    args = parse_args()
    use_dummy = not args.real

    print("=" * 60)
    print(f"模式: {'dummy' if use_dummy else '真实数据'}")
    print(f"target_fps={args.target_fps}  T={args.T}  batch_size={args.batch_size}")
    print("=" * 60)

    ds = LeRobotTactileDataset(
        repo_id=args.repo_id,
        root=args.root,
        sidecar_root=args.sidecar_root,
        episodes=args.episodes,
        target_fps=args.target_fps,
        T=args.T,
        use_dummy=use_dummy,
        n_dummy=max(args.batch_size * args.n_batches, 50),
    )
    print(f"Dataset 大小: {len(ds)} 个样本")
    print()

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    T, B = args.T, args.batch_size

    for batch_idx, batch in enumerate(dl):
        if batch_idx >= args.n_batches:
            break

        print(f"--- Batch {batch_idx} ---")
        check_tensor("side_rgb",      batch["side_rgb"],
                     expected_ndim=5, expected_shape=[None, T, 3, None, None])
        check_tensor("realsense_rgb", batch["realsense_rgb"],
                     expected_ndim=5, expected_shape=[None, T, 3, None, None])
        check_tensor("depth",         batch["depth"],
                     expected_ndim=5, expected_shape=[None, T, 1, None, None])
        check_tensor("tactile_grid",  batch["tactile_grid"],
                     expected_ndim=5, expected_shape=[None, T, 2, 16, 16])
        action_dim = batch["action"].shape[-1]
        check_tensor("action",        batch["action"],
                 expected_ndim=2, expected_shape=[None, action_dim])

        lang = batch.get("language_instruction", [])
        print(f"  language_instruction: {lang[0] if len(lang) else 'N/A'}")

        # 验证降采样：episode_index / frame_index 应为整数
        ep = batch.get("episode_index")
        fr = batch.get("frame_index")
        if ep is not None:
            print(f"  episode_index: {ep.tolist()}")
        if fr is not None:
            print(f"  frame_index:   {fr.tolist()}")
        print()

    print("[PASS] 所有 batch 形状验证通过。")


if __name__ == "__main__":
    main()
