"""
train/5_diffusion_policy.py

Stage 5a: Diffusion Policy + Tactile baseline training.

使用 Diffusion Policy (DDPM) 替代 SmolVLA 的 flow matching action expert,
将 Stage 3 预训练的触觉编码器通过 global conditioning 注入。

架构:
  DiffusionPolicy:
    ├─ ResNet18 + SpatialSoftmax (视觉编码, frozen or trainable)
    ├─ DualTactileGridEncoder (Stage 3, frozen/finetune)
    ├─ tactile_proj: Linear(512 → 128)
    └─ 1D UNet (trainable, FiLM conditioned on [state|img|tactile])

对比 SmolVLA:
  - 无语言条件 (Diffusion Policy 原生不支持语言)
  - 扁平 conditioning (非 token 序列)
  - DDPM noise schedule (非 flow matching)
  - 更简单的架构, 更快训练

运行:
  # 单卡
  python train/5_diffusion_policy.py \\
    --data_path ./data/inboxpicking-01 \\
    --stage3_ckpt outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt \\
    --epochs 100 --batch_size 64

  # 多卡
  accelerate launch --config_file multigpu_config.yaml \\
    train/5_diffusion_policy.py \\
    --data_path ./data/inboxpicking-01,./data/inboxpicking-02 \\
    --stage3_ckpt outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt \\
    --epochs 100 --batch_size 64
"""

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from accelerate import Accelerator

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# 配置
# ============================================================================

DEFAULT_CFG = {
    "seed": 42,
    "target_fps": 10,
    "window_T": 16,
    "batch_size": 64,
    "epochs": 100,
    "num_workers": 4,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "action_dim": 6,
    "state_dim": 6,
    # Diffusion Policy specific
    "n_obs_steps": 2,
    "horizon": 16,
    "n_action_steps": 8,
    "num_train_timesteps": 100,
    "vision_backbone": "resnet18",
    "lr": 1e-4,
    "weight_decay": 1e-6,
    "grad_clip_norm": 10.0,
    "amp": True,
    "freeze_tactile": True,
    "save_every": 10,
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage 5a: Diffusion Policy + Tactile Training")
    p.add_argument("--data_path", type=str, required=True, help="数据路径(逗号分隔)")
    p.add_argument("--stage3_ckpt", type=str,
                   default="outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--output_dir", type=str, default="outputs/exp_diffusion_tactile")
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--no_tactile", action="store_true", help="Ablation: disable tactile")
    return p.parse_args()


# ============================================================================
# Diffusion Policy 数据集 (adapts our multi-modal data to DP format)
# ============================================================================

class DiffusionTactileDataset(torch.utils.data.Dataset):
    """
    Wraps SmolVLATactileDataset to produce Diffusion Policy batch format.

    DP expects:
      observation.state:  [n_obs_steps, state_dim]
      observation.images: [n_obs_steps, n_cameras, C, H, W]
      observation.tactile: [T, 2, 16, 16]  (our extension)
      action:             [horizon, action_dim]
      action_is_pad:      [horizon]

    Since our dataset has single-frame images, we repeat along n_obs_steps.
    """

    def __init__(self, smolvla_dataset, n_obs_steps=2, horizon=16, image_size=96):
        self.ds = smolvla_dataset
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.image_size = image_size
        self._resize = torch.nn.functional.interpolate

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        raw = self.ds[idx]

        # Images → [n_obs_steps, n_cameras, C, H, W]
        side = raw["observation.images.side"]           # [3, H, W]
        realsense = raw["observation.images.realsense_rgb"]  # [3, H, W]

        # Resize to DP's default 96x96
        side = self._resize_img(side)
        realsense = self._resize_img(realsense)

        # Repeat along obs steps: [n_obs_steps, C, H, W]
        side = side.unsqueeze(0).expand(self.n_obs_steps, -1, -1, -1)
        realsense = realsense.unsqueeze(0).expand(self.n_obs_steps, -1, -1, -1)

        # State → [n_obs_steps, state_dim]
        state = raw["observation.state"]  # [state_dim]
        state = state.unsqueeze(0).expand(self.n_obs_steps, -1)

        # Action trajectory → [horizon, action_dim]
        full_action = raw["action"]  # [chunk_size, action_dim]
        if full_action.shape[0] >= self.horizon:
            action = full_action[:self.horizon]
        else:
            pad_len = self.horizon - full_action.shape[0]
            action = torch.cat([
                full_action,
                full_action[-1:].expand(pad_len, -1),
            ], dim=0)

        action_is_pad = torch.zeros(self.horizon, dtype=torch.bool)

        # Tactile passthrough
        tactile = raw["observation.tactile"]  # [T, 2, 16, 16]

        return {
            "observation.state": state.float(),
            "observation.images.side": side.float(),
            "observation.images.realsense_rgb": realsense.float(),
            "observation.tactile": tactile.float(),
            "action": action.float(),
            "action_is_pad": action_is_pad,
        }

    def _resize_img(self, img):
        """Resize [3, H, W] to [3, image_size, image_size]."""
        if img.shape[-1] != self.image_size or img.shape[-2] != self.image_size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return img


# ============================================================================
# 主训练
# ============================================================================

def main():
    args = parse_args()
    cfg = dict(DEFAULT_CFG)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.save_every is not None:
        cfg["save_every"] = args.save_every

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.get("amp") else "no",
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"[DiffusionPolicy+Tactile] device={device}  "
              f"mixed_precision={accelerator.mixed_precision}  "
              f"num_processes={accelerator.num_processes}")

    # ================================================================== #
    # 1. 构建 Diffusion Policy config
    # ================================================================== #
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.configs.types import FeatureType, PolicyFeature

    dp_config = DiffusionConfig(
        n_obs_steps=cfg["n_obs_steps"],
        horizon=cfg["horizon"],
        n_action_steps=cfg["n_action_steps"],
        vision_backbone=cfg["vision_backbone"],
        num_train_timesteps=cfg["num_train_timesteps"],
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(cfg["state_dim"],)),
            "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.images.realsense_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(cfg["action_dim"],)),
        },
    )

    # ================================================================== #
    # 2. 构建模型
    # ================================================================== #
    if args.no_tactile:
        # Ablation: standard diffusion policy (no tactile)
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy(dp_config)
        if is_main:
            print("[DiffusionPolicy] 无触觉 (ablation baseline)")
    else:
        from models.diffusion_tactile import build_tactile_diffusion
        policy = build_tactile_diffusion(
            config=dp_config,
            stage3_ckpt=args.stage3_ckpt,
            freeze_tactile=cfg["freeze_tactile"],
            device="cpu",
        )
        if is_main:
            print("[DiffusionPolicy+Tactile] 触觉已注入到 global conditioning")

    # ================================================================== #
    # 3. 数据集
    # ================================================================== #
    from dataset.smolvla_tactile_dataset import SmolVLATactileDataset

    data_paths = [p.strip() for p in args.data_path.split(",")]
    datasets = []

    for dp in data_paths:
        base_ds = SmolVLATactileDataset(
            data_path=dp,
            tokenizer=None,  # DP 不需要语言
            chunk_size=cfg["horizon"],
            target_fps=cfg["target_fps"],
            T=cfg["window_T"],
            action_dim=cfg["action_dim"],
            state_dim=cfg["state_dim"],
        )
        ds = DiffusionTactileDataset(
            base_ds,
            n_obs_steps=cfg["n_obs_steps"],
            horizon=cfg["horizon"],
        )
        datasets.append(ds)
        if is_main:
            print(f"  ↳ {dp}: {len(ds)} samples")

    train_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        drop_last=True,
        pin_memory=True,
        persistent_workers=bool(cfg["persistent_workers"]) and int(cfg["num_workers"]) > 0,
        prefetch_factor=int(cfg["prefetch_factor"]) if int(cfg["num_workers"]) > 0 else None,
    )

    if is_main:
        print(f"[DiffusionPolicy] 数据集: {len(train_ds)} samples, "
              f"{len(dl)} batches/epoch")

    # ================================================================== #
    # 4. 优化器
    # ================================================================== #
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        betas=(0.95, 0.999),
        eps=1e-8,
    )

    # ================================================================== #
    # 5. Accelerate prepare
    # ================================================================== #
    policy, optimizer, dl = accelerator.prepare(policy, optimizer, dl)

    # ================================================================== #
    # 6. 参数统计
    # ================================================================== #
    if is_main:
        uw = accelerator.unwrap_model(policy)
        total = sum(p.numel() for p in uw.parameters())
        trainable = sum(p.numel() for p in uw.parameters() if p.requires_grad)
        print(f"[DiffusionPolicy] 总参数: {total:,}  可训练: {trainable:,}")

    # ================================================================== #
    # 7. 训练
    # ================================================================== #
    output_dir = args.output_dir
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")
    start_epoch = 0

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        uw = accelerator.unwrap_model(policy)
        uw.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("loss", float("inf"))
        if is_main:
            print(f"[DiffusionPolicy] 从 epoch {start_epoch} 继续, best_loss={best_loss:.6f}")

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        policy.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg['epochs']}", leave=False, disable=not is_main)

        for batch in pbar:
            uw = accelerator.unwrap_model(policy)

            # Inject tactile if model supports it
            if hasattr(uw, "diffusion") and hasattr(uw.diffusion, "_current_tactile"):
                uw.diffusion._current_tactile = batch.get("observation.tactile")

            with accelerator.autocast():
                loss, _ = uw(batch)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(uw.parameters(), float(cfg["grad_clip_norm"]))

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Clear tactile ref
            if hasattr(uw, "diffusion") and hasattr(uw.diffusion, "_current_tactile"):
                uw.diffusion._current_tactile = None

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_loss = total_loss / max(n_batches, 1)
        if is_main:
            print(f"[DiffusionPolicy] Epoch {epoch}/{cfg['epochs']}  avg_loss={avg_loss:.6f}")

        if is_main and avg_loss < best_loss:
            best_loss = avg_loss
            _save_ckpt(accelerator, policy, optimizer, epoch, best_loss,
                       os.path.join(output_dir, "ckpt_diffusion_best.pt"))
            print(f"  → Best checkpoint (loss={best_loss:.6f})")

        if is_main and (epoch % cfg["save_every"] == 0):
            _save_ckpt(accelerator, policy, optimizer, epoch, best_loss,
                       os.path.join(output_dir, f"ckpt_diffusion_ep{epoch}.pt"))


def _save_ckpt(accelerator, policy, optimizer, epoch, loss, path):
    uw = accelerator.unwrap_model(policy)
    state = {
        "model": uw.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    if hasattr(uw, "diffusion") and hasattr(uw.diffusion, "tactile_encoder"):
        state["tactile_encoder"] = uw.diffusion.tactile_encoder.state_dict()
    torch.save(state, path)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
