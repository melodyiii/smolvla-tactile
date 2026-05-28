"""
train/6_pi0_policy.py

Stage 6: Pi0 / Pi0-FAST + Tactile policy training.

支持两种模式:
  --mode pi0      → Pi0 (flow matching, Gemma 2B + Gemma 300M expert)
  --mode pi0fast  → Pi0-FAST (autoregressive, Gemma 2B only, FAST tokenizer)

架构:
  Pi0 + Tactile:
    PaliGemma (SigLIP + Gemma 2B, frozen/LoRA)
    ├─ DualTactileGridEncoder (Stage 3, frozen/finetune)
    ├─ TactileMLPProjector: 512 → 8 × 2048  → prefix tokens
    ├─ Gemma 300M Action Expert (trainable)
    ├─ state_proj, action_in/out_proj (trainable)
    └─ Flow matching loss (MSE on velocity field)

  Pi0-FAST + Tactile:
    PaliGemma (SigLIP + Gemma 2B, frozen/LoRA)
    ├─ DualTactileGridEncoder (Stage 3, frozen/finetune)
    ├─ TactileMLPProjector: 512 → 8 × 2048  → prefix tokens
    └─ FAST action tokenizer + cross-entropy loss

运行:
  # Pi0 + Tactile (单卡)
  python train/6_pi0_policy.py --mode pi0 \\
    --data_path ./data/inboxpicking-01 \\
    --stage3_ckpt outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt \\
    --epochs 40 --batch_size 2

  # Pi0-FAST + Tactile (多卡)
  accelerate launch --config_file multigpu_config.yaml \\
    train/6_pi0_policy.py --mode pi0fast \\
    --data_path ./data/inboxpicking-01,./data/inboxpicking-02 \\
    --stage3_ckpt outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt \\
    --epochs 40 --batch_size 2
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
    "batch_size": 2,
    "epochs": 40,
    "num_workers": 4,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "action_dim": 6,
    "state_dim": 6,
    "chunk_size": 50,
    "n_tactile_tokens": 8,
    "tokenizer_max_length": 48,
    # Learning rates
    "lr_expert": 1e-4,
    "lr_tactile_proj": 5e-4,
    "lr_state_proj": 1e-4,
    "lr_tactile_encoder": 1e-5,
    "weight_decay": 1e-10,
    "grad_clip_norm": 10.0,
    "amp": True,
    "freeze_tactile": True,
    "save_every": 5,
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage 6: Pi0/Pi0Fast + Tactile Training")
    p.add_argument("--mode", type=str, choices=["pi0", "pi0fast"], default="pi0")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--pretrained", type=str, default=None,
                   help="预训练模型路径 (默认 lerobot/pi0_base 或 lerobot/pi0fast_base)")
    p.add_argument("--stage3_ckpt", type=str,
                   default="outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt")
    p.add_argument("--n_tactile_tokens", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--freeze_tactile", action="store_true", default=None)
    p.add_argument("--unfreeze_tactile", action="store_true", default=False)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--no_tactile", action="store_true", help="Ablation: no tactile input")
    return p.parse_args()


# ============================================================================
# 主训练
# ============================================================================

def main():
    args = parse_args()
    cfg = dict(DEFAULT_CFG)

    # CLI overrides
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.n_tactile_tokens is not None:
        cfg["n_tactile_tokens"] = args.n_tactile_tokens
    if args.save_every is not None:
        cfg["save_every"] = args.save_every
    if args.lr is not None:
        cfg["lr_expert"] = args.lr
        cfg["lr_tactile_proj"] = args.lr
        cfg["lr_state_proj"] = args.lr
    if args.unfreeze_tactile:
        cfg["freeze_tactile"] = False

    if args.output_dir is None:
        args.output_dir = f"outputs/exp_{args.mode}_tactile"

    pretrained = args.pretrained
    if pretrained is None:
        pretrained = "lerobot/pi0_base" if args.mode == "pi0" else "lerobot/pi0fast_base"

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
        print(f"[{args.mode.upper()}+Tactile] device={device}  "
              f"mixed_precision={accelerator.mixed_precision}  "
              f"num_processes={accelerator.num_processes}")

    # ================================================================== #
    # 1. 构建模型
    # ================================================================== #
    image_keys = [
        "observation.images.side",
        "observation.images.realsense_rgb",
    ]

    if args.mode == "pi0":
        if args.no_tactile:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            policy = PI0Policy.from_pretrained(pretrained)
        else:
            from models.pi0_tactile import build_tactile_pi0
            policy = build_tactile_pi0(
                pretrained_path=pretrained,
                n_tactile_tokens=cfg["n_tactile_tokens"],
                stage3_ckpt=args.stage3_ckpt,
                freeze_tactile=cfg["freeze_tactile"],
                device="cpu",
                image_keys=image_keys,
            )
        if is_main:
            print(f"[Pi0] Model loaded: {pretrained}")
    else:
        if args.no_tactile:
            from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
            policy = PI0FastPolicy.from_pretrained(pretrained)
        else:
            from models.pi0_tactile import build_tactile_pi0fast
            policy = build_tactile_pi0fast(
                pretrained_path=pretrained,
                n_tactile_tokens=cfg["n_tactile_tokens"],
                stage3_ckpt=args.stage3_ckpt,
                freeze_tactile=cfg["freeze_tactile"],
                device="cpu",
                image_keys=image_keys,
            )
        if is_main:
            print(f"[Pi0-FAST] Model loaded: {pretrained}")

    # ================================================================== #
    # 2. 数据集
    # ================================================================== #
    from dataset.smolvla_tactile_dataset import SmolVLATactileDataset

    # Get tokenizer from model
    if args.mode == "pi0":
        tokenizer = None  # Pi0 uses PaliGemma tokenizer internally
    else:
        tokenizer = None  # Pi0Fast also handles tokenization internally

    data_paths = [p.strip() for p in args.data_path.split(",")]
    datasets = []

    for dp in data_paths:
        ds = SmolVLATactileDataset(
            data_path=dp,
            tokenizer=tokenizer,
            chunk_size=cfg["chunk_size"],
            target_fps=cfg["target_fps"],
            T=cfg["window_T"],
            action_dim=cfg["action_dim"],
            state_dim=cfg["state_dim"],
            tokenizer_max_length=cfg["tokenizer_max_length"],
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
        print(f"[{args.mode.upper()}] 数据集: {len(train_ds)} samples, "
              f"{len(dl)} batches/epoch")

    # ================================================================== #
    # 3. 参数组 + 优化器
    # ================================================================== #
    param_groups = []
    uw_model = policy.model if hasattr(policy, 'model') else policy

    # Tactile-specific params
    if hasattr(uw_model, "tactile_encoder"):
        tac_train = [p for p in uw_model.tactile_encoder.parameters() if p.requires_grad]
        if tac_train:
            param_groups.append({
                "params": tac_train,
                "lr": float(cfg["lr_tactile_encoder"]),
                "name": "tactile_encoder",
            })

    if hasattr(uw_model, "tactile_proj"):
        proj_params = list(uw_model.tactile_proj.parameters())
        if proj_params:
            param_groups.append({
                "params": proj_params,
                "lr": float(cfg["lr_tactile_proj"]),
                "name": "tactile_proj",
            })

    # All other trainable params
    tactile_param_ids = set()
    if hasattr(uw_model, "tactile_encoder"):
        tactile_param_ids |= {id(p) for p in uw_model.tactile_encoder.parameters()}
    if hasattr(uw_model, "tactile_proj"):
        tactile_param_ids |= {id(p) for p in uw_model.tactile_proj.parameters()}

    other_params = [
        p for p in uw_model.parameters()
        if p.requires_grad and id(p) not in tactile_param_ids
    ]
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": float(cfg["lr_expert"]),
            "name": "model_trainable",
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg["weight_decay"]),
        betas=(0.9, 0.95),
    )

    # ================================================================== #
    # 4. Accelerate prepare
    # ================================================================== #
    policy, optimizer, dl = accelerator.prepare(policy, optimizer, dl)

    # ================================================================== #
    # 5. 参数统计
    # ================================================================== #
    if is_main:
        uw = accelerator.unwrap_model(policy)
        total = sum(p.numel() for p in uw.parameters())
        trainable = sum(p.numel() for p in uw.parameters() if p.requires_grad)
        for pg in param_groups:
            n = sum(p.numel() for p in pg["params"])
            print(f"  {pg['name']}: {n:,} params, lr={pg['lr']}")
        print(f"[{args.mode.upper()}] 总参数: {total:,}  可训练: {trainable:,}")

    # ================================================================== #
    # 6. 训练循环
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
            print(f"[{args.mode.upper()}] 从 epoch {start_epoch} 继续, best_loss={best_loss:.6f}")

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        policy.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg['epochs']}", leave=False, disable=not is_main)

        for batch in pbar:
            uw = accelerator.unwrap_model(policy)
            inner_model = uw.model if hasattr(uw, 'model') else uw

            # Inject tactile data
            if hasattr(inner_model, "_current_tactile"):
                inner_model._current_tactile = batch.get("observation.tactile")

            with accelerator.autocast():
                if args.mode == "pi0":
                    loss, loss_dict = uw(batch)
                else:
                    loss, loss_dict = uw(batch)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [p for p in uw.parameters() if p.requires_grad],
                    float(cfg["grad_clip_norm"]),
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if hasattr(inner_model, "_current_tactile"):
                inner_model._current_tactile = None

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_loss = total_loss / max(n_batches, 1)
        if is_main:
            print(f"[{args.mode.upper()}+Tactile] Epoch {epoch}/{cfg['epochs']}  "
                  f"avg_loss={avg_loss:.6f}")

        if is_main and avg_loss < best_loss:
            best_loss = avg_loss
            _save_ckpt(
                accelerator, policy, epoch, best_loss, args.mode,
                os.path.join(output_dir, f"ckpt_{args.mode}_best.pt"),
            )
            print(f"  → Best checkpoint (loss={best_loss:.6f})")

        if is_main and (epoch % cfg["save_every"] == 0):
            _save_ckpt(
                accelerator, policy, epoch, best_loss, args.mode,
                os.path.join(output_dir, f"ckpt_{args.mode}_ep{epoch}.pt"),
            )


def _save_ckpt(accelerator, policy, epoch, loss, mode, path):
    uw = accelerator.unwrap_model(policy)
    state = {
        "model": uw.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "mode": mode,
    }
    inner = uw.model if hasattr(uw, 'model') else uw
    if hasattr(inner, "tactile_encoder"):
        state["tactile_encoder"] = inner.tactile_encoder.state_dict()
    if hasattr(inner, "tactile_proj"):
        state["tactile_proj"] = inner.tactile_proj.state_dict()
    torch.save(state, path)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
