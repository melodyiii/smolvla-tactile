"""
train/4_smolvla_policy.py

Stage 4: 基于 SmolVLA 策略的触觉融合训练

使用真正的 SmolVLA 策略（flow matching action expert + 50-step action chunking），
将 Stage 3 预训练的触觉编码器注入 SmolVLA 的 prefix embedding 中。

架构:
  SmolVLA (SmolVLM2-500M VLM + 100M flow matching action expert)
    ├─ SigLIP 视觉编码器（冻结）: 编码相机图像
    ├─ SmolLM2 语言模型（冻结）: 编码语言指令
    ├─ DualTactileGridEncoder（冻结/微调）: 编码触觉序列 → 512-dim
    ├─ TactileMLPProjector（可训练）: 512 → n_tokens × 768
    ├─ state_proj（可训练）: 6-dim → 768
    └─ Action Expert（可训练）: flow matching, chunk_size=50

训练策略:
  - VLM (SigLIP + SmolLM2): 冻结
  - Action Expert (~100M params): 可训练, lr=1e-4
  - state_proj: 可训练, lr=1e-4
  - tactile_proj: 可训练, lr=5e-4
  - tactile_encoder: 默认冻结, 可选 lr=1e-5 微调

运行:
  # 单卡（debug）
  python train/4_smolvla_policy.py \\
    --data_path ./data/inboxpicking-01 \\
    --stage3_ckpt outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt \\
    --epochs 1 --batch_size 2

  # 多卡
  accelerate launch --config_file multigpu_config.yaml \\
    train/4_smolvla_policy.py \\
    --data_path ./data/inboxpicking-01,./data/inboxpicking-02 \\
    --stage3_ckpt outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt \\
    --epochs 40 --batch_size 4 --save_every 5
"""

import os
import sys
import argparse
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from accelerate import Accelerator

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# 配置
# ============================================================================

DEFAULT_CFG = {
    "seed": 42,
    "target_fps": 10,
    "window_T": 16,
    "batch_size": 4,
    "epochs": 40,
    "num_workers": 4,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "action_dim": 6,
    "state_dim": 6,
    "chunk_size": 50,
    "n_tactile_tokens": 8,
    "tokenizer_max_length": 48,
    "lr_expert": 1e-4,
    "lr_tactile_proj": 5e-4,
    "lr_tactile_encoder": 1e-5,
    "lr_state_proj": 1e-4,
    "weight_decay": 1e-10,
    "grad_clip_norm": 10.0,
    "amp": True,
    "freeze_tactile": True,
    "pretrained": "lerobot/smolvla_base",
    "save_every": 5,
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage 4: SmolVLA + Tactile Policy Training")
    p.add_argument("--data_path", type=str, required=True,
                   help="数据根目录（逗号分隔多路径）")
    p.add_argument("--config", type=str, default="configs/stage4_vla.yaml")
    p.add_argument("--pretrained", type=str, default=None,
                   help="SmolVLA 预训练模型路径 (默认 lerobot/smolvla_base)")
    p.add_argument("--stage3_ckpt", type=str,
                   default="outputs/exp002_stage3_full/ckpt_stage3_depth_final.pt",
                   help="Stage 3 触觉编码器 checkpoint")
    p.add_argument("--n_tactile_tokens", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader worker 数；4 卡时建议先从 4/进程起步")
    p.add_argument("--lr", type=float, default=None,
                   help="统一学习率（覆盖所有 param groups）")
    p.add_argument("--freeze_tactile", action="store_true", default=None,
                   help="冻结触觉编码器")
    p.add_argument("--unfreeze_tactile", action="store_true", default=False,
                   help="解冻触觉编码器微调")
    p.add_argument("--output_dir", type=str, default="outputs/stage4_smolvla")
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--repo_id", type=str, default="local/inboxpicking")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="从已有 checkpoint 继续训练")
    p.add_argument("--no_tactile", action="store_true", default=False,
                   help="不使用触觉（VLA-only baseline）")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(path):
        with open(path) as f:
            file_cfg = yaml.safe_load(f)
        if file_cfg:
            cfg.update(file_cfg)
    return cfg


# ============================================================================
# 主训练
# ============================================================================

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # CLI 覆盖配置
    if args.pretrained is not None:
        cfg["pretrained"] = args.pretrained
    if args.n_tactile_tokens is not None:
        cfg["n_tactile_tokens"] = args.n_tactile_tokens
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.save_every is not None:
        cfg["save_every"] = args.save_every
    if args.lr is not None:
        cfg["lr_expert"] = args.lr
        cfg["lr_tactile_proj"] = args.lr
        cfg["lr_state_proj"] = args.lr
    if args.unfreeze_tactile:
        cfg["freeze_tactile"] = False
    elif args.freeze_tactile is not None:
        cfg["freeze_tactile"] = True

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ================================================================== #
    # Accelerator
    # ================================================================== #
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.get("amp") else "no",
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"[Stage4-SmolVLA] device={device}  "
              f"mixed_precision={accelerator.mixed_precision}  "
              f"num_processes={accelerator.num_processes}")

    # ================================================================== #
    # 1. 构建模型
    # ================================================================== #
    no_tactile = args.no_tactile

    if is_main:
        print(f"[Stage4-SmolVLA] 加载模型: {cfg['pretrained']}")
        if no_tactile:
            print("[Stage4-SmolVLA] *** VLA-only baseline（无触觉）***")

    image_keys = [
        "observation.images.side",
        "observation.images.realsense_rgb",
    ]

    if no_tactile:
        # === VLA-only: 直接加载 base SmolVLA ===
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.configuration_smolvla import PolicyFeature, FeatureType

        policy = SmolVLAPolicy.from_pretrained(cfg["pretrained"])
        config = policy.config

        # 重映射 image_features 到我们的数据集相机名
        new_input_features = {}
        for k, v in config.input_features.items():
            if v.type != FeatureType.VISUAL:
                new_input_features[k] = v
        for key in image_keys:
            new_input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 512, 512),
            )
        config.input_features = new_input_features
        if is_main:
            print(f"[Stage4-SmolVLA] image_features 重映射: {list(config.image_features.keys())}")

        # ------------------------------------------------------------------ #
        # Patch VLAFlowMatching.forward to fix bf16/fp32 dtype mismatch
        # (DeepSpeed bf16 mode + lerobot's sample_noise/sample_time hardcode
        #  float32, causing crashes at action_in_proj and action_out_proj)
        # This mirrors the same fix in TactileVLAFlowMatching.
        # ------------------------------------------------------------------ #
        import types as _types
        import torch.nn.functional as _F
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks as _make_att_2d_masks

        def _bf16_safe_vla_forward(
            self, images, img_masks, lang_tokens, lang_masks, state, actions,
            noise=None, time=None
        ):
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            # Cast x_t to model weight dtype to avoid fp32 vs bf16 mismatch
            x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = _make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            (_, suffix_out), _ = self.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            # Cast suffix_out to action_out_proj dtype (lerobot upcasts to fp32)
            suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)
            losses = _F.mse_loss(u_t.float(), v_t.float(), reduction="none")
            return losses

        policy.model.forward = _types.MethodType(_bf16_safe_vla_forward, policy.model)
        if is_main:
            print("[Stage4-SmolVLA] 已 patch VLAFlowMatching.forward 以修复 bf16/fp32 dtype 问题")
    else:
        # === Tactile: 构建 TactileVLAFlowMatching ===
        from models.smolvla_tactile import build_tactile_smolvla

        policy = build_tactile_smolvla(
            pretrained_path=cfg["pretrained"],
            n_tactile_tokens=cfg["n_tactile_tokens"],
            tactile_feat_dim=512,
            stage3_ckpt=args.stage3_ckpt,
            device="cpu",
            image_keys=image_keys,
        )

    # 获取 tokenizer（用于数据集）
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

    # ================================================================== #
    # 2. 数据集
    # ================================================================== #
    from dataset.smolvla_tactile_dataset import SmolVLATactileDataset

    data_paths = [p.strip() for p in args.data_path.split(",")]
    datasets = []

    for dp in data_paths:
        ds = SmolVLATactileDataset(
            data_path=dp,
            tokenizer=tokenizer,
            repo_id=args.repo_id,
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
        print(f"[Stage4-SmolVLA] 数据集: {len(train_ds)} samples, "
              f"{len(dl)} batches/epoch  workers={cfg['num_workers']}  "
              f"persistent={cfg['persistent_workers']}  prefetch={cfg['prefetch_factor']}")

    # ================================================================== #
    # 3. 冻结策略 + 优化器
    # ================================================================== #
    model = policy.model

    # 构建参数组
    param_groups = []

    if no_tactile:
        # === VLA-only: 仅 expert + state_proj ===
        state_proj_ids = set(id(p) for p in model.state_proj.parameters())

        expert_params = []
        for name, p in model.named_parameters():
            if p.requires_grad and id(p) not in state_proj_ids:
                expert_params.append(p)

        if expert_params:
            param_groups.append({
                "params": expert_params,
                "lr": float(cfg["lr_expert"]),
                "name": "expert",
            })

        state_proj_params = [p for p in model.state_proj.parameters() if p.requires_grad]
        if state_proj_params:
            param_groups.append({
                "params": state_proj_params,
                "lr": float(cfg["lr_state_proj"]),
                "name": "state_proj",
            })
    else:
        # === Tactile: expert + state_proj + tactile_proj + tactile_encoder ===

        # 冻结触觉编码器（默认）
        if cfg["freeze_tactile"]:
            for p in model.tactile_encoder.parameters():
                p.requires_grad_(False)
            if is_main:
                print("[Stage4-SmolVLA] 触觉编码器已冻结")

        # (a) Action expert
        expert_params = []
        tactile_param_ids = set(
            id(p) for p in model.tactile_encoder.parameters()
        ) | set(
            id(p) for p in model.tactile_proj.parameters()
        ) | set(
            id(p) for p in model.state_proj.parameters()
        )

        for name, p in model.named_parameters():
            if p.requires_grad and id(p) not in tactile_param_ids:
                expert_params.append(p)

        if expert_params:
            param_groups.append({
                "params": expert_params,
                "lr": float(cfg["lr_expert"]),
                "name": "expert",
            })

        # (b) state_proj
        state_proj_params = [p for p in model.state_proj.parameters() if p.requires_grad]
        if state_proj_params:
            param_groups.append({
                "params": state_proj_params,
                "lr": float(cfg["lr_state_proj"]),
                "name": "state_proj",
            })

        # (c) tactile_proj
        proj_params = list(model.tactile_proj.parameters())
        if proj_params:
            param_groups.append({
                "params": proj_params,
                "lr": float(cfg["lr_tactile_proj"]),
                "name": "tactile_proj",
            })

        # (d) tactile_encoder（若未冻结）
        tac_trainable = [p for p in model.tactile_encoder.parameters() if p.requires_grad]
        if tac_trainable:
            param_groups.append({
                "params": tac_trainable,
                "lr": float(cfg["lr_tactile_encoder"]),
                "name": "tactile_encoder",
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
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for pg in param_groups:
            n = sum(p.numel() for p in pg["params"])
            print(f"  {pg['name']}: {n:,} params, lr={pg['lr']}")
        print(f"[Stage4-SmolVLA] 总参数: {total_params:,}  可训练: {trainable_params:,}")

    # ================================================================== #
    # 6. 训练
    # ================================================================== #
    output_dir = args.output_dir
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    best_loss = float("inf")
    start_epoch = 0

    # Resume
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        uw = accelerator.unwrap_model(policy)
        uw.model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("loss", float("inf"))
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                if is_main:
                    print(f"[Stage4-SmolVLA] 无法恢复 optimizer: {e}")
        if is_main:
            print(f"[Stage4-SmolVLA] 从 epoch {start_epoch} 继续, best_loss={best_loss:.6f}")

    # 确定 batch tensor 需要 cast 到的目标 dtype（DeepSpeed bf16 模式下参数是 bf16，
    # 但 DataLoader 输出 float32；autocast 无法自动转 input，需要手动对齐）
    compute_dtype = torch.bfloat16 if cfg.get("amp") else torch.float32

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        policy.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            dl,
            desc=f"Epoch {epoch}/{cfg['epochs']}",
            leave=False,
            disable=not is_main,
        )

        for batch in pbar:
            uw = accelerator.unwrap_model(policy)

            # --- 将浮点 batch 张量统一转为模型计算 dtype (bf16 or fp32) ---
            batch = {
                k: v.to(dtype=compute_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in batch.items()
            }

            # --- 注入触觉数据（仅 tactile 模式）---
            if not no_tactile:
                uw.model._current_tactile = batch["observation.tactile"]

            # --- Forward ---
            with accelerator.autocast():
                loss, loss_dict = uw(batch)

            # --- Backward ---
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    uw.parameters(), float(cfg["grad_clip_norm"])
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 清除触觉引用
            if not no_tactile:
                uw.model._current_tactile = None

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_loss = total_loss / max(n_batches, 1)
        if is_main:
            print(f"[Stage4-SmolVLA] Epoch {epoch}/{cfg['epochs']}  "
                  f"avg_loss={avg_loss:.6f}")

        # --- 保存最优 ---
        if is_main and avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(
                accelerator, policy, optimizer, epoch, best_loss,
                os.path.join(output_dir, "ckpt_stage4_smolvla_best.pt"),
                no_tactile=no_tactile,
            )
            print(f"  → Best checkpoint (loss={best_loss:.6f})")

        # --- 周期保存 ---
        if is_main and (epoch % cfg["save_every"] == 0):
            _save_checkpoint(
                accelerator, policy, optimizer, epoch, best_loss,
                os.path.join(output_dir, f"ckpt_stage4_ep{epoch}.pt"),
                no_tactile=no_tactile,
            )
            print(f"  → Checkpoint: epoch {epoch}")

    # ================================================================== #
    # 7. 最终保存
    # ================================================================== #
    if is_main:
        final_path = os.path.join(output_dir, "ckpt_stage4_smolvla_final.pt")
        _save_checkpoint(
            accelerator, policy, optimizer, cfg["epochs"], best_loss, final_path,
            no_tactile=no_tactile,
        )
        print(f"[Stage4-SmolVLA] 训练完成！最终权重: {final_path}")


def _save_checkpoint(accelerator, policy, optimizer, epoch, loss, path,
                     no_tactile=False):
    """保存 checkpoint（仅主进程）。"""
    uw = accelerator.unwrap_model(policy)
    save_dict = {
        "epoch": epoch,
        "model": uw.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "no_tactile": no_tactile,
    }
    if not no_tactile:
        save_dict["tactile_encoder"] = uw.model.tactile_encoder.state_dict()
        save_dict["tactile_proj"] = uw.model.tactile_proj.state_dict()
    torch.save(save_dict, path)


if __name__ == "__main__":
    main()
