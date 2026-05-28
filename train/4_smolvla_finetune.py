"""
train/4_smolvla_finetune.py

Stage 4: SmolVLA 微调（端到端动作预测）

数据流：
  LeRobotTactileDataset
    ├─ observation.images.realsense_rgb  [T, 3, 480, 640]  → SmolVLM vision encoder（冻结）
    ├─ observation.images.side           [T, 3, 480, 640]  → SmolVLM 额外视角（可选）
    ├─ depth_1ch                         [T, 1, 480, 640]  → 本脚本不用（Stage 3 已对齐）
    ├─ tactile_grid                      [T, 2, 16, 16]    → DualTactileGridEncoder → Projector
    ├─ language_instruction              str                → SmolVLM text encoder（冻结）
    └─ action                            [action_dim]       → MSE 监督目标

模型：
  TactileVLAAdapter（overfit/models.py）
    ├─ SmolVLM（冻结）: 提取 vision + text tokens
    ├─ DualTactileGridEncoder（可加载 Stage 3 权重）: 提取触觉 token
    ├─ TactileMLPProjector: 映射到 VLA hidden_size
    ├─ concat: [vision_tokens, tactile_tokens] → LLM backbone
    └─ ActionHead: mean-pool → Linear → action_pred [B, action_dim]

训练策略：
  - SmolVLM 全冻结
  - tactile_encoder: 可加载 Stage 3 预训练权重，小学习率微调
  - projector + action_head: 正常学习率

运行方式：
  # 单卡 dummy VLA（快速调试）
  python train/4_smolvla_finetune.py --data_path ./data/inboxpicking-01 --use_dummy_vla

  # 多卡 LoRA 微调
  accelerate launch --config_file multigpu_config.yaml train/4_smolvla_finetune.py \
    --data_path ./data/inboxpicking-01 --use_lora --lora_rank 16 --batch_size 8 --epochs 40
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from overfit.dataset import LeRobotTactileDataset
from overfit.models import TactileVLAAdapter


# ============================================================================
# 1. 辅助函数
# ============================================================================

def tensor_to_pil(rgb_tensor: torch.Tensor) -> Image.Image:
    """
    [3, H, W] float [0,1] → PIL.Image
    """
    arr = (rgb_tensor.float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)  # CHW → HWC
    return Image.fromarray(arr)


def batch_images_to_pil(
    rgb_batch: torch.Tensor,
    take_frame: int = -1,
) -> list:
    """
    rgb_batch: [B, T, 3, H, W] or [B, 3, H, W]
    取第 take_frame 帧（默认最后一帧），转 PIL list。
    """
    if rgb_batch.dim() == 5:
        rgb_batch = rgb_batch[:, take_frame]  # [B, 3, H, W]
    return [tensor_to_pil(rgb_batch[i]) for i in range(rgb_batch.shape[0])]


def batch_dual_images_to_pil(
    hand_batch: torch.Tensor,
    pano_batch: torch.Tensor,
    take_frame: int = -1,
) -> list:
    """
    hand_batch: [B, T, 3, H, W] or [B, 3, H, W]  (手眼视角)
    pano_batch: [B, T, 3, H, W] or [B, 3, H, W]  (全景视角)

    返回: List[List[PIL.Image]]，每个样本两个视角 [hand_eye, panoramic]
    """
    hand_list = batch_images_to_pil(hand_batch, take_frame=take_frame)
    pano_list = batch_images_to_pil(pano_batch, take_frame=take_frame)
    return [[hand_list[i], pano_list[i]] for i in range(len(hand_list))]


# ============================================================================
# 2. Stage 3 权重加载
# ============================================================================

def load_pretrained_tactile(model: TactileVLAAdapter, ckpt_path: str, device: str):
    """
    加载 Stage 3 预训练的触觉编码器权重到 TactileVLAAdapter.tactile_encoder。

    Stage 3 checkpoint 保存的是 DepthGuidedTLVModel 的 state_dict，
    其中 tactile_encoder.* 前缀对应我们需要的权重。

    DualTactileGridEncoder 内部的 encoder 就是 TactileEncoder，
    键名匹配: encoder.cnn.0.weight ↔ tactile_encoder.cnn.0.weight（加前缀映射）
    """
    if not os.path.exists(ckpt_path):
        print(f"[Stage4] 预训练权重不存在: {ckpt_path}，触觉编码器随机初始化。")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    s3_dict = ckpt.get("model", ckpt)

    # 收集 Stage 3 的 tactile_encoder.* 权重
    # 支持两种 checkpoint 格式：
    #   A) full_final: 扁平 state_dict  {"tactile_encoder.cnn.0.weight": ..., ...}
    #   B) depth_final: 嵌套 dict       {"tactile_encoder": OrderedDict({...}), ...}
    tac_weights = {}

    if "tactile_encoder" in s3_dict and isinstance(s3_dict["tactile_encoder"], dict):
        # 格式 B: depth_final 嵌套 dict → 直接展开添加 encoder. 前缀
        for k, v in s3_dict["tactile_encoder"].items():
            tac_weights[f"encoder.{k}"] = v
    else:
        # 格式 A: full_final 扁平 state_dict
        for k, v in s3_dict.items():
            if k.startswith("tactile_encoder."):
                # Stage 3 的 tactile_encoder 就是 TactileEncoder
                # DualTactileGridEncoder 里叫 self.encoder = TactileEncoder(...)
                new_k = k.replace("tactile_encoder.", "encoder.", 1)
                tac_weights[new_k] = v

    if not tac_weights:
        print("[Stage4] checkpoint 中无 tactile_encoder 权重，跳过。")
        return

    msg = model.tactile_encoder.load_state_dict(tac_weights, strict=False)
    print(f"[Stage4] 加载 {len(tac_weights)} 个触觉编码器参数层。")
    if msg.missing_keys:
        print(f"  missing keys（可能是右路 encoder_right）: {msg.missing_keys[:5]}...")


def load_resume_checkpoint(model: TactileVLAAdapter, ckpt_path: str, device: str):
    """
    从 Stage 4 checkpoint 续跑。

    当前默认恢复模型权重、epoch 和 best loss。
    如果 checkpoint 中带有 optimizer 状态，则一并恢复；否则优化器重新初始化。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"resume checkpoint 不存在: {ckpt_path}")

    print(f"[Stage4] Resume checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)

    start_epoch = int(ckpt.get("epoch", 0))
    best_loss = float(ckpt.get("loss", float("inf")))
    has_optimizer = "optimizer" in ckpt
    return model, start_epoch, best_loss, ckpt.get("optimizer") if has_optimizer else None


# ============================================================================
# 3. 配置
# ============================================================================

DEFAULT_CFG = """
seed: 42
window_T: 16
fps: 20.0
batch_size: 4
epochs: 40
lr_tactile: 1e-4
lr_adapter: 5e-4
lr_lora: 1e-4
weight_decay: 1e-4
action_dim: 6
n_tactile_tokens: 8
use_dummy_vla: true
amp: true
freeze_tactile: true
"""


def parse_args():
    p = argparse.ArgumentParser(description="Stage 4: SmolVLA Tactile LoRA Finetune")
    p.add_argument("--repo_id",      type=str, default="local/inboxpicking",
                   help="LeRobot 数据集 ID")
    p.add_argument("--data_path",    type=str, default=None,
                   help="本地数据根目录（同时用作 root 和 sidecar_root）")
    p.add_argument("--sidecar_root", type=str, default=None,
                   help="Sidecar 根目录，默认与 data_path 相同")
    p.add_argument("--config",       type=str, default="configs/stage4_vla.yaml")
    p.add_argument("--stage3_ckpt",  type=str, default="outputs/stage3_align/ckpt_stage3_depth_final.pt",
                   help="Stage 3 预训练 checkpoint")
    p.add_argument("--use_dummy_vla", action="store_true", default=False,
                   help="使用 DummyVLA 跳过下载大模型")
    p.add_argument("--vla_model_id", type=str, default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--has_right_tactile", action="store_true", default=True)
    # LoRA 参数
    p.add_argument("--use_lora",     action="store_true", default=False,
                   help="启用 LoRA 微调 LLM backbone")
    p.add_argument("--lora_rank",    type=int, default=16, help="LoRA rank (r)")
    p.add_argument("--lora_alpha",   type=int, default=32, help="LoRA alpha")
    # 训练超参覆盖
    p.add_argument("--learning_rate", type=float, default=None,
                   help="覆盖 lr_lora / lr_adapter")
    p.add_argument("--batch_size",   type=int, default=None, help="覆盖每卡 batch size")
    p.add_argument("--epochs",       type=int, default=None, help="覆盖总 epoch 数")
    p.add_argument("--output_dir",   type=str, default="outputs/stage4_lora",
                   help="权重输出目录")
    p.add_argument("--save_every",   type=int, default=5,
                   help="每 N 个 epoch 保存 checkpoint")
    p.add_argument("--resume_ckpt",  type=str, default=None,
                   help="从已有 Stage 4 checkpoint 继续训练")
    # 向后兼容
    p.add_argument("--root",         type=str, default=None)
    return p.parse_args()


def load_cfg(path: str) -> dict:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(DEFAULT_CFG)
    cfg = yaml.safe_load(open(path))
    cfg["seed"]             = int(cfg.get("seed", 42))
    cfg["window_T"]         = int(cfg.get("window_T", 16))
    cfg["fps"]              = float(cfg.get("fps", 20.0))
    cfg["batch_size"]       = int(cfg.get("batch_size", 4))
    cfg["epochs"]           = int(cfg.get("epochs", 40))
    cfg["lr_tactile"]       = float(cfg.get("lr_tactile", 1e-4))
    cfg["lr_adapter"]       = float(cfg.get("lr_adapter", 5e-4))
    cfg["lr_lora"]          = float(cfg.get("lr_lora", 1e-4))
    cfg["weight_decay"]     = float(cfg.get("weight_decay", 1e-4))
    cfg["action_dim"]       = int(cfg.get("action_dim", 7))
    cfg["n_tactile_tokens"] = int(cfg.get("n_tactile_tokens", 8))
    cfg["use_dummy_vla"]    = str(cfg.get("use_dummy_vla", "true")).lower() in ("1", "true", "yes")
    cfg["amp"]              = str(cfg.get("amp", "false")).lower() in ("1", "true", "yes")
    cfg["freeze_tactile"]   = str(cfg.get("freeze_tactile", "true")).lower() in ("1", "true", "yes")
    return cfg


# ============================================================================
# 4. 主训练循环
# ============================================================================

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    # CLI 参数覆盖配置
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.learning_rate is not None:
        cfg["lr_lora"] = args.learning_rate
        cfg["lr_adapter"] = args.learning_rate

    data_root    = args.data_path or args.root
    sidecar_root = args.sidecar_root or data_root
    use_dummy    = args.use_dummy_vla or cfg["use_dummy_vla"]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ================================================================== #
    # Accelerator
    # ================================================================== #
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.get("amp") else "no",
    )
    device  = accelerator.device
    is_main = accelerator.is_main_process

    if is_main:
        print(f"[Stage4] device={device}  mixed_precision={accelerator.mixed_precision}  "
              f"num_processes={accelerator.num_processes}  "
              f"dummy_vla={use_dummy}  use_lora={args.use_lora}")

    # ================================================================== #
    # 1. 数据集（统一使用 LeRobotTactileDataset）
    # ================================================================== #
    T   = cfg["window_T"]
    fps = cfg["fps"]

    # 支持逗号分隔的多数据集路径
    data_paths = [p.strip() for p in data_root.split(",")] if data_root else []
    if not data_paths:
        raise ValueError("请通过 --data_path 指定数据路径（支持逗号分隔多路径）")

    datasets = []
    for dp in data_paths:
        sr = sidecar_root if (sidecar_root and len(data_paths) == 1) else dp
        sub_ds = LeRobotTactileDataset(
            repo_id=args.repo_id,
            root=dp,
            sidecar_root=sr,
            target_fps=int(fps),
            T=T,
            has_right_tactile=args.has_right_tactile,
        )
        datasets.append(sub_ds)
        if is_main:
            print(f"  ↳ {dp}: {len(sub_ds)} samples")

    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    if is_main:
        print(f"[Stage4] 数据集: {len(ds)} 样本, {len(dl)} batches/epoch")

    # ================================================================== #
    # 2. 模型（含可选 LoRA）
    # ================================================================== #
    model = TactileVLAAdapter(
        vla_model_id=args.vla_model_id,
        tactile_feat_dim=512,
        n_tactile_tokens=cfg["n_tactile_tokens"],
        action_dim=cfg["action_dim"],
        device=str(device),
        use_dummy_vla=use_dummy,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    start_epoch = 0
    best_loss = float("inf")
    resume_optimizer_state = None

    if args.resume_ckpt:
        model, start_epoch, best_loss, resume_optimizer_state = load_resume_checkpoint(
            model, args.resume_ckpt, str(device)
        )

    # 非 resume 时加载 Stage 3 预训练触觉编码器
    if not args.resume_ckpt:
        load_pretrained_tactile(model, args.stage3_ckpt, str(device))

    # 按要求冻结触觉主干
    if cfg["freeze_tactile"]:
        for p in model.tactile_encoder.parameters():
            p.requires_grad_(False)
        if is_main:
            print("[Stage4] 触觉编码器已冻结")

    # ================================================================== #
    # 3. 分组优化器（差异学习率）
    # ================================================================== #
    param_groups = []

    # 触觉编码器（若未冻结）
    tac_params = [p for p in model.tactile_encoder.parameters() if p.requires_grad]
    if tac_params:
        param_groups.append({
            "params": tac_params,
            "lr": cfg["lr_tactile"],
            "name": "tactile_encoder",
        })

    # projector + action_head
    param_groups.append({
        "params": list(model.projector.parameters()) + list(model.action_head.parameters()),
        "lr": cfg["lr_adapter"],
        "name": "adapter_head",
    })

    # LoRA 参数
    if args.use_lora and model.llm_backbone is not None:
        lora_params = [p for p in model.llm_backbone.parameters() if p.requires_grad]
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": cfg["lr_lora"],
                "name": "lora_adapter",
            })

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])
    criterion = nn.MSELoss()

    # ================================================================== #
    # 4. Accelerate prepare
    # ================================================================== #
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            if is_main:
                print("[Stage4] 已恢复 optimizer 状态")
        except Exception as exc:
            if is_main:
                print(f"[Stage4] 恢复 optimizer 状态失败，改为仅恢复模型继续训练: {exc}")

    # ================================================================== #
    # 5. 参数统计
    # ================================================================== #
    if is_main:
        n_tac  = sum(p.numel() for p in model.parameters() if p.requires_grad and any(
            p is pp for pp in (accelerator.unwrap_model(model).tactile_encoder.parameters())))
        n_proj = sum(p.numel() for p in accelerator.unwrap_model(model).projector.parameters())
        n_head = sum(p.numel() for p in accelerator.unwrap_model(model).action_head.parameters())
        n_lora = sum(p.numel() for g in param_groups if g.get("name") == "lora_adapter" for p in g["params"])
        print(f"[Stage4] 可训练参数: tactile={n_tac:,}  proj={n_proj:,}  head={n_head:,}  lora={n_lora:,}")

    # ================================================================== #
    # 6. 训练循环
    # ================================================================== #
    output_dir = args.output_dir
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        model.train()
        # VLA 主体始终 eval（冻结 BN/Dropout）
        uw = accelerator.unwrap_model(model)
        uw.vla.eval()

        total_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{cfg['epochs']}", leave=False, disable=not is_main)

        for batch in pbar:
            dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
            # --- 触觉 [B, T, 2, 16, 16] ---
            tactile = batch["tactile_grid"].to(device, dtype=dtype)

            # --- 双视角 RGB ---
            hand_rgb = batch["side_rgb"].to(device, dtype=dtype)
            pano_rgb = batch["realsense_rgb"].to(device, dtype=dtype)
            images_pil = batch_dual_images_to_pil(hand_rgb, pano_rgb, take_frame=-1)

            # --- 文本指令 ---
            texts = batch.get("language_instruction", ["grasp the cloth"] * tactile.shape[0])
            if isinstance(texts, torch.Tensor):
                texts = [str(t) for t in texts]

            # --- 动作标签 ---
            action_gt = batch["action"].to(device, dtype=dtype)
            if action_gt.dim() == 3:
                action_gt = action_gt[:, 0, :]

            with accelerator.autocast():
                action_pred = model(
                    images=images_pil,
                    texts=list(texts),
                    tactile_grids=tactile,
                )
                loss = criterion(action_pred, action_gt)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_loss = total_loss / max(len(dl), 1)
        if is_main:
            print(f"[Stage4] Epoch {epoch}/{cfg['epochs']}  avg_loss={avg_loss:.6f}")

        # --- 保存最优 ---
        if is_main and avg_loss < best_loss:
            best_loss = avg_loss
            uw = accelerator.unwrap_model(model)
            torch.save(
                {
                    "epoch": epoch,
                    "model": uw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": best_loss,
                },
                os.path.join(output_dir, "ckpt_stage4_vla_best.pt"),
            )
            print(f"  → 最优 checkpoint (loss={best_loss:.6f})")

        # --- 周期保存 ---
        if is_main and (epoch % args.save_every == 0):
            uw = accelerator.unwrap_model(model)
            ckpt_path = os.path.join(output_dir, f"ckpt_stage4_ep{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": uw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": best_loss,
                },
                ckpt_path,
            )
            print(f"  → Checkpoint: {ckpt_path}")

    # ================================================================== #
    # 7. 最终保存
    # ================================================================== #
    if is_main:
        uw = accelerator.unwrap_model(model)
        final_path = os.path.join(output_dir, "ckpt_stage4_vla_final.pt")
        torch.save(
            {
                "epoch": cfg["epochs"],
                "model": uw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": best_loss,
            },
            final_path,
        )
        print(f"[Stage4] 训练完成！最终权重: {final_path}")


if __name__ == "__main__":
    main()
