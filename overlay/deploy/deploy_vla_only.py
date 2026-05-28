#!/usr/bin/env python3
"""
deploy_vla_only.py — SmolVLA（无触觉）真机推理部署。

完全基于 lerobot 官方 predict_action / record_loop 模式。
无需复制训练代码，仅使用 lerobot 官方 API。

用法:
    python deploy/deploy_vla_only.py \
        --ckpt train/ckpt_stage4_smolvla_final.pt \
        --task "grasp the cloth from the box" \
        --robot_port /dev/ttyACM0 \
        --realsense_serial 243522072793
"""

from __future__ import annotations

import argparse
import logging
import time
from copy import copy
from contextlib import nullcontext

import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import (
    build_inference_frame,
    make_robot_action,
    prepare_observation_for_inference,
)
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from deploy.safety import ActionSafetyWrapper, sanity_check_action, validate_action_range

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────
INFERENCE_HZ = 30          # 与采数据 fps=30 对齐
ACTION_DIM = 6
BASE_MODEL_ID = "lerobot/smolvla_base"

# 安全：每步最大变化度数（10Hz 下 10°/步 = 100°/s）
MAX_RELATIVE_TARGET = 10.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy SmolVLA (no tactile) on SO-101")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt path")
    p.add_argument("--task", type=str, required=True, help="Language task instruction")
    # 硬件端口（默认值与 test.sh 采数据一致）
    p.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    p.add_argument("--cam_side", type=int, default=0, help="Side camera index (OpenCV)")
    p.add_argument("--realsense_serial", type=str, default="243522072793",
                   help="RealSense D435i serial number")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_relative_target", type=float, default=MAX_RELATIVE_TARGET,
                   help="Safety: max degrees per step (higher=faster but riskier)")
    return p.parse_args()


def make_robot(args: argparse.Namespace) -> SO101Follower:
    """创建 SO-101 follower 机器人（与 test.sh 采数据相机配置对齐）。

    相机配置:
      - depth: RealSense RGBD → 观测: realsense_rgb (H,W,3) + realsense_depth (H,W,3)
      - side:  OpenCV USB 相机 → 观测: side (H,W,3)
    """
    config = SO101FollowerConfig(
        port=args.robot_port,
        use_degrees=True,
        max_relative_target=args.max_relative_target,
        cameras={
            "realsense_rgb": RealSenseCameraConfig(
                serial_number_or_name=args.realsense_serial,
                width=640, height=480, fps=30,
                use_depth=False,
            ),
            "side": OpenCVCameraConfig(
                index_or_path=args.cam_side, width=640, height=480, fps=30,
            ),
        },
    )
    robot = SO101Follower(config)
    robot.connect()
    return robot


def load_policy(ckpt_path: str, device: torch.device) -> SmolVLAPolicy:
    """
    加载 SmolVLA base + fine-tune checkpoint。

    checkpoint 结构: {"model": state_dict, "epoch": int, "loss": float, ...}
    model state_dict 的 key 前缀是 vlm_with_expert.* / state_proj.* / action_*.* 等，
    对应 policy.model (VLAFlowMatching) 的子模块。
    """
    logger.info("Loading SmolVLA base from HuggingFace: %s", BASE_MODEL_ID)
    policy = SmolVLAPolicy.from_pretrained(BASE_MODEL_ID)

    logger.info("Loading fine-tuned weights from: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # model state_dict 对应 policy.model (VLAFlowMatching)
    # 过滤掉触觉相关的 key（仅加载 VLA 部分）
    model_sd = ckpt["model"]
    vla_sd = {k: v for k, v in model_sd.items() if not k.startswith("tactile_")}

    missing, unexpected = policy.model.load_state_dict(vla_sd, strict=False)
    logger.info("  epoch=%s  loss=%.4f", ckpt.get("epoch", "?"), ckpt.get("loss", float("nan")))
    if missing:
        logger.warning("  Missing keys (expected for base model): %d", len(missing))
    if unexpected:
        logger.warning("  Unexpected keys: %s", unexpected[:5])

    # SmolVLA base 预定义 camera1/2/3 features, 替换为训练实际使用的 feature 名
    from lerobot.configs.types import PolicyFeature, FeatureType
    policy.config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.realsense_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }

    policy.to(device)
    policy.eval()
    return policy


@torch.no_grad()
def inference_loop(
    policy: SmolVLAPolicy,
    robot,
    preprocess,
    postprocess,
    dataset_features: dict,
    args: argparse.Namespace,
) -> dict:
    """
    主推理循环。

    使用 policy.select_action() 自动管理 action chunk queue。
    SmolVLA 输出 50 步 chunk，select_action 逐步弹出。

    动作空间: 绝对关节位置（度数），由 postprocessor 反归一化。
    """
    device = torch.device(args.device)
    interval = 1.0 / INFERENCE_HZ
    latencies: list[float] = []
    step = 0

    policy.reset()
    safe_robot = ActionSafetyWrapper(robot, max_relative_target=args.max_relative_target)

    logger.info("=" * 50)
    logger.info("Starting SmolVLA inference @ %d Hz", INFERENCE_HZ)
    logger.info("Task: '%s'", args.task)
    logger.info("Max steps: %d | Safety limit: %.1f°/step", args.max_steps, args.max_relative_target)
    logger.info("=" * 50)

    while step < args.max_steps:
        t_start = time.monotonic()

        # 1) 读取观测
        raw_obs = safe_robot.get_observation()

        # 2) 构建推理帧（参考 lerobot 官方 predict_action）
        observation = copy(raw_obs)
        observation = prepare_observation_for_inference(
            observation=observation,
            device=device,
            task=args.task,
            robot_type="so_follower",
        )

        # 3) 预处理（归一化 state, tokenize language）
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if device.type == "cuda" else nullcontext(),
        ):
            observation = preprocess(observation)

            # 4) 推理
            t_inf = time.monotonic()
            action = policy.select_action(observation)
            inf_latency = time.monotonic() - t_inf

            # 5) 后处理（反归一化 → 回到度数空间）
            action = postprocess(action)

        # 记录推理延迟（仅在实际做了新推理时）
        if inf_latency > 0.005:
            latencies.append(inf_latency)

        # 6) 安全检查
        if not sanity_check_action(action, ACTION_DIM):
            logger.error("Stopping due to unsafe action at step %d", step)
            break

        # 7) 转换为机器人动作字典 + 发送
        robot_action = make_robot_action(action, dataset_features)
        robot_action = validate_action_range(robot_action)
        safe_robot.send_action(robot_action)

        step += 1

        if step % 50 == 0:
            avg_lat = np.mean(latencies[-10:]) * 1000 if latencies else 0
            logger.info("Step %d/%d | recent avg latency: %.1f ms", step, args.max_steps, avg_lat)

        # 节拍控制
        elapsed = time.monotonic() - t_start
        if (sleep_t := interval - elapsed) > 0:
            time.sleep(sleep_t)

    avg_latency = np.mean(latencies) * 1000 if latencies else 0.0
    logger.info("Done: %d steps, %d inferences, avg latency %.1f ms", step, len(latencies), avg_latency)
    return {"total_steps": step, "n_inferences": len(latencies), "avg_latency_ms": avg_latency}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # 加载策略
    policy = load_policy(args.ckpt, device)

    # 创建预/后处理器（含归一化/反归一化统计量）
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        BASE_MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # 连接机器人
    logger.info("Connecting to SO-101 on %s ...", args.robot_port)
    robot = make_robot(args)
    logger.info("Robot connected. Cameras ready.")

    # 构建 dataset features（用于 obs/action key 映射）
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    try:
        stats = inference_loop(policy, robot, preprocess, postprocess, dataset_features, args)
        logger.info("Stats: %s", stats)
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    finally:
        robot.disconnect()
        logger.info("Robot disconnected.")


if __name__ == "__main__":
    main()
