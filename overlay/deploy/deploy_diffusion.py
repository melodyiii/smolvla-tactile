#!/usr/bin/env python3
"""
deploy_diffusion.py — Diffusion Policy 真机推理部署。

使用 lerobot DiffusionPolicy（ResNet18 + U-Net DDPM），适用于
单任务 image-conditioned 操控。

架构:
    观测: n_obs_steps=2 帧 (state + images)
    输出: horizon=16, 执行 n_action_steps=8 步后重新推理
    扩散: DDPM 100 步 → 推理时 num_inference_steps 可调

DiffusionPolicy 自带 action queue 管理（select_action 自动弹出），
每 n_action_steps(=8) 个 step 才做一次扩散推理。

用法:
    python deploy/deploy_diffusion.py \
        --ckpt_dir outputs/diffusion_so101/ \
        --robot_port /dev/ttyACM0 \
        --realsense_serial 243522072793

    # 或从 config 创建 + 加载 state_dict:
    python deploy/deploy_diffusion.py \
        --ckpt_state_dict outputs/diffusion_so101/model.pt \
        --state_dim 6 --action_dim 6 \
        --robot_port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import copy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from deploy.safety import ActionSafetyWrapper, sanity_check_action, validate_action_range

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# 默认参数
# ─────────────────────────────────────────────────────────────────
INFERENCE_HZ = 30          # 与采数据 fps=30 对齐
MAX_RELATIVE_TARGET = 10.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy Diffusion Policy on SO-101")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt_dir", type=str, help="Directory with pretrained DiffusionPolicy (HF format)")
    g.add_argument("--ckpt_state_dict", type=str, help="Raw state_dict .pt path (requires --state_dim/--action_dim)")

    p.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    p.add_argument("--cam_side", type=int, default=0, help="Side camera index (OpenCV)")
    p.add_argument("--realsense_serial", type=str, default="243522072793",
                   help="RealSense D435i serial number")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_relative_target", type=float, default=MAX_RELATIVE_TARGET)

    # 构建 config 时使用（仅 --ckpt_state_dict 模式）
    p.add_argument("--state_dim", type=int, default=6)
    p.add_argument("--action_dim", type=int, default=6)
    p.add_argument("--n_obs_steps", type=int, default=2)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=None,
                   help="Reverse diffusion steps (default: num_train_timesteps=100)")
    p.add_argument("--img_height", type=int, default=96)
    p.add_argument("--img_width", type=int, default=96)
    p.add_argument("--resize", type=int, nargs=2, default=None,
                   help="Resize images to (H, W) before backbone, e.g. --resize 96 96")
    return p.parse_args()


def make_robot(args: argparse.Namespace) -> SO101Follower:
    """创建 SO-101 follower 机器人（与 test.sh 采数据相机配置对齐）。"""
    config = SO101FollowerConfig(
        port=args.robot_port,
        use_degrees=True,
        max_relative_target=args.max_relative_target,
        cameras={
            "realsense_rgb": RealSenseCameraConfig(
                serial_number_or_name=args.realsense_serial,
                width=args.img_width, height=args.img_height, fps=30,
                use_depth=False,
            ),
            "side": OpenCVCameraConfig(
                index_or_path=args.cam_side, width=args.img_width, height=args.img_height, fps=30,
            ),
        },
    )
    robot = SO101Follower(config)
    robot.connect()
    return robot


def load_policy_from_dir(ckpt_dir: str, device: torch.device) -> DiffusionPolicy:
    """从 HuggingFace format 目录加载 DiffusionPolicy。"""
    logger.info("Loading DiffusionPolicy from: %s", ckpt_dir)
    policy = DiffusionPolicy.from_pretrained(ckpt_dir)
    policy.to(device)
    policy.eval()
    return policy


def load_policy_from_state_dict(
    state_dict_path: str,
    args: argparse.Namespace,
    device: torch.device,
) -> DiffusionPolicy:
    """
    从裸 state_dict 创建 DiffusionPolicy。

    需要手动指定 input/output features 维度。
    """
    logger.info("Creating DiffusionPolicy from config + state_dict: %s", state_dict_path)

    resize_shape = tuple(args.resize) if args.resize else None
    config = DiffusionConfig(
        n_obs_steps=args.n_obs_steps,
        horizon=args.horizon,
        n_action_steps=args.n_action_steps,
        num_inference_steps=args.num_inference_steps,
        resize_shape=resize_shape,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
            "observation.images.side": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, args.img_height, args.img_width),
            ),
            "observation.images.realsense_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, args.img_height, args.img_width),
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,)),
        },
    )
    policy = DiffusionPolicy(config)

    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    # 支持直接 state_dict 或 {"model": state_dict} 格式
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Missing keys: %d (first 5: %s)", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys: %d (first 5: %s)", len(unexpected), unexpected[:5])

    policy.to(device)
    policy.eval()
    return policy


@torch.no_grad()
def inference_loop(
    policy: DiffusionPolicy,
    robot,
    preprocess,
    postprocess,
    dataset_features: dict,
    args: argparse.Namespace,
) -> dict:
    """
    主推理循环。

    DiffusionPolicy.select_action() 内部自动管理 action chunk queue：
    - 每 n_action_steps(=8) 步做一次扩散推理
    - 中间步骤从 queue 弹出缓存的 action
    """
    device = torch.device(args.device)
    interval = 1.0 / INFERENCE_HZ
    latencies: list[float] = []
    step = 0
    action_dim = args.action_dim

    policy.reset()
    safe_robot = ActionSafetyWrapper(robot, max_relative_target=args.max_relative_target)

    n_action_steps = policy.config.n_action_steps
    logger.info("=" * 50)
    logger.info("Starting Diffusion Policy inference @ %d Hz", INFERENCE_HZ)
    logger.info("Diffusion: horizon=%d, n_action_steps=%d, n_obs_steps=%d",
                policy.config.horizon, n_action_steps, policy.config.n_obs_steps)
    logger.info("Max steps: %d | Safety limit: %.1f°/step", args.max_steps, args.max_relative_target)
    logger.info("=" * 50)

    while step < args.max_steps:
        t_start = time.monotonic()

        # 1) 读取观测
        raw_obs = safe_robot.get_observation()

        # 2) 准备推理帧
        observation = copy(raw_obs)
        observation = prepare_observation_for_inference(
            observation=observation,
            device=device,
            robot_type="so_follower",
        )

        # 3) 预处理 + 推理 + 后处理
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if device.type == "cuda" else nullcontext(),
        ):
            observation = preprocess(observation)

            t_inf = time.monotonic()
            action = policy.select_action(observation)
            inf_latency = time.monotonic() - t_inf

            action = postprocess(action)

        # 扩散推理较慢（~50-200ms），但只有每 n_action_steps 次才真正推理
        if inf_latency > 0.01:
            latencies.append(inf_latency)

        # 4) 安全检查 + 发送
        if not sanity_check_action(action, action_dim):
            logger.error("Stopping due to unsafe action at step %d", step)
            break

        robot_action = make_robot_action(action, dataset_features)
        robot_action = validate_action_range(robot_action)
        safe_robot.send_action(robot_action)

        step += 1
        if step % 50 == 0:
            avg_lat = np.mean(latencies[-10:]) * 1000 if latencies else 0
            logger.info("Step %d/%d | diffusion inferences: %d | avg latency: %.1f ms",
                        step, args.max_steps, len(latencies), avg_lat)

        elapsed = time.monotonic() - t_start
        if (sleep_t := interval - elapsed) > 0:
            time.sleep(sleep_t)

    avg_latency = np.mean(latencies) * 1000 if latencies else 0.0
    logger.info("Done: %d steps, %d diffusion calls, avg diffusion latency %.1f ms",
                step, len(latencies), avg_latency)
    return {"total_steps": step, "n_diffusion_calls": len(latencies), "avg_diffusion_latency_ms": avg_latency}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # 加载策略
    if args.ckpt_dir:
        policy = load_policy_from_dir(args.ckpt_dir, device)
    else:
        policy = load_policy_from_state_dict(args.ckpt_state_dict, args, device)

    # 创建预/后处理器
    pretrained_path = args.ckpt_dir if args.ckpt_dir else None
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # 连接机器人
    logger.info("Connecting to SO-101 on %s ...", args.robot_port)
    robot = make_robot(args)
    logger.info("Robot connected.")

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
