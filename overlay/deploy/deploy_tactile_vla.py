#!/usr/bin/env python3
"""
deploy_tactile_vla.py — SmolVLA + 触觉传感器真机推理部署。

在 SmolVLA 基础上注入触觉编码器的特征 token，通过 monkey-patch embed_prefix
将触觉 embedding 追加到 VLM 的 prefix 序列中，让 action expert 通过 KV-cache
交叉注意力读取触觉信息。

架构:
    embed_prefix 原始输出: [image_tokens, language_tokens, state_token]
    注入后:                 [image_tokens, language_tokens, state_token, tactile_tokens×8]
    tactile_tokens: TactileGridEncoder(CNN+GRU → 512) → MLP(512 → 7680) → reshape(8, 960)

用法:
    python deploy/deploy_tactile_vla.py \
        --ckpt train/ckpt_stage4_smolvla_final.pt \
        --task "grasp the cloth from the box" \
        --robot_port /dev/ttyACM0 \
        --tactile_left_port /dev/ttyUSB0 \
        --tactile_right_port /dev/ttyUSB1 \
        --realsense_serial 243522072793

    # 无触觉硬件时:
    python deploy/deploy_tactile_vla.py ... --tactile_mock
"""

from __future__ import annotations

import argparse
import logging
import time
import types
from copy import copy
from contextlib import nullcontext

import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.common.robot_devices.robots.tactile_so101 import TactileSO101Robot
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.config_so_follower import TactileSensorConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from deploy.tactile_modules import DualTactileGridEncoder, TactileMLPProjector, load_tactile_modules
from deploy.tactile_sensor_reader import TactileSensorReader
from deploy.safety import ActionSafetyWrapper, sanity_check_action, validate_action_range

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────
INFERENCE_HZ = 30          # 与采数据 fps=30 对齐
ACTION_DIM = 6
BASE_MODEL_ID = "lerobot/smolvla_base"
MAX_RELATIVE_TARGET = 10.0
VLM_HIDDEN_SIZE = 960          # SmolVLM2-500M hidden dim
TACTILE_TOKENS = 8             # 7680 / 960 = 8 tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy SmolVLA + Tactile on SO-101")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt path")
    p.add_argument("--task", type=str, required=True, help="Language task instruction")
    # 硬件端口（默认值与 test.sh 采数据一致）
    p.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    p.add_argument("--cam_side", type=int, default=0, help="Side camera index (OpenCV)")
    p.add_argument("--realsense_serial", type=str, default="243522072793",
                   help="RealSense D435i serial number")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_relative_target", type=float, default=MAX_RELATIVE_TARGET)
    # 触觉传感器端口（默认值与 test.sh 采数据一致）
    p.add_argument("--tactile_left_port", type=str, default="/dev/ttyUSB0")
    p.add_argument("--tactile_right_port", type=str, default="/dev/ttyUSB1")
    p.add_argument("--tactile_baudrate", type=int, default=115200)
    p.add_argument("--tactile_mock", action="store_true", help="Use mock tactile data (no hardware)")
    p.add_argument("--tactile_mock_mode", type=str, default="zero", choices=["zero", "random"])
    return p.parse_args()


def make_robot(args: argparse.Namespace) -> TactileSO101Robot:
    """创建 TactileSO101Robot（与 test.sh 采数据完全对齐）。

    相机配置:
      - depth: RealSense RGBD → 观测: realsense_rgb (H,W,3) + realsense_depth (H,W,3)
      - side:  OpenCV USB 相机 → 观测: side (H,W,3)
    触觉配置:
      - tactile_left / tactile_right → TactileSensor 自动连接（mock 模式下跳过）
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

    # 触觉传感器配置（mock 模式下不配置，TactileSO101Robot 会跳过传感器连接）
    if not args.tactile_mock:
        config.tactile_left = TactileSensorConfig(
            port=args.tactile_left_port,
            baudrate=args.tactile_baudrate,
            threshold=12.0,
            noise_scale=60.0,
        )
        config.tactile_right = TactileSensorConfig(
            port=args.tactile_right_port,
            baudrate=args.tactile_baudrate,
            threshold=12.0,
            noise_scale=60.0,
        )

    robot = TactileSO101Robot(config)
    robot.connect()
    return robot


def load_policy_with_tactile(
    ckpt_path: str,
    device: torch.device,
) -> tuple[SmolVLAPolicy, DualTactileGridEncoder, TactileMLPProjector]:
    """
    加载完整的 fine-tuned 模型（VLA + 触觉模块）。

    checkpoint["model"] 包含 vlm_with_expert.* + action_*.* + state_proj.* + tactile_encoder.* + tactile_proj.*
    checkpoint["tactile_encoder"] 和 checkpoint["tactile_proj"] 是独立存储的触觉模块权重。
    """
    logger.info("Loading SmolVLA base from HuggingFace: %s", BASE_MODEL_ID)
    policy = SmolVLAPolicy.from_pretrained(BASE_MODEL_ID)

    logger.info("Loading fine-tuned weights from: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 加载 VLA 部分（过滤掉 tactile_ 前缀的 key，因为基础模型没有这些子模块）
    model_sd = ckpt["model"]
    vla_sd = {k: v for k, v in model_sd.items() if not k.startswith("tactile_")}
    missing, unexpected = policy.model.load_state_dict(vla_sd, strict=False)
    logger.info("  VLA weights: missing=%d, unexpected=%d", len(missing), len(unexpected))

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

    # 加载触觉模块（使用独立存储的权重）
    encoder, projector = load_tactile_modules(ckpt_path, device)
    logger.info("  Tactile encoder + projector loaded.")
    logger.info("  epoch=%s  loss=%.4f", ckpt.get("epoch", "?"), ckpt.get("loss", float("nan")))

    policy.to(device)
    policy.eval()
    return policy, encoder, projector


def patch_embed_prefix(
    policy: SmolVLAPolicy,
    encoder: DualTactileGridEncoder,
    projector: TactileMLPProjector,
    reader: TactileSensorReader,
    device: torch.device,
    vlm_hidden: int = VLM_HIDDEN_SIZE,
) -> None:
    """
    Monkey-patch VLAFlowMatching.embed_prefix 以注入触觉 token。

    原始 prefix 序列: [img_start, img_embs, img_end, ..., lang_embs, state_emb] → pad to prefix_length
    注入后:           [...same..., tactile_tokens×8]  (追加在 padding 之后)

    触觉 token 使用 att_mask=1（与 state 相同的 causal boundary），
    让 image/language 不 attend 到触觉，但 action expert 通过 KV-cache 读取。

    position_ids = cumsum(pad_masks) - 1:
        padding tokens (pad_mask=0) 共享同一个 position → 不影响
        tactile tokens (pad_mask=1) 获得正确的递增 position
    """
    original_method = policy.model.embed_prefix

    def patched_embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state=None
    ):
        # 调用原始方法
        embs, pad_masks, att_masks = original_method(images, img_masks, lang_tokens, lang_masks, state)

        # 计算触觉 embedding
        tactile_window = reader.get_window(device)  # (1, 16, 2, 16, 16)
        with torch.no_grad():
            tactile_feat = encoder(tactile_window)    # (1, 512)
            tactile_emb = projector(tactile_feat)     # (1, 7680)

        B = embs.shape[0]
        n_tokens = tactile_emb.shape[-1] // vlm_hidden  # 8
        tactile_tokens = tactile_emb.reshape(B, n_tokens, vlm_hidden).to(embs.dtype)

        # 追加到 prefix
        embs = torch.cat([embs, tactile_tokens], dim=1)
        tactile_pad = torch.ones(B, n_tokens, dtype=pad_masks.dtype, device=pad_masks.device)
        pad_masks = torch.cat([pad_masks, tactile_pad], dim=1)
        tactile_att = torch.ones(B, n_tokens, dtype=att_masks.dtype, device=att_masks.device)
        att_masks = torch.cat([att_masks, tactile_att], dim=1)

        return embs, pad_masks, att_masks

    policy.model.embed_prefix = types.MethodType(patched_embed_prefix, policy.model)
    logger.info("Patched embed_prefix: +%d tactile tokens (hidden=%d)", TACTILE_TOKENS, vlm_hidden)


@torch.no_grad()
def inference_loop(
    policy: SmolVLAPolicy,
    robot,
    preprocess,
    postprocess,
    dataset_features: dict,
    args: argparse.Namespace,
) -> dict:
    """主推理循环（与 deploy_vla_only.py 相同结构，触觉注入通过 monkey-patch 自动生效）。"""
    device = torch.device(args.device)
    interval = 1.0 / INFERENCE_HZ
    latencies: list[float] = []
    step = 0

    policy.reset()
    safe_robot = ActionSafetyWrapper(robot, max_relative_target=args.max_relative_target)

    logger.info("=" * 50)
    logger.info("Starting SmolVLA + Tactile inference @ %d Hz", INFERENCE_HZ)
    logger.info("Task: '%s'", args.task)
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
            task=args.task,
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

        if inf_latency > 0.005:
            latencies.append(inf_latency)

        # 4) 安全检查 + 发送
        if not sanity_check_action(action, ACTION_DIM):
            logger.error("Stopping due to unsafe action at step %d", step)
            break

        robot_action = make_robot_action(action, dataset_features)
        robot_action = validate_action_range(robot_action)
        safe_robot.send_action(robot_action)

        step += 1
        if step % 50 == 0:
            avg_lat = np.mean(latencies[-10:]) * 1000 if latencies else 0
            logger.info("Step %d/%d | recent avg latency: %.1f ms", step, args.max_steps, avg_lat)

        elapsed = time.monotonic() - t_start
        if (sleep_t := interval - elapsed) > 0:
            time.sleep(sleep_t)

    avg_latency = np.mean(latencies) * 1000 if latencies else 0.0
    logger.info("Done: %d steps, %d inferences, avg latency %.1f ms", step, len(latencies), avg_latency)
    return {"total_steps": step, "n_inferences": len(latencies), "avg_latency_ms": avg_latency}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # 加载策略 + 触觉模块
    policy, encoder, projector = load_policy_with_tactile(args.ckpt, device)

    # 连接机器人（先于读取器创建，这样可以从 robot 取传感器引用）
    logger.info("Connecting to SO-101 on %s ...", args.robot_port)
    robot = make_robot(args)
    logger.info("Robot connected.")

    # 创建触觉读取器
    if args.tactile_mock:
        reader = TactileSensorReader(mock=True, mock_mode=args.tactile_mock_mode)
    else:
        # 直接复用 TactileSO101Robot 里已连接好的传感器（和采数据时同一个驱动）
        reader = TactileSensorReader(
            left_sensor=robot.tactile_sensor_left,
            right_sensor=robot.tactile_sensor_right,
        )
    reader.start()
    logger.info("Tactile reader started (mock=%s)", args.tactile_mock)

    # Monkey-patch embed_prefix 以注入触觉 token
    patch_embed_prefix(policy, encoder, projector, reader, device)

    # 创建预/后处理器
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        BASE_MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    try:
        stats = inference_loop(policy, robot, preprocess, postprocess, dataset_features, args)
        logger.info("Stats: %s", stats)
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    finally:
        reader.stop()
        robot.disconnect()
        logger.info("Cleanup done.")


if __name__ == "__main__":
    main()
