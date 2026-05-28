#!/usr/bin/env python3
"""
eval_real_robot.py — SO-101 真机统一评测脚本。

支持三种策略: tactile_vla / vla_only / diffusion
参考 lerobot 官方 record_loop 模式。

功能:
  - 多 episode 评测（--n_episodes）
  - 成功率统计（人工按键标记 success/fail）
  - 延迟统计 & JSON 输出
  - 安全保护（max_relative_target + 范围钳位）
  - 可选录像保存

用法:
    python deploy/eval_real_robot.py \
        --model vla_only \
        --ckpt train/ckpt_stage4_smolvla_final.pt \
        --task "grasp the cloth from the box" \
        --n_episodes 10 \
        --max_steps_per_episode 300 \
        --output_json eval_results.json \
        --realsense_serial 243522072793
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

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action, build_inference_frame
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

from deploy.safety import ActionSafetyWrapper, sanity_check_action, validate_action_range

OBS_STR = "observation"
ACTION_STR = "action"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INFERENCE_HZ = 30          # 与采数据 fps=30 对齐
MAX_RELATIVE_TARGET = 10.0
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLLOWER_CALIBRATION_DIR = REPO_ROOT / "calibration" / "robots" / "so101_follower"
DEFAULT_FOLLOWER_CALIBRATION_ID = "so101_follower_arm"
DEFAULT_FOLLOWER_CALIBRATION_FPATH = DEFAULT_FOLLOWER_CALIBRATION_DIR / f"{DEFAULT_FOLLOWER_CALIBRATION_ID}.json"


def resolve_follower_calibration(calibration_file: str | None) -> tuple[Path, str, Path]:
    if calibration_file:
        calibration_path = Path(calibration_file).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = (REPO_ROOT / calibration_path).resolve()
        else:
            calibration_path = calibration_path.resolve()
    else:
        calibration_path = DEFAULT_FOLLOWER_CALIBRATION_FPATH

    return calibration_path.parent, calibration_path.stem, calibration_path


def _override_smolvla_features(policy, state_dim: int = 6, action_dim: int = 6):
    """将 SmolVLA base model 的 camera1/2/3 features 替换为训练时实际使用的 feature 名。

    SmolVLA base 预定义 input_features 包含 camera1, camera2, camera3，
    但训练使用的数据集 feature 是 realsense_rgb + side。
    必须在创建 preprocessor 之前调用。
    """
    from lerobot.configs.types import PolicyFeature, FeatureType

    policy.config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        "observation.images.realsense_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified evaluation for SO-101 real-robot deployment")
    p.add_argument("--model", type=str, required=True, choices=["tactile_vla", "vla_only", "diffusion"])
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (.pt or directory)")
    p.add_argument("--task", type=str, default="", help="Language task instruction (VLA models)")
    p.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    p.add_argument("--follower_calibration_file", type=str, default=None,
                   help="Override follower calibration json file")
    p.add_argument("--cam_side", type=int, default=0)
    p.add_argument("--realsense_serial", type=str, default="243522072793",
                   help="RealSense D435i serial number")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_relative_target", type=float, default=MAX_RELATIVE_TARGET)

    # 评测参数
    p.add_argument("--n_episodes", type=int, default=10, help="Number of evaluation episodes")
    p.add_argument("--max_steps_per_episode", type=int, default=300)
    p.add_argument("--output_json", type=str, default="eval_results.json")
    p.add_argument("--warmup_steps", type=int, default=5,
                   help="Steps to skip before timing (warm up GPU cache)")
    p.add_argument("--fps", type=int, default=30,
                   help="Inference frequency (should match training dataset fps)")
    p.add_argument("--skip_calibration", action="store_true",
                   help="Skip motor calibration (reuse existing from data collection)")

    # 录像回放
    p.add_argument("--record_dataset", action="store_true",
                   help="Record evaluation episodes as LeRobotDataset for video replay")
    p.add_argument("--record_repo_id", type=str, default="natsuu/eval_replay",
                   help="Dataset repo_id for recorded evaluation episodes")
    p.add_argument("--record_root", type=str, default=None,
                   help="Local root dir for recorded dataset (default: ~/.cache/...)")

    # 触觉（仅 tactile_vla）
    p.add_argument("--tactile_mock", action="store_true")
    p.add_argument("--tactile_mock_mode", type=str, default="zero", choices=["zero", "random"])
    p.add_argument("--tactile_left_port", type=str, default="/dev/ttyUSB0")
    p.add_argument("--tactile_right_port", type=str, default="/dev/ttyUSB1")
    p.add_argument("--tactile_baudrate", type=int, default=115200)

    # Diffusion 参数
    p.add_argument("--state_dim", type=int, default=6)
    p.add_argument("--action_dim", type=int, default=6)
    return p.parse_args()


def make_robot(args: argparse.Namespace):
    """创建机器人（与 test.sh 采数据相机配置对齐）。

    tactile_vla 模式使用 TactileSO101Robot（自动连接触觉传感器），
    其他模式使用普通 SO101Follower。
    """
    follower_calibration_dir, follower_calibration_id, follower_calibration_fpath = resolve_follower_calibration(
        args.follower_calibration_file
    )

    if not follower_calibration_fpath.is_file():
        raise FileNotFoundError(
            f"Follower calibration file not found: {follower_calibration_fpath}"
        )

    cameras = {
        "realsense_rgb": RealSenseCameraConfig(
            serial_number_or_name=args.realsense_serial,
            width=640, height=480, fps=30,
            use_depth=False,
            warmup_s=3,
        ),
        "side": OpenCVCameraConfig(
            index_or_path=args.cam_side, width=640, height=480, fps=30,
        ),
    }

    if args.model == "tactile_vla":
        from lerobot.common.robot_devices.robots.tactile_so101 import TactileSO101Robot
        from lerobot.robots.so_follower.config_so_follower import TactileSensorConfig

        config = SO101FollowerConfig(
            id=follower_calibration_id,
            calibration_dir=follower_calibration_dir,
            port=args.robot_port,
            use_degrees=True,
            max_relative_target=args.max_relative_target,
            cameras=cameras,
        )
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
    else:
        config = SO101FollowerConfig(
            id=follower_calibration_id,
            calibration_dir=follower_calibration_dir,
            port=args.robot_port,
            use_degrees=True,
            max_relative_target=args.max_relative_target,
            cameras=cameras,
        )
        robot = SO101Follower(config)

    logger.info("Using follower calibration file: %s", follower_calibration_fpath)
    robot.connect(calibrate=not args.skip_calibration)
    return robot


def load_model(args: argparse.Namespace, device: torch.device, robot=None):
    """
    加载策略模型 + 预/后处理器。

    Args:
        robot: 已连接的 robot 实例（tactile_vla 模式需要，用于获取传感器引用）

    Returns:
        (policy, preprocess, postprocess, cleanup_fn)
        cleanup_fn: 在评测结束时调用（关闭触觉读取器等资源）
    """
    cleanup_fns = []

    if args.model == "vla_only":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        base_id = "lerobot/smolvla_base"
        policy = SmolVLAPolicy.from_pretrained(base_id)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        vla_sd = {k: v for k, v in ckpt["model"].items() if not k.startswith("tactile_")}
        policy.model.load_state_dict(vla_sd, strict=False)
        _override_smolvla_features(policy, state_dim=args.state_dim, action_dim=args.action_dim)
        policy.to(device).eval()

        preprocess, postprocess = make_pre_post_processors(
            policy.config, base_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

    elif args.model == "tactile_vla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from deploy.tactile_modules import load_tactile_modules
        from deploy.tactile_sensor_reader import TactileSensorReader
        from deploy.deploy_tactile_vla import patch_embed_prefix

        base_id = "lerobot/smolvla_base"
        policy = SmolVLAPolicy.from_pretrained(base_id)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        vla_sd = {k: v for k, v in ckpt["model"].items() if not k.startswith("tactile_")}
        policy.model.load_state_dict(vla_sd, strict=False)
        _override_smolvla_features(policy, state_dim=args.state_dim, action_dim=args.action_dim)
        policy.to(device).eval()

        encoder, projector = load_tactile_modules(args.ckpt, device)

        # 使用真实传感器或 mock
        if args.tactile_mock:
            reader = TactileSensorReader(mock=True, mock_mode=args.tactile_mock_mode)
        else:
            # robot 已在 make_robot 中连接，复用其传感器引用
            reader = TactileSensorReader(
                left_sensor=robot.tactile_sensor_left,
                right_sensor=robot.tactile_sensor_right,
            )
        reader.start()
        cleanup_fns.append(reader.stop)

        patch_embed_prefix(policy, encoder, projector, reader, device)

        preprocess, postprocess = make_pre_post_processors(
            policy.config, base_id,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

    elif args.model == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        ckpt_path = Path(args.ckpt)
        if ckpt_path.is_dir():
            policy = DiffusionPolicy.from_pretrained(str(ckpt_path))
        else:
            # state_dict 模式
            from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
            from lerobot.configs.types import PolicyFeature, FeatureType

            config = DiffusionConfig(
                n_obs_steps=2,
                horizon=16,
                n_action_steps=8,
                input_features={
                    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
                    "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
                    "observation.images.realsense_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
                },
                output_features={
                    "action": PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,)),
                },
            )
            policy = DiffusionPolicy(config)
            sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            policy.load_state_dict(sd, strict=False)

        policy.to(device).eval()

        pretrained_path = str(ckpt_path) if ckpt_path.is_dir() else None
        preprocess, postprocess = make_pre_post_processors(
            policy.config, pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    def cleanup():
        for fn in cleanup_fns:
            fn()

    return policy, preprocess, postprocess, cleanup


def ask_success() -> bool:
    """交互式询问当前 episode 是否成功。"""
    while True:
        resp = input("\n>>> Episode result? [s]uccess / [f]ail / [r]etry: ").strip().lower()
        if resp in ("s", "success"):
            return True
        if resp in ("f", "fail"):
            return False
        if resp in ("r", "retry"):
            return None  # type: ignore[return-value]
        print("Please enter 's', 'f', or 'r'.")


@torch.no_grad()
def run_episode(
    policy,
    robot,
    preprocess,
    postprocess,
    dataset_features: dict,
    args: argparse.Namespace,
    episode_idx: int,
    dataset: LeRobotDataset | None = None,
) -> dict:
    """运行单个评测 episode，返回统计信息。

    如果传入 dataset，每一步都会通过 build_dataset_frame + add_frame 记录
    RGB / depth / state / action，episode 结束后调用 save_episode 保存。
    """
    device = torch.device(args.device)
    interval = 1.0 / args.fps
    latencies: list[float] = []
    step = 0
    action_dim = args.action_dim
    is_vla = args.model in ("vla_only", "tactile_vla")
    task_str = args.task or ""

    policy.reset()
    safe_robot = ActionSafetyWrapper(robot, max_relative_target=args.max_relative_target)

    logger.info("-" * 40)
    logger.info("Episode %d/%d starting...", episode_idx + 1, args.n_episodes)

    # === 实时角度监控器 ===
    print("\n🔍 [调试模式] 正在实时监控关节角度 (请手动调整机械臂)")
    print("💡 调整到你满意的姿态后，按【Ctrl + C】退出监控，即可开始执行模型！\n")

    try:
        while True:
            obs = robot.get_observation()
            gripper_ang = obs.get("gripper.pos", 0.0)
            elbow_ang = obs.get("elbow_flex.pos", 0.0)
            shoulder_pan = obs.get("shoulder_pan.pos", 0.0)
            shoulder_lift = obs.get("shoulder_lift.pos", 0.0)
            wrist_flex = obs.get("wrist_flex.pos", 0.0)
            wrist_roll = obs.get("wrist_roll.pos", 0.0)

            print(
                f"\r  肩旋:{shoulder_pan:7.2f}° 肩抬:{shoulder_lift:7.2f}° "
                f"肘:{elbow_ang:7.2f}° 腕弯:{wrist_flex:7.2f}° "
                f"腕旋:{wrist_roll:7.2f}° 夹爪:{gripper_ang:5.1f}  ",
                end="", flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n✅ 角度调整完毕！准备启动模型推理...")
        time.sleep(0.5)
    # ======================

    logger.info("Running...")

    t_episode_start = time.monotonic()

    while step < args.max_steps_per_episode:
        t_start = time.monotonic()

        raw_obs = safe_robot.get_observation()

        # 如果需要录像，先构建 observation frame（在 build_inference_frame 之前，用原始 numpy obs）
        if dataset is not None:
            obs_frame = build_dataset_frame(dataset_features, raw_obs, prefix=OBS_STR)

        # build_inference_frame: build_dataset_frame (组装 motor floats → state array, 提取 image keys)
        #                       → prepare_observation_for_inference (numpy→tensor, HWC→CHW, 归一化)
        task_arg = args.task if is_vla else None
        observation = build_inference_frame(
            observation=copy(raw_obs),
            device=device,
            ds_features=dataset_features,
            task=task_arg,
            robot_type="so_follower",
        )

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if device.type == "cuda" else nullcontext(),
        ):
            observation = preprocess(observation)
            t_inf = time.monotonic()
            action = policy.select_action(observation)
            inf_latency = time.monotonic() - t_inf
            action = postprocess(action)

        # 记录有效推理延迟（排除从 queue 弹出的 cached action）
        if inf_latency > 0.005:
            latencies.append(inf_latency)

        if not sanity_check_action(action, action_dim):
            logger.error("Unsafe action at step %d, stopping episode", step)
            break

        robot_action = make_robot_action(action, dataset_features)
        robot_action = validate_action_range(robot_action)
        safe_robot.send_action(robot_action)

        # 录制数据帧（跟官方 record_loop 一致）
        if dataset is not None:
            action_frame = build_dataset_frame(dataset_features, robot_action, prefix=ACTION_STR)
            frame = {**obs_frame, **action_frame, "task": task_str}
            dataset.add_frame(frame)

        step += 1

        elapsed = time.monotonic() - t_start
        if (sleep_t := interval - elapsed) > 0:
            time.sleep(sleep_t)

    episode_duration = time.monotonic() - t_episode_start
    avg_latency = np.mean(latencies) * 1000 if latencies else 0.0

    logger.info("Episode %d done: %d steps in %.1f s, %d inferences, avg latency %.1f ms",
                episode_idx + 1, step, episode_duration, len(latencies), avg_latency)

    # 询问结果
    success = ask_success()
    if success is None:
        # retry — 清空当前 episode 缓冲
        if dataset is not None:
            dataset.clear_episode_buffer()
        return {"retry": True}

    # 保存 episode 到数据集
    if dataset is not None:
        dataset.save_episode()
        logger.info("Episode %d recorded to dataset.", episode_idx + 1)

    return {
        "episode": episode_idx + 1,
        "success": success,
        "total_steps": step,
        "duration_s": round(episode_duration, 2),
        "n_inferences": len(latencies),
        "avg_latency_ms": round(avg_latency, 2),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # 先连接机器人（tactile_vla 模式需要 robot 来获取传感器引用）
    robot = make_robot(args)
    logger.info("Robot connected.")

    # 加载模型
    logger.info("Loading model: %s", args.model)
    policy, preprocess, postprocess, cleanup = load_model(args, device, robot=robot)
    logger.info("Model loaded.")

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # 可选：创建评测录像数据集（用于回看视频 / 复盘动作效果）
    dataset: LeRobotDataset | None = None
    if args.record_dataset:
        from pathlib import Path as _Path
        record_root = _Path(args.record_root) if args.record_root else None
        dataset = LeRobotDataset.create(
            args.record_repo_id,
            fps=args.fps,
            root=record_root,
            robot_type=robot.robot_type,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=4 * len(robot.cameras),
            )
        logger.info("Recording enabled → %s (local: %s)", args.record_repo_id, dataset.root)

    # 评测循环
    results: list[dict] = []
    episode_idx = 0

    try:
        while episode_idx < args.n_episodes:
            ep_result = run_episode(
                policy, robot, preprocess, postprocess,
                dataset_features, args, episode_idx,
                dataset=dataset,
            )
            if ep_result.get("retry"):
                logger.info("Retrying episode %d...", episode_idx + 1)
                continue

            results.append(ep_result)
            episode_idx += 1

            # 实时打印统计
            successes = sum(1 for r in results if r["success"])
            logger.info("Running tally: %d/%d success (%.0f%%)",
                        successes, len(results), 100 * successes / len(results))

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
    finally:
        if dataset is not None:
            dataset.finalize()
            logger.info("Dataset finalized: %s (%d episodes)", dataset.root, dataset.num_episodes)
        cleanup()
        robot.disconnect()

    # 汇总统计
    if results:
        successes = sum(1 for r in results if r["success"])
        all_latencies = [r["avg_latency_ms"] for r in results if r["avg_latency_ms"] > 0]
        summary = {
            "model": args.model,
            "task": args.task,
            "checkpoint": args.ckpt,
            "n_episodes": len(results),
            "n_success": successes,
            "success_rate": round(successes / len(results), 4),
            "avg_steps_per_episode": round(np.mean([r["total_steps"] for r in results]), 1),
            "avg_duration_s": round(np.mean([r["duration_s"] for r in results]), 2),
            "avg_inference_latency_ms": round(np.mean(all_latencies), 2) if all_latencies else 0,
            "episodes": results,
        }

        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("  Model:        %s", summary["model"])
        logger.info("  Task:         %s", summary["task"])
        logger.info("  Episodes:     %d", summary["n_episodes"])
        logger.info("  Success rate: %.1f%% (%d/%d)", summary["success_rate"] * 100, successes, len(results))
        logger.info("  Avg latency:  %.1f ms", summary["avg_inference_latency_ms"])
        logger.info("=" * 50)

        # 保存结果
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        logger.info("Results saved to: %s", output_path)
    else:
        logger.warning("No episodes completed.")


if __name__ == "__main__":
    main()
