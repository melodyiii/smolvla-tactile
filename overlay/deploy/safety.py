"""
safety.py — SO-101 机械臂部署安全工具模块。

包含:
- 动作范围验证（绝对位置度数空间）
- 单步限幅（防瞬间乱飞）
- 平滑启动 & 紧急停止
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# SO-101 STS3215 关节安全参数（度数空间）
# ─────────────────────────────────────────────────────────────────────
# 关节名: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# 各关节绝对位置范围（度数）—— 保守估计，避免机械限位碰撞
JOINT_LIMITS_DEG = {
    "shoulder_pan":  (-150.0, 150.0),
    "shoulder_lift": (-150.0, 150.0),
    "elbow_flex":    (-150.0, 150.0),
    "wrist_flex":    (-150.0, 150.0),
    "wrist_roll":    (-150.0, 150.0),
    "gripper":       (0.0, 100.0),      # gripper 使用 0-100 范围
}

# 每步最大相对变化（度数/步）—— 防止模型输出异常导致猛冲
# 在 10Hz 下, 10°/步 = 100°/s，已经很快了
MAX_DELTA_PER_STEP_DEG = {
    "shoulder_pan":  10.0,
    "shoulder_lift": 10.0,
    "elbow_flex":    10.0,
    "wrist_flex":    10.0,
    "wrist_roll":    10.0,
    "gripper":       15.0,
}


def validate_action_range(action_dict: dict[str, float]) -> dict[str, float]:
    """
    将动作值钳位到安全关节范围内。

    如果任何关节超出范围，记录警告并钳位。
    """
    clamped = {}
    for key, val in action_dict.items():
        motor_name = key.removesuffix(".pos")
        if motor_name in JOINT_LIMITS_DEG:
            lo, hi = JOINT_LIMITS_DEG[motor_name]
            safe_val = max(lo, min(val, hi))
            if abs(safe_val - val) > 0.01:
                logger.warning(
                    "SAFETY CLAMP: %s %.2f° → %.2f° (limits [%.1f, %.1f])",
                    motor_name, val, safe_val, lo, hi,
                )
            clamped[key] = safe_val
        else:
            clamped[key] = val
    return clamped


def clip_action_delta(
    action_dict: dict[str, float],
    current_pos: dict[str, float],
    max_delta: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    限制每步最大变化量，防止模型输出异常时机械臂猛冲。

    Args:
        action_dict:  目标关节位置 {"shoulder_pan.pos": 45.0, ...}
        current_pos:  当前关节位置 {"shoulder_pan.pos": 40.0, ...}
        max_delta:    每步最大变化（度数），默认使用 MAX_DELTA_PER_STEP_DEG
    """
    if max_delta is None:
        max_delta = MAX_DELTA_PER_STEP_DEG

    clipped = {}
    for key, goal in action_dict.items():
        motor_name = key.removesuffix(".pos")
        curr = current_pos.get(key, goal)
        delta = goal - curr
        cap = max_delta.get(motor_name, 10.0)
        safe_delta = max(-cap, min(delta, cap))
        clipped[key] = curr + safe_delta

        if abs(safe_delta - delta) > 0.01:
            logger.warning(
                "DELTA CLAMP: %s goal=%.2f° curr=%.2f° delta=%.2f° → %.2f°",
                motor_name, goal, curr, delta, safe_delta,
            )
    return clipped


def sanity_check_action(action_tensor: torch.Tensor, action_dim: int = 6) -> bool:
    """
    推理输出的快速检查。检测明显异常的动作值。

    Returns:
        True = 正常, False = 检测到异常
    """
    if action_tensor.numel() < action_dim:
        logger.error("Action tensor has fewer elements (%d) than expected (%d)", action_tensor.numel(), action_dim)
        return False

    values = action_tensor[:action_dim].detach().cpu().float().numpy()

    # 检查 NaN / Inf
    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        logger.error("Action contains NaN or Inf: %s", values)
        return False

    # 检查异常大的值（度数空间下超过 ±200° 几乎不可能）
    if np.any(np.abs(values) > 200.0):
        logger.warning("Action has suspiciously large values: %s (max abs=%.1f)", values, np.max(np.abs(values)))
        # 不直接拒绝，但发出警告；后续 clamp 会处理

    return True


def smooth_start_check(
    robot,
    target_pos: dict[str, float],
    max_initial_delta: float = 5.0,
) -> bool:
    """
    检查首次动作是否与当前位置差距过大。

    如果差距 > max_initial_delta（度），发出警告。
    用于推理开始前的安全检查。
    """
    try:
        obs = robot.get_observation()
    except Exception as e:
        logger.warning("Cannot read robot state for smooth start check: %s", e)
        return True

    large_deltas = []
    for key, goal in target_pos.items():
        motor_name = key.removesuffix(".pos")
        curr = obs.get(key, obs.get(f"{motor_name}.pos"))
        if curr is not None:
            delta = abs(goal - curr)
            if delta > max_initial_delta:
                large_deltas.append((motor_name, curr, goal, delta))

    if large_deltas:
        logger.warning("=" * 50)
        logger.warning("SMOOTH START WARNING: Large initial deltas detected!")
        for name, curr, goal, delta in large_deltas:
            logger.warning("  %s: current=%.1f° goal=%.1f° delta=%.1f°", name, curr, goal, delta)
        logger.warning("The robot may move quickly. Ensure clear workspace.")
        logger.warning("=" * 50)
        return False
    return True


class ActionSafetyWrapper:
    """
    包装 robot.send_action()，自动添加安全检查。

    使用方法:
        safe_robot = ActionSafetyWrapper(robot, max_relative_target=10.0)
        safe_robot.send_action(action_dict)
    """

    def __init__(
        self,
        robot,
        max_relative_target: float = 10.0,
        enable_range_check: bool = True,
        enable_delta_clip: bool = True,
    ) -> None:
        self.robot = robot
        self.max_relative_target = max_relative_target
        self.enable_range_check = enable_range_check
        self.enable_delta_clip = enable_delta_clip
        self._last_pos: dict[str, float] | None = None

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        # 1) 范围钳位
        if self.enable_range_check:
            action = validate_action_range(action)

        # 2) 单步限幅
        if self.enable_delta_clip and self._last_pos is not None:
            action = clip_action_delta(
                action, self._last_pos,
                {k: self.max_relative_target for k in MAX_DELTA_PER_STEP_DEG},
            )

        # 3) 发送
        sent = self.robot.send_action(action)
        self._last_pos = dict(sent)
        return sent

    def get_observation(self):
        obs = self.robot.get_observation()
        # 更新 last_pos
        if self._last_pos is None:
            self._last_pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
        return obs

    def disconnect(self):
        self.robot.disconnect()

    def __getattr__(self, name):
        return getattr(self.robot, name)
