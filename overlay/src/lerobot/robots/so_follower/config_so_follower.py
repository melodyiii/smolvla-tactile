#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class TactileSensorConfig:
    """Optional config for 16x16 tactile sensor (used when robot.type=so101_tactile)."""

    port: str | None = None
    baudrate: int = 2_000_000
    threshold: float = 12.0
    noise_scale: float = 60.0


@dataclass
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True

    # Tactile sensor(s) (only required when using TactileSO101Robot / robot.type=so101_tactile)
    # Single sensor (legacy): set tactile. Dual left/right: set tactile_left and tactile_right.
    tactile: TactileSensorConfig | None = None
    tactile_left: TactileSensorConfig | None = None
    tactile_right: TactileSensorConfig | None = None


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@RobotConfig.register_subclass("so101_tactile")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    pass


SO100FollowerConfig = SOFollowerRobotConfig
SO101FollowerConfig = SOFollowerRobotConfig
