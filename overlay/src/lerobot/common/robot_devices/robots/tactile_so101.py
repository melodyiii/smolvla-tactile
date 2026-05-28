from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import logging

from lerobot.common.robot_devices.sensors.tactile import TactileSensor
from lerobot.robots.so_follower import SO101Follower


class TactileSO101Robot(SO101Follower):
    """
    带有 16x16 触觉阵列传感器的 SO-101 Manipulator 包装类。

    支持单触觉或左右双触觉：
    - 单触觉：配置中设置 ``tactile``（port/baudrate 等），观测键为 ``observation.tactile``。
    - 双触觉：配置中设置 ``tactile_left`` 与 ``tactile_right``（各自 port/baudrate），
      观测键为 ``observation.tactile_left``、``observation.tactile_right``。
    """

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)

        # Tactile visualization only (does not change raw tactile acquisition values)
        self._tactile_vis_size = 256
        self._tactile_vis_gain = 1.6

        tactile_left_cfg = getattr(config, "tactile_left", None)
        tactile_right_cfg = getattr(config, "tactile_right", None)
        tactile_cfg = getattr(config, "tactile", None)

        self._dual_tactile = tactile_left_cfg is not None and tactile_right_cfg is not None

        if self._dual_tactile:
            self.tactile_sensor_left = TactileSensor(tactile_left_cfg)
            self.tactile_sensor_right = TactileSensor(tactile_right_cfg)
            self.tactile_sensor = None
        elif tactile_cfg is not None:
            self.tactile_sensor = TactileSensor(config)
            self.tactile_sensor_left = None
            self.tactile_sensor_right = None
        else:
            self.tactile_sensor = None
            self.tactile_sensor_left = None
            self.tactile_sensor_right = None

    def connect(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        base_connected = False
        left_connected = False
        right_connected = False
        single_connected = False
        try:
            super().connect(*args, **kwargs)
            base_connected = True
            if self._dual_tactile:
                self.tactile_sensor_left.connect()
                left_connected = True
                self.tactile_sensor_right.connect()
                right_connected = True
            elif self.tactile_sensor is not None:
                self.tactile_sensor.connect()
                single_connected = True
        except Exception:
            # Best-effort rollback to avoid leaked serial/camera resources.
            if right_connected:
                self.tactile_sensor_right.disconnect()
            if left_connected:
                self.tactile_sensor_left.disconnect()
            if single_connected:
                self.tactile_sensor.disconnect()
            if base_connected:
                super().disconnect()
            raise

    @property
    def observation_features(self) -> dict[str, Any]:  # type: ignore[override]
        # Tactile sensors are saved as raw .npy sidecars only - no video stream.
        return dict(super().observation_features)

    def get_observation(self) -> dict[str, Any]:  # type: ignore[override]
        obs = super().get_observation()

        def _to_colormap_3c_from_raw(raw_16x16: np.ndarray) -> np.ndarray:
            # Use raw tactile values for visualization, only applying display gain/normalization.
            vis = np.clip(raw_16x16.astype(np.float32) * self._tactile_vis_gain, 0.0, 255.0).astype(np.uint8)
            colormap = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
            return cv2.resize(
                colormap,
                (self._tactile_vis_size, self._tactile_vis_size),
                interpolation=cv2.INTER_NEAREST,
            )

        if self._dual_tactile:
            raw_l = self.tactile_sensor_left.get_raw_frame()
            raw_r = self.tactile_sensor_right.get_raw_frame()
            # Only sidecar raw npy - no video stream added to obs.
            obs["__tactile_raw_left__"] = raw_l
            obs["__tactile_raw_right__"] = raw_r
        elif self.tactile_sensor is not None:
            raw = self.tactile_sensor.get_raw_frame()
            obs["__tactile_raw__"] = raw
        return obs

    def disconnect(self) -> None:  # type: ignore[override]
        disconnect_errors: list[Exception] = []
        if self._dual_tactile:
            try:
                self.tactile_sensor_left.disconnect()
            except Exception as e:
                disconnect_errors.append(e)
            try:
                self.tactile_sensor_right.disconnect()
            except Exception as e:
                disconnect_errors.append(e)
        elif self.tactile_sensor is not None:
            try:
                self.tactile_sensor.disconnect()
            except Exception as e:
                disconnect_errors.append(e)
        try:
            super().disconnect()
        except Exception as e:
            disconnect_errors.append(e)

        if disconnect_errors:
            logging.warning(
                "Disconnect completed with %d non-fatal error(s): %s",
                len(disconnect_errors),
                "; ".join(str(e) for e in disconnect_errors),
            )

