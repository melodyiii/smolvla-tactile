"""
TactileSensorReader — 触觉传感器滑动窗口读取器。

维护一个 window_T=16 帧的滑动窗口缓冲区，每次调用 get_window() 返回
[1, 16, 2, 16, 16] 的 tensor (batch, time, left/right, H, W)。

支持 mock 模式用于无硬件测试。
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import torch


class TactileSensorReader:
    """双触觉传感器滑动窗口读取器。"""

    WINDOW_T = 16        # 时间窗口长度
    GRID_H = 16          # 压力网格高度
    GRID_W = 16          # 压力网格宽度
    READ_HZ = 50         # 传感器原始读取频率

    def __init__(
        self,
        left_sensor=None,
        right_sensor=None,
        mock: bool = False,
        mock_mode: str = "zero",  # "zero" | "random"
    ) -> None:
        """
        Args:
            left_sensor:  TactileSensor 实例（左手）
            right_sensor: TactileSensor 实例（右手）
            mock:         True 时忽略实际传感器，生成模拟数据
            mock_mode:    mock=True 时的数据模式（"zero" 或 "random"）
        """
        self.mock = mock
        self.mock_mode = mock_mode
        self.left_sensor = left_sensor
        self.right_sensor = right_sensor

        # 滑动窗口：每个元素为 (left_16x16, right_16x16)
        self._buffer: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.WINDOW_T)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 后台采集线程
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动后台采集线程。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止后台采集线程。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _read_loop(self) -> None:
        interval = 1.0 / self.READ_HZ
        while self._running:
            t0 = time.monotonic()
            left, right = self._read_once()
            with self._lock:
                self._buffer.append((left, right))
            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _read_once(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mock:
            if self.mock_mode == "random":
                left = np.random.rand(self.GRID_H, self.GRID_W).astype(np.float32)
                right = np.random.rand(self.GRID_H, self.GRID_W).astype(np.float32)
            else:
                left = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
                right = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float32)
            return left, right

        left = self.left_sensor.get_raw_frame().astype(np.float32)
        right = self.right_sensor.get_raw_frame().astype(np.float32)
        return left, right

    # ------------------------------------------------------------------
    # 手动推入（非后台模式下，可由外部循环调用）
    # ------------------------------------------------------------------

    def push(self, left: np.ndarray, right: np.ndarray) -> None:
        """手动将一帧触觉数据推入缓冲区。"""
        with self._lock:
            self._buffer.append((left.astype(np.float32), right.astype(np.float32)))

    # ------------------------------------------------------------------
    # 获取当前窗口
    # ------------------------------------------------------------------

    def get_window(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """
        返回 [1, 16, 2, 16, 16] 的 tensor。

        如果缓冲区不满 16 帧，左侧用零填充。
        """
        with self._lock:
            frames = list(self._buffer)

        n = len(frames)
        window = np.zeros((self.WINDOW_T, 2, self.GRID_H, self.GRID_W), dtype=np.float32)

        if n > 0:
            # 右对齐：最新帧放在末尾
            start = self.WINDOW_T - n
            for i, (left, right) in enumerate(frames):
                window[start + i, 0] = left
                window[start + i, 1] = right

        tensor = torch.from_numpy(window).unsqueeze(0)  # [1, 16, 2, 16, 16]
        return tensor.to(device)

    def clear(self) -> None:
        """清空缓冲区。"""
        with self._lock:
            self._buffer.clear()
