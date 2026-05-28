import threading
import time
from typing import Any

import numpy as np
import serial
import torch

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]


class TactileSensor:
    """
    16x16 触觉阵列传感器的串口驱动封装。

    该类被设计为与 LeRobot 兼容的独立传感器模块，负责：
    - 串口连接与后台读取线程
    - 上电后若干帧的中值基线估计
    - 去基线、阈值裁剪与归一化
    - 简单的时间滤波（指数滑动平均）
    """

    def __init__(self, config: Any) -> None:
        """
        Args:
            config: 一般为 Hydra/LeRobot 的 robot config。
                - 必须字段/推荐字段：
                  - id: 机器人/传感器实例 id
                  - port / baudrate (或 baud)：串口配置
                  - threshold / noise_scale: 触觉预处理参数
                - 也支持在 `config.tactile` 小节中定义上述字段。
        """

        self.id = getattr(config, "id", "tactile_sensor")

        # 兼容两种写法：
        # 1) 直接在 config 上写 port/baudrate
        # 2) 在 config.tactile 子配置中写这些字段（config.tactile 为 None 时退回 config）
        _tactile = getattr(config, "tactile", None)
        sensor_cfg = _tactile if _tactile is not None else config

        self.port: str | None = getattr(sensor_cfg, "port", None)
        self.baudrate: int = int(
            getattr(sensor_cfg, "baudrate", getattr(sensor_cfg, "baud", 2_000_000)),
        )

        self.threshold: float = float(getattr(sensor_cfg, "threshold", 12.0))
        self.noise_scale: float = float(getattr(sensor_cfg, "noise_scale", 60.0))

        # 为了兼容之前代码中用到的字段名
        self.THRESHOLD = self.threshold
        self.NOISE_SCALE = self.noise_scale
        self.BAUD = self.baudrate

        # 数据缓存
        # raw_frame：串口直接读到的 16x16 原始压力值（int 转 float）
        # contact_data_norm / output_frame：后处理用的归一化 & 时间滤波结果，只用于可视化等
        self.raw_frame = np.zeros((16, 16), dtype=np.float32)
        self.contact_data_norm = np.zeros((16, 16), dtype=np.float32)
        self.prev_frame = np.zeros((16, 16), dtype=np.float32)
        self.output_frame = np.zeros((16, 16), dtype=np.float32)

        # 状态标志
        self._initialized = False  # 完成基线标定
        self._is_running = False
        self._thread: threading.Thread | None = None
        self._median = np.zeros((16, 16), dtype=np.float32)
        self._frame_lock = threading.Lock()
        self.raw_timestamp: float = 0.0

        self._ser: serial.Serial | None = None

    # ---------------------------------------------------------------------
    # 串口与后台读取
    # ---------------------------------------------------------------------
    def _read_loop(self) -> None:
        """后台串口读取与数据处理线程。"""
        assert self._ser is not None

        data_tac: list[np.ndarray] = []
        num = 0
        current: list[list[int]] = []

        def _finish_frame() -> None:
            nonlocal num, data_tac, current
            if not current or len(current) != 16:
                return

            backup = np.array(current, dtype=np.float32)
            # 始终保留一份「原始压力矩阵」供上层读取
            with self._frame_lock:
                self.raw_frame = backup
                self.raw_timestamp = time.perf_counter()

            if not self._initialized:
                # 初始化阶段：收集 30 帧计算中值作为静态基线
                data_tac.append(backup)
                num += 1
                if num >= 30:
                    self._median = np.median(np.stack(data_tac, axis=0), axis=0)
                    self._initialized = True
            else:
                # 正常运行阶段：去基线 + 阈值裁剪 + 归一化 + 时间滤波
                processed = backup - self._median - self.THRESHOLD
                processed = np.clip(processed, 0.0, 100.0)

                if np.max(processed) < self.THRESHOLD:
                    self.contact_data_norm = processed / max(self.NOISE_SCALE, 1e-6)
                else:
                    self.contact_data_norm = processed / (float(np.max(processed)) + 1e-6)

                # 简单的指数滑动平均（alpha=0.2）
                self.output_frame = 0.2 * self.contact_data_norm + 0.8 * self.prev_frame
                self.prev_frame = self.output_frame

        while self._is_running:
            if self._ser.in_waiting <= 0:
                time.sleep(0.001)
                continue

            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                # 解码失败时丢弃该行
                continue

            if len(line) < 10:
                # 一帧结束：16 行采集完成（兼容空行作为分隔符）
                _finish_frame()
                current = []
                continue

            # 逐行累积 16x16 整型原始数据
            try:
                str_values = line.split()
                if len(str_values) == 16:
                    # 若已经累积了完整一帧，则先结束上一帧，再开始新一帧
                    if len(current) == 16:
                        _finish_frame()
                        current = []
                    current.append([int(val) for val in str_values])
            except ValueError:
                continue

    # ---------------------------------------------------------------------
    # 公共接口
    # ---------------------------------------------------------------------
    def connect(self, timeout_s: float = 30.0) -> None:
        """建立串口连接并启动后台读取线程。"""
        if not self.port:
            raise ValueError("错误：未在配置文件中指定触觉传感器的 port 路径！")

        print(f"正在连接触觉传感器，端口: {self.port}")

        self._ser = serial.Serial(self.port, self.baudrate, timeout=1)
        self._ser.reset_input_buffer()

        self._is_running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        # 等待初始化完成（基线估计）
        start_time = time.time()
        while not self._initialized:
            time.sleep(0.05)
            if time.time() - start_time > timeout_s:
                self.disconnect()
                raise TimeoutError("触觉传感器初始化超时，请检查数据流。")

    def disconnect(self) -> None:
        """停止后台线程并关闭串口。"""
        self._is_running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # LeRobot 期望的接口形式（松耦合的传感器观测定义）
    # ------------------------------------------------------------------
    @property
    def observation_features(self) -> dict[str, Any]:
        """
        返回该传感器产生的观测特征定义。

        这里只描述 16x16 的单通道触觉地图，命名为 ``tactile_map``。
        上层 Robot 可以选择将其挂到 ``observation.tactile`` 等具体键上。
        """

        return {
            "tactile_map": {
                "shape": (16, 16, 1),
                "dtype": "uint8",
                "names": ["height", "width"],
            }
        }

    def get_observation(self) -> dict[str, torch.Tensor]:
        """
        获取当前帧的触觉观测值（原始压力矩阵）。

       
        数值对应串口读取的原始压力值（仅做 dtype 转换，不做归一化）。
        """

        with self._frame_lock:
            raw_copy = self.raw_frame.copy()
        data_uint8 = np.clip(raw_copy, 0, 255).astype(np.uint8)
            
        frame = torch.from_numpy(data_uint8).unsqueeze(2).clone()
        return {"tactile_map": frame}

    def get_raw_frame(self) -> np.ndarray:
        """Return latest raw tactile matrix (16x16, float32)."""
        with self._frame_lock:
            return self.raw_frame.copy()

    # ------------------------------------------------------------------
    # 可选：调试可视化
    # ------------------------------------------------------------------
    def visualize(self, window_name: str = "LeRobot Tactile") -> None:
        """使用 OpenCV 热力图实时可视化触觉数据（仅调试用）。"""
        if cv2 is None:
            return

        if not self._initialized:
            return

        scaled = np.clip(self.output_frame, 0.0, 1.0)
        scaled = (scaled * 255.0).astype(np.uint8)
        colormap = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
        cv2.imshow(window_name, colormap)
        cv2.waitKey(1)


# 向后兼容：老代码中可能还在使用 TactileRobot 名称
TactileRobot = TactileSensor

