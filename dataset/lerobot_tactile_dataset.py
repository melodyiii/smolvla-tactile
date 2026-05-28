"""
dataset/lerobot_tactile_dataset.py

包装 LeRobotDataset，注入 sidecar 触觉数据 + 深度单通道转换。

数据格式（你的实际采集结构）：
  主视频流（LeRobot Parquet + video）:
    observation.images.realsense_rgb   : video, (480, 640, 3), uint8
    observation.images.realsense_depth : video, (480, 640, 3), uint8  ← np.repeat 3ch
    observation.images.side            : video, (480, 640, 3), uint8

  Sidecar（.npy 文件，不在主流）:
    <dataset_root>/tactile_raw_left/episode-XXXXXX/frame-XXXXXX.npy   float32, (16,16)
    <dataset_root>/tactile_raw_right/episode-XXXXXX/frame-XXXXXX.npy  float32, (16,16)
    <dataset_root>/depth_mm/episode-XXXXXX/frame-XXXXXX.npy           uint16,  (480,640)

本模块做三件事：
  1. 代理 LeRobotDataset.__getitem__()，正常返回 RGB/depth/side/action/text
  2. 按 episode_index + frame_index 从 sidecar 读触觉 .npy，
     拼成 [T, 2, 16, 16]（左右双传感器），插入到 batch dict
  3. 将 3ch depth 转为 1ch（取第一通道），供 DepthEncoder 使用

使用方式：
    from dataset.lerobot_tactile_dataset import LeRobotTactileDataset

    ds = LeRobotTactileDataset(
        repo_id="your_org/your_dataset",
        root="/path/to/local/dataset",
        delta_timestamps={...},          # 同 LeRobotDataset
        sidecar_root="/path/to/local/dataset",
    )
    sample = ds[0]
    # sample["tactile_grid"]        : [T, 2, 16, 16]  float32  左+右
    # sample["depth_1ch"]           : [T, 1, H, W]    float32  单通道深度
    # 其余键与 LeRobotDataset 完全一致
"""

import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

# 兼容 lerobot 新旧版本路径
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        LeRobotDataset = None


class LeRobotTactileDataset(Dataset):
    """
    在 LeRobotDataset 基础上注入 sidecar 触觉 + 深度单通道。

    Args:
        repo_id:           LeRobot 数据集 ID（HuggingFace Hub 或本地）
        root:              数据集本地根目录
        delta_timestamps:  时序窗口配置（传给 LeRobotDataset）
        sidecar_root:      sidecar 文件根目录（包含 tactile_raw_left/ 等）
                           若为 None，则从 root 推断
        tactile_keys:      左右触觉 sidecar 子目录名
        window_T:          时序窗口帧数（与 delta_timestamps 一致）
        fps:               采集帧率（Hz）
        depth_video_key:   主流中 3ch 深度的键名
        has_right_tactile: 是否有右触觉传感器（False 则只用左，右侧用零填充）
    """

    def __init__(
        self,
        repo_id: str,
        root: str = None,
        delta_timestamps: dict = None,
        sidecar_root: str = None,
        tactile_keys: tuple = ("tactile_raw_left", "tactile_raw_right"),
        window_T: int = 16,
        fps: float = 20.0,
        depth_video_key: str = "observation.images.realsense_depth",
        has_right_tactile: bool = True,
    ):
        super().__init__()

        # 底层 LeRobotDataset
        if LeRobotDataset is None:
            raise ImportError(
                "无法导入 LeRobotDataset。请安装兼容版本的 lerobot，"
                "或改用 overfit.dataset.LeRobotTactileDataset（支持纯本地回退）。"
            )
        self.base_ds = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            delta_timestamps=delta_timestamps,
        )

        self.sidecar_root = Path(sidecar_root or root)
        self.tactile_left_dir = self.sidecar_root / tactile_keys[0]
        self.tactile_right_dir = self.sidecar_root / tactile_keys[1]
        self.has_right_tactile = has_right_tactile
        self.depth_video_key = depth_video_key
        self.window_T = window_T
        self.fps = fps

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> dict:
        # 1. 从 LeRobotDataset 取主流数据
        sample = self.base_ds[idx]

        # 2. 解析当前样本的 episode_index 与 frame_index
        ep_idx = sample.get("episode_index", 0)
        frame_idx = sample.get("frame_index", sample.get("index", idx))

        # 如果是标量 tensor 转 int
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()
        if isinstance(frame_idx, torch.Tensor):
            frame_idx = frame_idx.item()

        # 3. 读取 sidecar 触觉序列 [T, 2, 16, 16]
        tactile_grid = self._load_tactile_window(int(ep_idx), int(frame_idx))
        sample["tactile_grid"] = tactile_grid  # [T, 2, 16, 16]

        # 4. 将 3ch depth 转 1ch（若存在）
        if self.depth_video_key in sample:
            depth_3ch = sample[self.depth_video_key]  # [T, 3, H, W] 或 [3, H, W]
            if depth_3ch.dim() == 4:
                # [T, 3, H, W] -> [T, 1, H, W]，取第一通道
                sample["depth_1ch"] = depth_3ch[:, 0:1, :, :]
            elif depth_3ch.dim() == 3:
                # [3, H, W] -> [1, H, W]
                sample["depth_1ch"] = depth_3ch[0:1, :, :]

        return sample

    def _load_tactile_window(self, ep_idx: int, frame_idx: int) -> torch.Tensor:
        """
        加载以 frame_idx 为终点、向前 window_T 帧的触觉序列。

        返回: [T, 2, 16, 16]  float32
              channel 0 = left, channel 1 = right
        """
        frames = []
        for t_offset in range(self.window_T - 1, -1, -1):
            # 向前取帧：frame_idx - t_offset
            fi = max(0, frame_idx - t_offset)

            left = self._read_npy(self.tactile_left_dir, ep_idx, fi)     # [16, 16]

            if self.has_right_tactile:
                right = self._read_npy(self.tactile_right_dir, ep_idx, fi)  # [16, 16]
            else:
                right = np.zeros_like(left)

            # stack 左右 -> [2, 16, 16]
            frame = np.stack([left, right], axis=0)
            frames.append(frame)

        # [T, 2, 16, 16]
        tactile_seq = np.stack(frames, axis=0).astype(np.float32)

        # min-max 归一化到 [0, 1]
        mn, mx = tactile_seq.min(), tactile_seq.max()
        if mx - mn > 1e-8:
            tactile_seq = (tactile_seq - mn) / (mx - mn)

        return torch.from_numpy(tactile_seq)

    @staticmethod
    def _read_npy(base_dir: Path, ep_idx: int, frame_idx: int) -> np.ndarray:
        """
        读取 sidecar .npy。支持两种命名约定：
          约定 A: tactile_raw_left/episode-000000/frame-000000.npy
          约定 B: tactile_raw_left/episode_000/frame_000.npy
        """
        # 约定 A（LeRobot 默认 sidecar 格式）
        path_a = base_dir / f"episode-{ep_idx:06d}" / f"frame-{frame_idx:06d}.npy"
        if path_a.exists():
            return np.load(path_a)

        # 约定 B（简化格式）
        path_b = base_dir / f"episode_{ep_idx:03d}" / f"frame_{frame_idx:03d}.npy"
        if path_b.exists():
            return np.load(path_b)

        # 回退：返回全零矩阵（静默处理缺失帧，不阻塞训练）
        return np.zeros((16, 16), dtype=np.float32)
