"""
overfit/dataset.py  -  LeRobot v3.0 + Sidecar 混合数据加载管道

设计原则：
  - 包装器模式：内部实例化官方 LeRobotDataset，绝不手写解帧逻辑
  - 降频策略：原始 30 FPS -> 目标 target_fps (默认 10 FPS)，stride=3 帧取 1 帧
  - 历史窗口：每个 __getitem__ 返回长度 T 的序列，episode 开头不足时首帧填充
  - Sidecar 精准读取：按 episode_index + frame_index 索引触觉 / 深度 .npy
  - 零崩溃保证：sidecar 文件缺失时静默返回 zeros，绝不 crash DataLoader

键名输出规范：
  side_rgb            [T, 3, H, W]    手眼相机 (observation.images.side)
  realsense_rgb       [T, 3, H, W]    全景相机 (observation.images.realsense_rgb)
  depth               [T, 1, H, W]    外挂高精度深度 (depth_mm/episode_*.npy)
  tactile_grid        [T, 2, 16, 16]  双触觉拼接 (left ch0 + right ch1)
  action              [action_dim]    当前窗口最后一帧的 action
  language_instruction  str           文本指令

用法：
    ds = LeRobotTactileDataset(
        repo_id="your_org/chip-moving-03",
        sidecar_root="/data/chip-moving-03",
        target_fps=10,
        T=16,
    )
    batch = next(iter(DataLoader(ds, batch_size=4)))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ============================================================================
# 全局常量
# ============================================================================

SRC_FPS            = 30   # 原始采集帧率
TGT_FPS            = 10   # 默认目标帧率
DEFAULT_T          = 16
DEFAULT_ACTION_DIM = 7
DEFAULT_TAC_H      = 16
DEFAULT_TAC_W      = 16

_DEFAULT_INSTRUCTIONS = [
    "grasp the cloth from the box",
    "pick up the soft fabric",
    "lift the cloth out of the bin",
    "grab the textile material",
    "retrieve the flexible cloth",
]


# ============================================================================
# 主 Dataset
# ============================================================================

class LeRobotTactileDataset(Dataset):
    """
    LeRobot v3.0 + Sidecar 多模态数据加载器。

    Parameters
    ----------
    repo_id : str
        LeRobot 数据集 ID，例如 "your_org/chip-moving-03"。
    root : str | None
        数据集本地根目录，None 则从 HuggingFace Hub 拉取。
    sidecar_root : str | None
        Sidecar 文件根目录（depth_mm/, tactile_raw_left/, tactile_raw_right/
        所在父目录）。为 None 时 sidecar 全部退化为零张量。
    target_fps : int
        降采样目标帧率（默认 10）。stride = round(30 / target_fps)。
    T : int
        历史观测窗口长度（按降采样后帧数计算，默认 16）。
    action_dim : int
        动作向量维度（默认 7-DoF）。
    tac_h / tac_w : int
        触觉矩阵分辨率（默认 16x16）。
    has_right_tactile : bool
        是否有右手触觉传感器（False 时右路全零）。
    use_dummy : bool
        True 时完全使用随机 dummy 数据，跳过 LeRobot 加载。
    n_dummy : int
        use_dummy=True 时的虚拟样本数。
    delta_timestamps : dict | None
        若传入则覆盖自动生成的 delta_timestamps（供 Stage4 向后兼容）。
    fps : float
        向后兼容旧参数名，效果同 target_fps。
    window_T : int | None
        向后兼容旧参数名，效果同 T。
    """

    def __init__(
        self,
        repo_id: str = "your_org/chip-moving-03",
        root: Optional[str] = None,
        sidecar_root: Optional[str] = None,
        target_fps: int = TGT_FPS,
        T: int = DEFAULT_T,
        action_dim: int = DEFAULT_ACTION_DIM,
        tac_h: int = DEFAULT_TAC_H,
        tac_w: int = DEFAULT_TAC_W,
        has_right_tactile: bool = True,
        use_dummy: bool = False,
        n_dummy: int = 50,
        episodes: Optional[list[int]] = None,
        # 向后兼容 Stage4 旧参数名
        delta_timestamps: Optional[dict] = None,
        fps: float = float(TGT_FPS),
        window_T: Optional[int] = None,
    ):
        super().__init__()

        # ---- 向后兼容旧参数名 ----
        if window_T is not None:
            T = window_T
        if fps != float(TGT_FPS) and target_fps == TGT_FPS:
            target_fps = int(fps)

        self.T                 = T
        self.action_dim        = action_dim
        self.tac_h             = tac_h
        self.tac_w             = tac_w
        self.has_right_tactile = has_right_tactile
        self.use_dummy         = use_dummy
        self.sidecar_root      = Path(sidecar_root) if sidecar_root else None
        self.target_fps        = target_fps
        self.episodes          = episodes
        # stride = 原始FPS / 目标FPS (30/10=3)：控制 LeRobot item 之间的最小帧距
        self.stride            = max(1, round(SRC_FPS / target_fps))
        self._video_caps       = {}
        self._video_shapes     = {}
        self._local_mode       = False
        self._reported_local_fallback = False
        self._reported_local_init = False

        # ------------------------------------------------------------------ #
        # Dummy 模式：跳过 LeRobot，快速验证 pipeline 连通性
        # ------------------------------------------------------------------ #
        if use_dummy:
            self._dummy_n   = n_dummy
            self._lerobot   = None
            self._index_map = []
            print(f"[Dataset] Dummy 模式：{n_dummy} 个虚拟样本，stride={self.stride}")
            return

        # ------------------------------------------------------------------ #
        # 真实模式：实例化官方 LeRobotDataset
        # ------------------------------------------------------------------ #
        local_root = self._resolve_local_dataset_root(repo_id, root)

        LeRobotDataset = None
        lerobot_import_error = None
        for import_path in (
            "lerobot.common.datasets.lerobot_dataset",
            "lerobot.datasets.lerobot_dataset",
        ):
            try:
                module = __import__(import_path, fromlist=["LeRobotDataset"])
                LeRobotDataset = getattr(module, "LeRobotDataset")
                break
            except Exception as exc:
                lerobot_import_error = exc

        if LeRobotDataset is None and local_root is None:
            raise ImportError(
                "无法导入兼容的 LeRobotDataset，且未找到本地数据集根目录。\n"
                "请安装兼容版本的 lerobot，或传入包含 meta/info.json 的 root。\n"
                f"最后一次导入错误: {lerobot_import_error}"
            ) from lerobot_import_error

        # 自动构造 delta_timestamps：
        # 以 target_fps 步长，向过去取 T 帧历史窗口的时间偏移（单位：秒）
        # 例：target_fps=10, T=16 -> [-1.5, -1.4, ..., -0.1, 0.0]
        dt = 1.0 / target_fps
        ts = [-(T - 1 - i) * dt for i in range(T)]

        if delta_timestamps is None:
            delta_timestamps = {
                "observation.images.side":            ts,
                "observation.images.realsense_rgb":   ts,
                "observation.images.realsense_depth": ts,
                "action":                             [0.0],
            }

        if LeRobotDataset is not None:
            try:
                kwargs = {
                    "repo_id": repo_id,
                    "root": root,
                    "delta_timestamps": delta_timestamps,
                }
                if episodes is not None:
                    kwargs["episodes"] = episodes
                self._lerobot = LeRobotDataset(**kwargs)
            except Exception as exc:
                if local_root is None:
                    raise RuntimeError(
                        f"LeRobotDataset 初始化失败，且无法回退到本地数据读取: {exc}"
                    ) from exc
                self._init_local_dataset(local_root)
                return
        else:
            self._init_local_dataset(local_root)
            return

        # ------------------------------------------------------------------ #
        # 构建降采样索引映射
        #
        # LeRobot len() = 可作为「当前步」的帧总数（已排除开头窗口不足帧）。
        # 在此基础上以 stride 步长再次过滤，避免相邻 item 的时序窗口过度重叠。
        # 实际降采效果：delta_timestamps 已保证每帧间隔 1/target_fps 秒，
        # stride 过滤进一步保证不同 item 的「当前步」间隔 >= stride 原始帧。
        # ------------------------------------------------------------------ #
        total = len(self._lerobot)
        self._index_map = list(range(0, total, self.stride))
        print(
            f"[Dataset] repo={repo_id}  原始条目={total}，"
            f"stride={self.stride}  降采后条目={len(self._index_map)}"
        )

    # ---------------------------------------------------------------------- #
    # 长度
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        if self.use_dummy:
            return self._dummy_n
        return len(self._index_map)

    # ---------------------------------------------------------------------- #
    # 主入口
    # ---------------------------------------------------------------------- #

    def __getitem__(self, idx: int) -> dict:
        if self.use_dummy:
            return self._make_dummy_sample(idx)
        if self._local_mode:
            return self._load_local_sample(idx)
        return self._load_real_sample(idx)

    def _resolve_local_dataset_root(self, repo_id: str, root: Optional[str]) -> Optional[Path]:
        candidates = []
        if root:
            root_path = Path(root)
            candidates.append(root_path)
            candidates.append(root_path / repo_id)
        repo_path = Path(repo_id)
        if repo_path.exists():
            candidates.append(repo_path)

        for candidate in candidates:
            if (candidate / "meta" / "info.json").exists():
                return candidate
        return None

    def _init_local_dataset(self, dataset_root: Path) -> None:
        self._local_mode = True
        self._lerobot = None
        self._dataset_root = dataset_root

        data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"未找到 parquet 数据文件: {dataset_root / 'data'}")

        frames_df = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)
        if self.episodes is not None:
            frames_df = frames_df[frames_df["episode_index"].isin(self.episodes)].copy()
        if frames_df.empty:
            raise ValueError(f"给定 episodes={self.episodes} 后无可用样本")
        self._frames_df = frames_df.reset_index(drop=True)

        ep_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not ep_files:
            raise FileNotFoundError(f"未找到 episode 元数据: {dataset_root / 'meta' / 'episodes'}")
        episodes_df = pd.concat([pd.read_parquet(p) for p in ep_files], ignore_index=True)
        if self.episodes is not None:
            episodes_df = episodes_df[episodes_df["episode_index"].isin(self.episodes)].copy()
        self._episodes_df = episodes_df.reset_index(drop=True)
        self._episode_meta = {
            int(row["episode_index"]): row.to_dict()
            for _, row in self._episodes_df.iterrows()
        }

        # 过滤掉无元数据的 episode（parquet 中可能多于 info.json 声明的数量）
        valid_eps = set(self._episode_meta.keys())
        before = len(self._frames_df)
        self._frames_df = self._frames_df[
            self._frames_df["episode_index"].isin(valid_eps)
        ].reset_index(drop=True)
        if len(self._frames_df) < before and not self._reported_local_init:
            print(f"[Dataset] 过滤无元数据 episode: {before} → {len(self._frames_df)} 帧")

        with open(dataset_root / "meta" / "info.json", "r", encoding="utf-8") as handle:
            self._info = json.load(handle)

        total = len(self._frames_df)
        self._index_map = list(range(0, total, self.stride))
        if not self._reported_local_init:
            print(
                f"[Dataset] local_root={dataset_root}  原始条目={total}，"
                f"stride={self.stride}  降采后条目={len(self._index_map)}"
            )
            self._reported_local_init = True

    # ====================================================================== #
    # 真实样本加载
    # ====================================================================== #

    def _load_real_sample(self, idx: int) -> dict:
        """
        通过降采样索引映射取出 LeRobot 样本，并附加 sidecar 数据。
        """
        # 将外部 idx 映射回 LeRobot 内部 idx
        lerobot_idx = self._index_map[idx]
        try:
            raw = self._lerobot[lerobot_idx]
        except RuntimeError as exc:
            # torchcodec 无法加载时（如 libnvrtc.so.13 缺失），自动回退到 cv2 本地模式
            if "torchcodec" in str(exc).lower() or "libnvrtc" in str(exc).lower():
                local_root = self._resolve_local_dataset_root(
                    self._lerobot.repo_id if hasattr(self._lerobot, "repo_id") else "",
                    str(self._lerobot.root) if hasattr(self._lerobot, "root") else None,
                )
                if local_root is not None:
                    if not self._reported_local_fallback:
                        print(f"[Dataset] torchcodec 不可用，自动回退到 cv2 本地模式: {local_root}")
                        self._reported_local_fallback = True
                    self._init_local_dataset(local_root)
                    return self._load_local_sample(idx)
            raise

        # ------------------------------------------------------------------ #
        # 1) 视觉数据
        #    LeRobot 返回 tensor [T, C, H, W]，float，已归一化到 [0, 1]
        # ------------------------------------------------------------------ #
        side_rgb      = self._safe_tensor(raw, "observation.images.side")            # [T,3,H,W]
        realsense_rgb = self._safe_tensor(raw, "observation.images.realsense_rgb")   # [T,3,H,W]
        depth_video   = self._safe_tensor(raw, "observation.images.realsense_depth") # [T,C,H,W]

        # 确保深度图是单通道
        if depth_video.dim() == 4 and depth_video.shape[1] != 1:
            depth_video = depth_video[:, :1, :, :]   # 取第 0 通道

        # ------------------------------------------------------------------ #
        # 2) 提取 episode_index 和 frame_index（当前步 t=0 对应的原始位置）
        # ------------------------------------------------------------------ #
        ep_idx = int(raw["episode_index"])  # 所属 episode 编号
        fr_idx = int(raw["frame_index"])    # 当前步在该 episode 中的原始帧号（30FPS计）

        # ------------------------------------------------------------------ #
        # 3) Sidecar：外挂高精度深度（优先级高于 LeRobot 视频流深度）
        #
        # 文件：{sidecar_root}/depth_mm/episode_{ep_idx:06d}.npy
        # 形状 [N_frames, H, W] 或 [N_frames, 1, H, W]，uint16，单位 mm
        # 提取 T 帧窗口后归一化到 [0, 1]。
        # ------------------------------------------------------------------ #
        depth_sidecar = self._load_sidecar_depth(ep_idx, fr_idx)   # [T,1,H,W] or None
        depth = depth_sidecar if depth_sidecar is not None else depth_video

        # ------------------------------------------------------------------ #
        # 4) Sidecar：双触觉序列
        #
        # 文件：
        #   {sidecar_root}/tactile_raw_left/episode_{ep_idx:06d}.npy  [N,16,16]
        #   {sidecar_root}/tactile_raw_right/episode_{ep_idx:06d}.npy [N,16,16]
        # 提取 T 帧窗口并在 channel 维拼接为 [T, 2, 16, 16]。
        # ------------------------------------------------------------------ #
        tactile_grid = self._load_sidecar_tactile(ep_idx, fr_idx)  # [T,2,16,16]

        # ------------------------------------------------------------------ #
        # 5) Action（取当前步 t=0 的 action）
        #    LeRobot 在 delta_timestamps={"action":[0.0]} 时返回 [1, action_dim]
        # ------------------------------------------------------------------ #
        action_raw = raw["action"]
        if isinstance(action_raw, torch.Tensor):
            action = action_raw.squeeze(0).float()    # [action_dim]
        else:
            action = torch.tensor(action_raw, dtype=torch.float32)

        # ------------------------------------------------------------------ #
        # 6) 语言指令
        # ------------------------------------------------------------------ #
        lang = raw.get("language_instruction", None)
        if not lang:
            lang = _DEFAULT_INSTRUCTIONS[idx % len(_DEFAULT_INSTRUCTIONS)]

        return {
            "side_rgb":             side_rgb.float(),       # [T, 3, H, W]  手眼视角
            "realsense_rgb":        realsense_rgb.float(),  # [T, 3, H, W]  全景视角
            "depth":                depth.float(),          # [T, 1, H, W]
            "tactile_grid":         tactile_grid.float(),   # [T, 2, 16, 16]
            "action":               action,                 # [action_dim]
            "language_instruction": lang,
            "episode_index":        ep_idx,
            "frame_index":          fr_idx,
            # 向后兼容旧键名（Stage4 等脚本直接使用这些键）
            "observation.images.side":          side_rgb.float(),
            "observation.images.realsense_rgb": realsense_rgb.float(),
            "tactile_grid":                     tactile_grid.float(),
        }

    def _load_local_sample(self, idx: int) -> dict:
        row = self._frames_df.iloc[self._index_map[idx]]
        ep_idx = int(row["episode_index"])
        fr_idx = int(row["frame_index"])

        side_rgb = self._load_local_video_window("observation.images.side", ep_idx, fr_idx)
        realsense_rgb = self._load_local_video_window("observation.images.realsense_rgb", ep_idx, fr_idx)
        depth_video = self._load_local_video_window("observation.images.realsense_depth", ep_idx, fr_idx)
        depth_video = depth_video[:, :1, :, :]

        depth_sidecar = self._load_sidecar_depth(ep_idx, fr_idx)
        depth = depth_sidecar if depth_sidecar is not None else depth_video
        tactile_grid = self._load_sidecar_tactile(ep_idx, fr_idx)

        action = torch.tensor(np.asarray(row["action"], dtype=np.float32))

        episode_meta = self._episode_meta.get(ep_idx, {})
        tasks = episode_meta.get("tasks", [])
        lang = tasks[0] if isinstance(tasks, (list, tuple)) and tasks else _DEFAULT_INSTRUCTIONS[idx % len(_DEFAULT_INSTRUCTIONS)]

        return {
            "side_rgb": side_rgb.float(),
            "realsense_rgb": realsense_rgb.float(),
            "depth": depth.float(),
            "tactile_grid": tactile_grid.float(),
            "action": action,
            "language_instruction": lang,
            "episode_index": ep_idx,
            "frame_index": fr_idx,
            "observation.images.side": side_rgb.float(),
            "observation.images.realsense_rgb": realsense_rgb.float(),
        }

    # ====================================================================== #
    # Sidecar 辅助函数
    # ====================================================================== #

    def _build_sidecar_window(self, arr: np.ndarray, fr_idx: int) -> np.ndarray:
        """
        从 episode 全帧数组中提取长度 T 的历史窗口。

        参数
        ----
        arr    : [N, ...]  episode 全帧数据（30 FPS 原始帧）
        fr_idx : int       当前步在 30 FPS 序列中的帧号

        返回
        ----
        window : [T, ...]  T 帧历史窗口
                 窗口最后一帧 = fr_idx，相邻帧间隔 stride 原始帧
                 episode 开头不足时用第 0 帧前向填充（padding）
        """
        N = len(arr)
        # 计算 T 个采样点在原始 30 FPS 序列中的帧号
        # 最新帧 = fr_idx，向前每隔 stride 帧取一个点
        frame_ids = [fr_idx - (self.T - 1 - i) * self.stride for i in range(self.T)]

        frames = []
        for fid in frame_ids:
            if fid < 0:
                # 超出 episode 开头：用第 0 帧填充（padding）
                frames.append(arr[0])
            elif fid >= N:
                # 超出 episode 末尾（理论不应发生，保险起见取末帧）
                frames.append(arr[-1])
            else:
                frames.append(arr[fid])

        return np.stack(frames, axis=0)   # [T, ...]

    def _load_sidecar_depth(self, ep_idx: int, fr_idx: int):
        """
        读取外挂高精度深度 .npy，返回归一化后的 [T, 1, H, W] float32 tensor。
        文件不存在或读取失败时返回 None（调用方 fallback 到视频流深度）。
        """
        if self.sidecar_root is None:
            return None

        depth_path = self.sidecar_root / "depth_mm" / f"episode_{ep_idx:06d}.npy"
        if not depth_path.exists():
            depth_dir = self.sidecar_root / "depth_mm" / f"episode-{ep_idx:06d}"
            if not depth_dir.exists():
                return None

            frame_ids = [fr_idx - (self.T - 1 - i) * self.stride for i in range(self.T)]
            frames = []
            last = None
            for fid in frame_ids:
                path = depth_dir / f"frame-{max(fid, 0):06d}.npy"
                if path.exists():
                    arr = np.load(str(path)).astype(np.float32)
                    last = arr
                elif last is None:
                    last = np.zeros((480, 640), dtype=np.float32)
                frames.append(last)

            window = np.stack(frames, axis=0) / 5000.0
            window = np.clip(window, 0.0, 1.0)
            return torch.from_numpy(window[:, np.newaxis, :, :])

        try:
            arr = np.load(str(depth_path))   # [N, H, W] 或 [N, 1, H, W]，uint16 mm
        except Exception:
            return None

        # 统一形状为 [N, H, W]
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, :, :]

        # 提取 T 帧窗口
        window = self._build_sidecar_window(arr, fr_idx)   # [T, H, W]

        # uint16 mm -> float [0,1]（假设最大量程 5000 mm = 5 m）
        depth_max = 5000.0
        window = window.astype(np.float32) / depth_max
        window = np.clip(window, 0.0, 1.0)

        # 增加通道维 -> [T, 1, H, W]
        window = window[:, np.newaxis, :, :]
        return torch.from_numpy(window)   # [T, 1, H, W]

    def _load_sidecar_tactile(
        self, ep_idx: int, fr_idx: int
    ) -> torch.Tensor:
        """
        读取左右手触觉 .npy，拼接为 [T, 2, 16, 16] float32 tensor。
        任意一路文件缺失时该路全零，整体不崩溃。
        """
        def _load_one(subdir: str) -> np.ndarray:
            """读取单路触觉，返回 [N, 16, 16] ndarray；失败返回 None。"""
            if self.sidecar_root is None:
                return None
            p = self.sidecar_root / subdir / f"episode_{ep_idx:06d}.npy"
            p_dir = self.sidecar_root / subdir / f"episode-{ep_idx:06d}"
            if not p.exists():
                if not p_dir.exists():
                    return None
                frame_ids = [fr_idx - (self.T - 1 - i) * self.stride for i in range(self.T)]
                frames = []
                last = None
                for fid in frame_ids:
                    frame_path = p_dir / f"frame-{max(fid, 0):06d}.npy"
                    if frame_path.exists():
                        arr = np.load(str(frame_path)).astype(np.float32)
                        last = arr
                    elif last is None:
                        last = np.zeros((self.tac_h, self.tac_w), dtype=np.float32)
                    frames.append(last)
                return np.stack(frames, axis=0)
            try:
                arr = np.load(str(p))   # [N, 16, 16] 或 [N, 1, 16, 16]
                if arr.ndim == 4 and arr.shape[1] == 1:
                    arr = arr[:, 0, :, :]
                return arr.astype(np.float32)
            except Exception:
                return None

        def _window_or_zeros(arr) -> np.ndarray:
            """提取 T 帧窗口；arr 为 None 时返回全零。"""
            if arr is None:
                return np.zeros((self.T, self.tac_h, self.tac_w), dtype=np.float32)
            raw_win = self._build_sidecar_window(arr, fr_idx)   # [T, 16, 16]
            # min-max 归一化（防止量程差异）
            mn, mx = raw_win.min(), raw_win.max()
            if mx - mn > 1e-6:
                raw_win = (raw_win - mn) / (mx - mn)
            return raw_win

        left_arr  = _load_one("tactile_raw_left")
        right_arr = _load_one("tactile_raw_right") if self.has_right_tactile else None

        left_win  = _window_or_zeros(left_arr)    # [T, 16, 16]
        right_win = _window_or_zeros(right_arr)   # [T, 16, 16]

        # 在 channel 维拼接：[T, 16, 16] -> [T, 1, 16, 16]
        left_t  = torch.from_numpy(left_win[:, np.newaxis, :, :])   # [T,1,16,16]
        right_t = torch.from_numpy(right_win[:, np.newaxis, :, :])  # [T,1,16,16]
        return torch.cat([left_t, right_t], dim=1)   # [T, 2, 16, 16]

    def _load_local_video_window(self, video_key: str, ep_idx: int, fr_idx: int) -> torch.Tensor:
        meta = self._episode_meta[ep_idx]
        start_frame = int(round(float(meta[f"videos/{video_key}/from_timestamp"]) * SRC_FPS))
        frame_ids = [fr_idx - (self.T - 1 - i) * self.stride for i in range(self.T)]
        frames = []
        for fid in frame_ids:
            safe_fid = max(fid, 0)
            frames.append(self._read_video_frame(video_key, start_frame + safe_fid))
        return torch.stack(frames, dim=0)

    def _read_video_frame(self, video_key: str, frame_idx: int) -> torch.Tensor:
        cap = self._get_video_cap(video_key)
        cap.set(1, frame_idx)
        ok, frame = cap.read()
        if not ok:
            shape = self._video_shapes.get(video_key, (480, 640, 3))
            frame = np.zeros(shape, dtype=np.uint8)
        else:
            self._video_shapes[video_key] = frame.shape

        try:
            import cv2
        except ImportError as exc:
            raise ImportError("本地视频回退模式需要安装 opencv-python") from exc

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return tensor

    def _get_video_cap(self, video_key: str):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("本地视频回退模式需要安装 opencv-python") from exc

        if video_key in self._video_caps:
            return self._video_caps[video_key]

        episode_meta = next(iter(self._episode_meta.values()))
        chunk_idx = int(episode_meta[f"videos/{video_key}/chunk_index"])
        file_idx = int(episode_meta[f"videos/{video_key}/file_index"])
        path = (
            self._dataset_root
            / "videos"
            / video_key
            / f"chunk-{chunk_idx:03d}"
            / f"file-{file_idx:03d}.mp4"
        )
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件: {path}")
        self._video_caps[video_key] = cap
        return cap

    # ====================================================================== #
    # 辅助：安全取 tensor
    # ====================================================================== #

    @staticmethod
    def _safe_tensor(raw: dict, key: str) -> torch.Tensor:
        """
        从 LeRobot 返回的字典中安全取 tensor。
        若 key 不存在或取值失败，返回形状 [1, 1, 1, 1] 的零 tensor（不崩溃）。
        """
        try:
            v = raw[key]
            if isinstance(v, torch.Tensor):
                return v
            return torch.tensor(v, dtype=torch.float32)
        except (KeyError, Exception):
            return torch.zeros(1, 1, 1, 1)

    # ====================================================================== #
    # Dummy 样本生成
    # ====================================================================== #

    def _make_dummy_sample(self, idx: int) -> dict:
        """
        生成随机 dummy 样本，所有 tensor 形状与真实数据完全一致。
        视角说明：side = 手眼相机，realsense_rgb = 全景相机。
        """
        rng = np.random.default_rng(seed=idx)   # 固定 seed，可复现

        # 触觉: [T, 2, 16, 16]，channel 0=left, 1=right
        tactile_grid = torch.from_numpy(
            rng.random((self.T, 2, self.tac_h, self.tac_w)).astype(np.float32)
        )
        # 手眼 RGB: [T, 3, 224, 224]
        side_rgb = torch.from_numpy(
            rng.random((self.T, 3, 224, 224)).astype(np.float32)
        )
        # 全景 RGB: [T, 3, 224, 224]
        realsense_rgb = torch.from_numpy(
            rng.random((self.T, 3, 224, 224)).astype(np.float32)
        )
        # 深度: [T, 1, 224, 224]
        depth = torch.from_numpy(
            rng.random((self.T, 1, 224, 224)).astype(np.float32)
        )
        # 动作: [action_dim]
        action = torch.from_numpy(
            rng.random(self.action_dim).astype(np.float32)
        )

        lang = _DEFAULT_INSTRUCTIONS[idx % len(_DEFAULT_INSTRUCTIONS)]

        return {
            "side_rgb":             side_rgb,      # [T, 3, H, W]  手眼视角
            "realsense_rgb":        realsense_rgb, # [T, 3, H, W]  全景视角
            "depth":                depth,         # [T, 1, H, W]
            "tactile_grid":         tactile_grid,  # [T, 2, 16, 16]
            "action":               action,        # [action_dim]
            "language_instruction": lang,
            "episode_index":        idx,
            "frame_index":          0,
            # 向后兼容旧键名
            "observation.images.side":          side_rgb,
            "observation.images.realsense_rgb": realsense_rgb,
        }


# ============================================================================
# 向后兼容别名（旧代码 import OverfitDataset 不会炸）
# ============================================================================

class OverfitDataset(LeRobotTactileDataset):
    """
    向后兼容旧名称。新代码请直接使用 LeRobotTactileDataset。

    旧参数映射：
      episode_dir  ->  忽略（LeRobot 用 repo_id / root）
      use_dummy    ->  use_dummy
      n_dummy      ->  n_dummy
      T            ->  T
    """

    def __init__(
        self,
        episode_dir: str = "episodes",
        use_dummy: bool = True,
        n_dummy: int = 50,
        T: int = DEFAULT_T,
        episodes: Optional[list[int]] = None,
        # 以下旧参数静默忽略，避免调用方报错
        tac_h: int = DEFAULT_TAC_H,
        tac_w: int = DEFAULT_TAC_W,
        rgb_h: int = 224,
        rgb_w: int = 224,
        depth_h: int = 224,
        depth_w: int = 224,
        action_dim: int = DEFAULT_ACTION_DIM,
        **kwargs,
    ):
        super().__init__(
            use_dummy=use_dummy,
            n_dummy=n_dummy,
            T=T,
            episodes=episodes,
            tac_h=tac_h,
            tac_w=tac_w,
            action_dim=action_dim,
            **kwargs,
        )
