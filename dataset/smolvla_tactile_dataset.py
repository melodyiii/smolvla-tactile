"""
dataset/smolvla_tactile_dataset.py

将 LeRobotTactileDataset 的输出转换为 SmolVLA 策略所需的 batch 格式。

SmolVLA 期望的 batch:
  observation.images.side                [3, H, W]          最后一帧（float [0,1]）
  observation.images.realsense_rgb       [3, H, W]          最后一帧
  observation.state                      [6]                关节位置
  observation.language.tokens            [max_length]       tokenized 文本
  observation.language.attention_mask    [max_length]       attention mask
  observation.tactile                    [T, 2, 16, 16]    触觉序列（传给触觉编码器）
  action                                 [chunk_size, 6]    未来动作轨迹

设计:
  - 底层数据读取全部委托给 LeRobotTactileDataset（视频解码、sidecar 加载等）
  - 本类只做格式转换 + action 轨迹构建
  - action 轨迹通过直接读取 parquet 获取未来帧（LeRobotTactileDataset 只返回当前帧 action）
  - observation.state 从 parquet 直接读取
  - 语言 tokenization 使用 SmolVLA 的 processor.tokenizer

用法:
    ds = SmolVLATactileDataset(
        data_path="./data/inboxpicking-01",
        tokenizer=policy.model.vlm_with_expert.processor.tokenizer,
        chunk_size=50,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from overfit.dataset import LeRobotTactileDataset, _DEFAULT_INSTRUCTIONS


class SmolVLATactileDataset(Dataset):
    """
    SmolVLA 格式的触觉多模态数据集。

    包装 LeRobotTactileDataset，将输出转换为 SmolVLA 策略可直接消费的 dict。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        repo_id: str = "local/inboxpicking",
        chunk_size: int = 50,
        target_fps: int = 10,
        T: int = 16,
        action_dim: int = 6,
        state_dim: int = 6,
        tokenizer_max_length: int = 48,
        has_right_tactile: bool = True,
        episodes: Optional[list[int]] = None,
        use_dummy: bool = False,
        n_dummy: int = 50,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.target_fps = target_fps
        self.T = T

        # 底层 LeRobot + Sidecar 数据集
        self._inner = LeRobotTactileDataset(
            repo_id=repo_id,
            root=data_path,
            sidecar_root=data_path,
            target_fps=target_fps,
            T=T,
            action_dim=action_dim,
            has_right_tactile=has_right_tactile,
            use_dummy=use_dummy,
            n_dummy=n_dummy,
            episodes=episodes,
        )

        # 加载 parquet 获取 observation.state 和 action 轨迹
        self._data_path = Path(data_path)
        self._use_dummy = use_dummy

        if not use_dummy:
            self._load_parquet_data()

    def _load_parquet_data(self):
        """加载 parquet 数据以获取 observation.state 和 action 轨迹。"""
        data_dir = self._data_path / "data"
        parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"未找到 parquet 文件: {data_dir}")

        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

        # 按 episode 分组，存储 per-episode 的 state 和 action 数组
        self._episode_states = {}
        self._episode_actions = {}
        self._episode_lengths = {}

        for ep_idx in df["episode_index"].unique():
            ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
            states = np.stack(ep_df["observation.state"].values).astype(np.float32)
            actions = np.stack(ep_df["action"].values).astype(np.float32)
            self._episode_states[int(ep_idx)] = states    # [N, state_dim]
            self._episode_actions[int(ep_idx)] = actions   # [N, action_dim]
            self._episode_lengths[int(ep_idx)] = len(ep_df)

        # 加载 tasks 元数据（用于 language instruction）
        self._task_text = self._load_task_text()

    def _load_task_text(self) -> str:
        """从 meta/tasks.parquet 加载任务描述文本。"""
        tasks_file = self._data_path / "meta" / "tasks.parquet"
        if tasks_file.exists():
            tasks_df = pd.read_parquet(tasks_file)
            if len(tasks_df) > 0:
                # 取第一个 task 的文本
                for col in ["task", "text", "instruction", "description"]:
                    if col in tasks_df.columns:
                        return str(tasks_df[col].iloc[0])
        return _DEFAULT_INSTRUCTIONS[0]

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        if self._use_dummy:
            return self._make_dummy_sample()

        # 从底层数据集获取基础数据
        raw = self._inner[idx]

        ep_idx = int(raw["episode_index"])
        fr_idx = int(raw["frame_index"])

        # ---- 图像: 取最后一帧 [3, H, W] ----
        side_rgb = raw["side_rgb"]             # [T, 3, H, W]
        realsense_rgb = raw["realsense_rgb"]   # [T, 3, H, W]

        side_last = side_rgb[-1]                # [3, H, W]
        realsense_last = realsense_rgb[-1]      # [3, H, W]

        # ---- observation.state: 从 parquet 读取 ----
        state = self._get_state(ep_idx, fr_idx)  # [state_dim]

        # ---- 触觉 ----
        tactile = raw["tactile_grid"]  # [T, 2, 16, 16]

        # ---- 语言 tokenization ----
        lang_text = raw.get("language_instruction", self._task_text)
        lang_tokens, lang_mask = self._tokenize(lang_text)

        # ---- Action 轨迹: [chunk_size, action_dim] ----
        action_chunk = self._get_action_chunk(ep_idx, fr_idx)

        return {
            "observation.images.side": side_last.float(),
            "observation.images.realsense_rgb": realsense_last.float(),
            "observation.state": state,
            "observation.language.tokens": lang_tokens,
            "observation.language.attention_mask": lang_mask,
            "observation.tactile": tactile.float(),
            "action": action_chunk,
            "episode_index": ep_idx,
            "frame_index": fr_idx,
        }

    def _get_state(self, ep_idx: int, fr_idx: int) -> torch.Tensor:
        """获取当前帧的 observation.state。"""
        states = self._episode_states.get(ep_idx)
        if states is None:
            return torch.zeros(self.state_dim, dtype=torch.float32)
        fr_idx_clamped = min(fr_idx, len(states) - 1)
        return torch.tensor(states[fr_idx_clamped], dtype=torch.float32)

    def _get_action_chunk(self, ep_idx: int, fr_idx: int) -> torch.Tensor:
        """
        构建从 fr_idx 开始的 chunk_size 步 action 轨迹。

        在 episode 末尾不足 chunk_size 帧时，用最后一个有效 action 填充。
        """
        actions = self._episode_actions.get(ep_idx)
        if actions is None:
            return torch.zeros(self.chunk_size, self.action_dim, dtype=torch.float32)

        n_frames = len(actions)
        chunk = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

        for i in range(self.chunk_size):
            src_idx = min(fr_idx + i, n_frames - 1)
            chunk[i] = actions[src_idx]

        return torch.tensor(chunk, dtype=torch.float32)

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """用 SmolVLA tokenizer 将文本编码为 token ids + attention mask。"""
        if self.tokenizer is None:
            # 无 tokenizer 时返回填充的 dummy tokens
            tokens = torch.ones(self.tokenizer_max_length, dtype=torch.long)
            mask = torch.ones(self.tokenizer_max_length, dtype=torch.bool)
            return tokens, mask

        encoded = self.tokenizer(
            text,
            max_length=self.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].squeeze(0)          # [max_length]
        mask = encoded["attention_mask"].squeeze(0).bool() # [max_length]
        return tokens, mask

    def _make_dummy_sample(self) -> dict:
        """生成 dummy 样本用于 pipeline 连通性测试。"""
        return {
            "observation.images.side": torch.rand(3, 480, 640),
            "observation.images.realsense_rgb": torch.rand(3, 480, 640),
            "observation.state": torch.randn(self.state_dim),
            "observation.language.tokens": torch.ones(self.tokenizer_max_length, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(self.tokenizer_max_length, dtype=torch.bool),
            "observation.tactile": torch.randn(self.T, 2, 16, 16),
            "action": torch.randn(self.chunk_size, self.action_dim),
            "episode_index": 0,
            "frame_index": 0,
        }
