"""
overfit/models.py

目标：
- 不重新定义触觉编码器结构，直接 import 你本地已有的 Encoder
- 使用 transformers 加载小型 VLA 占位基座（默认 SmolVLM）并冻结
- 将 tactile token 拼接到 vision token 后，再送入（可用时）VLA 的 LLM backbone
- 输出连续动作向量
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1) 只 import 本地触觉编码器，不在这里重新造 CNN/ResNet
# ============================================================================
# 期望你后续提供：models/tactile_encoder.py 里的 MyGridTactileEncoder
# 兼容当前仓库：若暂时没有该文件，回退到已有 TactileEncoder（tlv_student.py）。

try:
    from models.tactile_encoder import DualTactileGridEncoder as MyGridTactileEncoder
except Exception:
    from models.tlv_student import TactileEncoder as MyGridTactileEncoder


# ============================================================================
# 2) 两层 MLP Projector: tactile_feat -> tactile_tokens(hidden_size)
# ============================================================================

class TactileMLPProjector(nn.Module):
    """
    输入 : [B, tactile_dim]
    输出 : [B, n_tactile_tokens, hidden_size]
    """

    def __init__(self, tactile_dim: int, hidden_size: int, n_tactile_tokens: int = 8):
        super().__init__()
        self.n_tactile_tokens = n_tactile_tokens
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(tactile_dim, tactile_dim * 2),
            nn.GELU(),
            nn.Linear(tactile_dim * 2, n_tactile_tokens * hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, tactile_dim]
        b = x.shape[0]
        y = self.mlp(x)  # [B, n_tactile_tokens * hidden_size]
        return y.view(b, self.n_tactile_tokens, self.hidden_size)


# ============================================================================
# 3) TactileVLAAdapter
# ============================================================================

class TactileVLAAdapter(nn.Module):
    """
    forward 输入:
      - images: List[PIL.Image], 长度 B
      - texts:  List[str], 长度 B
      - tactile_grids: [B, T, 1, 16, 16]（或可被 encoder 接收的形状）

    输出:
      - action_pred: [B, action_dim]

    参数训练策略:
      - VLA 全冻结
      - tactile_encoder / projector / action_head 可训练
    """

    DEFAULT_VLA = "HuggingFaceTB/SmolVLM-Instruct"

    def __init__(
        self,
        vla_model_id: Optional[str] = None,
        tactile_feat_dim: int = 512,
        n_tactile_tokens: int = 8,
        action_dim: int = 7,
        device: str = "cpu",
        use_dummy_vla: bool = True,
        # LoRA 参数
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.device = device
        self.use_dummy_vla = use_dummy_vla
        self.use_lora = use_lora

        if use_dummy_vla:
            # 本地连通性测试：不下载大模型
            self.processor = None
            self.vla = _DummyVLA(hidden_size=512).to(device)
            self.hidden_size = 512
            self.llm_backbone = self.vla.llm_backbone
        else:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            mid = vla_model_id or self.DEFAULT_VLA
            self.processor = AutoProcessor.from_pretrained(mid)
            self.vla = AutoModelForImageTextToText.from_pretrained(mid).to(device)

            # 冻结 VLA 全部参数
            for p in self.vla.parameters():
                p.requires_grad_(False)

            cfg = self.vla.config
            self.hidden_size = getattr(
                cfg,
                "hidden_size",
                getattr(getattr(cfg, "text_config", cfg), "hidden_size", 1024),
            )

            # 尝试拿到可接受 inputs_embeds 的 LLM backbone
            self.llm_backbone = self._locate_llm_backbone(self.vla)

            # ---- LoRA 注入 ----
            if use_lora and self.llm_backbone is not None:
                from peft import LoraConfig, get_peft_model

                # 自动探测目标模块（默认: attention + MLP 线性层）
                if lora_target_modules is None:
                    lora_target_modules = self._detect_lora_targets(self.llm_backbone)

                lora_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=lora_target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.llm_backbone = get_peft_model(self.llm_backbone, lora_cfg)
                self.llm_backbone.print_trainable_parameters()

        # === 只实例化本地触觉编码器，不定义新结构 ===
        self.tactile_encoder = MyGridTactileEncoder()

        self.projector = TactileMLPProjector(
            tactile_dim=tactile_feat_dim,
            hidden_size=self.hidden_size,
            n_tactile_tokens=n_tactile_tokens,
        )
        self.action_head = nn.Linear(self.hidden_size, action_dim)

    @staticmethod
    def _detect_lora_targets(backbone: nn.Module) -> List[str]:
        """自动探测 LLM backbone 中适合接 LoRA 的线性层名称模式。"""
        names = set()
        for name, mod in backbone.named_modules():
            if isinstance(mod, nn.Linear):
                # 取最后一段（如 q_proj, k_proj, v_proj, o_proj, gate_proj 等）
                short = name.rsplit(".", 1)[-1]
                if any(kw in short for kw in ("proj", "gate", "dense", "fc")):
                    names.add(short)
        if not names:
            # 回退: 所有线性层
            names = {"q_proj", "v_proj"}
        return sorted(names)

    @staticmethod
    def _locate_llm_backbone(vla_model: nn.Module):
        """尽量找到支持 inputs_embeds 的 text backbone。"""
        candidates = [
            getattr(vla_model, "language_model", None),
            getattr(getattr(vla_model, "model", None), "language_model", None),
            getattr(getattr(vla_model, "model", None), "text_model", None),
            getattr(vla_model, "text_model", None),
        ]
        for c in candidates:
            if c is not None:
                return c
        return None

    def trainable_params(self):
        """
        返回可训练模块参数：
        - tactile_encoder (可选冻结)
        - projector
        - action_head
        - LoRA adapter 参数（若启用）
        """
        params = (
            list(self.tactile_encoder.parameters())
            + list(self.projector.parameters())
            + list(self.action_head.parameters())
        )
        if self.use_lora and self.llm_backbone is not None:
            # 仅收集 LoRA 注入的可训练参数
            for p in self.llm_backbone.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

    def _normalize_image_batch(self, images: List):
        """
        统一图像输入格式。
        支持:
          - 单视角: List[PIL.Image], 长度 B
          - 多视角: List[List[PIL.Image]], 外层长度 B
        返回:
          image_batch: List[List[PIL.Image]]
          n_views: int
        """
        if len(images) == 0:
            raise ValueError("images 不能为空")

        first = images[0]
        if isinstance(first, (list, tuple)):
            image_batch = [list(x) for x in images]
            n_views = len(image_batch[0])
        else:
            image_batch = [[img] for img in images]
            n_views = 1

        return image_batch, n_views

    def _extract_vla_tokens(self, images: List, texts: List[str]) -> torch.Tensor:
        """
        提取 VLA token 序列。
        images 支持单视角或多视角。
        返回: [B, N_vis, hidden_size]
        """
        image_batch, n_views = self._normalize_image_batch(images)

        if self.use_dummy_vla:
            return self.vla(len(texts))

        prompts = []
        for t in texts:
            content = [{"type": "image"} for _ in range(n_views)]
            content.append({"type": "text", "text": t})
            prompts.append(
                self.processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    add_generation_prompt=False,
                )
            )

        inputs = self.processor(
            text=prompts,
            images=image_batch if n_views > 1 else [x[0] for x in image_batch],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            out = self.vla(**inputs, output_hidden_states=True, return_dict=True)
            tokens = out.hidden_states[-1]  # [B, N_vis+N_text, D]
        return tokens

    def _encode_tactile(self, tactile_grids: torch.Tensor) -> torch.Tensor:
        """
        兼容多种输入格式，统一得到 [B, tactile_feat_dim]。

        支持:
          [B, T, 2, 16, 16]  双触觉（DualTactileGridEncoder 原生接受）
          [B, T, 1, 16, 16]  单触觉
          [B, T, 16, 16]     无通道维
        """
        x = tactile_grids

        # DualTactileGridEncoder 接受 [B,T,C,16,16]，C=1或2
        # 也接受 [B,T,16,16]（内部会 unsqueeze）
        # 仅对老版 TactileEncoder（期望 [B,T,16,16]）做 squeeze
        if not hasattr(self.tactile_encoder, 'share_weights'):
            # 老版 TactileEncoder，需 squeeze 通道维
            if x.dim() == 5 and x.shape[2] == 1:
                x = x.squeeze(2)
            elif x.dim() == 5 and x.shape[2] == 2:
                # 双触觉但用老 encoder：取左路
                x = x[:, :, 0, :, :]

        out = self.tactile_encoder(x)

        # 兼容返回 tuple/list
        if isinstance(out, (tuple, list)):
            out = out[0]

        # 若是序列 [B,T,D]，取最后时刻
        if out.dim() == 3:
            out = out[:, -1, :]

        # 若不是 [B,D]，在最后维之外做平均池化
        if out.dim() > 2:
            out = out.flatten(start_dim=1)

        return out  # [B, tactile_feat_dim]

    def forward(self, images: List, texts: List[str], tactile_grids: torch.Tensor) -> torch.Tensor:
        # 1) VLA 提取视觉/文本 token（冻结）
        vla_tokens = self._extract_vla_tokens(images, texts)  # [B, N_vis, D]

        # 2) 本地触觉 encoder
        tactile_feat = self._encode_tactile(tactile_grids.to(self.device))  # [B, tactile_feat_dim]

        # 3) MLP 映射为触觉 token
        tactile_tokens = self.projector(tactile_feat)  # [B, N_tac, D]

        # 4) 拼接到 vision token 后
        fused_tokens = torch.cat([vla_tokens, tactile_tokens], dim=1)  # [B, N_vis+N_tac, D]

        # 5) 送入 LLM backbone（若可用），否则直接用 fused tokens
        if self.llm_backbone is not None:
            try:
                # LoRA 启用时需要梯度流过 backbone
                ctx = torch.no_grad() if not self.use_lora else torch.enable_grad()
                with ctx:
                    llm_out = self.llm_backbone(
                        inputs_embeds=fused_tokens,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    if hasattr(llm_out, "last_hidden_state") and llm_out.last_hidden_state is not None:
                        hidden = llm_out.last_hidden_state
                    else:
                        hidden = llm_out.hidden_states[-1]
            except Exception:
                hidden = fused_tokens
        else:
            hidden = fused_tokens

        # 6) 聚合并预测 action
        pooled = hidden.mean(dim=1)             # [B, D]
        action_pred = self.action_head(pooled)  # [B, action_dim]
        return action_pred


class _DummyVLA(nn.Module):
    """本地连通性测试占位：模拟 VLA token 与 LLM backbone。"""

    def __init__(self, hidden_size: int = 512, n_vis_tokens: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_vis_tokens = n_vis_tokens
        self.embed = nn.Linear(hidden_size, hidden_size)
        self.llm_backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                batch_first=True,
                dim_feedforward=hidden_size * 2,
            ),
            num_layers=2,
        )
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, bsz: int) -> torch.Tensor:
        x = torch.zeros(bsz, self.n_vis_tokens, self.hidden_size, device=next(self.parameters()).device)
        return self.embed(x)
