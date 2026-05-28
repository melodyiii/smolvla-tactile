import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# 降维和可视化库
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# 导入模型
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.tlv_student import TactileEncoder


class CrossModalAttentionPool(nn.Module):
    """跨模态注意力池化（与 train/2_sentence_vision.py 保持一致）"""
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, tactile_seq, anchor_emb):
        query = anchor_emb.unsqueeze(1)
        attn_out, _ = self.multihead_attn(query, tactile_seq, tactile_seq)
        x = self.norm(query + attn_out)
        x = self.norm_ffn(x + self.ffn(x))
        return x.squeeze(1)


class VisionGuidedTactileModel(nn.Module):
    """完整的 Stage 2 模型结构，用于正确加载检查点"""
    def __init__(self, tau_init=0.07, feature_dim=512):
        super().__init__()
        self.tactile_encoder = TactileEncoder(proj_dim=feature_dim, tau=tau_init)
        self.cross_attn = CrossModalAttentionPool(dim=feature_dim)
        self.logit_scale_tv = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_tactile(self, x_tactile, z_vision=None):
        """
        推理接口：
        - z_vision 不为 None：返回视觉引导的对齐特征 [B, 512]
        - z_vision 为 None：返回原始全局触觉特征 [B, 512]
        """
        if z_vision is not None:
            z_t_seq, z_t_global, _ = self.tactile_encoder(x_tactile, return_seq=True)
            return self.cross_attn(z_t_seq, z_vision)
        else:
            z_t_global, _ = self.tactile_encoder(x_tactile, return_seq=False)
            return z_t_global


def visualize_embeddings(
    ckpt_path="runs/ckpt_stage2_vision_attn.pt",  # 修正：与 Stage 2 实际保存路径一致
    num_samples=20,
    method="umap",
    text_type="phrases",
    use_vision_guide=True,
    seed=42
):
    """
    主函数：加载完整模型，提取三模态特征，降维并可视化。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # =========================================================
    # 1. 加载完整模型（Stage 2 结构，含 CrossModalAttentionPool）
    # =========================================================
    model = VisionGuidedTactileModel()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    print(f"已加载完整模型（含视觉引导注意力）: {ckpt_path}")

    # 加载 Teacher 特征库
    text_repo = torch.load("teachers/text_embed.pt", map_location="cpu")
    visual_repo = torch.load("teachers/visual_embed.pt", map_location="cpu")

    # =========================================================
    # 2. 选择样本并提取三模态特征
    # =========================================================
    items = sorted([d for d in glob("dataset/seq_*") if os.path.isdir(d)])
    if num_samples > len(items):
        num_samples = len(items)

    np.random.seed(seed)
    selected_items = np.random.choice(items, num_samples,         replace=False)

    all_features = []
    all_modalities = []
    all_object_ids = []

    print(f"正在从 {num_samples} 个样本中提取三模态特征...")
    for item_path in tqdm(selected_items):
        sample_id = os.path.basename(item_path)

        # 读取视觉特征（Teacher）
        z_v = F.normalize(visual_repo[sample_id].unsqueeze(0), dim=-1).to(device)  # [1, 512]

        # 提取触觉特征（Student）
        tac_data = torch.from_numpy(
            np.load(f"{item_path}/tactile.npy")
        ).float()[:16].unsqueeze(0).to(device)  # [1, T, 16, 16]

        with torch.no_grad():
            if use_vision_guide:
                # 使用视觉引导的完整 Stage 2 特征
                z_t = model.encode_tactile(tac_data, z_vision=z_v)
            else:
                # 仅使用原始全局触觉特征（Stage 1 基准）
                z_t = model.encode_tactile(tac_data, z_vision=None)
            z_t_np = z_t.cpu().numpy().flatten()

        z_v_np = z_v.cpu().numpy().flatten()

        # 提取文本特征（Teacher）
        j = json.load(open(f"{item_path}/text.json"))
        text = j[text_type][0]
        z_text_np = text_repo[text_type][text].numpy().flatten()

        all_features.extend([z_t_np, z_v_np, z_text_np])
        all_modalities.extend(["Tactile", "Visual", "Text"])
        all_object_ids.extend([sample_id] * 3)

    features_np = np.array(all_features)

    # =========================================================
    # 3. 降维（t-SNE 或 UMAP）
    # =========================================================
    print(f"正在使用 {method.upper()} 进行降维...")
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=15, random_state=seed,
                       init="pca", learning_rate="auto")
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                            n_components=2, random_state=seed)

    features_2d = reducer.fit_transform(features_np)

    # =========================================================
    # 4. 绘图
    # =========================================================
    df = pd.DataFrame({
        "x": features_2d[:, 0],
        "y": features_2d[:, 1],
        "object_id": all_object_ids,
        "modality": all_modalities
    })

    print("正在生成图像...")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(14, 10))

    guide_tag = "vision-guided" if use_vision_guide else "global-only"
    sns.scatterplot(
        data=df, x="x", y="y",
        hue="object_id", style="modality",
        s=150, alpha=0.8
    )

    plt.title(f"{method.upper()} Visualization - {guide_tag}", fontsize=16)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_path = f"viz/{method}_{guide_tag}_visualization.png"
    os.makedirs("viz", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"图像已保存到: {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Multimodal Embeddings")
    parser.add_argument("--ckpt", type=str, default="runs/ckpt_stage2_vision_attn.pt")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"])
    parser.add_argument("--text_type", type=str, default="phrases", choices=["phrases", "sentences"])
    parser.add_argument("--use_vision_guide", action="store_true",
                        help="使用视觉引导触觉特征（Stage 2 完整能力）")
    args = parser.parse_args()

    visualize_embeddings(
        ckpt_path=args.ckpt,
        num_samples=args.samples,
        method=args.method,
        text_type=args.text_type,
        use_vision_guide=args.use_vision_guide,
    )
