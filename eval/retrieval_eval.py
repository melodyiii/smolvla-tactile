import os, json, argparse, csv, torch, numpy as np
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm

# 导入完整的多模态模型（Stage 2 结构）
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.tlv_student import TactileEncoder, improved_multi_pos_infonce
import torch.nn as nn


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
        推理接口：根据是否传入视觉特征，返回不同的触觉表示。
        - 有视觉特征：返回视觉引导后的对齐特征 z_t_aligned [B, 512]
        - 无视觉特征：返回全局触觉特征 z_t_global [B, 512]
        """
        if z_vision is not None:
            z_t_seq, z_t_global, s = self.tactile_encoder(x_tactile, return_seq=True)
            z_t_aligned = self.cross_attn(z_t_seq, z_vision)
            return z_t_aligned
        else:
            z_t_global, s = self.tactile_encoder(x_tactile, return_seq=False)
            return z_t_global


def recall_at_k(sim, labels, k):
    """计算 Recall@K 指标"""
    topk = sim.topk(k, dim=1).indices  # [B, k]
    hit = sum([labels[i] in topk[i].tolist() for i in range(sim.size(0))])
    return hit / sim.size(0)


def main():
    ap = argparse.ArgumentParser()
    # 修正默认路径：与 Stage 2 实际保存路径一致
    ap.add_argument("--ckpt", type=str, default="runs/ckpt_stage2_vision_attn.pt",
                    help="模型检查点路径（Stage 2: ckpt_stage2_vision_attn.pt）")
    ap.add_argument("--mode", type=str, default="t2t", choices=["t2t", "t2v", "v2t"],
                    help="评测模式: t2t (触觉->文本), t2v (触觉->视觉), v2t (视觉->文本)")
    ap.add_argument("--bank", type=str, default="phrases", choices=["phrases", "sentences"],
                    help="使用的文本库类型")
    ap.add_argument("--csv", type=str, default="", help="CSV输出路径，留空则不保存")
    ap.add_argument("--T", type=int, default=16, help="触觉序列长度")
    ap.add_argument("--use_vision_guide", action="store_true",
                    help="t2t/t2v 模式下是否使用视觉引导触觉特征（需要 visual_embed.pt）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[信息] 使用设备: {device}")
    print(f"[信息] 加载检查点: {args.ckpt}")

    # =========================================================
    # 1. 加载完整模型（Stage 2 结构）
    # =========================================================
    model = VisionGuidedTactileModel()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    print(f"[信息] 模型加载成功（含 CrossModalAttentionPool）")

    # =========================================================
    # 2. 加载 Teacher 特征库
    # =========================================================
    text_repo = torch.load("teachers/text_embed.pt", map_location="cpu")
    bank_texts = sorted(list(text_repo[args.bank].keys()))
    Zx_bank = F.normalize(
        torch.stack([text_repo[args.bank][t] for t in bank_texts]), dim=-1
    ).to(device)  # [N_text, 512]

    visual_repo = torch.load("teachers/visual_embed.pt", map_location="cpu")

    # =========================================================
    # 3. 为整个数据集提取特征
    # =========================================================
    Zt_list, Zv_list, text_labels, seq_names = [], [], [], []
    items = sorted([d for d in glob("dataset/seq_*") if os.path.isdir(d)])

    print(f"[信息] 正在从 {len(items)} 个样本中提取特征...")
    for d in tqdm(items):
        seq_name = os.path.basename(d)
        seq_names.append(seq_name)

        # 读取视觉特征（用于视觉引导）
        z_v = F.normalize(visual_repo[seq_name].unsqueeze(0), dim=-1).to(device)  # [1, 512]
        Zv_list.append(z_v.squeeze(0).cpu())

        # 提取触觉特征
        tac = torch.from_numpy(
            np.load(f"{d}/tactile.npy")
        ).float()[:args.T].unsqueeze(0).to(device)  # [1, T, 16, 16]

        with torch.no_grad():
            if args.use_vision_guide and args.mode in ["t2t", "t2v"]:
                # 使用视觉引导的对齐特征（Stage 2/3 完整能力）
                z_t = model.encode_tactile(tac, z_vision=z_v)  # [1, 512]
            else:
                # 仅使用触觉全局特征（Stage 1 基准）
                z_t = model.encode_tactile(tac, z_vision=None)  # [1, 512]
            Zt_list.append(z_t.cpu())

        # 文本标签（在文本库中的索引）
        j = json.load(open(f"{d}/text.json"))
        pick = j["phrases"][0] if args.bank == "phrases" else j["sentences"][0]
        text_labels.append(bank_texts.index(pick))

    Zt = torch.cat(Zt_list, 0).to(device)                          # [B, 512]
    Zv = F.normalize(torch.stack(Zv_list, 0), dim=-1).to(device)   # [B, 512]
    text_labels = torch.tensor(text_labels, device=device)          # [B]

    # =========================================================
    # 4. 根据评测模式选择 Query 和 Bank
    # =========================================================
    if args.mode == "t2t":
        print("[信息] 评测模式: 触觉 -> 文本")
        Z_query, Z_bank, labels = Zt, Zx_bank, text_labels
        query_names, bank_names = seq_names, bank_texts
    elif args.mode == "t2v":
        print("[信息] 评测模式: 触觉 -> 视觉")
        Z_query, Z_bank = Zt, Zv
        labels = torch.arange(len(items), device=device)
        query_names, bank_names = seq_names, seq_names
    elif args.mode == "v2t":
        print("[信息] 评测模式: 视觉 -> 文本")
        Z_query, Z_bank, labels = Zv, Zx_bank, text_labels
        query_names, bank_names = seq_names, bank_texts

    # =========================================================
    # 5. 计算相似度并评估 Recall@K
    # =========================================================
    sim = Z_query @ Z_bank.t()  # [B_query, N_bank]

    r1  = recall_at_k(sim, labels, 1)
    r5  = recall_at_k(sim, labels, 5)
    r10 = recall_at_k(sim, labels, 10)
    guide_str = "(vision-guided)" if args.use_vision_guide else "(global-only)"
    print(f"[{args.mode.upper()} {guide_str} Eval:{args.bank}] R@1/5/10 = {r1:.3f} / {r5:.3f} / {r10:.3f}")

    # =========================================================
    # 6. 保存 CSV 结果
    # =========================================================
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "ground_truth", "R1_hit", "top5_retrieved_items"])
            top5_indices = sim.topk(5, dim=1).indices.tolist()
            labels_list = labels.tolist()
            for i, idxs in enumerate(top5_indices):
                query_id = query_names[i]
                gt_label_idx = labels_list[i]
                ground_truth = bank_names[gt_label_idx]
                r1_hit = int(idxs[0] == gt_label_idx)
                top5_names = "|".join(bank_names[j] for j in idxs)
                w.writerow([query_id, ground_truth, r1_hit, top5_names])
        print(f"CSV 结果已保存 -> {args.csv}")


if __name__ == "__main__":
    main()
