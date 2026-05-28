"""
visualize/tsne_stage3.py - E002 Stage 3 嵌入空间 t-SNE 可视化

验证目标：
  1. 三模态嵌入（触觉 / 深度 / RGB）是否在 512d 空间中对齐
  2. 同一 episode 的不同模态向量是否聚拢
  3. 不同数据集来源是否存在明显 domain gap

用法：
  python visualize/tsne_stage3.py \
    --ckpt outputs/exp002_stage3_full/ckpt_stage3_full_final.pt \
    --data_dirs data/inboxpicking-01 data/inboxpicking-02 ... \
    --max_samples 500 --output_dir outputs/exp002_stage3_full/viz
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib

_stage3_mod = importlib.import_module("train.3_3D_align")
DepthGuidedTLVModel = _stage3_mod.DepthGuidedTLVModel

from overfit.dataset import LeRobotTactileDataset


def build_model(cfg):
    model = DepthGuidedTLVModel(
        feature_dim=cfg["feature_dim"],
        tau_init=cfg["tau_init"],
        use_gtr_side_head=cfg["use_gtr_side_head"],
        gtr_mode=cfg["gtr_mode"],
        gtr_q_pos_dim=cfg["gtr_q_pos_dim"],
        gtr_map_size=cfg["gtr_map_size"],
        patch_proj_dim=cfg["patch_proj_dim"],
        patch_temperature=cfg["patch_temperature"],
    )
    return model


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)
    print(f"[Viz] Loaded checkpoint (epoch {ckpt.get('epoch', '?')}): {ckpt_path}")
    return model


@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples):
    model.eval()

    all_z_tactile = []
    all_z_depth = []
    all_z_rgb = []
    all_z_aligned = []
    all_episode = []
    all_dataset = []

    n_collected = 0
    for batch in dataloader:
        if n_collected >= max_samples:
            break

        tactile = batch["tactile_grid"].to(device)  # [B, T, 2, 16, 16]
        depth = batch["depth"].to(device)  # [B, T, 1, H, W]
        side_rgb = batch["side_rgb"].to(device)  # [B, T, 3, H, W]

        B = tactile.shape[0]

        # 取左通道触觉: [B, T, 2, 16, 16] -> [B, T, 16, 16]
        tac_left = tactile[:, :, 0, :, :]

        out = model(
            x_tac=tac_left,
            x_dep=depth,
            x_side_rgb=side_rgb,
        )

        z_t = out["z_t_raw"].cpu()  # [B, 512]
        z_d = out["z_depth"].cpu()  # [B, 512]
        z_s = out["z_side"].cpu()  # [B, 512]
        z_a = out["z_t_aligned"].cpu()  # [B, 512]

        n_take = min(B, max_samples - n_collected)
        all_z_tactile.append(z_t[:n_take])
        all_z_depth.append(z_d[:n_take])
        all_z_rgb.append(z_s[:n_take])
        all_z_aligned.append(z_a[:n_take])
        all_episode.append(batch["episode_index"][:n_take].numpy())
        all_dataset.append(batch["dataset_index"][:n_take].numpy())

        n_collected += n_take

    return {
        "z_tactile": torch.cat(all_z_tactile).numpy(),
        "z_depth": torch.cat(all_z_depth).numpy(),
        "z_rgb": torch.cat(all_z_rgb).numpy(),
        "z_aligned": torch.cat(all_z_aligned).numpy(),
        "episode": np.concatenate(all_episode),
        "dataset": np.concatenate(all_dataset),
    }


def compute_retrieval_metrics(emb_dict):
    """Cross-modal retrieval: tactile->depth, depth->RGB, tactile->RGB"""
    metrics = {}
    pairs = [
        ("tactile", "depth", "z_tactile", "z_depth"),
        ("depth", "rgb", "z_depth", "z_rgb"),
        ("tactile", "rgb", "z_tactile", "z_rgb"),
        ("aligned", "depth", "z_aligned", "z_depth"),
    ]
    for name_a, name_b, key_a, key_b in pairs:
        a = emb_dict[key_a]
        b = emb_dict[key_b]
        # Cosine similarity matrix
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        sim = a_norm @ b_norm.T  # [N, N]
        N = sim.shape[0]

        # R@1, R@5, R@10
        ranks = []
        for i in range(N):
            row = sim[i]
            sorted_idx = np.argsort(-row)
            rank = np.where(sorted_idx == i)[0][0] + 1
            ranks.append(rank)
        ranks = np.array(ranks)

        r1 = (ranks <= 1).mean() * 100
        r5 = (ranks <= 5).mean() * 100
        r10 = (ranks <= 10).mean() * 100
        median_rank = np.median(ranks)

        key = f"{name_a}->{name_b}"
        metrics[key] = {"R@1": r1, "R@5": r5, "R@10": r10, "MedR": median_rank}
        print(f"  {key}: R@1={r1:.1f}% R@5={r5:.1f}% R@10={r10:.1f}% MedR={median_rank:.0f}")

    return metrics


def plot_tsne_by_modality(emb_2d, labels_modality, n_per_modality, output_path):
    """t-SNE colored by modality type (3 colors)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors_map = {"tactile": "#e74c3c", "depth": "#3498db", "rgb": "#2ecc71", "aligned": "#9b59b6"}
    modality_names = ["tactile", "depth", "rgb", "aligned"]

    for i, name in enumerate(modality_names):
        mask = labels_modality == i
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=colors_map[name],
            s=8,
            alpha=0.5,
            label=name,
        )

    ax.legend(fontsize=11, markerscale=3)
    ax.set_title("t-SNE: Stage 3 Embeddings by Modality", fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_tsne_by_episode(emb_2d, labels_modality, episodes, n_per_modality, output_path):
    """t-SNE colored by episode, shape by modality."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    markers = ["o", "^", "s", "D"]
    modality_names = ["tactile", "depth", "rgb", "aligned"]
    unique_eps = np.unique(episodes)

    cmap = plt.cm.get_cmap("tab20", len(unique_eps))
    ep_color = {ep: cmap(i) for i, ep in enumerate(unique_eps)}

    # Repeat episodes for all 4 modalities
    ep_repeated = np.tile(episodes, 4)

    for i, name in enumerate(modality_names):
        mask = labels_modality == i
        for ep in unique_eps:
            ep_mask = mask & (ep_repeated == ep)
            if ep_mask.sum() == 0:
                continue
            ax.scatter(
                emb_2d[ep_mask, 0],
                emb_2d[ep_mask, 1],
                c=[ep_color[ep]],
                marker=markers[i],
                s=12,
                alpha=0.5,
            )

    # Legend: modality shapes
    legend_mod = [
        Line2D([0], [0], marker=m, color="gray", linestyle="None", markersize=8, label=n)
        for m, n in zip(markers, modality_names)
    ]
    ax.legend(handles=legend_mod, fontsize=10, title="Modality", loc="upper right")
    ax.set_title("t-SNE: Stage 3 Embeddings by Episode (color) × Modality (shape)", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_tsne_by_dataset(emb_2d, labels_modality, datasets, n_per_modality, output_path):
    """t-SNE colored by dataset source."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    unique_ds = np.unique(datasets)
    cmap = plt.cm.get_cmap("Set1", max(len(unique_ds), 3))
    ds_color = {ds: cmap(i) for i, ds in enumerate(unique_ds)}

    ds_repeated = np.tile(datasets, 4)

    for ds in unique_ds:
        mask = ds_repeated == ds
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=[ds_color[ds]],
            s=8,
            alpha=0.4,
            label=f"dataset-{ds:02d}",
        )

    ax.legend(fontsize=9, markerscale=3)
    ax.set_title("t-SNE: Stage 3 Embeddings by Dataset Source", fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cosine_similarity_heatmap(emb_dict, output_path):
    """Cross-modal average cosine similarity heatmap."""
    keys = ["z_tactile", "z_depth", "z_rgb", "z_aligned"]
    names = ["tactile", "depth", "rgb", "aligned"]
    sim_matrix = np.zeros((4, 4))

    for i, ki in enumerate(keys):
        ai = emb_dict[ki]
        ai = ai / (np.linalg.norm(ai, axis=1, keepdims=True) + 1e-8)
        for j, kj in enumerate(keys):
            aj = emb_dict[kj]
            aj = aj / (np.linalg.norm(aj, axis=1, keepdims=True) + 1e-8)
            # Mean diagonal cosine similarity (paired samples)
            sim_matrix[i, j] = np.mean(np.sum(ai * aj, axis=1))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(sim_matrix, vmin=-0.2, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_yticks(range(4))
    ax.set_yticklabels(names, fontsize=11)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{sim_matrix[i, j]:.3f}", ha="center", va="center", fontsize=11)
    fig.colorbar(im)
    ax.set_title("Paired Cosine Similarity (diagonal)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def collate_fn(batch):
    """Custom collate that handles variable-sized tensors and adds metadata."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], (int, float, np.integer)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out


class WrappedDataset(torch.utils.data.Dataset):
    """Wraps LeRobotTactileDataset to add episode_index and dataset_index."""

    def __init__(self, base_dataset, dataset_index):
        self.base = base_dataset
        self.dataset_index = dataset_index

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        sample["dataset_index"] = self.dataset_index
        # episode_index should already be in sample from the base dataset
        if "episode_index" not in sample:
            sample["episode_index"] = 0
        return sample


def main():
    parser = argparse.ArgumentParser(description="E002 Stage 3 t-SNE Visualization")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/exp002_stage3_full/ckpt_stage3_full_final.pt",
    )
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        default=[f"data/inboxpicking-{i:02d}" for i in range(1, 8)],
    )
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_fps", type=int, default=5, help="Lower FPS = sparser sampling")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/exp002_stage3_full/viz",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    cfg = {
        "feature_dim": 512,
        "tau_init": 0.07,
        "use_gtr_side_head": True,
        "gtr_mode": "mlp",
        "gtr_q_pos_dim": 6,
        "gtr_map_size": 16,
        "patch_proj_dim": 128,
        "patch_temperature": 0.1,
    }
    print("[Viz] Building model...")
    model = build_model(cfg)
    model = load_checkpoint(model, args.ckpt, device)
    model = model.to(device)

    # ---- Data ----
    print("[Viz] Loading datasets...")
    datasets = []
    for di, data_dir in enumerate(args.data_dirs):
        if not os.path.exists(data_dir):
            print(f"  [Skip] {data_dir} does not exist")
            continue
        ds = LeRobotTactileDataset(
            repo_id="local/inboxpicking",
            root=data_dir,
            sidecar_root=data_dir,
            T=16,
            target_fps=args.target_fps,
        )
        datasets.append(WrappedDataset(ds, dataset_index=di))
        print(f"  [{di}] {data_dir}: {len(ds)} samples")

    if not datasets:
        print("[Error] No datasets loaded!")
        return

    concat_ds = torch.utils.data.ConcatDataset(datasets)
    loader = DataLoader(
        concat_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False,
    )
    print(f"[Viz] Total samples: {len(concat_ds)}, will use up to {args.max_samples}")

    # ---- Extract embeddings ----
    print("[Viz] Extracting embeddings...")
    emb_dict = extract_embeddings(model, loader, device, args.max_samples)
    N = emb_dict["z_tactile"].shape[0]
    print(f"[Viz] Collected {N} samples")

    # ---- Quantitative: cross-modal retrieval ----
    print("\n[Viz] Cross-modal retrieval metrics:")
    metrics = compute_retrieval_metrics(emb_dict)

    # Save metrics
    import json

    with open(os.path.join(args.output_dir, "retrieval_metrics.json"), "w") as f:
        json.dump({k: {mk: float(mv) for mk, mv in v.items()} for k, v in metrics.items()}, f, indent=2)

    # ---- t-SNE ----
    print("\n[Viz] Running t-SNE (this may take a minute)...")
    all_emb = np.concatenate(
        [emb_dict["z_tactile"], emb_dict["z_depth"], emb_dict["z_rgb"], emb_dict["z_aligned"]]
    )
    labels_modality = np.array([0] * N + [1] * N + [2] * N + [3] * N)

    tsne = TSNE(
        n_components=2,
        perplexity=min(args.tsne_perplexity, len(all_emb) / 4 - 1),
        random_state=args.seed,
        init="pca",
        max_iter=1000,
    )
    emb_2d = tsne.fit_transform(all_emb)

    # ---- Plots ----
    print("[Viz] Generating plots...")
    plot_tsne_by_modality(
        emb_2d,
        labels_modality,
        N,
        os.path.join(args.output_dir, "tsne_by_modality.png"),
    )
    plot_tsne_by_episode(
        emb_2d,
        labels_modality,
        emb_dict["episode"],
        N,
        os.path.join(args.output_dir, "tsne_by_episode.png"),
    )
    plot_tsne_by_dataset(
        emb_2d,
        labels_modality,
        emb_dict["dataset"],
        N,
        os.path.join(args.output_dir, "tsne_by_dataset.png"),
    )
    plot_cosine_similarity_heatmap(
        emb_dict,
        os.path.join(args.output_dir, "cosine_similarity_heatmap.png"),
    )

    # Save raw embeddings for further analysis
    np.savez_compressed(
        os.path.join(args.output_dir, "embeddings.npz"),
        z_tactile=emb_dict["z_tactile"],
        z_depth=emb_dict["z_depth"],
        z_rgb=emb_dict["z_rgb"],
        z_aligned=emb_dict["z_aligned"],
        episode=emb_dict["episode"],
        dataset=emb_dict["dataset"],
        tsne_2d=emb_2d,
        labels_modality=labels_modality,
    )
    print(f"\n[Viz] All outputs saved to {args.output_dir}/")
    print("  - tsne_by_modality.png")
    print("  - tsne_by_episode.png")
    print("  - tsne_by_dataset.png")
    print("  - cosine_similarity_heatmap.png")
    print("  - retrieval_metrics.json")
    print("  - embeddings.npz")


if __name__ == "__main__":
    main()
