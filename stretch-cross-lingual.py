"""
Stretch 6B-S2: Cross-Lingual Embedding Comparison
Module 6 Week B — Honors Track

Investigates whether bert-base-multilingual-cased creates a shared
embedding space for Arabic and English climate texts, enabling
cross-lingual semantic similarity without separate per-language models.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch


# ---------------------------------------------------------------------------
# Text selection: 10 EN + 10 AR, paired by topic
# Each EN[i] and AR[i] cover the same subject (IPCC report, COP28, etc.)
# ---------------------------------------------------------------------------
EN_INDICES = [0, 1, 2, 3, 10, 11, 15, 17, 22, 26]
AR_INDICES = [0, 2, 1, 6,  7,  8,  9, 12, 15, 17]

# Human-readable short labels for heatmap axes (40 chars max)
EN_LABELS = [
    "EN: IPCC Sixth Assessment Report",
    "EN: COP28 fossil fuel transition",
    "EN: World Bank $12.5B fund",
    "EN: Jordan climate policy 31%",
    "EN: NASA 2023 warmest year",
    "EN: Greenland ice loss 270Gt",
    "EN: CO2 reaches 424 ppm",
    "EN: Sea level +4.5mm/yr",
    "EN: Dead Sea surface loss 1/3",
    "EN: Jordan water scarcity 80m3",
]

AR_LABELS = [
    "AR: IPCC Sixth Assessment (AR)",
    "AR: COP28 Dubai fossil fuels",
    "AR: Jordan-World Bank $250M",
    "AR: Jordan renewable energy law",
    "AR: NASA 2023 warmest year (AR)",
    "AR: Greenland ice sheet loss",
    "AR: CO2 424 ppm May 2024",
    "AR: WMO sea level rise 4.5mm",
    "AR: Dead Sea surface area loss",
    "AR: Jordan water scarcity 80m3",
]


def load_texts(filepath):
    """Load climate_articles.csv and return selected EN and AR texts."""
    df = pd.read_csv(filepath)
    en_all = df[df["language"] == "en"].reset_index(drop=True)
    ar_all = df[df["language"] == "ar"].reset_index(drop=True)
    en_texts = [en_all.iloc[i]["text"] for i in EN_INDICES]
    ar_texts = [ar_all.iloc[i]["text"] for i in AR_INDICES]
    return en_texts, ar_texts


def mean_pool(last_hidden_state, attention_mask):
    """Attention-mask weighted mean pooling over token dimension."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).squeeze(0)


def embed_texts(texts, tokenizer, model):
    """Compute multilingual BERT embeddings via mean pooling.

    Args:
        texts: List of strings (any language).
        tokenizer: bert-base-multilingual-cased tokenizer.
        model: bert-base-multilingual-cased model in eval mode.

    Returns:
        numpy array of shape (n, 768).
    """
    embeddings = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**encoded)
        emb = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings.append(emb.numpy())
    return np.array(embeddings)


def cosine_similarity_matrix(a, b):
    """Compute cosine similarity between every row in a and every row in b.

    Args:
        a: numpy array (m, d)
        b: numpy array (n, d)

    Returns:
        numpy array (m, n)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def plot_heatmap(sim_matrix, en_labels, ar_labels, output_path):
    """Save a labeled cosine similarity heatmap as a PNG.

    Args:
        sim_matrix: (10, 10) numpy array — EN rows × AR columns.
        en_labels: List of 10 EN axis labels.
        ar_labels: List of 10 AR axis labels.
        output_path: File path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.5,
        vmax=1.0,
        xticklabels=ar_labels,
        yticklabels=en_labels,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(
        "Cross-Lingual Cosine Similarity: English vs Arabic Climate Texts\n"
        "(bert-base-multilingual-cased, mean pooling)",
        fontsize=13,
        pad=14,
    )
    ax.set_xlabel("Arabic texts", fontsize=11)
    ax.set_ylabel("English texts", fontsize=11)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {output_path}")


def analyze_similarity(sim_matrix):
    """Print diagonal (same-topic) vs off-diagonal (different-topic) stats."""
    diagonal = np.diag(sim_matrix)
    # Off-diagonal: all cross-lingual pairs that are NOT same-topic
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    off_diag = sim_matrix[mask]

    print("\n=== Cross-Lingual Similarity Analysis ===")
    print(f"Same-topic pairs  (diagonal) — mean: {diagonal.mean():.4f}, "
          f"min: {diagonal.min():.4f}, max: {diagonal.max():.4f}")
    print(f"Diff-topic pairs (off-diag)  — mean: {off_diag.mean():.4f}, "
          f"min: {off_diag.min():.4f}, max: {off_diag.max():.4f}")
    print(f"Delta (same - diff): {diagonal.mean() - off_diag.mean():.4f}")

    print("\nTop 5 same-topic pairs:")
    for i in np.argsort(diagonal)[::-1][:5]:
        print(f"  [{i}] {EN_LABELS[i][:45]} <-> {AR_LABELS[i][:45]}: {diagonal[i]:.4f}")

    print("\nLowest same-topic pairs:")
    for i in np.argsort(diagonal)[:3]:
        print(f"  [{i}] {EN_LABELS[i][:45]} <-> {AR_LABELS[i][:45]}: {diagonal[i]:.4f}")

    # Within-language comparison baseline using EN embeddings only
    return diagonal, off_diag


if __name__ == "__main__":
    DATA_PATH = "data/climate_articles.csv"
    HEATMAP_PATH = "cross_lingual_heatmap.png"
    MODEL_NAME = "bert-base-multilingual-cased"

    print(f"Loading model: {MODEL_NAME} (~680MB, may take a moment)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")

    print("\nLoading texts...")
    en_texts, ar_texts = load_texts(DATA_PATH)
    print(f"  {len(en_texts)} English texts, {len(ar_texts)} Arabic texts")

    print("\nEmbedding English texts...")
    en_embs = embed_texts(en_texts, tokenizer, model)
    print(f"  EN embeddings: {en_embs.shape}")

    print("Embedding Arabic texts...")
    ar_embs = embed_texts(ar_texts, tokenizer, model)
    print(f"  AR embeddings: {ar_embs.shape}")

    print("\nComputing 10x10 cross-lingual cosine similarity matrix...")
    sim_matrix = cosine_similarity_matrix(en_embs, ar_embs)
    print("Similarity matrix (EN rows x AR cols):")
    print(np.round(sim_matrix, 3))

    plot_heatmap(sim_matrix, EN_LABELS, AR_LABELS, HEATMAP_PATH)
    analyze_similarity(sim_matrix)

    # Within-language EN baseline for comparison
    en_sim = cosine_similarity_matrix(en_embs, en_embs)
    mask_en = ~np.eye(10, dtype=bool)
    print(f"\nWithin-language EN off-diagonal mean: {en_sim[mask_en].mean():.4f}")
    print(f"Cross-lingual off-diagonal mean:      {sim_matrix[mask_en].mean():.4f}")
    print("\nDone.")