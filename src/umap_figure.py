"""Figure 4: UMAP embeddings of real and synthetic patients.

For each synthesizer (run with seed 0) and the reshuffled reference, the 18 real patients and the 1,000
synthetic records are embedded together in two dimensions. Categorical variables are one-hot encoded and
numerical variables standardized (encoders fit on the real data); UMAP uses a fixed random seed.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (BASELINE, COLORS, DISPLAY, INK, MODELS, as_str, build_metadata, generate, independent_marginals,
                    load_real, numeric_columns, save, style)

warnings.filterwarnings("ignore")
SEED = 0


def encoder(real):
    num = numeric_columns(real)
    cat = [c for c in real.columns if c not in num]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(real[cat].apply(as_str))
    scaler = StandardScaler().fit(real[num])

    def encode(df):
        x_num = scaler.transform(df[num].apply(pd.to_numeric, errors="coerce").fillna(real[num].median()))
        return np.hstack([x_num, ohe.transform(df[cat].apply(as_str))])
    return encode


def main():
    real = load_real()
    metadata = build_metadata(real)
    encode = encoder(real)
    sources = ["GaussianCopula", "TVAE", "CTGAN", "CopulaGAN", BASELINE]  # same order as Figure 3
    datasets = {m: generate(real, metadata, m, SEED) for m in MODELS}
    datasets[BASELINE] = independent_marginals(real, SEED)

    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    axes = axes.ravel()
    for i, (ax, source) in enumerate(zip(axes, sources)):
        x = np.vstack([encode(real), encode(datasets[source])])
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(x)
        r, s = emb[: len(real)], emb[len(real):]
        ax.scatter(s[:, 0], s[:, 1], s=4, color=COLORS[source], alpha=0.35, lw=0, label="Synthetic")
        ax.scatter(r[:, 0], r[:, 1], s=22, color=INK, edgecolor="white", lw=0.6, label="Real", zorder=3)
        ax.set_title(DISPLAY.get(source, source).replace("\n", " "), fontsize=8, loc="left", x=0.08)
        ax.text(0, 1.02, "ABCDE"[i], transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=INK, markeredgecolor="white", ms=6,
                          label="Real patients (n = 18)"),
               plt.Line2D([], [], marker="o", ls="", color="#8a8983", alpha=0.6, ms=4,
                          label="Synthetic records (n = 1,000; colored by model)")]
    axes[-1].legend(handles=handles, loc="center", fontsize=7.5, frameon=False)
    fig.text(0.5, 0.01, "UMAP dimension 1", ha="center", fontsize=8)
    fig.text(0.005, 0.5, "UMAP dimension 2", va="center", rotation=90, fontsize=8)
    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    save(fig, "Figure4_umap")


if __name__ == "__main__":
    main()
