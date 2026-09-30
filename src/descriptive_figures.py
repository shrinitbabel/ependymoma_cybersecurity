"""Supplementary Figures S3-S5 (descriptive comparisons), regenerated from the final synthetic runs.

S3: symptom prevalence across the clinical course (pre-op, post-op, follow-up), real cohort vs each model.
S4: discharge disposition proportions. S5: extent of resection proportions.
Synthetic values are means over 10 runs; error bars show the range across runs.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, INK, INK_2, as_str, build_metadata, generate, load_real, save, style

MODELS = ["GaussianCopula", "TVAE", "CTGAN", "CopulaGAN"]  # same order as the main figures
SEEDS = range(10)
STAGES = ["Pre-op", "Post-op", "Follow-up"]
SYMPTOMS = {
    "Hydrocephalus": ["Preop Hydrocephalus", "Postop Hydrocephalus", None],
    "Cranial nerve palsy": ["Preop CN Palsy (Y/N)", "New/worse Postop CN Palsy Y/N", "Followup CN Palsy Y/N"],
    "Cerebellar signs": ["Preop Cerebellar Signs", "Postop cerebellar deficit",
                         "Cerebellar function at followup compared to preop"],
    "Nausea/vomiting": ["Preop Nausea/Vomiting", None, None],
    "Weakness": ["Preop Weakness", "Postop Weakness", "Followup Weakness"],
    "Sensory loss": ["Preop Sensory Loss", "Postop Sensory Changes", "Followup Sensory Changes"],
    "Vertigo": ["Preop vertigo", None, "Vertigo at followup"],
    "Papilledema": ["Preop Papilledema (Y/N)", None, None],
    "Intraoperative complications": [None, "Intraop Complications", None],
}


def pct_yes(df, col):
    return 100 * (as_str(df[col]) == "Y").mean()


def load():
    real = load_real()
    metadata = build_metadata(real)
    return real, {m: [generate(real, metadata, m, s) for s in SEEDS] for m in MODELS}


def trajectories(real, synth):
    style()
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0), sharey=True)
    offsets = dict(zip(["Real"] + MODELS, np.linspace(-0.12, 0.12, 5)))
    for i, (ax, (name, cols)) in enumerate(zip(axes.ravel(), SYMPTOMS.items())):
        idx = [k for k, c in enumerate(cols) if c]
        xs = np.array(idx, dtype=float)
        for source in ["Real"] + MODELS:
            if source == "Real":
                y, lo, hi = [pct_yes(real, cols[k]) for k in idx], None, None
            else:
                runs = np.array([[pct_yes(d, cols[k]) for k in idx] for d in synth[source]])
                y, lo, hi = runs.mean(0), runs.min(0), runs.max(0)
            x = xs + offsets[source]
            color = INK if source == "Real" else COLORS[source]
            ax.plot(x, y, "-o", color=color, lw=2.2 if source == "Real" else 1.3, ms=4.5 if source == "Real" else 3.5,
                    zorder=4 if source == "Real" else 3, label=source)
            if lo is not None:
                ax.vlines(x, lo, hi, color=color, lw=0.8, alpha=0.6, zorder=2)
        ax.set_title(name, fontsize=8.5, loc="left")
        ax.text(-0.02, 1.08, "ABCDEFGHI"[i], transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right")
        ax.set_xticks(range(3))
        ax.set_xticklabels(STAGES, fontsize=7)
        ax.set_xlim(-0.4, 2.4)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Patients, %", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels = ["Real cohort (n = 18)" if l == "Real" else l for l in labels]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "S_symptom_trajectories")


def category_bars(real, synth, column, categories, labels, name):
    style()
    sources = ["Real"] + MODELS
    width = 0.8 / len(sources)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for j, source in enumerate(sources):
        if source == "Real":
            vals = np.array([(as_str(real[column]) == c).mean() for c in categories])
            err = None
        else:
            runs = np.array([[(as_str(d[column]) == c).mean() for c in categories] for d in synth[source]])
            vals = runs.mean(0)
            err = np.vstack([vals - runs.min(0), runs.max(0) - vals])
        x = np.arange(len(categories)) - 0.4 + width * (j + 0.5)
        color = INK if source == "Real" else COLORS[source]
        ax.bar(x, 100 * vals, width=width * 0.92, color=color, label="Real cohort (n = 18)" if source == "Real" else source,
               yerr=None if err is None else 100 * err, error_kw=dict(ecolor=INK_2, lw=0.8, capsize=1.5))
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Patients, %", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, name)


def main():
    real, synth = load()
    trajectories(real, synth)
    category_bars(real, synth, "Discharge dispo", ["Home", "Rehab", "Hospice", "Deceased before discharge"],
                  ["Home", "Rehabilitation", "Hospice", "Died before discharge"], "S_discharge_disposition")
    category_bars(real, synth, "EOR (GTR, NTR, STR, biopsy)", ["GTR", "NTR", "STR"],
                  ["Gross total resection", "Near-total resection", "Subtotal resection"], "S_extent_of_resection")


if __name__ == "__main__":
    main()
