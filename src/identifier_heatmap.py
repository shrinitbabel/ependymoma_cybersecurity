"""Figure 5: privacy by variable, without vs with feature-level anonymization (heatmap, original figure layout).

Left block, direct identifiers: real patients whose identifier was reproduced exactly in the synthetic data,
minus the number expected by chance (mean over 10 runs); * = above chance in a paired Wilcoxon test (p < 0.05).
Right block, clinical variables: gain in attack success over a prior-only guess (strongest attacker per run),
percentage points, mean over 10 runs. "With anonymization" = AnonymizedFaker; its clinical data are identical to
the main analysis runs.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from common import INK, INK_2, OUT, save, style

MODELS = ["GaussianCopula", "TVAE", "CTGAN", "CopulaGAN"]  # same order as Figures 3, 4 and 6
IDENTIFIERS = ["MRN", "Initials", "Date of Birth", "Date of Sx"]
ID_LABELS = ["MRN", "Initials", "Date of\nbirth", "Date of\nsurgery"]
TARGETS = ["Death (Y/N)", "Recurrence", "Postop complication", "EOR (GTR, NTR, STR, biopsy)", "Radiotherapy (Y/N)",
           "Discharge dispo", "Overall Survival (yrs)", "PFS (After Sx; Months)"]
TARGET_LABELS = ["Death", "Recur-\nrence", "Postop\ncompli-\ncation", "Extent of\nresection", "Radio-\ntherapy",
                 "Discharge\ndispo-\nsition", "Overall\nsurvival", "PFS"]
# Diverging: blue = no risk (at or below chance), gray = chance, red = risk above chance.
CMAP = LinearSegmentedColormap.from_list("risk", ["#2a78d6", "#f0efec", "#e34948"])


def identifier_matrix():
    t = pd.read_csv(OUT / "identifier_reproduction_tests.csv")
    vals, stars = np.zeros((8, 4)), np.zeros((8, 4), bool)
    for r, (version, model) in enumerate([(v, m) for v in ("Identifiers retained", "AnonymizedFaker") for m in MODELS]):
        for c, ident in enumerate(IDENTIFIERS):
            row = t[(t.Model == model) & (t.Version == version) & (t.Identifier == ident)].iloc[0]
            vals[r, c] = row.Exact_mean - row.Chance_mean
            stars[r, c] = row.Exact_mean > row.Chance_mean and row.p_Wilcoxon < 0.05
    return vals, stars


def clinical_matrix():
    retained = pd.read_csv(OUT / "privacy_identifiers_retained_by_run.csv")
    anonymized = pd.read_csv(OUT / "privacy_revised_by_run.csv")
    vals = np.zeros((8, 8))
    for block, runs in enumerate([retained, anonymized]):
        best = runs.loc[runs.groupby(["Source", "Seed", "Target"]).Correct.idxmax()]
        mean = best.groupby(["Source", "Target"]).Lift_pp.mean()
        for i, model in enumerate(MODELS):
            vals[block * 4 + i] = [mean[(model, t)] for t in TARGETS]
    return vals


def draw(ax, vals, xlabels, fmt, norm, stars=None):
    im = ax.imshow(vals, cmap=CMAP, norm=norm, aspect="auto")
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            text = fmt.format(round(vals[r, c], 1 if "1f" in fmt else 0) + 0.0) + ("*" if stars is not None and stars[r, c] else "")
            ax.text(c, r, text, ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=7, rotation=0)
    ax.set_yticks(range(8))
    ax.set_yticklabels(MODELS * 2, fontsize=7.5)
    ax.axhline(3.5, color="white", lw=3)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    return im


def main():
    style()
    id_vals, id_stars = identifier_matrix()
    clin_vals = clinical_matrix()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [4, 8.5]})
    im1 = draw(ax1, id_vals, ID_LABELS, "{:.1f}", TwoSlopeNorm(0, vmin=-3, vmax=3), id_stars)
    im2 = draw(ax2, clin_vals, TARGET_LABELS, "{:.0f}", TwoSlopeNorm(0, vmin=-30, vmax=30))
    ax2.set_yticklabels([])
    ax1.set_title("Direct identifiers", fontsize=9, loc="left")
    ax2.set_title("Clinical variables (attribute inference)", fontsize=9, loc="left")
    for ax, letter in [(ax1, "A"), (ax2, "B")]:
        ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")
    for y, label in [(0.75, "Without\nanonymization"), (0.25, "With\nanonymization")]:
        ax1.text(-0.72, y, label, transform=ax1.transAxes, ha="center", va="center", fontsize=7.5,
                 style="italic", color=INK_2, rotation=90)
    for y0, y1 in [(0.52, 0.98), (0.02, 0.48)]:
        ax1.plot([-0.62, -0.62], [y0, y1], transform=ax1.transAxes, color=INK_2, lw=1, clip_on=False)
    c1 = fig.colorbar(im1, ax=ax1, orientation="horizontal", fraction=0.06, pad=0.2)
    c1.set_label("Patients reproduced beyond chance", fontsize=7)
    c2 = fig.colorbar(im2, ax=ax2, orientation="horizontal", fraction=0.06, pad=0.2)
    c2.set_label("Attack success above guessing, percentage points", fontsize=7)
    for c in (c1, c2):
        c.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    save(fig, "Figure5_privacy_by_variable")


if __name__ == "__main__":
    main()
