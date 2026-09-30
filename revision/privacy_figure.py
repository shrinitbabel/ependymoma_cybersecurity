"""Figure 6: attribute inference attacks, in the style of the original manuscript figure (marker per attacker).

A: gain in attack success over the prior-only guess (percentage points), averaged over targets, for each of
   the 10 runs and each attacker.
B: training patients minus held-out patients (holdout analysis): difference in attack gain, mean and
   bootstrap 95% CI per attacker. Values above 0 indicate memorization of training patients.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BASELINE, DISPLAY, INK, INK_2, MODELS, OUT, save, style

SOURCES = ["GaussianCopula", "TVAE", "CTGAN", "CopulaGAN", BASELINE]  # same order as Figures 3-4
ATTACKERS = {"Naive Bayes": ("^", -0.2), "kNN": ("x", 0.0), "Random Forest": ("s", 0.2)}
N_BOOT = 2000


def marker_kwargs(marker):
    if marker == "x":
        return dict(marker="x", color=INK, linewidths=0.9)
    return dict(marker=marker, facecolors="#d9d8d4", edgecolors=INK, linewidths=0.7)


def panel_gain(ax, retained, title):
    """Attack gain per run and attacker. retained=True uses the synthetic data trained with direct identifiers
    retained; otherwise the anonymized data (AnonymizedFaker; identical clinical data to the main runs).
    The reshuffled reference is the same in both panels."""
    main = pd.read_csv(OUT / "privacy_revised_by_run.csv")
    runs = pd.read_csv(OUT / "privacy_identifiers_retained_by_run.csv") if retained else main
    runs = pd.concat([runs[runs.Source != BASELINE], main[main.Source == BASELINE]])
    per_run = runs.groupby(["Source", "Seed", "Attacker"]).Lift_pp.mean().reset_index()
    rng = np.random.default_rng(0)
    for i, source in enumerate(SOURCES):
        for attacker, (marker, dx) in ATTACKERS.items():
            v = per_run[(per_run.Source == source) & (per_run.Attacker == attacker)].Lift_pp.to_numpy()
            ax.scatter(i + dx + rng.uniform(-0.05, 0.05, len(v)), v, s=18, zorder=3, **marker_kwargs(marker))
    ax.axhline(0, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    ax.set_ylabel("Attack success above\nguessing, pp", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")


def panel_membership(ax):
    aia = pd.read_csv(OUT / "holdout_aia_by_patient.csv")
    aia["Lift"] = aia.Correct.astype(float) - aia.Prior_correct.astype(float)
    per_patient = aia.groupby(["Source", "Attacker", "Repeat", "Fold", "Patient", "Member"]).Lift.mean().reset_index()
    rng = np.random.default_rng(0)
    for i, source in enumerate(SOURCES):
        for attacker, (marker, dx) in ATTACKERS.items():
            g = per_patient[(per_patient.Source == source) & (per_patient.Attacker == attacker)]
            mem, non = g[g.Member].Lift.to_numpy(), g[~g.Member].Lift.to_numpy()
            diff = 100 * (mem.mean() - non.mean())
            boots = [100 * (rng.choice(mem, len(mem)).mean() - rng.choice(non, len(non)).mean())
                     for _ in range(N_BOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ax.plot([i + dx, i + dx], [lo, hi], color=INK_2, lw=0.9, zorder=2)
            ax.scatter([i + dx], [diff], s=30, zorder=3, **marker_kwargs(marker))
    ax.axhline(0, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    ax.set_ylabel("Training minus held-out patients,\npercentage points (95% CI)", fontsize=8)
    ax.set_title("Attack success on training vs held-out patients (memorization test)", fontsize=9, loc="left")


def anonymization_figure():
    """Supplementary figure: attacks without vs with anonymization, plus the memorization test."""
    style()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.6), sharex=True)
    panel_gain(axes[0], True, "Attribute inference without anonymization (direct identifiers retained in training)")
    panel_gain(axes[1], False, "Attribute inference with feature-level anonymization (AnonymizedFaker)")
    lo = min(a.get_ylim()[0] for a in axes[:2])
    hi = max(a.get_ylim()[1] for a in axes[:2])
    for a in axes[:2]:
        a.set_ylim(lo, hi)
    panel_membership(axes[2])
    axes[2].set_title("Training vs held-out patients (memorization test, anonymized data)", fontsize=9, loc="left")
    for ax, letter in zip(axes, "ABC"):
        ax.set_xticks(range(len(SOURCES)))
        ax.set_xticklabels([DISPLAY.get(s, s).replace("\n", " ") for s in SOURCES], fontsize=8)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.text(-0.1, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    axes[2].set_ylabel("Training minus held-out,\npp (95% CI)", fontsize=8)
    handles = [plt.Line2D([], [], ls="", markersize=6,
                          **({"marker": "x", "color": INK} if m == "x" else
                             {"marker": m, "markerfacecolor": "#d9d8d4", "markeredgecolor": INK}), label=a)
               for a, (m, _) in ATTACKERS.items()]
    axes[0].legend(handles=handles, title="Attacker", fontsize=7, title_fontsize=7.5, loc="upper right",
                   frameon=True)
    fig.tight_layout(h_pad=2.2)
    save(fig, "S_privacy_by_anonymization")


def main_figure():
    """Figure 6: attack gain (anonymized data) and the memorization test."""
    style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4))
    panel_gain(axes[0], False, "Attack success above a prior-only guess (10 runs per model)")
    panel_membership(axes[1])
    for ax, letter in zip(axes, "AB"):
        ax.set_xticks(range(len(SOURCES)))
        ax.set_xticklabels([DISPLAY.get(s, s).replace(chr(10), " ") for s in SOURCES], fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.text(-0.09, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
    handles = [plt.Line2D([], [], ls="", markersize=6,
                          **({"marker": "x", "color": INK} if m == "x" else
                             {"marker": m, "markerfacecolor": "#d9d8d4", "markeredgecolor": INK}), label=a)
               for a, (m, _) in ATTACKERS.items()]
    axes[0].legend(handles=handles, title="Attacker", fontsize=7, title_fontsize=7.5, loc="upper right",
                   frameon=True)
    fig.tight_layout(h_pad=2.5)
    save(fig, "Figure6_privacy_attacks")


def main():
    main_figure()
    anonymization_figure()


if __name__ == "__main__":
    main()
