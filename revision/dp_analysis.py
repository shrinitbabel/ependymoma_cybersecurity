"""Differentially private (DP) synthesizers: privacy-utility trade-off at n = 18.

DP synthesizers (formal epsilon-DP guarantee for every patient's record):
  * DPCopula (Li et al., 2014): DP counterpart of GaussianCopula. Half of epsilon is spent on the
    61 marginal histograms (Laplace noise on counts; numerical variables binned over public
    clinical bounds), half on Kendall's tau for every variable pair (Laplace noise, sensitivity
    4 / (n + 1)), converted to a correlation matrix (rho = sin(pi * tau / 2)) and projected to the
    nearest positive-definite correlation matrix.
  * DP-CTGAN and PATE-CTGAN (SmartNoise): DP counterparts of CTGAN.
  * MST (SmartNoise): marginal-based DP synthesizer (winner of the NIST 2018 DP synthetic data
    challenge).
There is no maintained DP implementation of TVAE.

Numerical variables use fixed public bounds (clinically plausible ranges chosen without looking at
the data), because DP bound estimation is not possible with 18 records. Each synthesizer is run at
epsilon = 1, 3, 10 and 30 across several seeds and evaluated with the same utility (Spearman
correlation-structure agreement) and privacy (attribute inference lift over a prior-only guess;
records closer to a real patient than any two real patients) measures as the non-DP synthesizers.
"""
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm

from common import (BASELINE, COLORS, INK, INK_2, OUT, SYNTH_DIR, as_str, load_real, match_fraction,
                    numeric_columns, save, seed_all, style)
from privacy_analysis import ATTACKERS, CAT_TARGETS, NUM_TARGETS, attack, attack_applicable, prior_guess
from utility_analysis import encoded_matrix, upper

warnings.filterwarnings("ignore")

N_SYNTH = 1000
EPSILONS = [1.0, 3.0, 10.0, 30.0]
BOUNDS = {"Age at Diagnosis": (0, 100), "Age at Sx": (0, 100), "BMI": (10, 60), "Tumor Size 1 (cm)": (0, 10),
          "Tumor Size 2 (cm)": (0, 10), "Tumor Size 3 (cm)": (0, 10), "Mitotic figures in 10 hpf": (0, 50),
          "Duration of symptoms preop (months)": (0, 60), "EBL": (0, 3000), "LOS (days)": (0, 100),
          "PFS (After Sx; Months)": (0, 240), "Overall Survival (yrs)": (0, 20),
          "Length of followup (months)": (0, 240)}
N_BINS = 10
DP_COLORS = {"DPCopula": "#2a78d6", "DP-CTGAN": "#1baf7a", "PATE-CTGAN": "#4a3aa7", "MST": "#e87ba4"}


def prepared(real):
    df = real.copy()
    num = numeric_columns(real)
    for c in real.columns:
        df[c] = df[c].clip(*BOUNDS[c]).astype(float) if c in num else as_str(df[c])
    return df, num, [c for c in real.columns if c not in num]


# ---------- DPCopula ----------

def nearest_correlation(r):
    vals, vecs = np.linalg.eigh((r + r.T) / 2)
    r = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(r))
    return r / np.outer(d, d)


def dp_copula(df, num, cat, eps, seed):
    rng = np.random.default_rng(seed)
    cols = list(df.columns)
    n = len(df)
    eps_marg = eps / 2 / len(cols)
    eps_corr = eps / 2

    # Noisy marginals.
    marginals, codes = {}, {}
    for c in cols:
        if c in num:
            edges = np.linspace(*BOUNDS[c], N_BINS + 1)
            idx = np.clip(np.digitize(df[c], edges[1:-1]), 0, N_BINS - 1)
            levels = np.arange(N_BINS)
        else:
            levels = np.array(sorted(df[c].unique()))
            idx = np.searchsorted(levels, df[c].to_numpy())
        counts = np.bincount(idx, minlength=len(levels)).astype(float) + rng.laplace(0, 1 / eps_marg, len(levels))
        p = np.clip(counts, 0, None)
        p = p / p.sum() if p.sum() > 0 else np.full(len(levels), 1 / len(levels))
        marginals[c] = (levels, p)
        codes[c] = idx

    # Noisy Kendall's tau for every pair -> correlation matrix.
    k = len(cols)
    n_pairs = k * (k - 1) / 2
    scale = (4 / (n + 1)) * n_pairs / eps_corr
    r = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            tau = kendalltau(codes[cols[i]], codes[cols[j]]).statistic
            tau = 0.0 if np.isnan(tau) else tau
            tau = np.clip(tau + rng.laplace(0, scale), -1, 1)
            r[i, j] = r[j, i] = np.sin(np.pi * tau / 2)
    r = nearest_correlation(r)

    # Sample: Gaussian copula -> inverse of each noisy marginal CDF.
    u = norm.cdf(rng.multivariate_normal(np.zeros(k), r, N_SYNTH))
    out = {}
    for j, c in enumerate(cols):
        levels, p = marginals[c]
        pick = np.clip(np.searchsorted(np.cumsum(p), u[:, j]), 0, len(levels) - 1)
        if c in num:
            edges = np.linspace(*BOUNDS[c], N_BINS + 1)
            out[c] = rng.uniform(edges[pick], edges[pick + 1])
        else:
            out[c] = levels[pick]
    return pd.DataFrame(out)


# ---------- SmartNoise synthesizers ----------

def smartnoise(df, num, cat, name, eps, seed):
    from snsynth import Synthesizer
    from snsynth.transform import BinTransformer, MinMaxTransformer
    from snsynth.transform.table import TableTransformer

    seed_all(seed)
    if name == "MST":
        cons = {c: BinTransformer(bins=N_BINS, lower=lo, upper=hi) for c, (lo, hi) in BOUNDS.items()}
        tt = TableTransformer.create(df, style="cube", categorical_columns=cat, continuous_columns=num,
                                     constraints=cons)
        synth = Synthesizer.create("mst", epsilon=eps)
    else:
        cons = {c: MinMaxTransformer(lower=lo, upper=hi, negative=False) for c, (lo, hi) in BOUNDS.items()}
        tt = TableTransformer.create(df, style="gan", categorical_columns=cat, continuous_columns=num,
                                     constraints=cons)
        synth = Synthesizer.create({"DP-CTGAN": "dpctgan", "PATE-CTGAN": "patectgan"}[name], epsilon=eps,
                                   verbose=False)
    synth.fit(df, transformer=tt, preprocessor_eps=0.0)
    return synth.sample(N_SYNTH)


def generate_dp(df, num, cat, name, eps, seed):
    path = SYNTH_DIR / "dp" / f"{name}_eps{eps:g}_seed{seed}.csv"
    if path.exists():
        return pd.read_csv(path)
    out = dp_copula(df, num, cat, eps, seed) if name == "DPCopula" else smartnoise(df, num, cat, name, eps, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


# ---------- evaluation (same measures as the non-DP synthesizers) ----------

def evaluate(real, synth, seed, real_sp, real_min_dcr):
    sp = encoded_matrix(synth, real).corr(method="spearman").fillna(0.0)
    spearman_r = np.corrcoef(upper(real_sp), upper(sp))[0, 1]
    lifts = []
    for t in CAT_TARGETS + NUM_TARGETS:
        prior = prior_guess(synth, real, real, t).sum()
        best = max(attack(synth, real, real, t, a, seed).sum() for a in ATTACKERS if attack_applicable(a, t))
        lifts.append(100 * (best - prior) / len(real))
    dcr = 1 - match_fraction(synth, real, real).max(axis=1)
    return {"Spearman_r": spearman_r, "Attack_lift_pp": float(np.mean(lifts)),
            "Too_close_pct": 100 * float(np.mean(dcr < real_min_dcr))}


def plot(res, reference):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    metrics = [("Spearman_r", "Utility: correlation-structure\nagreement (higher = better)"),
               ("Attack_lift_pp", "Privacy: attack success above\nguessing, pp (lower = better)"),
               ("Too_close_pct", "Privacy: records closer than any two\nreal patients, % (lower = better)")]
    for ax, (metric, title) in zip(axes, metrics):
        for name, g in res.groupby("Model"):
            m = g.groupby("Epsilon")[metric].agg(["mean", "min", "max"])
            ax.fill_between(m.index, m["min"], m["max"], color=DP_COLORS[name], alpha=0.15, lw=0)
            ax.plot(m.index, m["mean"], "-o", color=DP_COLORS[name], lw=1.6, ms=4, label=name)
        for ref, style_ in [("GaussianCopula", (0, (4, 2))), ("TVAE", (0, (1, 1.5)))]:
            ax.axhline(reference[ref][metric], color=COLORS[ref], lw=1.2, ls=style_)
        ax.axhline(reference[BASELINE][metric], color=COLORS[BASELINE], lw=1.2, ls=(0, (3, 3)))
        ax.set_xscale("log")
        ax.set_xticks(EPSILONS)
        ax.set_xticklabels([f"{e:g}" for e in EPSILONS])
        ax.set_xlabel("Privacy budget ε (smaller = stronger privacy)", fontsize=7)
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=COLORS["GaussianCopula"], ls=(0, (4, 2)), lw=1.2),
                plt.Line2D([], [], color=COLORS["TVAE"], ls=(0, (1, 1.5)), lw=1.2),
                plt.Line2D([], [], color=COLORS[BASELINE], ls=(0, (3, 3)), lw=1.2)]
    labels += ["GaussianCopula (no DP)", "TVAE (no DP)", "Independent baseline"]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "S_dp_tradeoff")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["DPCopula", "DP-CTGAN", "PATE-CTGAN", "MST"])
    args = parser.parse_args()

    real = load_real()
    df, num, cat = prepared(real)
    real_sp = encoded_matrix(real, real).corr(method="spearman").fillna(0.0)
    rr = match_fraction(real, real, real)
    np.fill_diagonal(rr, 0)
    real_min_dcr = (1 - rr.max(axis=1)).min()

    rows = []
    for name in args.models:
        for eps in EPSILONS:
            for seed in range(args.seeds):
                synth = generate_dp(df, num, cat, name, eps, seed)
                rows.append({"Model": name, "Epsilon": eps, "Seed": seed,
                             **evaluate(real, synth, seed, real_sp, real_min_dcr)})
                print(f"{name} eps={eps:g} seed {seed} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "dp_by_run.csv", index=False)

    # Non-DP reference values from the main analyses (means over 10 seeds).
    rel = pd.read_csv(OUT / "relationships_summary.csv")
    priv = pd.read_csv(OUT / "privacy_revised_by_model.csv")
    reid = pd.read_csv(OUT / "reidentification_summary.csv")
    reference = {}
    for src in ["GaussianCopula", "TVAE", BASELINE]:
        reference[src] = {
            "Spearman_r": rel[(rel.Source == src) & (rel.Metric == "Spearman_r")].Mean.iloc[0],
            "Attack_lift_pp": priv[(priv.Source == src) & (priv.Metric == "Mean_lift_pp")].Mean.iloc[0],
            "Too_close_pct": reid[(reid.Source == src) & (reid.Metric == "Too_close_pct")].Mean.iloc[0],
        }
    summary = res.groupby(["Model", "Epsilon"])[["Spearman_r", "Attack_lift_pp", "Too_close_pct"]].agg(
        ["mean", "std"]).round(3)
    summary.to_csv(OUT / "dp_summary.csv")
    plot(res, reference)

    pd.set_option("display.width", 200)
    print(summary.to_string())
    print(pd.DataFrame(reference).round(3).to_string())


if __name__ == "__main__":
    main()
