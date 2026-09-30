"""Univariate and pairwise distributional similarity (KS statistic, Jensen-Shannon divergence).

For each synthesizer run and the independent-marginals baseline:
  * KS statistic per numerical variable (real vs synthetic);
  * JSD per categorical variable (category proportions) and per numerical variable
    (binned at the real-data quintiles);
  * pairwise JSD on the joint distribution of every pair of binary variables, which
    measures whether combinations of values (relationships) are preserved, not only
    each variable's own distribution.
JSD uses base 2 (0 = identical, 1 = no overlap). Lower is better for all measures.
"""
import argparse
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from common import (DISPLAY, BASELINE, COLORS, INK, MODELS, OUT, as_str, build_metadata, compare_sources, generate,
                    independent_marginals, load_real, mean_ci, numeric_columns, save, style)


def jsd(p, q):
    return float(jensenshannon(p, q, base=2))


def proportions(values, levels):
    counts = pd.Series(values).value_counts()
    return np.array([counts.get(level, 0) for level in levels], dtype=float) / len(values)


def univariate(real, synth, num):
    rows = []
    for c in real.columns:
        if c in num:
            r = real[c].to_numpy(dtype=float)
            s = pd.to_numeric(synth[c], errors="coerce").dropna().to_numpy()
            edges = np.unique(np.quantile(r, [0.2, 0.4, 0.6, 0.8]))
            levels = range(len(edges) + 1)
            rows.append({"Variable": c, "Type": "Numerical", "KS": ks_2samp(r, s).statistic,
                         "JSD": jsd(proportions(np.digitize(r, edges), levels),
                                    proportions(np.digitize(s, edges), levels))})
        else:
            r, s = as_str(real[c]), as_str(synth[c])
            levels = sorted(set(r) | set(s))
            rows.append({"Variable": c, "Type": "Categorical", "KS": np.nan,
                         "JSD": jsd(proportions(r, levels), proportions(s, levels))})
    return rows


def pairwise(real, synth, binary):
    out = []
    for a, b in combinations(binary, 2):
        joint_r = as_str(real[a]) + "|" + as_str(real[b])
        joint_s = as_str(synth[a]) + "|" + as_str(synth[b])
        levels = sorted(set(joint_r) | set(joint_s))
        out.append(jsd(proportions(joint_r, levels), proportions(joint_s, levels)))
    return np.array(out)


def plot(per_run):
    style()
    sources = MODELS + [BASELINE]
    metrics = [("Mean_KS_numerical", "KS statistic, numerical variables"),
               ("Mean_JSD_categorical", "JSD, categorical variables"),
               ("Mean_JSD_pairs", "JSD, joint distribution of binary pairs")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    rng = np.random.default_rng(0)
    ys = np.arange(len(sources))[::-1]
    for ax, (metric, title) in zip(axes, metrics):
        for y, s in zip(ys, sources):
            v = per_run[per_run.Source == s][metric].to_numpy()
            ax.scatter(v, y + rng.uniform(-0.12, 0.12, len(v)), s=14, color=COLORS[s], alpha=0.75, lw=0)
            ax.plot([v.mean()] * 2, [y - 0.28, y + 0.28], color=INK, lw=1.6)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("Lower = more similar", fontsize=8)
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([DISPLAY.get(s, s) for s in sources], fontsize=8)
    axes[0].tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "S_distribution_tests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    real = load_real()
    metadata = build_metadata(real)
    num = numeric_columns(real)
    binary = [c for c in real.columns if c not in num and real[c].nunique() == 2]

    var_rows, run_rows = [], []
    for seed in range(args.seeds):
        datasets = {m: generate(real, metadata, m, seed) for m in MODELS}
        datasets[BASELINE] = independent_marginals(real, seed)
        for source, synth in datasets.items():
            uni = pd.DataFrame(univariate(real, synth, num))
            pairs = pairwise(real, synth, binary)
            var_rows += [{"Source": source, "Seed": seed, **r} for r in uni.to_dict("records")]
            run_rows.append({
                "Source": source, "Seed": seed,
                "Mean_KS_numerical": uni.KS.mean(),
                "Mean_JSD_numerical": uni[uni.Type == "Numerical"].JSD.mean(),
                "Mean_JSD_categorical": uni[uni.Type == "Categorical"].JSD.mean(),
                "Pct_categorical_JSD_below_0.1": 100 * (uni[uni.Type == "Categorical"].JSD < 0.1).mean(),
                "Mean_JSD_pairs": pairs.mean(),
            })
    per_var = pd.DataFrame(var_rows)
    per_run = pd.DataFrame(run_rows)

    summary = []
    for source, g in per_run.groupby("Source"):
        for metric in [c for c in per_run.columns if c not in ("Source", "Seed")]:
            m, sd, lo, hi = mean_ci(g[metric])
            summary.append({"Source": source, "Metric": metric, "Mean": m, "SD": sd, "CI95_low": lo, "CI95_high": hi})
    summary = pd.DataFrame(summary)
    tests = pd.concat([compare_sources(per_run, m) for m in
                       ["Mean_KS_numerical", "Mean_JSD_categorical", "Mean_JSD_pairs"]])
    by_variable = per_var.groupby(["Variable", "Type", "Source"])[["KS", "JSD"]].mean().unstack("Source")

    per_run.to_csv(OUT / "distribution_tests_by_run.csv", index=False)
    summary.to_csv(OUT / "distribution_tests_summary.csv", index=False)
    tests.to_csv(OUT / "distribution_tests_tests.csv", index=False)
    by_variable.to_csv(OUT / "distribution_tests_by_variable.csv")
    plot(per_run)

    pd.set_option("display.width", 200)
    print(summary.pivot(index="Source", columns="Metric", values="Mean").round(3).to_string())
    print(tests.round(4).to_string())


if __name__ == "__main__":
    main()
