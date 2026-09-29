"""Fidelity across independently seeded synthesizer runs (Reviewer 1 comment 2).

Runs the SDMetrics QualityReport (column shapes, column pair trends, overall) on every
cached synthetic dataset, plus the independent-marginals baseline, and reports mean,
SD and 95% CI per model with Kruskal-Wallis and Holm-corrected pairwise comparisons.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sdmetrics.reports.single_table import QualityReport

from common import (BASELINE, COLORS, INK, MODELS, OUT, build_metadata, compare_sources, generate,
                    independent_marginals, load_real, mean_ci, metadata_columns, save, style, tick_label)

METRICS = ["Overall", "Column Shapes", "Column Pair Trends"]


def quality(real, synth, meta):
    report = QualityReport()
    report.generate(real, synth, meta, verbose=False)
    props = report.get_properties().set_index("Property")["Score"]
    return {"Overall": report.get_score(), "Column Shapes": props["Column Shapes"],
            "Column Pair Trends": props["Column Pair Trends"]}


def plot(scores):
    style()
    sources = MODELS + [BASELINE]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    rng = np.random.default_rng(0)
    ys = np.arange(len(sources))[::-1]
    for ax, metric in zip(axes, METRICS):
        for y, s in zip(ys, sources):
            v = 100 * scores[scores.Source == s][metric].to_numpy()
            ax.scatter(v, y + rng.uniform(-0.12, 0.12, len(v)), s=14, color=COLORS[s], alpha=0.75, lw=0)
            m, _, lo, hi = mean_ci(v)
            ax.plot([m, m], [y - 0.28, y + 0.28], color=INK, lw=1.6)
        ax.set_title(metric, fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("SDMetrics score, %")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([s.replace(" baseline", "\nbaseline") if s == BASELINE else s for s in sources],
                            fontsize=8)
    axes[0].tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "S_fidelity_by_seed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    real = load_real()
    metadata = build_metadata(real)
    meta = {"columns": metadata_columns(real)}
    rows = []
    for seed in range(args.seeds):
        for model in MODELS:
            rows.append({"Source": model, "Seed": seed, **quality(real, generate(real, metadata, model, seed), meta)})
        rows.append({"Source": BASELINE, "Seed": seed, **quality(real, independent_marginals(real, seed), meta)})
        print(f"seed {seed} done", flush=True)
    scores = pd.DataFrame(rows)

    summary = []
    for source, g in scores.groupby("Source"):
        for metric in METRICS:
            m, sd, lo, hi = mean_ci(100 * g[metric])
            summary.append({"Source": source, "Metric": metric, "Mean": m, "SD": sd, "CI95_low": lo, "CI95_high": hi,
                            "Min": 100 * g[metric].min(), "Max": 100 * g[metric].max()})
    summary = pd.DataFrame(summary)
    tests = pd.concat([compare_sources(scores, m) for m in METRICS])

    scores.to_csv(OUT / "fidelity_by_run.csv", index=False)
    summary.to_csv(OUT / "fidelity_summary.csv", index=False)
    tests.to_csv(OUT / "fidelity_tests.csv", index=False)
    plot(scores)

    pd.set_option("display.width", 200)
    print(summary.round(2).to_string())
    print(tests.round(4).to_string())


if __name__ == "__main__":
    main()
