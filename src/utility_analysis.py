"""Analytic utility: do synthetic datasets reproduce associations seen in the real cohort?

Addresses Reviewer 2 comment 3 / Reviewer 1 comment 5.

  1. Prespecified clinical associations (log odds ratios), real vs synthetic.
     Synthetic effects are reported on the full synthetic sample and on
     repeated n=18 subsamples, so significance is not inflated by n=1000.
  2. Global structure: agreement between the real and synthetic Spearman
     correlation matrices over all binary + numerical variables, benchmarked
     against an "independent marginals" baseline that keeps every column's
     distribution but destroys all relationships between columns.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from common import (DISPLAY, BASELINE, COLORS, INK, INK_2, MODELS, OUT, build_metadata, compare_sources, generate,
                    independent_marginals, indicator, load_real, mean_ci, numeric_columns, save, style, tick_label)

N_SUBSAMPLE = 500

# (label, exposure column, exposure level, outcome column, outcome level)
# Chosen on clinical grounds, before looking at the real-data estimates.
ASSOCIATIONS = [
    ("Infratentorial → preop hydrocephalus",
     "Cranial Location (Infratentorial, supratentorial)", "Infratentorial", "Preop Hydrocephalus", "Y"),
    ("Infratentorial → preop cerebellar signs",
     "Cranial Location (Infratentorial, supratentorial)", "Infratentorial", "Preop Cerebellar Signs", "Y"),
    ("Infratentorial → preop nausea/vomiting",
     "Cranial Location (Infratentorial, supratentorial)", "Infratentorial", "Preop Nausea/Vomiting", "Y"),
    ("WHO grade 3 → radiotherapy", "WHO Grade", "3", "Radiotherapy (Y/N)", "Y"),
    ("WHO grade 3 → death", "WHO Grade", "3", "Death (Y/N)", "Y"),
    ("Non-GTR → radiotherapy", "EOR (GTR, NTR, STR, biopsy)", "!GTR", "Radiotherapy (Y/N)", "Y"),
    ("Ki-67 ≥7% → WHO grade 3", "Ki-67_index_7%", "1", "WHO Grade", "3"),
]


def two_by_two(df, exp_col, exp_level, out_col, out_level):
    e = indicator(df, exp_col, exp_level)
    o = indicator(df, out_col, out_level)
    return np.array([[(e & o).sum(), (e & ~o).sum()], [(~e & o).sum(), (~e & ~o).sum()]])


def log_or(table):
    """Haldane-Anscombe corrected log odds ratio and its standard error."""
    a, b, c, d = table.ravel() + 0.5
    return np.log(a * d / (b * c)), np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)


def association_rows(real, datasets):
    rows = []
    for label, ec, el, oc, ol in ASSOCIATIONS:
        t_real = two_by_two(real, ec, el, oc, ol)
        lor, se = log_or(t_real)
        rows.append({
            "Association": label, "Source": "Real", "Seed": None,
            "logOR": lor, "CI_low": lor - 1.96 * se, "CI_high": lor + 1.96 * se,
            "Fisher_p": fisher_exact(t_real)[1], "Table_a_b_c_d": t_real.ravel().tolist(),
        })
        for (source, seed), df in datasets.items():
            lor_full, _ = log_or(two_by_two(df, ec, el, oc, ol))
            rng = np.random.default_rng(seed)
            sub = np.array([
                log_or(two_by_two(df.iloc[rng.choice(len(df), len(real), replace=False)], ec, el, oc, ol))[0]
                for _ in range(N_SUBSAMPLE)])
            rows.append({
                "Association": label, "Source": source, "Seed": seed,
                "logOR": lor_full,
                "Same_direction": float(np.sign(lor_full) == np.sign(lor)),
                "Same_direction_n18_pct": 100 * np.mean(np.sign(sub) == np.sign(lor)),
                "Inside_real_CI": float(lor - 1.96 * se <= lor_full <= lor + 1.96 * se),
            })
    return pd.DataFrame(rows)


def encoded_matrix(df, real):
    """Binary columns -> indicator of the real data's minority level; numerical columns as-is."""
    num = numeric_columns(real)
    enc = {}
    for c in real.columns:
        if c in num:
            enc[c] = pd.to_numeric(df[c], errors="coerce")
        elif real[c].nunique() == 2:
            minority = str(real[c].value_counts().index[-1])
            enc[c] = indicator(df, c, minority).astype(float)
    return pd.DataFrame(enc)


def upper(corr):
    return corr.to_numpy()[np.triu_indices(len(corr), 1)]


def global_rows(real, datasets):
    r_real = upper(encoded_matrix(real, real).corr(method="spearman"))
    strong = np.abs(r_real) >= 0.3
    rows = []
    for (source, seed), df in datasets.items():
        r_syn = upper(encoded_matrix(df, real).corr(method="spearman"))
        # A variable that is constant in the synthetic data has no correlation; count it as 0 (no structure kept).
        r_syn = np.nan_to_num(r_syn)
        ok = ~np.isnan(r_real)
        rows.append({
            "Source": source, "Seed": seed,
            "Corr_of_correlations": np.corrcoef(r_real[ok], r_syn[ok])[0, 1],
            "Mean_abs_diff": np.mean(np.abs(r_real[ok] - r_syn[ok])),
            "Sign_agreement_strong_pct": 100 * np.mean(np.sign(r_real[ok & strong]) == np.sign(r_syn[ok & strong])),
            "N_pairs": int(ok.sum()), "N_strong_pairs": int((ok & strong).sum()),
        })
    return pd.DataFrame(rows)


def summarize(assoc, glob):
    syn = assoc[assoc.Source != "Real"]
    real = assoc[assoc.Source == "Real"].set_index("Association")
    per_assoc = syn.groupby(["Association", "Source"]).agg(
        logOR_mean=("logOR", "mean"), logOR_min=("logOR", "min"), logOR_max=("logOR", "max"),
        Same_direction_runs_pct=("Same_direction", lambda s: 100 * s.mean()),
        Same_direction_n18_pct=("Same_direction_n18_pct", "mean"),
        Inside_real_CI_runs_pct=("Inside_real_CI", lambda s: 100 * s.mean()),
    ).reset_index()
    per_assoc = per_assoc.join(real[["logOR", "CI_low", "CI_high", "Fisher_p", "Table_a_b_c_d"]]
                               .add_prefix("Real_"), on="Association")
    per_model = syn.groupby("Source").agg(
        Same_direction_runs_pct=("Same_direction", lambda s: 100 * s.mean()),
        Same_direction_n18_pct=("Same_direction_n18_pct", "mean"),
    )
    glob_rows = []
    for source, g in glob.groupby("Source"):
        for metric in ["Corr_of_correlations", "Mean_abs_diff", "Sign_agreement_strong_pct"]:
            m, sd, lo, hi = mean_ci(g[metric])
            glob_rows.append({"Source": source, "Metric": metric, "Mean": m, "SD": sd, "CI95_low": lo, "CI95_high": hi})
    return per_assoc, per_model, pd.DataFrame(glob_rows)


def plot_associations(assoc):
    style()
    real = assoc[assoc.Source == "Real"].set_index("Association")
    syn = assoc[assoc.Source != "Real"]
    labels = [a[0] for a in ASSOCIATIONS]
    sources = MODELS + [BASELINE]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    step = 0.13
    for i, label in enumerate(labels):
        y = len(labels) - 1 - i
        r = real.loc[label]
        ax.plot([r.CI_low, r.CI_high], [y + 0.36, y + 0.36], color=INK, lw=1.4, solid_capstyle="round")
        ax.plot(r.logOR, y + 0.36, "D", color=INK, ms=6, mec="white", mew=1)
        for j, s in enumerate(sources):
            vals = syn[(syn.Association == label) & (syn.Source == s)].logOR
            yy = y + 0.36 - step * (j + 1)
            ax.plot([vals.min(), vals.max()], [yy, yy], color=COLORS[s], lw=1.4, solid_capstyle="round")
            ax.plot(vals.mean(), yy, "o", color=COLORS[s], ms=5.5, mec="white", mew=1)
    ax.axvline(0, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks([len(labels) - 1 - i + 0.03 for i in range(len(labels))])
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Log odds ratio (Haldane-corrected)")
    handles = [plt.Line2D([], [], color=INK, marker="D", lw=1.4, ms=6, mec="white", label="Real cohort (95% CI)")]
    handles += [plt.Line2D([], [], color=COLORS[s], marker="o", lw=1.4, ms=5.5, mec="white",
                           label=f"{DISPLAY.get(s, s).replace(chr(10), chr(32))} (mean, range over seeds)") for s in sources]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, fontsize=8)
    save(fig, "S_utility_associations")


def plot_global(glob):
    style()
    metrics = [("Corr_of_correlations", "Correlation between real and\nsynthetic correlation matrices"),
               ("Sign_agreement_strong_pct", "Sign agreement for strong real\ncorrelations (|ρ| ≥ 0.3), %")]
    sources = MODELS + [BASELINE]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    rng = np.random.default_rng(0)
    for ax, (metric, title) in zip(axes, metrics):
        for i, s in enumerate(sources):
            v = glob[glob.Source == s][metric].to_numpy()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=COLORS[s], alpha=0.75, lw=0)
            m, _, lo, hi = mean_ci(v)
            ax.plot([i - 0.28, i + 0.28], [m, m], color=INK, lw=1.6)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([tick_label(s) for s in sources], fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", visible=False)
    axes[1].axhline(50, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    axes[1].text(-0.45, 51, "chance", ha="left", va="bottom", fontsize=7, color=INK_2)
    fig.tight_layout()
    save(fig, "S_utility_global_structure")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    real = load_real()
    metadata = build_metadata(real)
    datasets = {}
    for model in MODELS:
        for seed in range(args.seeds):
            print(f"{model} seed {seed}", flush=True)
            datasets[(model, seed)] = generate(real, metadata, model, seed)
    for seed in range(args.seeds):
        datasets[(BASELINE, seed)] = independent_marginals(real, seed)

    assoc = association_rows(real, datasets)
    glob = global_rows(real, datasets)
    per_assoc, per_model, glob_summary = summarize(assoc, glob)
    stats = pd.concat([compare_sources(glob, "Corr_of_correlations"),
                       compare_sources(glob, "Sign_agreement_strong_pct")])

    OUT.mkdir(parents=True, exist_ok=True)
    assoc.to_csv(OUT / "utility_associations_by_run.csv", index=False)
    glob.to_csv(OUT / "utility_global_by_run.csv", index=False)
    per_assoc.to_csv(OUT / "utility_associations_summary.csv", index=False)
    per_model.to_csv(OUT / "utility_by_model.csv")
    glob_summary.to_csv(OUT / "utility_global_summary.csv", index=False)
    stats.to_csv(OUT / "utility_global_tests.csv", index=False)
    plot_associations(assoc)
    plot_global(glob)

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("\n=== Real-data associations ===")
    print(assoc[assoc.Source == "Real"][["Association", "Table_a_b_c_d", "logOR", "CI_low", "CI_high", "Fisher_p"]]
          .round(2).to_string())
    print("\n=== Per association x model ===")
    print(per_assoc.drop(columns=["Real_Table_a_b_c_d", "Real_Fisher_p"]).round(2).to_string())
    print("\n=== Per model ===")
    print(per_model.round(1))
    print("\n=== Global structure ===")
    print(glob_summary.round(3).to_string())
    print(stats.round(4).to_string())


if __name__ == "__main__":
    main()
