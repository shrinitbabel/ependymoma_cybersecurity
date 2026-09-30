"""Multivariate (relationship) metrics: do synthetic data preserve how variables go together?

All metrics are computed for every seeded synthesizer run and the independent-marginals
baseline (no relationships by construction), and all are reported:
  * Spearman and Kendall correlation matrices: correlation between the real and synthetic
    pairwise coefficients (higher = better) and correlation matrix distance
    (CMD = 1 - tr(R1 R2) / (||R1||_F ||R2||_F); 0 = identical structure, lower = better);
  * normalized mutual information (NMI) between every pair of variables (numerical variables
    binned at real-data quintiles): correlation between the real and synthetic NMI matrices;
  * Cramér's V (symmetric) and Theil's U (asymmetric uncertainty coefficient) between every
    pair of the 48 categorical variables, including multiclass ones: correlation between the
    real and synthetic association matrices;
  * log-cluster metric (Goncalves et al. 2020): k-means (k = 5) on pooled real + synthetic
    records, synthetic data subsampled to n = 18 (100 repeats) so both groups are equal size;
    lower (more negative) = real and synthetic records are more evenly mixed.
Also draws real vs synthetic Spearman correlation heatmaps side by side.
"""
import argparse
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (BASELINE, COLORS, INK, INK_2, MODELS, OUT, as_str, build_metadata, compare_sources, generate,
                    independent_marginals, load_real, mean_ci, numeric_columns, save, style)
from utility_analysis import encoded_matrix, upper

N_CLUSTERS = 5
N_LOGCLUSTER_REPEATS = 100
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

METRICS = [
    ("Spearman_r", "Spearman: correlation of\nreal vs synthetic matrices", "higher"),
    ("Kendall_r", "Kendall: correlation of\nreal vs synthetic matrices", "higher"),
    ("Spearman_CMD", "Spearman: correlation\nmatrix distance", "lower"),
    ("NMI_r", "Mutual information: correlation\nof real vs synthetic matrices", "higher"),
    ("CramersV_r", "Cramér's V (categorical):\ncorrelation of matrices", "higher"),
    ("CramersV_corrected_r", "Bias-corrected Cramér's V:\ncorrelation of matrices", "higher"),
    ("TheilsU_r", "Theil's U (categorical):\ncorrelation of matrices", "higher"),
    ("Log_cluster", "Log-cluster metric", "lower"),
]


def corr_matrix(df, real, method):
    c = encoded_matrix(df, real).corr(method=method)
    c = c.fillna(0.0)  # constant synthetic columns carry no structure
    np.fill_diagonal(c.values, 1.0)
    return c


def cmd(r1, r2):
    return 1 - np.trace(r1 @ r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))


def discretized(df, real, num):
    out = {}
    for c in real.columns:
        if c in num:
            edges = np.unique(np.quantile(real[c].to_numpy(dtype=float), [0.2, 0.4, 0.6, 0.8]))
            out[c] = np.digitize(pd.to_numeric(df[c], errors="coerce").fillna(real[c].median()), edges)
        else:
            out[c] = as_str(df[c]).to_numpy()
    return out


def nmi_vector(disc, cols):
    return np.array([normalized_mutual_info_score(disc[a], disc[b]) for a, b in combinations(cols, 2)])


def entropy(x):
    p = pd.Series(x).value_counts(normalize=True).to_numpy()
    return float(-(p * np.log(p)).sum())


def conditional_entropy(x, y):
    """H(X | Y)."""
    df = pd.DataFrame({"x": x, "y": y})
    h = 0.0
    for _, g in df.groupby("y"):
        h += len(g) / len(df) * entropy(g.x)
    return h


def cramers_v(x, y):
    table = pd.crosstab(pd.Series(x), pd.Series(y)).to_numpy()
    if min(table.shape) < 2:
        return 0.0
    n = table.sum()
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / n / (min(table.shape) - 1)))


def cramers_v_corrected(x, y):
    """Bias-corrected Cramér's V (Bergsma 2013); plain V is inflated at small n, more so with many categories."""
    table = pd.crosstab(pd.Series(x), pd.Series(y)).to_numpy()
    r, k = table.shape
    if min(r, k) < 2:
        return 0.0
    n = table.sum()
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    phi2 = ((table - expected) ** 2 / expected).sum() / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr, k_corr = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def theils_u(x, y):
    """U(X | Y): fraction of the uncertainty in X explained by knowing Y (asymmetric)."""
    hx = entropy(x)
    return 0.0 if hx == 0 else (hx - conditional_entropy(x, y)) / hx


def categorical_vectors(disc, cat):
    v = np.array([cramers_v(disc[a], disc[b]) for a, b in combinations(cat, 2)])
    vc = np.array([cramers_v_corrected(disc[a], disc[b]) for a, b in combinations(cat, 2)])
    u = np.array([theils_u(disc[a], disc[b]) for a in cat for b in cat if a != b])
    return v, vc, u


def log_cluster(real, synth, num, seed):
    cat = [c for c in real.columns if c not in num]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(real[cat].apply(as_str))
    scaler = StandardScaler().fit(real[num])

    def features(df):
        x_num = scaler.transform(df[num].apply(pd.to_numeric, errors="coerce").fillna(real[num].median()))
        return np.hstack([x_num, enc.transform(df[cat].apply(as_str))])

    x_real = features(real)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(N_LOGCLUSTER_REPEATS):
        sub = synth.iloc[rng.choice(len(synth), len(real), replace=False)]
        x = np.vstack([x_real, features(sub)])
        labels = KMeans(N_CLUSTERS, n_init=5, random_state=int(rng.integers(1e9))).fit_predict(x)
        is_real = np.r_[np.ones(len(real)), np.zeros(len(real))]
        c = 0.5
        terms = [(is_real[labels == k].mean() - c) ** 2 for k in np.unique(labels)]
        scores.append(np.mean(terms))
    # Average before the log: a perfectly mixed subsample has a term of 0 and log(0) = -inf.
    return float(np.log(np.mean(scores)))


def plot_heatmaps(real, datasets):
    style()
    r_real = corr_matrix(real, real, "spearman")
    order = leaves_list(linkage(1 - np.abs(r_real.values[np.triu_indices(len(r_real), 1)]), "average"))
    panels = [("Real cohort (n = 18)", r_real)] + [
        (name, corr_matrix(datasets[(name, 0)], real, "spearman")) for name in MODELS + [BASELINE]]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))
    for ax, (title, c) in zip(axes.ravel(), panels):
        im = ax.imshow(c.values[np.ix_(order, order)], cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    save(fig, "S_relationships_heatmaps")


# Clinically interpretable subset for the labeled heatmap (binary variables coded as the real minority level).
KEY_VARIABLES = {
    "Age at Diagnosis": "Age at diagnosis",
    "Sex": "Male sex",
    "Cranial Location (Infratentorial, supratentorial)": "Supratentorial location",
    "WHO Grade": "WHO grade 3",
    "Ki-67_index_7%": "Ki-67 ≥ 7%",
    "Tumor Size 1 (cm)": "Tumor size",
    "Recurrent Tumor (Y/N)": "Recurrent at presentation",
    "Preop Hydrocephalus": "Preop hydrocephalus",
    "Preop Cerebellar Signs": "No preop cerebellar signs",
    "Preop Nausea/Vomiting": "Preop nausea/vomiting",
    "Radiotherapy (Y/N)": "No radiotherapy",
    "Postop complication": "Postop complication",
    "LOS (days)": "Length of stay",
    "Death (Y/N)": "Death",
    "Overall Survival (yrs)": "Overall survival",
}


def plot_key_heatmaps(real, datasets, seeds):
    style()
    cols = list(KEY_VARIABLES)
    real_c = corr_matrix(real, real, "spearman").loc[cols, cols]
    panels = [("Real cohort (n = 18)", real_c)]
    for name in ["GaussianCopula", "TVAE", "CTGAN"]:
        mats = [corr_matrix(datasets[(name, s)], real, "spearman").loc[cols, cols].values for s in range(seeds)]
        panels.append((f"{name} (mean of {seeds} runs)", pd.DataFrame(np.mean(mats, axis=0), cols, cols)))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
    labels = [KEY_VARIABLES[c] for c in cols]
    for i, (ax, (title, c)) in enumerate(zip(axes.ravel(), panels)):
        im = ax.imshow(c.values, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels if i >= 2 else [], rotation=90, fontsize=6)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(labels if i % 2 == 0 else [], fontsize=6)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    save(fig, "S_relationships_key_variables")


def plot_key_heatmaps_main(real, datasets, seeds, summary):
    """Main-text figure: real cohort, all four synthesizers and the reshuffled reference (synthetic matrices
    averaged over runs), each titled with its whole-matrix Spearman agreement with the real data."""
    from common import DISPLAY
    style()
    cols = list(KEY_VARIABLES)
    labels = [KEY_VARIABLES[c] for c in cols]
    agreement = summary[summary.Metric == "Spearman_r"].set_index("Source").Mean
    panels = [("Real cohort (n = 18)", corr_matrix(real, real, "spearman").loc[cols, cols])]
    for name in ["GaussianCopula", "TVAE", "CTGAN", "CopulaGAN", BASELINE]:
        mats = [corr_matrix(datasets[(name, s)], real, "spearman").loc[cols, cols].values for s in range(seeds)]
        title = f"{DISPLAY.get(name, name).replace(chr(10), ' ')} (r = {round(agreement[name], 2) + 0.0:.2f})"
        panels.append((title, pd.DataFrame(np.mean(mats, axis=0), cols, cols)))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.4))
    for i, (ax, (title, c)) in enumerate(zip(axes.ravel(), panels)):
        im = ax.imshow(c.values, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"     {title}", fontsize=8, loc="left")
        ax.text(0, 1.02, "ABCDEF"[i], transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom",
                ha="left")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels if i >= 3 else [], rotation=90, fontsize=5.5)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(labels if i % 3 == 0 else [], fontsize=5.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    fig.subplots_adjust(bottom=0.24, hspace=0.18, wspace=0.08)
    cax = fig.add_axes([0.38, 0.035, 0.3, 0.014])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Spearman ρ", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    save(fig, "Figure3_correlation_heatmaps")


def plot_metrics(per_run):
    style()
    sources = MODELS + [BASELINE]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.8), sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(0)
    ys = np.arange(len(sources))[::-1]
    for ax, (metric, title, better) in zip(axes, METRICS):
        for y, s in zip(ys, sources):
            v = per_run[per_run.Source == s][metric].to_numpy()
            ax.scatter(v, y + rng.uniform(-0.12, 0.12, len(v)), s=10, color=COLORS[s], alpha=0.75, lw=0)
            ax.plot([v.mean()] * 2, [y - 0.28, y + 0.28], color=INK, lw=1.4)
        ax.set_title(title, fontsize=7, loc="left")
        ax.set_xlabel(f"{better.capitalize()} = better", fontsize=7, color=INK_2)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    for ax in (axes[0], axes[4]):
        ax.set_yticklabels([s.replace(" baseline", "\nbaseline") for s in sources], fontsize=7.5)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "S_relationships_metrics")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    real = load_real()
    metadata = build_metadata(real)
    num = numeric_columns(real)
    cols = list(real.columns)
    real_sp, real_kd = corr_matrix(real, real, "spearman"), corr_matrix(real, real, "kendall")
    cat = [c for c in cols if c not in num]
    real_disc = discretized(real, real, num)
    real_nmi = nmi_vector(real_disc, cols)
    real_v, real_vc, real_u = categorical_vectors(real_disc, cat)

    datasets, rows = {}, []
    for seed in range(args.seeds):
        for m in MODELS:
            datasets[(m, seed)] = generate(real, metadata, m, seed)
        datasets[(BASELINE, seed)] = independent_marginals(real, seed)
        for m in MODELS + [BASELINE]:
            df = datasets[(m, seed)]
            sp, kd = corr_matrix(df, real, "spearman"), corr_matrix(df, real, "kendall")
            disc = discretized(df, real, num)
            v, vc, u = categorical_vectors(disc, cat)
            rows.append({
                "Source": m, "Seed": seed,
                "Spearman_r": np.corrcoef(upper(real_sp), upper(sp))[0, 1],
                "Kendall_r": np.corrcoef(upper(real_kd), upper(kd))[0, 1],
                "Spearman_CMD": cmd(real_sp.values, sp.values),
                "Kendall_CMD": cmd(real_kd.values, kd.values),
                "NMI_r": np.corrcoef(real_nmi, nmi_vector(disc, cols))[0, 1],
                "CramersV_r": np.corrcoef(real_v, v)[0, 1],
                "CramersV_corrected_r": np.corrcoef(real_vc, vc)[0, 1] if vc.std() > 0 else 0.0,
                "TheilsU_r": np.corrcoef(real_u, u)[0, 1],
                "Log_cluster": log_cluster(real, df, num, seed),
            })
        print(f"seed {seed} done", flush=True)
    per_run = pd.DataFrame(rows)

    summary = []
    for source, g in per_run.groupby("Source"):
        for metric in [c for c in per_run.columns if c not in ("Source", "Seed")]:
            m, sd, lo, hi = mean_ci(g[metric])
            summary.append({"Source": source, "Metric": metric, "Mean": m, "SD": sd, "CI95_low": lo, "CI95_high": hi})
    summary = pd.DataFrame(summary)
    tests = pd.concat([compare_sources(per_run, m) for m in per_run.columns if m not in ("Source", "Seed")])

    per_run.to_csv(OUT / "relationships_by_run.csv", index=False)
    summary.to_csv(OUT / "relationships_summary.csv", index=False)
    tests.to_csv(OUT / "relationships_tests.csv", index=False)
    plot_heatmaps(real, datasets)
    plot_key_heatmaps(real, datasets, args.seeds)
    plot_metrics(per_run)

    pd.set_option("display.width", 200)
    print(summary.pivot(index="Source", columns="Metric", values="Mean").round(3).to_string())
    print(tests[tests.Comparison.str.contains("baseline|Kruskal")].round(4).to_string())


if __name__ == "__main__":
    main()
