"""Record-level re-identification risk (identity disclosure), complementing the attribute inference attacks.

For every seeded synthesizer run and the independent-marginals baseline:
  1. Record copying: share of synthetic records that match some real patient on >= 90%, >= 95%
     and 100% of the 61 variables (categorical exact; numerical within 5% of the real range).
  2. Distance to closest record (DCR): 1 - similarity of each synthetic record to its closest real
     patient, compared with the same distance between each real patient and the closest *other*
     real patient. Synthetic records closer to a real patient than any real patient is to another
     (DCR below the minimum real-to-real distance) are flagged as "too close".
  3. Quasi-identifier (QI) linkage, the Safe-Harbor-relevant scenario: QIs = 10-year age band,
     sex, cranial location, WHO grade. For each real patient, the synthetic records sharing the
     patient's QI combination are retrieved; a disclosure occurs if the majority value of a
     sensitive attribute among those records equals the patient's true value AND differs from the
     cohort majority (i.e. the QI match reveals something a population guess would not).
     k-anonymity of the real cohort on the same QIs is reported for context.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (DISPLAY, BASELINE, COLORS, INK, INK_2, MODELS, OUT, as_str, build_metadata, compare_sources, generate,
                    independent_marginals, load_real, match_fraction, mean_ci, save, style)

QI = ["Age band", "Sex", "Cranial Location (Infratentorial, supratentorial)", "WHO Grade"]
SENSITIVE = ["Death (Y/N)", "Postop complication", "EOR (GTR, NTR, STR, biopsy)", "Radiotherapy (Y/N)",
             "Recurrence", "Discharge dispo"]


def with_age_band(df):
    df = df.copy()
    age = pd.to_numeric(df["Age at Diagnosis"], errors="coerce")
    df["Age band"] = (age // 10 * 10).astype("Int64").astype(str)
    return df


def qi_key(df):
    return df[QI[:1]].astype(str).agg("|".join, axis=1) + "|" + df[QI[1:]].apply(as_str).agg("|".join, axis=1)


def qi_linkage_per_patient(real, synth):
    """Per real patient: (QI combination found in synthetic data, number of non-obvious sensitive values revealed)."""
    real, synth = with_age_band(real), with_age_band(synth)
    rk, sk = qi_key(real), qi_key(synth)
    majority = {s: as_str(real[s]).mode().iloc[0] for s in SENSITIVE}
    matched, leaked = np.zeros(len(real), bool), np.zeros(len(real), int)
    for i in range(len(real)):
        hits = synth[sk == rk.iloc[i]]
        if hits.empty:
            continue
        matched[i] = True
        for s in SENSITIVE:
            truth = as_str(real[s]).iloc[i]
            leaked[i] += as_str(hits[s]).mode().iloc[0] == truth and truth != majority[s]
    return matched, leaked


def qi_linkage(real, synth):
    matched, leaked = qi_linkage_per_patient(real, synth)
    return int(matched.sum()), int((leaked > 0).sum()), int(leaked.sum())


def holdout_qi_linkage(real, metadata, repeats=2, n_folds=6):
    """QI disclosure for patients in vs out of the synthesizer's training folds (reuses holdout_analysis runs)."""
    from sklearn.model_selection import KFold
    rows = []
    for rep in range(repeats):
        for fold, (tr, _) in enumerate(KFold(n_folds, shuffle=True, random_state=100 + rep).split(real)):
            seed = 1000 * (rep + 1) + fold
            train = real.iloc[tr].reset_index(drop=True)
            member = np.isin(np.arange(len(real)), tr)
            datasets = {m: generate(train, metadata, m, seed, tag=f"_holdout_r{rep}f{fold}") for m in MODELS}
            datasets[BASELINE] = independent_marginals(train, seed)
            for source, synth in datasets.items():
                _, leaked = qi_linkage_per_patient(real, synth)
                rows += [{"Source": source, "Repeat": rep, "Fold": fold, "Patient": i, "Member": bool(member[i]),
                          "Disclosed": bool(leaked[i] > 0)} for i in range(len(real))]
    df = pd.DataFrame(rows)
    out = []
    rng = np.random.default_rng(0)
    for source, g in df.groupby("Source"):
        mem, non = g[g.Member].Disclosed.to_numpy(float), g[~g.Member].Disclosed.to_numpy(float)
        boots = [rng.choice(mem, len(mem)).mean() - rng.choice(non, len(non)).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"Source": source, "Disclosed_members_pct": 100 * mem.mean(),
                    "Disclosed_heldout_pct": 100 * non.mean(), "Difference_pp": 100 * (mem.mean() - non.mean()),
                    "CI95_low": 100 * lo, "CI95_high": 100 * hi})
    return pd.DataFrame(out)


def k_anonymity(real):
    counts = qi_key(with_age_band(real)).value_counts()
    return counts.min(), int((counts == 1).sum())


def plot(per_run, real_min_dcr):
    style()
    sources = MODELS + [BASELINE]
    metrics = [("Near_copies_95_pct", "Synthetic records matching a real\npatient on ≥ 95% of variables, %"),
               ("Too_close_pct", "Synthetic records closer to a real patient\nthan any two real patients are, %"),
               ("QI_disclosed_patients", "Real patients whose non-obvious outcome\nis revealed by a QI match (of 18)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    rng = np.random.default_rng(0)
    ys = np.arange(len(sources))[::-1]
    for ax, (metric, title) in zip(axes, metrics):
        for y, s in zip(ys, sources):
            v = per_run[per_run.Source == s][metric].to_numpy(dtype=float)
            ax.scatter(v, y + rng.uniform(-0.12, 0.12, len(v)), s=12, color=COLORS[s], alpha=0.75, lw=0)
            ax.plot([v.mean()] * 2, [y - 0.28, y + 0.28], color=INK, lw=1.4)
        ax.set_title(title, fontsize=7, loc="left")
        ax.set_xlim(left=-0.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([DISPLAY.get(s, s) for s in sources], fontsize=7.5)
    axes[0].tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "S_reidentification")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    real = load_real()
    metadata = build_metadata(real)
    rr = match_fraction(real, real, real)
    np.fill_diagonal(rr, 0)
    real_nearest_dcr = 1 - rr.max(axis=1)
    real_min_dcr = real_nearest_dcr.min()
    k_min, n_unique = k_anonymity(real)

    rows = []
    for seed in range(args.seeds):
        datasets = {m: generate(real, metadata, m, seed) for m in MODELS}
        datasets[BASELINE] = independent_marginals(real, seed)
        for source, synth in datasets.items():
            sim = match_fraction(synth, real, real).max(axis=1)
            dcr = 1 - sim
            matched, disclosed_any, disclosures = qi_linkage(real, synth)
            rows.append({
                "Source": source, "Seed": seed,
                "Exact_copies_pct": 100 * np.mean(sim >= 1.0),
                "Near_copies_95_pct": 100 * np.mean(sim >= 0.95),
                "Near_copies_90_pct": 100 * np.mean(sim >= 0.90),
                "Median_DCR": float(np.median(dcr)),
                "DCR_5th_pct": float(np.percentile(dcr, 5)),
                "Too_close_pct": 100 * np.mean(dcr < real_min_dcr),
                "QI_matched_patients": matched,
                "QI_disclosed_patients": disclosed_any,
                "QI_disclosures_total": disclosures,
            })
        print(f"seed {seed} done", flush=True)
    per_run = pd.DataFrame(rows)

    summary = []
    for source, g in per_run.groupby("Source"):
        for metric in [c for c in per_run.columns if c not in ("Source", "Seed")]:
            m, sd, lo, hi = mean_ci(g[metric])
            summary.append({"Source": source, "Metric": metric, "Mean": m, "SD": sd, "CI95_low": lo,
                            "CI95_high": hi, "Min": g[metric].min(), "Max": g[metric].max()})
    summary = pd.DataFrame(summary)
    tests = pd.concat([compare_sources(per_run, m) for m in
                       ["Near_copies_95_pct", "Too_close_pct", "DCR_5th_pct", "QI_disclosed_patients"]])
    context = pd.DataFrame([{
        "Real_nearest_neighbor_DCR_median": float(np.median(real_nearest_dcr)),
        "Real_nearest_neighbor_DCR_min": float(real_min_dcr),
        "Real_k_anonymity_min_k": int(k_min), "Real_unique_patients_on_QIs": n_unique,
    }])

    holdout = holdout_qi_linkage(real, metadata)
    holdout.to_csv(OUT / "reidentification_qi_holdout.csv", index=False)
    per_run.to_csv(OUT / "reidentification_by_run.csv", index=False)
    summary.to_csv(OUT / "reidentification_summary.csv", index=False)
    tests.to_csv(OUT / "reidentification_tests.csv", index=False)
    context.to_csv(OUT / "reidentification_real_context.csv", index=False)
    plot(per_run, real_min_dcr)

    pd.set_option("display.width", 220)
    print(context.round(3).to_string(index=False))
    print(summary.pivot(index="Source", columns="Metric", values="Mean").round(2).to_string())
    print(tests[tests.Comparison.str.contains("baseline|Kruskal")].round(4).to_string())
    print("\n=== QI disclosure: training vs held-out patients ===")
    print(holdout.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
