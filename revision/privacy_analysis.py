"""Attribute inference attacks across seeded synthesizer runs (Reviewer 1 comments 2-4, 6; Reviewer 2 comment 1).

A. Original method, re-run on every seed: SDMetrics categorical attackers with the manuscript's
   key and sensitive fields. Records how many real patients the attacker could evaluate at all:
   these attackers treat every key as categorical, so a real patient whose exact continuous
   key values (age, BMI) never occur in the synthetic data gets no prediction and is scored
   as "protected".
B. Revised attack: attacker-known keys and targets are disjoint; continuous keys are scaled,
   categorical keys one-hot encoded; categorical targets are scored by exact match and
   continuous targets by falling within 10% of the real interquartile range (robust to the
   implausible survival outliers in the source data). Every attack is compared with
   (i) a prior-only guess (synthetic majority class / median, no keys) and (ii) the same attack
   trained on independent-marginals data. Counts are out of 18 with exact (Clopper-Pearson) CIs.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sdmetrics.single_table.privacy.categorical_sklearn import (CategoricalKNNAttacker, CategoricalNBAttacker,
                                                                  CategoricalRFAttacker)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (BASELINE, COLORS, INK, INK_2, MODELS, OUT, as_str, build_metadata, compare_sources, generate,
                    independent_marginals, load_real, mean_ci, save, style, tick_label)

# Manuscript's attacker-known fields (kept for continuity with the original analysis).
KEYS = ["Age at Diagnosis", "Sex", "BMI", "Cranial Location (Infratentorial, supratentorial)", "WHO Grade",
        "Age at Sx"]
NUM_KEYS = ["Age at Diagnosis", "BMI", "Age at Sx"]
CAT_KEYS = [k for k in KEYS if k not in NUM_KEYS]

# Revised targets: none overlaps with the keys.
CAT_TARGETS = ["Death (Y/N)", "Recurrence", "Postop complication", "EOR (GTR, NTR, STR, biopsy)",
               "Radiotherapy (Y/N)", "Discharge dispo"]
NUM_TARGETS = ["Overall Survival (yrs)", "PFS (After Sx; Months)"]
NUM_TOLERANCE = 0.10
SHORT = {"Death (Y/N)": "Death", "Recurrence": "Recurrence", "Postop complication": "Postop complication",
         "EOR (GTR, NTR, STR, biopsy)": "Extent of resection", "Radiotherapy (Y/N)": "Radiotherapy",
         "Discharge dispo": "Discharge disposition", "Overall Survival (yrs)": "Overall survival (±10%)",
         "PFS (After Sx; Months)": "PFS (±10%)"}
ATTACKERS = ["kNN", "Naive Bayes", "Random Forest"]

# Manuscript's original sensitive fields (clinical ones; identifiers are not modeled).
ORIGINAL_SENSITIVE = ["Death (Y/N)", "Recurrence", "Postop complication", "Age at Diagnosis", "BMI",
                      "Cranial Location (Infratentorial, supratentorial)", "EOR (GTR, NTR, STR, biopsy)", "WHO Grade"]
ORIGINAL_ATTACKERS = {"kNN": CategoricalKNNAttacker, "Naive Bayes": CategoricalNBAttacker,
                      "Random Forest": CategoricalRFAttacker}


def clopper_pearson(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


# ---------- A. original method ----------

def align_like_real(synth, real):
    synth = synth.copy()
    for c in real.columns:
        if pd.api.types.is_numeric_dtype(real[c]):
            synth[c] = pd.to_numeric(synth[c], errors="coerce")
        else:
            synth[c] = synth[c].astype(str)
    return synth


def original_attack(real, synth, attacker_cls, sensitive):
    synth = align_like_real(synth, real)
    attacker = attacker_cls()
    attacker.fit(synth, KEYS, [sensitive])
    evaluable = correct = 0
    for i in range(len(real)):
        pred = attacker.predict(tuple(real[KEYS].iloc[i]))
        if pred is None:
            continue
        evaluable += 1
        correct += pred == (real[sensitive].iloc[i],)
    return evaluable, correct


# ---------- B. revised attack ----------

def features(df):
    x = df[KEYS].copy()
    for k in NUM_KEYS:
        x[k] = pd.to_numeric(x[k], errors="coerce")
    for k in CAT_KEYS:
        x[k] = as_str(x[k])
    return x


def attacker_model(name, continuous, seed):
    pre = ColumnTransformer([("num", StandardScaler(), NUM_KEYS),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_KEYS)])
    if continuous:
        est = {"kNN": KNeighborsRegressor(5), "Random Forest": RandomForestRegressor(200, random_state=seed)}[name]
    else:
        est = {"kNN": KNeighborsClassifier(5), "Naive Bayes": GaussianNB(),
               "Random Forest": RandomForestClassifier(200, random_state=seed)}[name]
    return make_pipeline(pre, est)


def target_values(df, target):
    if target in NUM_TARGETS:
        return pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    return as_str(df[target]).to_numpy()


def is_correct(pred, truth, target, real):
    if target in NUM_TARGETS:
        tol = NUM_TOLERANCE * (real[target].quantile(0.75) - real[target].quantile(0.25))
        return np.abs(pred - truth) <= tol
    return pred == truth


def attack(train, test, real, target, name, seed):
    """Per-patient correctness of one attack: fit keys->target on `train` (synthetic), predict `test` (real)."""
    y = target_values(train, target)
    keep = ~pd.isna(y)
    model = attacker_model(name, target in NUM_TARGETS, seed)
    model.fit(features(train)[keep], y[keep])
    return is_correct(model.predict(features(test)), target_values(test, target), target, real)


def prior_guess(train, test, real, target):
    """Attacker with the synthetic data but no keys: always guess the majority class / median."""
    y = pd.Series(target_values(train, target)).dropna()
    guess = y.median() if target in NUM_TARGETS else y.mode().iloc[0]
    return is_correct(np.full(len(test), guess, dtype=object if target in CAT_TARGETS else float),
                      target_values(test, target), target, real)


def attack_applicable(name, target):
    return not (name == "Naive Bayes" and target in NUM_TARGETS)


# ---------- figures ----------

def plot_original(orig):
    style()
    sources = MODELS + [BASELINE]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    rng = np.random.default_rng(0)
    for i, s in enumerate(sources):
        v = orig[orig.Source == s].Evaluable.to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v + rng.uniform(-0.15, 0.15, len(v)), s=10,
                   color=COLORS[s], alpha=0.6, lw=0)
    ax.axhline(18, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    ax.text(len(sources) - 0.5, 17.6, "all 18 patients", ha="right", va="top", fontsize=7, color=INK_2)
    ax.set_ylim(-0.8, 19)
    ax.set_yticks(range(0, 19, 3))
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([tick_label(s) for s in sources], fontsize=8)
    ax.set_ylabel("Real patients the attacker\ncould make a prediction for")
    ax.grid(axis="x", visible=False)
    save(fig, "S_privacy_original_method_evaluable")


def plot_by_target(worst):
    """Strongest attacker per run vs prior-only guess, per target."""
    style()
    targets = CAT_TARGETS + NUM_TARGETS
    sources = MODELS + [BASELINE]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    step = 0.15
    for i, t in enumerate(targets):
        y = len(targets) - 1 - i
        sub = worst[worst.Target == t]
        prior = 100 * sub.Prior_correct.mean() / 18
        ax.plot([prior, prior], [y - 0.42, y + 0.42], color=INK, lw=1.6)
        for j, s in enumerate(sources):
            v = 100 * sub[sub.Source == s].Correct.to_numpy() / 18
            yy = y + 0.3 - step * j
            ax.plot([v.min(), v.max()], [yy, yy], color=COLORS[s], lw=1.4, solid_capstyle="round")
            ax.plot(v.mean(), yy, "o", color=COLORS[s], ms=5.5, mec="white", mew=1)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([SHORT[t] for t in reversed(targets)])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Real patients whose value was correctly inferred, % (strongest attacker per run)")
    handles = [plt.Line2D([], [], color=INK, lw=1.6, label="Prior-only guess (no attacker knowledge)")]
    handles += [plt.Line2D([], [], color=COLORS[s], marker="o", lw=1.4, ms=5.5, mec="white",
                           label=f"{s} (mean, range over seeds)") for s in sources]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.11), ncol=2, fontsize=8)
    save(fig, "S_privacy_revised_attack_by_target")


def plot_lift(per_seed):
    style()
    sources = MODELS + [BASELINE]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    rng = np.random.default_rng(0)
    for i, s in enumerate(sources):
        v = per_seed[per_seed.Source == s].Mean_lift_pp.to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=COLORS[s], alpha=0.75, lw=0)
        ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=INK, lw=1.6)
    ax.axhline(0, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([tick_label(s) for s in sources], fontsize=8)
    ax.set_ylabel("Attack success above prior-only\nguess, percentage points")
    ax.grid(axis="x", visible=False)
    save(fig, "S_privacy_revised_lift_by_seed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    real = load_real()
    metadata = build_metadata(real)
    n = len(real)
    orig_rows, rows = [], []
    for seed in range(args.seeds):
        datasets = {m: generate(real, metadata, m, seed) for m in MODELS}
        datasets[BASELINE] = independent_marginals(real, seed)
        for source, synth in datasets.items():
            for atk, cls in ORIGINAL_ATTACKERS.items():
                for s in ORIGINAL_SENSITIVE:
                    evaluable, correct = original_attack(real, synth, cls, s)
                    orig_rows.append({"Source": source, "Seed": seed, "Attacker": atk, "Sensitive": s,
                                      "Evaluable": evaluable, "Correct": correct,
                                      "Reported_risk_pct": 100 * correct / n})
            for t in CAT_TARGETS + NUM_TARGETS:
                prior = int(prior_guess(synth, real, real, t).sum())
                for atk in ATTACKERS:
                    if not attack_applicable(atk, t):
                        continue
                    correct = int(attack(synth, real, real, t, atk, seed).sum())
                    rows.append({"Source": source, "Seed": seed, "Attacker": atk, "Target": t,
                                 "Correct": correct, "Prior_correct": prior, "Lift_pp": 100 * (correct - prior) / n})
        print(f"seed {seed} done", flush=True)
    orig = pd.DataFrame(orig_rows)
    res = pd.DataFrame(rows)

    # Strongest attacker per (source, seed, target) = worst case for privacy.
    worst = res.loc[res.groupby(["Source", "Seed", "Target"]).Correct.idxmax()]
    per_seed = worst.groupby(["Source", "Seed"]).agg(Mean_success_pct=("Correct", lambda c: 100 * c.mean() / n),
                                                     Mean_lift_pp=("Lift_pp", "mean")).reset_index()

    target_summary = []
    for (source, t), g in worst.groupby(["Source", "Target"]):
        med = int(np.median(g.Correct))
        lo, hi = clopper_pearson(med, n)
        target_summary.append({
            "Source": source, "Target": t, "Median_correct_of_18": med,
            "Median_run_CI95": f"{100 * lo:.1f}-{100 * hi:.1f}%",
            "Mean_success_pct": 100 * g.Correct.mean() / n, "SD_pct": 100 * g.Correct.std() / n,
            "Range_correct": f"{g.Correct.min()}-{g.Correct.max()}",
            "Prior_correct_of_18": round(g.Prior_correct.mean(), 1), "Mean_lift_pp": g.Lift_pp.mean(),
        })
    target_summary = pd.DataFrame(target_summary)
    model_summary = []
    for source, g in per_seed.groupby("Source"):
        for metric in ["Mean_success_pct", "Mean_lift_pp"]:
            m, sd, lo, hi = mean_ci(g[metric])
            model_summary.append({"Source": source, "Metric": metric, "Mean": m, "SD": sd, "CI95_low": lo,
                                  "CI95_high": hi})
    model_summary = pd.DataFrame(model_summary)
    tests = compare_sources(per_seed, "Mean_lift_pp")
    orig_summary = orig.groupby("Source").agg(
        Evaluable_mean=("Evaluable", "mean"), Evaluable_max=("Evaluable", "max"),
        Reported_risk_mean_pct=("Reported_risk_pct", "mean"), Reported_risk_max_pct=("Reported_risk_pct", "max"))
    orig_by_seed = orig.groupby(["Source", "Seed"]).Reported_risk_pct.mean().groupby("Source").agg(["mean", "std",
                                                                                                    "min", "max"])

    res.to_csv(OUT / "privacy_revised_by_run.csv", index=False)
    target_summary.to_csv(OUT / "privacy_revised_by_target.csv", index=False)
    model_summary.to_csv(OUT / "privacy_revised_by_model.csv", index=False)
    tests.to_csv(OUT / "privacy_revised_tests.csv", index=False)
    orig.to_csv(OUT / "privacy_original_by_run.csv", index=False)
    orig_summary.to_csv(OUT / "privacy_original_summary.csv")
    orig_by_seed.to_csv(OUT / "privacy_original_risk_by_seed.csv")
    plot_original(orig)
    plot_by_target(worst)
    plot_lift(per_seed)

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("=== A. Original method ===")
    print(orig_summary.round(2).to_string())
    print(orig_by_seed.round(2).to_string())
    print("\n=== B. Revised attack, per target (strongest attacker) ===")
    print(target_summary.round(1).to_string())
    print(model_summary.round(2).to_string())
    print(tests.round(4).to_string())


if __name__ == "__main__":
    main()
