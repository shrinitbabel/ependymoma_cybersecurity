"""Holdout analyses: membership signal and train-on-synthetic / test-on-real utility.

Each synthesizer is refit inside repeated 6-fold cross-validation (15 training patients,
3 held out), so every real patient is scored both as a member (in the generator's training
data) and as a non-member.

1. Memorization / membership (Reviewer 1 comments 1 and 6): for every real patient, the
   similarity of the closest synthetic record. If generators memorize, training patients
   have closer synthetic neighbours than held-out patients (AUC > 0.5). The revised attribute
   inference attack is also scored separately on members and non-members: success on
   non-members is population-level inference, not leakage of an individual's record.
2. TSTR vs TRTR (Reviewer 1 comment 5, Reviewer 2 comment 3): a classifier trained on
   synthetic data (TSTR) or on the 15 real training patients (TRTR) predicts the 3 held-out
   patients; predictions are pooled across folds into one AUC over all 18 patients.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (BASELINE, COLORS, INK, INK_2, MODELS, OUT, as_str, build_metadata, generate,
                    independent_marginals, load_real, match_fraction, save, style, tick_label)
from privacy_analysis import ATTACKERS, CAT_TARGETS, NUM_TARGETS, attack, attack_applicable, prior_guess

N_FOLDS = 6
N_BOOT = 2000

ENDPOINTS = ["Postop complication", "Radiotherapy (Y/N)"]
NUM_PREDICTORS = ["Age at Diagnosis", "BMI", "Tumor Size 1 (cm)", "Tumor Size 2 (cm)", "Tumor Size 3 (cm)",
                  "Duration of symptoms preop (months)", "Mitotic figures in 10 hpf"]
CAT_PREDICTORS = [
    "Sex", "Cranial Location (Infratentorial, supratentorial)", "Intra vs Extra Ventricular vs Both", "Laterality",
    "Cyst?", "WHO Grade", "Recurrent Tumor (Y/N)", "Prior surgery", "Prior Chemo", "Prior radiation", "TERT_marker",
    "Synaptophysin_marker", "GFAP_marker", "Olig2_marker", "Ki-67_index_7%", "Preop Hydrocephalus",
    "Preop CN Palsy (Y/N)", "Preop Papilledema (Y/N)", "Preop HA", "Preop Nausea/Vomiting", "Preop vertigo",
    "Preop Cerebellar Signs", "Preop Weakness", "Preop Sensory Loss", "Enhancement", "Surgical Approach",
    "EOR (GTR, NTR, STR, biopsy)"]


def bootstrap_auc(y, p, seed):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def predictor_frame(df):
    x = df[NUM_PREDICTORS + CAT_PREDICTORS].copy()
    for c in NUM_PREDICTORS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in CAT_PREDICTORS:
        x[c] = as_str(x[c])
    return x


def predict_endpoint(train, test, endpoint, seed):
    y = (as_str(train[endpoint]) == "Y").to_numpy().astype(int)
    if len(set(y)) < 2:
        return np.full(len(test), y.mean())
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_PREDICTORS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_PREDICTORS)])
    model = make_pipeline(pre, RandomForestClassifier(300, min_samples_leaf=2, random_state=seed))
    model.fit(predictor_frame(train), y)
    return model.predict_proba(predictor_frame(test))[:, 1]


def plot_membership(member):
    style()
    sources = MODELS + [BASELINE]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    rng = np.random.default_rng(0)
    for i, s in enumerate(sources):
        g = member[member.Source == s]
        for m, color, dx in [(True, COLORS[s], -0.17), (False, INK_2, 0.17)]:
            v = 100 * g[g.Member == m].Closest_match.to_numpy()
            axes[0].scatter(i + dx + rng.uniform(-0.07, 0.07, len(v)), v, s=5, color=color, alpha=0.35, lw=0)
            axes[0].plot([i + dx - 0.12, i + dx + 0.12], [np.median(v)] * 2, color=INK, lw=1.4)
    axes[0].set_ylabel("Similarity to closest synthetic record, %")
    axes[0].set_title("Training (colored) vs held-out (gray) patients", fontsize=9, loc="left")
    auc = member.groupby(["Source", "Repeat"]).apply(
        lambda g: roc_auc_score(g.Member.astype(int), g.Closest_match)).rename("AUC").reset_index()
    for i, s in enumerate(sources):
        v = auc[auc.Source == s].AUC.to_numpy()
        axes[1].scatter(np.full(len(v), i), v, s=22, color=COLORS[s], lw=0)
    axes[1].axhline(0.5, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    axes[1].text(len(sources) - 0.6, 0.51, "no membership signal", ha="right", va="bottom", fontsize=7, color=INK_2)
    axes[1].set_ylabel("Membership AUC (per CV repeat)")
    axes[1].set_title("Distance-based membership inference", fontsize=9, loc="left")
    for ax in axes:
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([tick_label(s) for s in sources], fontsize=7)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "S_holdout_membership")


def plot_tstr(tstr):
    style()
    sources = ["Real (TRTR)"] + MODELS + [BASELINE]
    colors = {**COLORS, "Real (TRTR)": INK}
    fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(7.2, 3.0), sharey=True)
    for ax, endpoint in zip(axes, ENDPOINTS):
        g = tstr[tstr.Endpoint == endpoint].set_index("Source")
        for i, s in enumerate(sources):
            r = g.loc[s]
            ax.plot([i, i], [r.CI_low, r.CI_high], color=colors[s], lw=1.4, solid_capstyle="round")
            ax.plot(i, r.AUC, "o", color=colors[s], ms=6, mec="white", mew=1)
        ax.axhline(0.5, color=INK_2, lw=0.8, ls=(0, (3, 3)))
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([tick_label(s) for s in sources], fontsize=7)
        ax.set_title(endpoint.replace(" (Y/N)", ""), fontsize=9, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("AUC on held-out real patients (95% CI)")
    fig.tight_layout()
    save(fig, "S_holdout_tstr")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    real = load_real()
    metadata = build_metadata(real)
    n = len(real)
    member_rows, aia_rows, pred_rows = [], [], []
    for rep in range(args.repeats):
        folds = KFold(N_FOLDS, shuffle=True, random_state=100 + rep).split(real)
        for fold, (tr, te) in enumerate(folds):
            train = real.iloc[tr].reset_index(drop=True)
            seed = 1000 * (rep + 1) + fold
            datasets = {m: generate(train, metadata, m, seed, tag=f"_holdout_r{rep}f{fold}") for m in MODELS}
            datasets[BASELINE] = independent_marginals(train, seed)
            is_member = np.isin(np.arange(n), tr)
            for source, synth in datasets.items():
                closest = match_fraction(real, synth, real).max(axis=1)
                for i in range(n):
                    member_rows.append({"Source": source, "Repeat": rep, "Fold": fold, "Patient": i,
                                        "Member": bool(is_member[i]), "Closest_match": closest[i]})
                for t in CAT_TARGETS + NUM_TARGETS:
                    prior = prior_guess(synth, real, real, t)
                    for atk in ATTACKERS:
                        if not attack_applicable(atk, t):
                            continue
                        ok = attack(synth, real, real, t, atk, seed)
                        for i in range(n):
                            aia_rows.append({"Source": source, "Repeat": rep, "Fold": fold, "Patient": i,
                                             "Member": bool(is_member[i]), "Target": t, "Attacker": atk,
                                             "Correct": bool(ok[i]), "Prior_correct": bool(prior[i])})
                for endpoint in ENDPOINTS:
                    p = predict_endpoint(synth, real.iloc[te], endpoint, seed)
                    pred_rows += [{"Source": source, "Repeat": rep, "Endpoint": endpoint, "Patient": int(i),
                                   "Prob": pi} for i, pi in zip(te, p)]
            for endpoint in ENDPOINTS:
                p = predict_endpoint(train, real.iloc[te], endpoint, seed)
                pred_rows += [{"Source": "Real (TRTR)", "Repeat": rep, "Endpoint": endpoint, "Patient": int(i),
                               "Prob": pi} for i, pi in zip(te, p)]
            print(f"repeat {rep} fold {fold} done", flush=True)

    member = pd.DataFrame(member_rows)
    aia = pd.DataFrame(aia_rows)
    preds = pd.DataFrame(pred_rows)

    # 1a. distance-based membership AUC, pooled over folds, per repeat
    auc = member.groupby(["Source", "Repeat"]).apply(
        lambda g: roc_auc_score(g.Member.astype(int), g.Closest_match)).rename("AUC").reset_index()
    membership = auc.groupby("Source").AUC.agg(["mean", "min", "max"]).join(
        member.groupby(["Source", "Member"]).Closest_match.median().unstack().rename(
            columns={True: "Median_similarity_members", False: "Median_similarity_nonmembers"}))

    # 1b. attribute inference on members vs non-members (lift over prior-only guess)
    aia["Lift"] = aia.Correct.astype(float) - aia.Prior_correct.astype(float)
    per_patient = aia.groupby(["Source", "Repeat", "Fold", "Patient", "Member"]).agg(
        Success=("Correct", "mean"), Lift=("Lift", "mean")).reset_index()
    aia_summary = []
    for source, g in per_patient.groupby("Source"):
        mem, non = g[g.Member], g[~g.Member]
        diff = mem.Lift.mean() - non.Lift.mean()
        rng = np.random.default_rng(0)
        boots = [rng.choice(mem.Lift.to_numpy(), len(mem)).mean() - rng.choice(non.Lift.to_numpy(), len(non)).mean()
                 for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        aia_summary.append({"Source": source, "Success_members_pct": 100 * mem.Success.mean(),
                            "Success_nonmembers_pct": 100 * non.Success.mean(),
                            "Lift_members_pp": 100 * mem.Lift.mean(), "Lift_nonmembers_pp": 100 * non.Lift.mean(),
                            "Member_minus_nonmember_pp": 100 * diff, "CI95_low": 100 * lo, "CI95_high": 100 * hi})
    aia_summary = pd.DataFrame(aia_summary)

    # 2. TSTR / TRTR: average predicted probability over repeats, one AUC over 18 patients
    y_true = {e: (as_str(real[e]) == "Y").astype(int).to_numpy() for e in ENDPOINTS}
    tstr_rows = []
    for (source, endpoint), g in preds.groupby(["Source", "Endpoint"]):
        p = g.groupby("Patient").Prob.mean().reindex(range(n)).to_numpy()
        y = y_true[endpoint]
        lo, hi = bootstrap_auc(y, p, 0)
        tstr_rows.append({"Source": source, "Endpoint": endpoint, "AUC": roc_auc_score(y, p), "CI_low": lo,
                          "CI_high": hi, "Events": int(y.sum()), "N": n})
    tstr = pd.DataFrame(tstr_rows)

    member.to_csv(OUT / "holdout_membership_by_patient.csv", index=False)
    membership.to_csv(OUT / "holdout_membership_summary.csv")
    aia.to_csv(OUT / "holdout_aia_by_patient.csv", index=False)
    aia_summary.to_csv(OUT / "holdout_aia_member_vs_nonmember.csv", index=False)
    preds.to_csv(OUT / "holdout_tstr_predictions.csv", index=False)
    tstr.to_csv(OUT / "holdout_tstr_summary.csv", index=False)
    plot_membership(member)
    plot_tstr(tstr)

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("=== Membership (closest-record similarity) ===")
    print(membership.round(3).to_string())
    print("\n=== Attribute inference: members vs non-members ===")
    print(aia_summary.round(1).to_string())
    print("\n=== TSTR vs TRTR ===")
    print(tstr.round(2).to_string())


if __name__ == "__main__":
    main()
