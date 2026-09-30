"""Feature-level anonymization of direct identifiers: does it prevent identifier leakage, and does it change
the privacy or utility of the clinical variables?

Three variants, each 4 synthesizers x 10 seeds:
  * Identifiers retained: trained on the full dataset with MRN as the table's primary key (SDV generates new
    IDs), initials as PII, and dates of birth and surgery modeled as dates.
  * AnonymizedFaker (the manuscript's method): the same data, with initials and both dates declared as PII and
    replaced during training by SDV's AnonymizedFaker transformer (random letters / random dates).
  * Identifiers excluded: the main analysis runs, trained without the four identifier columns.

Measures (identical for every variant):
  * identifier reproduction: real patients whose MRN, initials, date of birth or date of surgery appears
    exactly in the synthetic data, with chance references: for dates, matches to the real dates shifted by
    8-30 days; for initials, the expected number of chance matches of uniformly random two-letter strings;
  * clinical privacy: attribute inference gain over the prior-only guess (strongest attacker per target),
    records closer to a real patient than the closest real pair, patients with a non-obvious outcome revealed
    by quasi-identifier linkage;
  * utility: Spearman correlation-structure agreement.
The anonymized variants are compared with the retained variant within each synthesizer (two-sided
Mann-Whitney, Holm across synthesizers); exact vs chance identifier matches with paired Wilcoxon tests.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdt.transformers import AnonymizedFaker
from scipy.stats import mannwhitneyu, wilcoxon
from sdv.metadata import Metadata
from statsmodels.stats.multitest import multipletests

from common import (PRIVATE, COLORS, INK_2, MODELS, N_SYNTH, ORIGINAL_PARAMS, OUT, ROOT, SYNTH_DIR, SYNTHESIZERS, as_str, build_metadata,
                    generate, load_real, match_fraction, metadata_columns, save, seed_all, style)
from privacy_analysis import ATTACKERS, CAT_TARGETS, NUM_TARGETS, attack, attack_applicable, prior_guess
from reidentification_analysis import qi_linkage
from utility_analysis import encoded_matrix, upper

DATE_FORMAT = "%m/%d/%Y"
DATE_SHIFTS = [d for d in range(-30, 31) if abs(d) >= 8]
RETAINED, FAKER, EXCLUDED = "Identifiers retained", "AnonymizedFaker", "Identifiers excluded"
VERSIONS = [(RETAINED, "o", 0.35), (FAKER, "s", 0.65), (EXCLUDED, "D", 0.95)]
METRICS = [("Attack_lift_pp", "Attack success above\nguessing, pp"),
           ("Too_close_pct", "Records closer than closest\nreal pair, %"),
           ("QI_disclosed_patients", "Patients with QI\ndisclosure, of 18"),
           ("Spearman_r", "Correlation-structure\nagreement (utility)")]


def identifier_metadata(real_clinical, faker):
    cols = metadata_columns(real_clinical)
    date_spec = {"sdtype": "unknown", "pii": True} if faker else {"sdtype": "datetime",
                                                                   "datetime_format": DATE_FORMAT}
    cols.update({"MRN": {"sdtype": "id"}, "Initials": {"sdtype": "unknown", "pii": True},
                 "Date of Birth": date_spec, "Date of Sx": dict(date_spec)})
    # MRN must be the table's primary key: declared only as an ID column, SDV 1.38 GaussianCopula reproduced
    # the real MRNs verbatim.
    return Metadata.load_from_dict({"tables": {"table": {"primary_key": "MRN", "columns": cols}},
                                    "METADATA_SPEC_VERSION": "V1"})


def generate_faker(full, metadata, model, seed):
    path = SYNTH_DIR / f"{model}_faker_seed{seed}.csv"
    if path.exists():
        return pd.read_csv(path)
    seed_all(seed)
    synth = SYNTHESIZERS[model](metadata, **ORIGINAL_PARAMS[model])
    synth.auto_assign_transformers(full)
    date_faker = dict(provider_name="date_time", function_name="date", function_kwargs={"pattern": DATE_FORMAT})
    synth.update_transformers({
        "Initials": AnonymizedFaker(function_name="lexify",
                                    function_kwargs={"text": "??", "letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}),
        "Date of Birth": AnonymizedFaker(**date_faker),
        "Date of Sx": AnonymizedFaker(**date_faker),
    })
    synth.fit(full)
    synth._set_random_state(seed)
    # AnonymizedFaker keeps its own Faker seed, which SDV's random state does not reach; without this every run
    # would draw the same fake identifiers.
    for column, transformer in synth.get_transformers().items():
        if isinstance(transformer, AnonymizedFaker):
            transformer._faker_random_seed = seed
            transformer.faker.seed_instance(seed)
    df = synth.sample(N_SYNTH)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def identifier_reproduction(full, synth):
    out = {}
    real_mrn, real_init = as_str(full["MRN"]), as_str(full["Initials"]).str.upper()
    syn_init = set(as_str(synth["Initials"]).str.upper())
    out["Exact_MRN"] = int(real_mrn.isin(set(as_str(synth["MRN"]))).sum())
    out["Exact_Initials"] = int(real_init.isin(syn_init).sum())
    # Chance matches if the synthetic initials were uniformly random two-letter strings.
    is_two_letter = pd.Series(list(syn_init)).str.fullmatch(r"[A-Z]{2}").mean() if syn_init else 0
    out["Chance_Initials"] = float(is_two_letter * len(real_init) * (1 - (1 - 1 / 676) ** len(synth)))
    for c in ["Date of Birth", "Date of Sx"]:
        real_d = pd.to_datetime(full[c], format=DATE_FORMAT)
        syn_d = set(pd.to_datetime(synth[c], format=DATE_FORMAT, errors="coerce").dropna())
        out[f"Exact_{c}"] = int(real_d.isin(syn_d).sum())
        out[f"Chance_{c}"] = float(np.mean([(real_d + pd.Timedelta(days=d)).isin(syn_d).sum()
                                            for d in DATE_SHIFTS]))
    return out


def clinical_privacy_and_utility(real, synth, seed, real_sp, real_min_dcr):
    lifts = []
    for t in CAT_TARGETS + NUM_TARGETS:
        prior = prior_guess(synth, real, real, t).sum()
        best = max(attack(synth, real, real, t, a, seed).sum() for a in ATTACKERS if attack_applicable(a, t))
        lifts.append(100 * (best - prior) / len(real))
    dcr = 1 - match_fraction(synth, real, real).max(axis=1)
    sp = encoded_matrix(synth, real).corr(method="spearman").fillna(0.0)
    return {"Attack_lift_pp": float(np.mean(lifts)),
            "Too_close_pct": 100 * float(np.mean(dcr < real_min_dcr)),
            "QI_disclosed_patients": qi_linkage(real, synth)[1],
            "Spearman_r": float(np.corrcoef(upper(real_sp), upper(sp))[0, 1])}


def plot(res):
    style()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.2, 2.7))
    rng = np.random.default_rng(0)
    offsets = {RETAINED: -0.24, FAKER: 0.0, EXCLUDED: 0.24}
    for ax, (metric, title) in zip(axes, METRICS):
        for i, m in enumerate(MODELS):
            for version, marker, alpha in VERSIONS:
                v = res[(res.Model == m) & (res.Version == version)][metric].to_numpy(dtype=float)
                x = i + offsets[version]
                ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=8, marker=marker, color=COLORS[m],
                           alpha=alpha, lw=0)
                ax.plot([x - 0.09, x + 0.09], [v.mean()] * 2, color=INK_2, lw=1.2)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([{"GaussianCopula": "Gaussian-\nCopula", "CopulaGAN": "Copula-\nGAN"}.get(m, m)
                            for m in MODELS], fontsize=6.5)
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], marker=mk, ls="", color=INK_2, alpha=a, label=v) for v, mk, a in VERSIONS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "S_identifier_comparison")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    full = pd.read_csv(PRIVATE / "clean_data.csv")
    real = load_real()
    meta_retained, meta_faker = identifier_metadata(real, faker=False), identifier_metadata(real, faker=True)
    meta_clin = build_metadata(real)
    real_sp = encoded_matrix(real, real).corr(method="spearman").fillna(0.0)
    rr = match_fraction(real, real, real)
    np.fill_diagonal(rr, 0)
    real_min_dcr = (1 - rr.max(axis=1)).min()

    rows, ident_rows = [], []
    for seed in range(args.seeds):
        for m in MODELS:
            retained = generate(full, meta_retained, m, seed, tag="_withids")
            faker = generate_faker(full, meta_faker, m, seed)
            for version, synth in [(RETAINED, retained), (FAKER, faker)]:
                ident_rows.append({"Model": m, "Version": version, "Seed": seed,
                                   **identifier_reproduction(full, synth)})
            versions = {RETAINED: retained[real.columns], FAKER: faker[real.columns],
                        EXCLUDED: generate(real, meta_clin, m, seed)}
            for version, synth in versions.items():
                rows.append({"Model": m, "Version": version, "Seed": seed,
                             **clinical_privacy_and_utility(real, synth, seed, real_sp, real_min_dcr)})
        print(f"seed {seed} done", flush=True)
    res, ident = pd.DataFrame(rows), pd.DataFrame(ident_rows)

    # Identifier reproduction: exact vs chance, per model and variant.
    ident_tests = []
    for (m, version), g in ident.groupby(["Model", "Version"]):
        for c in ["Initials", "Date of Birth", "Date of Sx"]:
            exact, chance = g[f"Exact_{c}"], g[f"Chance_{c}"]
            diff = exact - chance
            p = wilcoxon(exact, chance).pvalue if (diff != 0).any() else 1.0
            ident_tests.append({"Model": m, "Version": version, "Identifier": c, "Exact_mean": exact.mean(),
                                "Exact_range": f"{exact.min()}-{exact.max()}", "Chance_mean": chance.mean(),
                                "Runs_above_chance": int((diff > 0).sum()), "p_Wilcoxon": p})
        ident_tests.append({"Model": m, "Version": version, "Identifier": "MRN",
                            "Exact_mean": g.Exact_MRN.mean(), "Exact_range": f"{g.Exact_MRN.min()}-{g.Exact_MRN.max()}",
                            "Chance_mean": 0.0, "Runs_above_chance": int((g.Exact_MRN > 0).sum()), "p_Wilcoxon": np.nan})
    ident_tests = pd.DataFrame(ident_tests)

    # Clinical privacy and utility: each anonymized variant vs retained.
    tests = []
    for comparison in (FAKER, EXCLUDED):
        for metric, _ in METRICS:
            ps, rows_m = [], []
            for m in MODELS:
                a = res[(res.Model == m) & (res.Version == RETAINED)][metric]
                b = res[(res.Model == m) & (res.Version == comparison)][metric]
                ps.append(mannwhitneyu(a, b).pvalue)
                rows_m.append({"Comparison": f"{comparison} vs {RETAINED}", "Metric": metric, "Model": m,
                               "Retained_mean": a.mean(), "Anonymized_mean": b.mean(), "Difference": b.mean() - a.mean()})
            for r, p, ph in zip(rows_m, ps, multipletests(ps, method="holm")[1]):
                tests.append({**r, "p": p, "p_Holm": ph})
    tests = pd.DataFrame(tests)

    res.to_csv(OUT / "identifier_comparison_by_run.csv", index=False)
    ident.to_csv(OUT / "identifier_reproduction_by_run.csv", index=False)
    ident_tests.to_csv(OUT / "identifier_reproduction_tests.csv", index=False)
    tests.to_csv(OUT / "identifier_comparison_tests.csv", index=False)
    plot(res)

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("=== Identifier reproduction: exact vs chance ===")
    print(ident_tests.round(3).to_string(index=False))
    print("\n=== Clinical privacy and utility: anonymized vs retained ===")
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
