"""Assemble the Supplementary Material (.docx) from the methods/results drafts, analysis CSVs and figures.

Run after the analysis scripts. Output: results/supplement/Supplementary_Material_revision.docx
"""
import re

import pandas as pd
from PIL import Image
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from common import BASELINE, FIG_DIR, MODELS, OUT, SUPPLEMENT_DIR

SOURCES = MODELS + [BASELINE]
SHORT_SOURCE = {BASELINE: "Reshuffled reference"}
MAX_FIG_HEIGHT = 7.8  # inches

SAFE_HARBOR = [
    ("1. Names", "Patient initials in source data", "Excluded from training; never modeled"),
    ("2. Geographic subdivisions smaller than a state", "None collected", "—"),
    ("3. All elements of dates (except year); ages > 89",
     "Date of birth, date of surgery; age at diagnosis and at surgery",
     "Dates excluded from training. Remaining time variables are durations (survival, follow-up, "
     "length of stay), not dates. Maximum age 79 in real and synthetic data (no ages > 89)"),
    ("4. Telephone numbers", "None collected", "—"),
    ("5. Fax numbers", "None collected", "—"),
    ("6. Email addresses", "None collected", "—"),
    ("7. Social Security numbers", "None collected", "—"),
    ("8. Medical record numbers", "MRN in source data", "Excluded from training; never modeled"),
    ("9. Health plan beneficiary numbers", "None collected", "—"),
    ("10. Account numbers", "None collected", "—"),
    ("11. Certificate/license numbers", "None collected", "—"),
    ("12. Vehicle identifiers", "None collected", "—"),
    ("13. Device identifiers and serial numbers", "None collected", "—"),
    ("14. Web URLs", "None collected", "—"),
    ("15. IP addresses", "None collected", "—"),
    ("16. Biometric identifiers", "None collected", "—"),
    ("17. Full-face photographs", "None collected", "—"),
    ("18. Any other unique identifying number, characteristic or code",
     "No free-text fields (≤ 6 categories per variable); rare combinations of clinical characteristics",
     "Combinations of clinical characteristics evaluated with attribute inference and membership analyses "
     "(Supplementary Methods S5–S6)"),
]

FIGURES = [
    ("orig:fig1", "Supplementary Figure S1. Molecular and imaging feature prevalence in the real cohort."),
    ("orig:fig2", "Supplementary Figure S2. Symptom trajectories in the real cohort."),
    ("S_symptom_trajectories", "Supplementary Figure S3. Symptom prevalence across the clinical course in the real "
     "cohort (black) and in each model's synthetic data (mean of 10 runs; vertical lines show the range across "
     "runs)."),
    ("S_discharge_disposition", "Supplementary Figure S4. Discharge disposition in the real cohort and in synthetic "
     "data (mean of 10 runs; error bars show the range across runs)."),
    ("S_extent_of_resection", "Supplementary Figure S5. Extent of resection in the real cohort and in synthetic data "
     "(mean of 10 runs; error bars show the range across runs)."),
    ("S_missingness", "Supplementary Figure S6. Variable-level missingness before imputation (variables with at "
     "least one missing value; labels give the number of missing patients out of 18)."),
    ("S_distribution_tests", "Supplementary Figure S7. Univariate and pairwise distributional similarity across 10 "
     "runs: mean Kolmogorov–Smirnov statistic over numerical variables, mean Jensen–Shannon divergence over "
     "categorical variables, and mean Jensen–Shannon divergence of the joint distribution of binary variable "
     "pairs (lower = more similar). The reshuffled reference is the ceiling for univariate similarity."),
    ("S_utility_associations", "Supplementary Figure S8. Prespecified clinical associations in the real cohort "
     "(black; log odds ratio with 95% CI) and in synthetic data (mean and range across 10 runs)."),
    ("S_relationships_heatmaps", "Supplementary Figure S9. Spearman correlation matrices of the real cohort and of "
     "one synthetic dataset per model (55 binary and numerical variables; same variable order in every panel, "
     "from hierarchical clustering of the real correlations)."),
    ("S_relationships_metrics", "Supplementary Figure S10. Nine relationship metrics across 10 runs per model. "
     "Points are individual runs; bars are means."),
    ("S_privacy_by_anonymization", "Supplementary Figure S11. Attribute inference with direct identifiers (A) "
     "retained in training or (B) anonymized with AnonymizedFaker, and (C) the memorization test. Markers show "
     "each attacker for each of 10 runs; dashed lines indicate guessing (A, B) and no difference between training "
     "and held-out patients (C). Anonymization did not change attack success on clinical variables."),
    ("S_privacy_revised_attack_by_target", "Supplementary Figure S12. Attribute inference by target: percentage of "
     "the 18 real patients whose value was correctly inferred by the strongest attacker in each run (mean and "
     "range across 10 runs), compared with a prior-only guess that ignores the attacker's knowledge (black bar)."),
    ("S_holdout_membership", "Supplementary Figure S13. Membership inference with models retrained in 6-fold "
     "cross-validation (2 repetitions). Left: similarity of each real patient to the closest synthetic record "
     "when the patient was in the model's training data (colored) or held out (gray). Right: area under the ROC "
     "curve for separating training from held-out patients; the reshuffled reference gives the reference level."),
    ("S_reidentification", "Supplementary Figure S14. Record-level re-identification risk across 10 runs: "
     "synthetic records matching a real patient on at least 95% of variables; synthetic records closer to a real "
     "patient than the closest pair of real patients; and real patients whose non-obvious sensitive value is "
     "revealed by linking four quasi-identifiers to the synthetic data."),
    ("S_privacy_original_method_evaluable", "Supplementary Figure S15. Re-analysis of the original attack: number "
     "of real patients (of 18) for whom the original SDMetrics attacker could make any prediction, per attack, "
     "sensitive field and run. Patients without a prediction were scored as protected by default."),
]


# ---------- document helpers ----------

def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_runs(paragraph, text):
    """Add text with **bold** spans."""
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part:
            paragraph.add_run(part)


def add_markdown(doc, path):
    for line in open(path, encoding="utf-8").read().splitlines():
        if not line.strip():
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("- "):
            add_runs(doc.add_paragraph(style="List Bullet"), line[2:])
        else:
            add_runs(doc.add_paragraph(), line)


def display(text):
    """Figures and text call the relationship-free reference the 'reshuffled reference'."""
    return (str(text).replace("Independent baseline", "Reshuffled reference")
            .replace("independent-marginals baseline", "reshuffled reference"))


def add_table(doc, title, header, rows, widths, note=None, font=8):
    title, header, note = display(title), [display(h) for h in header], note and display(note)
    rows = [[display(c) for c in row] for row in rows]
    heading = doc.add_paragraph()
    heading.add_run(title).bold = True
    heading.paragraph_format.keep_with_next = True
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text, w in zip(table.rows[0].cells, header, widths):
        cell.width = Inches(w)
        shade(cell, "E4E3DF")
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(font)
    for row in rows:
        cells = table.add_row().cells
        for cell, text, w in zip(cells, row, widths):
            cell.width = Inches(w)
            cell.paragraphs[0].add_run(str(text)).font.size = Pt(font)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(18)
    if note:
        r = p.add_run(note)
        r.italic = True
        r.font.size = Pt(8)


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ---------- tables from analysis outputs ----------

def table_missingness(doc):
    t = pd.read_csv(OUT / "missingness_table.csv")
    t = t[t.Missing > 0]
    rows = [(r.Variable, r.Type, f"{r.Missing}/18 ({r.Missing_pct:.1f}%)", r.Imputation) for r in t.itertuples()]
    add_table(doc, "Supplementary Table S2. Variable-level missingness before imputation",
              ["Variable", "Type", "Missing", "Imputation"], rows, [3.0, 1.0, 1.2, 1.6],
              note="Variables not listed had no missing values (21 of 61). Blank cells and cells recorded as "
                   "'Missing' in the source database were counted as missing.")


def table_safe_harbor(doc):
    add_table(doc, "Supplementary Table S3. Audit of study variables against the 18 HIPAA Safe Harbor identifier "
                   "categories (45 CFR §164.514(b)(2))",
              ["Identifier category", "Present in study data", "Handling"], SAFE_HARBOR, [2.0, 2.1, 2.7],
              note="This audit documents how each identifier category was handled; it is not a formal "
                   "Safe Harbor or Expert Determination certification of the released synthetic datasets.")


def table_fidelity(doc):
    s = pd.read_csv(OUT / "fidelity_summary.csv").set_index(["Source", "Metric"])
    d = pd.read_csv(OUT / "distribution_tests_summary.csv").set_index(["Source", "Metric"])
    rows = []
    for src in SOURCES:
        row = [src]
        for m in ["Overall", "Column Shapes", "Column Pair Trends"]:
            r = s.loc[(src, m)]
            row.append(f"{r.Mean:.1f} ± {r.SD:.1f} ({r.Min:.1f}–{r.Max:.1f})")
        row.append(f"{d.loc[(src, 'Mean_KS_numerical')].Mean:.2f}")
        row.append(f"{d.loc[(src, 'Mean_JSD_categorical')].Mean:.3f} "
                   f"({d.loc[(src, 'Pct_categorical_JSD_below_0.1')].Mean:.0f}%)")
        rows.append(row)
    tests = pd.read_csv(OUT / "fidelity_tests.csv")
    kw = tests[tests.Comparison.str.startswith("Kruskal")].set_index("Metric").p
    add_table(doc, "Supplementary Table S5. Fidelity across 10 runs per synthesizer: SDMetrics quality report and "
                   "distributional tests",
              ["Source", "SDMetrics overall, %", "Column shapes, %", "Column pair trends, %", "KS, numerical",
               "JSD, categorical (% < 0.1)"], rows, [1.35, 1.25, 1.25, 1.25, 0.8, 1.1], font=7,
              note="SDMetrics: mean ± SD (range) across 10 runs; Kruskal–Wallis across the four synthesizers: overall "
                   f"p {fmt_p(kw['Overall'])}, column shapes p {fmt_p(kw['Column Shapes'])}, column pair trends "
                   f"p {fmt_p(kw['Column Pair Trends'])}. KS, mean Kolmogorov–Smirnov statistic over numerical "
                   "variables; JSD, mean Jensen–Shannon divergence over categorical variables (percentage of "
                   "variables with JSD < 0.1); lower is more similar. The reshuffled reference resamples observed "
                   "values and is the ceiling for univariate similarity.")


def table_associations(doc):
    a = pd.read_csv(OUT / "utility_associations_summary.csv")
    order = list(dict.fromkeys(a.Association))
    rows = []
    for assoc in order:
        g = a[a.Association == assoc].set_index("Source")
        r0 = g.iloc[0]
        row = [assoc, f"{r0.Real_logOR:.2f} ({r0.Real_CI_low:.2f} to {r0.Real_CI_high:.2f})"]
        for src in SOURCES:
            r = g.loc[src]
            row.append(f"{r.logOR_mean:.2f} [{r.Same_direction_runs_pct:.0f}%]")
        rows.append(row)
    add_table(doc, "Supplementary Table S6. Reproduction of prespecified clinical associations",
              ["Association", "Real log OR (95% CI)"] + [SHORT_SOURCE.get(s, s) for s in SOURCES], rows,
              [1.75, 1.15] + [0.8] * 5, font=7,
              note="Synthetic entries: mean log odds ratio across 10 runs [percentage of runs with the same "
                   "direction as the real estimate]. Log odds ratios are Haldane–Anscombe corrected.")


def table_global(doc):
    g = pd.read_csv(OUT / "utility_global_summary.csv").set_index(["Source", "Metric"])
    rows = []
    for src in SOURCES:
        c = g.loc[(src, "Corr_of_correlations")]
        d = g.loc[(src, "Mean_abs_diff")]
        s = g.loc[(src, "Sign_agreement_strong_pct")]
        rows.append([src, f"{c.Mean:.2f} ({c.CI95_low:.2f} to {c.CI95_high:.2f})",
                     f"{d.Mean:.3f} ({d.CI95_low:.3f} to {d.CI95_high:.3f})",
                     f"{s.Mean:.1f} ({s.CI95_low:.1f} to {s.CI95_high:.1f})"])
    add_table(doc, "Supplementary Table S5. Preservation of the global correlation structure (55 variables, "
                   "1,485 pairs)",
              ["Source", "Correlation of correlations", "Mean absolute difference", "Sign agreement, strong pairs, %"],
              rows, [1.5, 1.8, 1.8, 1.8], note="Mean (95% CI) across 10 runs. Strong pairs: |ρ| ≥ 0.3 in the "
                                               "real data (410 pairs).")


def table_original_attack(doc):
    s = pd.read_csv(OUT / "privacy_original_summary.csv").set_index("Source")
    r = pd.read_csv(OUT / "privacy_original_risk_by_seed.csv").set_index("Source")
    rows = [[src, f"{s.loc[src].Evaluable_mean:.1f} (max {s.loc[src].Evaluable_max:.0f})",
             f"{r.loc[src]['mean']:.1f} ({r.loc[src]['min']:.1f}–{r.loc[src]['max']:.1f})"] for src in SOURCES]
    add_table(doc, "Supplementary Table S13. Re-analysis of the original SDMetrics attack specification across "
                   "10 runs",
              ["Source", "Patients evaluable per attack, of 18", "Reported risk per run, % mean (range)"], rows,
              [1.8, 2.4, 2.6],
              note="Original key fields and clinical sensitive fields; kNN, naive Bayes and random-forest "
                   "SDMetrics attackers. Reported risk = 1 − SDMetrics privacy score, averaged over sensitive "
                   "fields and attackers within each run.")


def table_revised_by_target(doc):
    t = pd.read_csv(OUT / "privacy_revised_by_target.csv")
    targets = list(dict.fromkeys(t.Target))
    rows = []
    for tgt in targets:
        g = t[t.Target == tgt].set_index("Source")
        row = [tgt, f"{g.iloc[0].Prior_correct_of_18:g}"]
        for src in SOURCES:
            r = g.loc[src]
            row.append(f"{r.Median_correct_of_18}/18 ({round(r.Mean_lift_pp, 1) + 0.0:+.1f})")
        rows.append(row)
    add_table(doc, "Supplementary Table S9. Attribute inference attack by target (strongest attacker per run)",
              ["Target", "Prior-only guess, correct of 18"] + [SHORT_SOURCE.get(s, s) for s in SOURCES], rows,
              [1.55, 0.9] + [0.87] * 5, font=7,
              note="Median number of real patients correctly inferred across 10 runs (mean lift over the "
                   "prior-only guess, percentage points). Continuous targets: correct if within ±10% of the "
                   "real interquartile range. Exact 95% CIs for a count k of 18 are wide (for example, 15/18: "
                   "58.6–96.4%); per-run values and CIs are in privacy_revised_by_target.csv.")


def table_revised_by_model(doc):
    m = pd.read_csv(OUT / "privacy_revised_by_model.csv").set_index(["Source", "Metric"])
    tests = pd.read_csv(OUT / "privacy_revised_tests.csv")
    vs_base = {c.split(" vs ")[0]: p for c, p in zip(tests.Comparison, tests.p_Holm) if c.endswith(BASELINE)}
    rows = []
    for src in SOURCES:
        s = m.loc[(src, "Mean_success_pct")]
        lift = m.loc[(src, "Mean_lift_pp")]
        rows.append([src, f"{s.Mean:.1f} ({s.CI95_low:.1f} to {s.CI95_high:.1f})",
                     f"{lift.Mean:+.1f} ({lift.CI95_low:+.1f} to {lift.CI95_high:+.1f})",
                     "—" if src == BASELINE else fmt_p(vs_base[src])])
    kw = tests[tests.Comparison.str.startswith("Kruskal")].p.iloc[0]
    add_table(doc, "Supplementary Table S10. Attribute inference attack by synthesizer",
              ["Source", "Mean success, % (95% CI)", "Lift over prior-only guess, pp (95% CI)",
               "Holm p vs reference"], rows, [1.5, 1.8, 2.1, 1.3],
              note=f"Averaged over eight targets per run (strongest attacker); 95% CI across 10 runs. "
                   f"Kruskal–Wallis across synthesizers p {fmt_p(kw)}.")


def table_holdout(doc):
    mem = pd.read_csv(OUT / "holdout_membership_summary.csv").set_index("Source")
    aia = pd.read_csv(OUT / "holdout_aia_member_vs_nonmember.csv").set_index("Source")
    rows = []
    for src in SOURCES:
        a, b = mem.loc[src], aia.loc[src]
        rows.append([src, f"{a['mean']:.2f} ({a['min']:.2f}–{a['max']:.2f})",
                     f"{100 * a.Median_similarity_members:.1f} / {100 * a.Median_similarity_nonmembers:.1f}",
                     f"{b.Success_members_pct:.1f} / {b.Success_nonmembers_pct:.1f}",
                     f"{b.Member_minus_nonmember_pp:+.1f} ({b.CI95_low:+.1f} to {b.CI95_high:+.1f})"])
    add_table(doc, "Supplementary Table S11. Memorization and membership analyses (6-fold cross-validation, "
                   "2 repetitions)",
              ["Source", "Membership AUC, mean (range)", "Closest-record similarity, %, members / held-out",
               "Attack success, %, members / held-out", "Difference in lift, pp (95% CI)"], rows,
              [1.4, 1.3, 1.5, 1.4, 1.4], font=7,
              note="Members: patients in the synthesizer's training folds; held-out: patients excluded from "
                   "training. The independent-marginals baseline resamples exact values from training patients, "
                   "so its AUC is the reference level for this measure.")


def table_tstr(doc):
    t = pd.read_csv(OUT / "holdout_tstr_summary.csv").set_index(["Source", "Endpoint"])
    rows = []
    for src in ["Real (TRTR)"] + SOURCES:
        rows.append([src] + [f"{t.loc[(src, e)].AUC:.2f} ({t.loc[(src, e)].CI_low:.2f}–{t.loc[(src, e)].CI_high:.2f})"
                             for e in ["Postop complication", "Radiotherapy (Y/N)"]])
    add_table(doc, "Supplementary Table S10. Train-on-synthetic, test-on-real discrimination",
              ["Training data", "Postoperative complication (6/18), AUC (95% CI)",
               "Radiotherapy (12/18), AUC (95% CI)"], rows, [1.8, 2.5, 2.5],
              note="Random forest, 34 pre-/intraoperative predictors; predictions for held-out patients pooled "
                   "over 6 folds and averaged over 2 repetitions; bootstrap 95% CI.")


def _summary_cell(s, source, metric, digits=2, ci=True):
    r = s.loc[(source, metric)]
    def f(x):
        return f"{max(round(x, digits), 0.0) + 0.0 if metric.endswith(('pct', 'patients')) else round(x, digits) + 0.0:.{digits}f}"
    return f"{f(r.Mean)} ({f(r.CI95_low)} to {f(r.CI95_high)})" if ci else f(r.Mean)


def table_distribution(doc):
    s = pd.read_csv(OUT / "distribution_tests_summary.csv").set_index(["Source", "Metric"])
    rows = [[src, _summary_cell(s, src, "Mean_KS_numerical"), _summary_cell(s, src, "Mean_JSD_numerical"),
             _summary_cell(s, src, "Mean_JSD_categorical", 3), _summary_cell(s, src, "Pct_categorical_JSD_below_0.1", 0),
             _summary_cell(s, src, "Mean_JSD_pairs", 3)] for src in SOURCES]
    add_table(doc, "Supplementary Table S11. Univariate and pairwise distributional similarity (KS statistic and "
                   "Jensen–Shannon divergence)",
              ["Source", "KS, numerical", "JSD, numerical", "JSD, categorical", "Categorical with JSD < 0.1, %",
               "JSD, binary pairs"], rows, [1.3, 1.1, 1.1, 1.15, 1.1, 1.15], font=7,
              note="Mean (95% CI) across 10 runs; lower KS and JSD indicate greater similarity. The "
                   "independent-marginals baseline resamples observed values and is the ceiling for univariate "
                   "similarity.")


def table_relationships(doc):
    s = pd.read_csv(OUT / "relationships_summary.csv").set_index(["Source", "Metric"])
    tests = pd.read_csv(OUT / "relationships_tests.csv")
    metrics = [("Spearman_r", "Spearman matrix correlation (↑)"), ("Kendall_r", "Kendall matrix correlation (↑)"),
               ("Spearman_CMD", "Spearman CMD (↓)"), ("Kendall_CMD", "Kendall CMD (↓)"),
               ("NMI_r", "Mutual information (↑)"), ("CramersV_r", "Cramér's V (↑)"),
               ("CramersV_corrected_r", "Bias-corrected Cramér's V (↑)"), ("TheilsU_r", "Theil's U (↑)"),
               ("Log_cluster", "Log-cluster (↓)")]
    rows = []
    for metric, label in metrics:
        row = [label]
        t = tests[tests.Metric == metric]
        higher_better = "(↑)" in label
        base = s.loc[(BASELINE, metric)].Mean
        for src in SOURCES:
            mean = s.loc[(src, metric)].Mean
            cell = f"{round(mean, 2) + 0.0:.2f}"
            if src != BASELINE:
                p = t[t.Comparison == f"{src} vs {BASELINE}"].p_Holm.iloc[0]
                if p < 0.05:
                    cell += "*" if (mean > base) == higher_better else "†"
            row.append(cell)
        rows.append(row)
    g = pd.read_csv(OUT / "utility_global_summary.csv").set_index(["Source", "Metric"])
    gt = pd.read_csv(OUT / "utility_global_tests.csv")
    gt = gt[gt.Metric == "Sign_agreement_strong_pct"]
    row = ["Sign agreement, strong correlations, % (↑)"]
    for src in SOURCES:
        cell = f"{g.loc[(src, 'Sign_agreement_strong_pct')].Mean:.1f}"
        if src != BASELINE and gt[gt.Comparison == f"{src} vs {BASELINE}"].p_Holm.iloc[0] < 0.05:
            cell += "*"
        row.append(cell)
    rows.insert(4, row)
    add_table(doc, "Supplementary Table S7. Relationship (multivariable) metrics across 10 runs per synthesizer",
              ["Metric"] + [SHORT_SOURCE.get(src, src) for src in SOURCES], rows, [2.1] + [0.94] * 5, font=7,
              note="Mean across 10 runs. * significantly better and † significantly worse than the "
                   "independent-marginals baseline (Holm-adjusted p < 0.05, two-sided Mann–Whitney). CMD, correlation matrix distance. Uncorrected Cramér's V, Theil's U and mutual "
                   "information are positively biased at n = 18; the bias-corrected Cramér's V was added post hoc.")


def table_reidentification(doc):
    s = pd.read_csv(OUT / "reidentification_summary.csv").set_index(["Source", "Metric"])
    h = pd.read_csv(OUT / "reidentification_qi_holdout.csv").set_index("Source")
    ctx = pd.read_csv(OUT / "reidentification_real_context.csv").iloc[0]
    rows = []
    for src in SOURCES:
        g = h.loc[src]
        rows.append([src, f"{s.loc[(src, 'Exact_copies_pct')].Mean:.1f} / {s.loc[(src, 'Near_copies_95_pct')].Mean:.1f}",
                     _summary_cell(s, src, "Too_close_pct", 1), _summary_cell(s, src, "DCR_5th_pct", 2, ci=False),
                     _summary_cell(s, src, "QI_disclosed_patients", 1),
                     f"{g.Disclosed_members_pct:.1f} / {g.Disclosed_heldout_pct:.1f} "
                     f"({g.Difference_pp:+.1f}; {g.CI95_low:+.1f} to {g.CI95_high:+.1f})"])
    add_table(doc, "Supplementary Table S12. Record-level re-identification risk",
              ["Source", "Exact / ≥ 95% copies, % of records", "Closer than closest real pair, % (95% CI)",
               "DCR, 5th percentile", "Patients with QI disclosure, of 18 (95% CI)",
               "QI disclosure, % training / held-out (difference; 95% CI)"], rows,
              [1.2, 1.0, 1.25, 0.8, 1.25, 1.6], font=7,
              note=f"Mean across 10 runs (holdout column: 6-fold cross-validation, 2 repetitions). Real cohort: "
                   f"closest pair of distinct patients differs on {100 * ctx.Real_nearest_neighbor_DCR_min:.1f}% of "
                   f"variables; {int(ctx.Real_unique_patients_on_QIs)} of 18 patients unique on the four "
                   f"quasi-identifiers (k-anonymity = {int(ctx.Real_k_anonymity_min_k)}). QI, quasi-identifier "
                   "(10-year age band, sex, cranial location, WHO grade); DCR, distance to closest record.")


def table_tumor(doc):
    """Supplementary Table S1 from the original submission, unchanged."""
    rows = [
        ("Cranial location", "13 (72%) infratentorial; 5 (28%) supratentorial"),
        ("Laterality", "8 (44%) midline; 6 (33%) left; 4 (22%) right"),
        ("Intraventricular vs extraventricular vs both", "9 (50%) intraventricular; 5 (28%) extraventricular; "
                                                         "3 (17%) both*"),
        ("Orthogonal dimensions (cm)", "3.14 (IQR 2.15–3.95); 2.40 (IQR 1.78–2.95); 2.40 (IQR 1.78–3.40)"),
    ]
    add_table(doc, "Supplementary Table S1. Tumor characteristics of the real-world ependymoma cohort (n = 18)",
              ["Tumor characteristic", "Value"], rows, [2.4, 4.4])


def table_settings(doc):
    rows = [
        ("GaussianCopula", "—", "Per-variable parametric (normal, truncated normal, gamma); default truncated normal",
         "Rounded to real-data precision"),
        ("CopulaGAN", "1,000", "Per-variable parametric (as GaussianCopula); default beta", "None"),
        ("CTGAN", "1,000", "Learned (mode-specific normalization)", "None"),
        ("TVAE", "500", "Learned", "None"),
    ]
    add_table(doc, "Supplementary Table S4. Synthesizer settings",
              ["Model", "Training epochs", "Numerical marginal distributions", "Rounding"], rows,
              [1.3, 1.0, 3.2, 1.3],
              note="Settings reproduce those of the models fitted in the original analysis (SDV 1.17.3); all other "
                   "hyperparameters, including network architectures, batch size (500) and learning rates, were SDV "
                   "defaults. Each model was trained and sampled under 10 random seeds (1,000 records per run).")


def table_identifiers(doc):
    t = pd.read_csv(OUT / "identifier_reproduction_tests.csv")
    rows = []
    for version, label in [("Identifiers retained", "Retained"), ("AnonymizedFaker", "AnonymizedFaker")]:
        for model in MODELS:
            row = [f"{model} ({label})"]
            for ident in ["MRN", "Initials", "Date of Birth", "Date of Sx"]:
                r = t[(t.Model == model) & (t.Version == version) & (t.Identifier == ident)].iloc[0]
                star = "*" if r.Exact_mean > r.Chance_mean and r.p_Wilcoxon < 0.05 else ""
                row.append(f"{r.Exact_mean:.1f} ({r.Chance_mean:.1f}){star}")
            rows.append(row)
    add_table(doc, "Supplementary Table S8. Reproduction of real direct identifiers with identifiers retained in "
                   "training vs feature-level anonymization (AnonymizedFaker)",
              ["Model (variant)", "MRN", "Initials", "Date of birth", "Date of surgery"], rows,
              [2.4, 1.0, 1.1, 1.1, 1.1], font=7.5,
              note="Real patients (of 18) whose identifier appeared exactly in the synthetic data, mean over 10 runs "
                   "(number expected by chance). Chance reference: for dates, matches to the real dates shifted by "
                   "8–30 days; for initials, expected matches of uniformly random two-letter strings. * above "
                   "chance, paired Wilcoxon p < 0.05. MRN was the table's primary key, for which SDV generates new "
                   "identifiers.")


def main():
    doc = Document()
    for section in doc.sections:
        section.page_width, section.page_height = Inches(8.5), Inches(11)
        for side in ["left_margin", "right_margin", "top_margin", "bottom_margin"]:
            setattr(section, side, Inches(0.8))
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    for level in (1, 2, 3):
        h = doc.styles[f"Heading {level}"]
        h.font.name = "Arial"
        # Built-in heading styles use theme fonts, which override font.name.
        fonts = h.element.get_or_add_rPr().get_or_add_rFonts()
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            fonts.attrib.pop(qn(attr), None)
        for attr in ("w:ascii", "w:hAnsi", "w:cs"):
            fonts.set(qn(attr), "Arial")
        h.font.color.rgb = RGBColor(0x0B, 0x0B, 0x0B)
        h.font.size = Pt({1: 14, 2: 12, 3: 10.5}[level])

    title = doc.add_paragraph()
    title.add_run("Supplementary Material").bold = True
    title.runs[0].font.size = Pt(16)
    doc.add_paragraph("Cybersecurity and AI in Neuro-oncology: An Exploratory Evaluation of Synthetic Data "
                      "Privacy and Fidelity in a Rare Brain Tumor Cohort")

    add_markdown(doc, SUPPLEMENT_DIR / "Supplementary_Methods_draft.md")
    doc.add_page_break()
    add_markdown(doc, SUPPLEMENT_DIR / "Supplementary_Results_draft.md")

    doc.add_heading("Supplementary Tables", level=1).paragraph_format.page_break_before = True
    table_tumor(doc)
    table_missingness(doc)
    table_safe_harbor(doc)
    table_settings(doc)
    table_fidelity(doc)
    table_associations(doc)
    table_relationships(doc)
    table_identifiers(doc)
    table_revised_by_target(doc)
    table_revised_by_model(doc)
    table_holdout(doc)
    table_reidentification(doc)
    table_original_attack(doc)

    doc.add_heading("Supplementary Figures", level=1).paragraph_format.page_break_before = True
    for name, caption in FIGURES:
        path = (SUPPLEMENT_DIR / "original_figures" / f"{name[5:]}.png") if name.startswith("orig:") else FIG_DIR / f"{name}.png"
        with Image.open(path) as im:
            aspect = im.height / im.width
        # Cap height so each figure fits on a page with its caption.
        doc.add_picture(str(path), width=Inches(min(6.6, MAX_FIG_HEIGHT / aspect)))
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(18)
        add_runs(p, caption)
        p.runs[0].font.size = Pt(9)

    out = SUPPLEMENT_DIR / "Supplementary_Material_revision.docx"
    doc.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
