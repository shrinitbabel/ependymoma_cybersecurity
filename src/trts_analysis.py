"""Train-on-real, test-on-synthetic (TRTS).

A random forest (same predictors and settings as the TSTR analysis) is trained on the 18 real
patients and evaluated on each synthetic dataset (10 runs per synthesizer) and on the
independent-marginals baseline. High TRTS AUC means the synthetic data follow the predictor-outcome
relationships the real-data model learned; note that synthesizers reproducing their training
records are favoured, because the model and the generator were trained on the same patients.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import BASELINE, MODELS, OUT, as_str, build_metadata, generate, independent_marginals, load_real, mean_ci
from holdout_analysis import ENDPOINTS, predict_endpoint


def main(seeds=10):
    real = load_real()
    metadata = build_metadata(real)
    rows = []
    for seed in range(seeds):
        datasets = {m: generate(real, metadata, m, seed) for m in MODELS}
        datasets[BASELINE] = independent_marginals(real, seed)
        for source, synth in datasets.items():
            for endpoint in ENDPOINTS:
                y = (as_str(synth[endpoint]) == "Y").astype(int).to_numpy()
                if len(set(y)) < 2:
                    continue
                p = predict_endpoint(real, synth, endpoint, seed)
                rows.append({"Source": source, "Seed": seed, "Endpoint": endpoint, "AUC": roc_auc_score(y, p),
                             "Synthetic_prevalence_pct": 100 * y.mean()})
    res = pd.DataFrame(rows)
    summary = []
    for (source, endpoint), g in res.groupby(["Source", "Endpoint"]):
        m, sd, lo, hi = mean_ci(g.AUC)
        summary.append({"Source": source, "Endpoint": endpoint, "AUC_mean": m, "SD": sd, "CI95_low": lo,
                        "CI95_high": hi, "Min": g.AUC.min(), "Max": g.AUC.max(),
                        "Synthetic_prevalence_pct": g.Synthetic_prevalence_pct.mean()})
    summary = pd.DataFrame(summary)
    res.to_csv(OUT / "trts_by_run.csv", index=False)
    summary.to_csv(OUT / "trts_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
