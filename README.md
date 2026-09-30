# Synthetic data privacy and fidelity in a rare brain tumor cohort

Code and aggregate results for **"Cybersecurity and AI in Neuro-oncology: An Exploratory Evaluation of Synthetic
Data Generation in Rare Brain Tumors."**

We evaluate four tabular synthetic data generators from the
[Synthetic Data Vault](https://github.com/sdv-dev/SDV) (GaussianCopula, CopulaGAN, CTGAN, TVAE) on a
single-institution cohort of 18 patients with cranial ependymoma (61 clinical variables), across 10 independently
trained runs per model:

- **Fidelity:** SDMetrics quality report, Kolmogorov–Smirnov and Jensen–Shannon tests, UMAP.
- **Utility:** prespecified clinical associations, correlation structure, and nine relationship metrics.
- **Privacy:** attribute inference attacks compared with a chance-level baseline, membership inference with held-out patients, and record-level re-identification.
- **Anonymization:** feature-level anonymization of direct identifiers (AnonymizedFaker) compared with identifiers retained in training.

Every analysis is benchmarked against a *reshuffled reference*: each real variable resampled independently, which
preserves every variable's distribution but contains no relationships between variables.

## Repository structure

```
├── src/                      Analysis code (one script per analysis; shared setup in common.py)
├── scripts/run_all.sh        Reproduces every analysis, figure and the supplement
├── data/
│   ├── metadata.json         SDV metadata (column types)
│   ├── synthetic/            Publicly released synthetic datasets (direct identifiers anonymized)
│   └── private/              Real patient data and models fitted on it — NOT distributed (gitignored)
├── results/
│   ├── figures/              Manuscript and supplementary figures (PNG, 300 dpi)
│   └── tables/               Aggregate result tables (CSV)
├── notebooks/                Exploratory notebooks from the original submission (outputs stripped)
├── docs/                     TRIPOD-AI reporting checklist
└── legacy/original_submission/  Figures and quality reports from the original submission (superseded)
```

### Analysis scripts (`src/`)

| Script | Analysis | Manuscript |
|---|---|---|
| `common.py` | Data loading, synthesizer settings, seeded generation, reshuffled reference, statistics, plotting style | — |
| `missingness.py` | Variable-level missingness before imputation | Suppl. Table S2, Fig. S6 |
| `fidelity_analysis.py` | SDMetrics quality report across runs | Fig. 2, Suppl. Table S5 |
| `distribution_tests.py` | KS / Jensen–Shannon distributional tests | Suppl. Table S5, Fig. S7 |
| `utility_analysis.py` | Prespecified clinical associations; correlation structure | Suppl. Table S6, Fig. S8 |
| `relationship_metrics.py` | Nine relationship metrics; correlation heatmaps | Fig. 3, Suppl. Table S7, Figs. S9–S10 |
| `umap_figure.py` | UMAP embeddings of real and synthetic patients | Fig. 4 |
| `identifier_comparison.py` | Identifiers retained vs AnonymizedFaker: identifier reproduction, privacy, utility | Suppl. Table S8 |
| `privacy_analysis.py` | Attribute inference attacks; re-analysis of the original attack | Suppl. Tables S9–S10, S13 |
| `holdout_analysis.py` | Membership inference (6-fold cross-validation) | Suppl. Table S11, Fig. S13 |
| `reidentification_analysis.py` | Record copying, distance to closest record, quasi-identifier linkage | Suppl. Table S12, Fig. S14 |
| `attack_by_anonymization.py` | Per-attacker attacks on identifier-retained data | Fig. 6, Suppl. Fig. S11 |
| `identifier_heatmap.py`, `privacy_figure.py`, `descriptive_figures.py` | Combined privacy and descriptive figures | Figs. 5–6, Suppl. Figs. S3–S5, S11 |
| `build_supplement.py` | Assembles the supplementary document | Supplementary Material |
| `dp_analysis.py`, `trts_analysis.py` | Exploratory analyses (differential privacy; train-on-real/test-on-synthetic), not reported | — |

## Reproducing the analyses

```bash
pip install -r requirements.txt
bash scripts/run_all.sh
```

The full pipeline trains 168 synthesizers (roughly 3 hours on a CPU). Synthesizer settings reproduce those of the
original analysis (`ORIGINAL_PARAMS` in `src/common.py`): CTGAN and CopulaGAN are trained for 1,000 epochs and
TVAE for 500, because with 18 records each epoch is a single optimization step.

## Data availability

The real patient data cannot be shared. The pipeline expects `data/private/clean_data.csv` (the imputed analytic
dataset) and, for the missingness analysis, the source database; neither is distributed. The synthetic datasets in
`data/synthetic/` contain no real direct identifiers; medical record numbers, initials and dates are randomly
generated surrogate values.

This study was approved by the University of South Florida Institutional Review Board (STUDY000422).

## License

MIT — see [LICENSE](LICENSE).
