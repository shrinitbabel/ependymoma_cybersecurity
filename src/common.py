"""Shared setup for all analyses: data loading, SDV metadata, cached synthesizer runs, plotting style.

All analyses read the real cohort from data/private/ (patient data; gitignored, never commit) and write tables to
results/tables/, figures to results/figures/, and the synthetic datasets they generate to
results/synthetic_runs/ (gitignored).
"""
import json
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sdv.metadata import Metadata
from sdv.single_table import (
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private"
RESULTS = ROOT / "results"
OUT = RESULTS / "tables"
SYNTH_DIR = RESULTS / "synthetic_runs"
FIG_DIR = RESULTS / "figures"
SUPPLEMENT_DIR = RESULTS / "supplement"

IDENTIFIERS = ["MRN", "Date of Sx", "Initials", "Date of Birth"]
SYNTHESIZERS = {
    "GaussianCopula": GaussianCopulaSynthesizer,
    "CopulaGAN": CopulaGANSynthesizer,
    "CTGAN": CTGANSynthesizer,
    "TVAE": TVAESynthesizer,
}
MODELS = list(SYNTHESIZERS)

# Settings of the original analysis, read from its fitted synthesizers (SDV 1.17.3): the revision reproduces them
# exactly. (With SDV's defaults, e.g. 300 epochs, CTGAN and CopulaGAN are badly undertrained on 18 rows.)
NUMERICAL_DISTRIBUTIONS = {
    "Age at Diagnosis": "norm", "Age at Sx": "norm", "BMI": "truncnorm", "Tumor Size 1 (cm)": "gamma",
    "Tumor Size 2 (cm)": "gamma", "Tumor Size 3 (cm)": "gamma", "Duration of symptoms preop (months)": "gamma",
    "EBL": "gamma", "Mitotic figures in 10 hpf": "gamma", "PFS (After Sx; Months)": "truncnorm",
    "Overall Survival (yrs)": "truncnorm", "Length of followup (months)": "truncnorm", "LOS (days)": "gamma",
}
ORIGINAL_PARAMS = {
    "GaussianCopula": dict(enforce_rounding=True, numerical_distributions=NUMERICAL_DISTRIBUTIONS,
                           default_distribution="truncnorm"),
    "CopulaGAN": dict(enforce_rounding=False, epochs=1000, numerical_distributions=NUMERICAL_DISTRIBUTIONS,
                      default_distribution="beta"),
    "CTGAN": dict(enforce_rounding=False, epochs=1000),
    "TVAE": dict(enforce_rounding=False, epochs=500),
}
BASELINE = "Independent baseline"
N_SYNTH = 1000

# Reference categorical palette (slots 1-4) for the models; gray for the baseline, ink for real data.
COLORS = {
    "GaussianCopula": "#2a78d6",
    "CopulaGAN": "#eb6834",
    "CTGAN": "#1baf7a",
    "TVAE": "#eda100",
    BASELINE: "#8a8983",
    "Real": "#0b0b0b",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_real():
    return pd.read_csv(PRIVATE / "clean_data.csv").drop(columns=IDENTIFIERS)


def metadata_columns(real):
    with open(ROOT / "data" / "metadata.json") as f:
        cols = json.load(f)["tables"]["table"]["columns"]
    cols = {c: spec for c, spec in cols.items() if c not in IDENTIFIERS}
    # Surgical Approach was flagged as PII in the original metadata, so SDV replaced it
    # with random strings instead of modeling it. It is a clinical variable.
    cols["Surgical Approach"] = {"sdtype": "categorical"}
    assert set(cols) == set(real.columns)
    return cols


def build_metadata(real):
    return Metadata.load_from_dict(
        {"tables": {"table": {"columns": metadata_columns(real)}}, "METADATA_SPEC_VERSION": "V1"})


def generate(train, metadata, model, seed, tag=""):
    """Fit one synthesizer on `train` and sample N_SYNTH rows; cached by model/seed/tag."""
    path = SYNTH_DIR / f"{model}{tag}_seed{seed}.csv"
    if path.exists():
        return pd.read_csv(path)
    seed_all(seed)
    synth = SYNTHESIZERS[model](metadata, **ORIGINAL_PARAMS[model])
    synth.fit(train)
    # SDV otherwise samples from a fixed internal random state, making every GaussianCopula run identical.
    synth._set_random_state(seed)
    df = synth.sample(N_SYNTH)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def independent_marginals(real, seed, n=N_SYNTH):
    """Baseline: every column resampled independently from the real data (keeps marginals, no relationships)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.choice(real[c].to_numpy(), n) for c in real.columns})


def as_str(series):
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def indicator(df, col, level):
    values = as_str(df[col])
    if level.startswith("!"):
        return values != level[1:]
    return values == level


def numeric_columns(real):
    return [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c]) and real[c].nunique() > 2]


def match_fraction(a, b, real):
    """Share of variables on which each row of `a` matches each row of `b`.

    Categorical: exact match. Numerical: within 5% of the real variable's range.
    Returns an (len(a), len(b)) array.
    """
    num = numeric_columns(real)
    cat = [c for c in real.columns if c not in num]
    rng = (real[num].max() - real[num].min()).replace(0, 1).to_numpy()
    ca = np.stack([as_str(a[c]).to_numpy() for c in cat], axis=1)
    cb = np.stack([as_str(b[c]).to_numpy() for c in cat], axis=1)
    na = a[num].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    nb = b[num].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    matches = (ca[:, None, :] == cb[None, :, :]).sum(2)
    matches += (np.abs(na[:, None, :] - nb[None, :, :]) / rng <= 0.05).sum(2)
    return matches / len(real.columns)


def compare_sources(df, value):
    """Kruskal-Wallis across the four models, then Holm-corrected pairwise Mann-Whitney (incl. vs baseline)."""
    groups = {s: g[value].to_numpy() for s, g in df.groupby("Source")}
    kw = kruskal(*[groups[m] for m in MODELS])
    pairs = [(a, b) for i, a in enumerate(MODELS) for b in MODELS[i + 1:]] + [(m, BASELINE) for m in MODELS]
    p = [mannwhitneyu(groups[a], groups[b]).pvalue for a, b in pairs]
    p_holm = multipletests(p, method="holm")[1]
    rows = [{"Metric": value, "Comparison": "Kruskal-Wallis (4 models)", "Statistic": kw.statistic, "p": kw.pvalue,
             "p_Holm": np.nan}]
    rows += [{"Metric": value, "Comparison": f"{a} vs {b}", "Statistic": np.nan, "p": pi, "p_Holm": ph}
             for (a, b), pi, ph in zip(pairs, p, p_holm)]
    return pd.DataFrame(rows)


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    m, sd = values.mean(), values.std(ddof=1)
    half = 1.96 * sd / np.sqrt(len(values))
    return m, sd, m - half, m + half


# Display names for figures (the data files keep the internal label BASELINE).
DISPLAY = {BASELINE: "Reshuffled\nreference"}


def tick_label(source):
    return {"GaussianCopula": "Gaussian-\nCopula", BASELINE: "Reshuffled\nreference",
            "Real (TRTR)": "Real\n(TRTR)"}.get(source, source)


def style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "Arial", "font.size": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "legend.frameon": False,
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{name}.tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
