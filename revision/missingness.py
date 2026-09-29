"""Variable-level missingness before imputation (Reviewer 2 comment 4).

Source: the 'CranialEPEdits' sheet of the master database (same 18 patients as clean_data.csv).
Derived columns are traced to their source column (IHC markers -> 'molecular marker(s)',
Ki-67 index -> 'Ki-67').
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import COLORS, INK_2, OUT, ROOT, load_real, numeric_columns, save, style

DERIVED_FROM = {"TERT_marker": "molecular marker(s)", "Synaptophysin_marker": "molecular marker(s)",
                "GFAP_marker": "molecular marker(s)", "Olig2_marker": "molecular marker(s)",
                "Ki-67_index_7%": "Ki-67"}


def main():
    real = load_real()
    raw = pd.read_excel(ROOT / "Ependymoma Database.xlsx", sheet_name="CranialEPEdits")
    num = numeric_columns(real)
    rows = []
    for c in real.columns:
        src = DERIVED_FROM.get(c, c)
        # Blank cells and the literal "Missing" token used in the sheet both count as missing.
        missing = int((raw[src].isna() | raw[src].astype(str).str.strip().str.lower().eq("missing")).sum())
        rows.append({"Variable": c, "Source column": src, "Missing": missing, "Missing_pct": 100 * missing / len(raw),
                     "Type": "Numerical" if c in num else "Categorical",
                     "Imputation": ("—" if missing == 0 else
                                    "Iterative (multivariate)" if c in num else "Most frequent category")})
    table = pd.DataFrame(rows).sort_values(["Missing", "Variable"], ascending=[False, True])
    table.to_csv(OUT / "missingness_table.csv", index=False)

    style()
    shown = table[table.Missing > 0]
    fig, ax = plt.subplots(figsize=(5.5, max(1.6, 0.26 * len(shown) + 0.8)))
    ax.barh(range(len(shown)), shown.Missing_pct, color=COLORS["GaussianCopula"], height=0.62)
    for i, (pct, k) in enumerate(zip(shown.Missing_pct, shown.Missing)):
        ax.text(pct + 0.8, i, f"{k}/18", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(shown.Variable, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Missing before imputation, %")
    ax.grid(axis="y", visible=False)
    save(fig, "S_missingness")

    print(f"{(table.Missing > 0).sum()} of {len(table)} variables had missing values; "
          f"total missing cells {table.Missing.sum()} / {len(table) * len(raw)} "
          f"({100 * table.Missing.sum() / (len(table) * len(raw)):.1f}%)")
    print(shown[["Variable", "Missing", "Missing_pct", "Imputation"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
