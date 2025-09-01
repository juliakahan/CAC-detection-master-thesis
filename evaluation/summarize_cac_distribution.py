import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Summarize the distribution of total Agatston scores into standard risk categories.

- Reads a master CSV with (at least) a 'total_alt_agatston' column.
- Buckets values into: 0, 1–10, 11–100, 101–400, 401–1000, >1000.
- Saves a CSV table with counts and percentages, a LaTeX table snippet,
  and a bar chart PNG.

Edit AGATSTON_CSV / OUTDIR below or call main() with your own paths.
"""

AGATSTON_CSV = "/path/to/cac_master_agatston.csv"
OUTDIR       = "/path/to/summary_dist"
os.makedirs(OUTDIR, exist_ok=True)

# Categories
BINS   = [-0.5, 0.5, 10.5, 100.5, 400.5, 1000.5, np.inf]
LABELS = ["0", "1–10", "11–100", "101–400", "401–1000", ">1000"]

def main(ag_csv: str = None, outdir: str = None) -> None:
    """
    Create distribution summary for total Agatston categories.

    Args:
        ag_csv:  Path to the master Agatston CSV.
        outdir:  Output directory where artifacts will be written.
    """
    ag_csv = ag_csv or AGATSTON_CSV
    outdir = outdir or OUTDIR
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(ag_csv)

    # Column to analyze (change if you want another source, e.g., 'total_ref_agatston')
    col = "total_alt_agatston"
    if col not in df.columns:
        raise SystemExit(f"Column '{col}' not found in {ag_csv}")

    s = df[col].astype(float).dropna()
    cats = pd.cut(s, bins=BINS, labels=LABELS, right=True, ordered=True)
    counts = cats.value_counts().reindex(LABELS).fillna(0).astype(int)
    total_n = int(counts.sum())
    pct = (counts / total_n * 100.0).round(1)

    # CSV: counts + percentages
    dist = pd.DataFrame({"category": LABELS, "n": counts.values, "percent": pct.values})
    csv_path = os.path.join(outdir, "agatston_category_distribution_alt.csv")
    dist.to_csv(csv_path, index=False)

    # LaTeX table
    tex_path = os.path.join(outdir, "agatston_category_distribution_tabular_alt.tex")
    lines = []
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Category & $n$ & \\% \\\\")
    lines.append("\\midrule")
    for cat, n, p in zip(dist["category"], dist["n"], dist["percent"]):
        lines.append(f"{cat} & {n} & {p:.1f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    # Bar chart
    png_path = os.path.join(outdir, "agatston_category_distribution.png")
    plt.figure()
    plt.bar(dist["category"], dist["n"])
    plt.xlabel("Agatston risk category (Alternative method)")
    plt.ylabel("Number of patients")
    plt.title(f"Distribution of patients by CAC category (n={total_n})")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {tex_path}")
    print(f"[OK] Saved: {png_path}")

if __name__ == "__main__":
    main()
