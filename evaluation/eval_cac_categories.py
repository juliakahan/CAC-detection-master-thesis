import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import matplotlib.pyplot as plt

"""
Evaluate categorical agreement for total Agatston score.

- Uses standard risk categories: 0, 1–10, 11–100, 101–400, 401–1000, >1000.
- Produces confusion matrices (CSV + heatmap PNG), overall accuracy,
  quadratic-weighted Cohen’s kappa, and per-category recall/specificity (one-vs-rest).
- Exports LaTeX tables for inclusion in a thesis/manuscript.
"""

AGATSTON_CSV = os.environ.get("AGATSTON_CSV", "/path/to/cac_master_agatston.csv")
OUTDIR       = os.environ.get("OUTDIR", "/path/to/cat_eval")

# ---------- Category definition ----------
# Disjoint bins for total Agatston score:
# 0, 1–10, 11–100, 101–400, 401–1000, >1000
BINS  = [-0.5, 0.5, 10.5, 100.5, 400.5, 1000.5, np.inf]
LABEL = ["0", "1-10", "11-100", "101-400", "401-1000", ">1000"]


def agat_category(x: pd.Series) -> pd.Categorical:
    """Bucket a numeric Agatston series into the predefined risk categories."""
    return pd.cut(x.astype(float), bins=BINS, labels=LABEL, right=True, ordered=True)


def confusion_and_stats(y_true_cat: pd.Series, y_pred_cat: pd.Series) -> Dict[str, float]:
    """
    Build the confusion matrix (True x Pred) and compute accuracy + quadratic-weighted kappa.
    Returns a dict with:
      - 'confusion' (DataFrame), 'n', 'accuracy', 'kappa_quadratic'
    """
    idx = pd.Index(LABEL, name="True")
    cols = pd.Index(LABEL, name="Pred")
    cm = pd.crosstab(y_true_cat, y_pred_cat).reindex(index=idx, columns=cols).fillna(0).astype(int)

    n = cm.values.sum()
    acc = np.trace(cm.values) / n if n else np.nan

    # Quadratic-weighted Cohen's kappa
    r = len(LABEL)
    rows = cm.sum(axis=1).values.reshape(-1, 1)
    cols_ = cm.sum(axis=0).values.reshape(1, -1)

    W = np.zeros((r, r), dtype=float)
    for i in range(r):
        for j in range(r):
            W[i, j] = ((i - j) ** 2) / ((r - 1) ** 2)

    O = cm.values / n if n else np.zeros_like(cm.values, dtype=float)
    E = (rows @ cols_) / (n * n) if n else np.zeros_like(cm.values, dtype=float)
    kw = 1 - (W * O).sum() / (W * E).sum() if n and (W * E).sum() > 0 else np.nan

    return {"confusion": cm, "n": int(n), "accuracy": float(acc), "kappa_quadratic": float(kw)}


def binary_metrics(y_true: pd.Series, y_pred: pd.Series, thr: float) -> Dict[str, float]:
    """
    Binary classification relative to a threshold 'thr'.
    - For thr == 0: use '>' (presence of any CAC).
    - For thr > 0:  use '>=' (e.g., clinical cutoffs 100, 400, 1000).
    """
    yt = y_true.astype(float)
    yp = y_pred.astype(float)
    if thr == 0:
        t = (yt > 0).astype(int)
        p = (yp > 0).astype(int)
    else:
        t = (yt >= thr).astype(int)
        p = (yp >= thr).astype(int)

    tp = int(((t == 1) & (p == 1)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan
    return {"thr": thr, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "sens": sens, "spec": spec, "acc": acc}


def heatmap(cm: pd.DataFrame, title: str, out_png: str) -> None:
    """Render and save a confusion matrix heatmap with counts and percentages."""
    plt.figure()
    plt.imshow(cm.values, interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    plt.yticks(range(len(cm.index)), cm.index)
    total = cm.values.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            pct = 100 * v / total if total else 0
            plt.text(j, i, f"{v}\n({pct:.0f}%)", ha="center", va="center", fontsize=8)
    plt.title(title)
    plt.xlabel("Predicted category")
    plt.ylabel("True category")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def to_latex_cm(cm: pd.DataFrame, caption: str, label: str) -> str:
    """Return a LaTeX table string for a confusion matrix DataFrame."""
    return cm.to_latex(na_rep="0", caption=caption, label=label, index=True, escape=False)


def main(ag_csv: str = None, outdir: str = None) -> None:
    """
    Run category agreement evaluation for total Agatston score.

    Args:
        ag_csv: Path to master Agatston CSV (with columns like total_corr_agatston, total_alt_agatston, etc.).
        outdir: Output directory for CSVs, PNGs and LaTeX snippets.
    """
    ag_csv = ag_csv or AGATSTON_CSV
    outdir = outdir or OUTDIR
    os.makedirs(outdir, exist_ok=True)

    # Category labels in the exact order used by agat_category(...)
    CAT_LABELS = ["0", "1-10", "11-100", "101-400", "401-1000", ">1000"]

    def safe_key(lbl: str) -> str:
        """Make a filesystem/CSV-safe key from a human-readable label."""
        return (lbl.replace(">", "gt")
                   .replace("–", "-")
                   .replace("—", "-")
                   .replace(" ", "")
                   .replace("-", "_"))

    def esc_latex(lbl: str) -> str:
        """Escape '>' for LaTeX."""
        return lbl.replace(">", "$>$")

    df = pd.read_csv(ag_csv)

    # Pairs to compare (columns must exist in the master file)
    pairs = [
        ("Alt vs SyngoVia (corr)", "total_corr_agatston", "total_alt_agatston"),
        ("Ref vs SyngoVia (corr)", "total_corr_agatston", "total_ref_agatston"),
        ("Alt vs SyngoVia (raw) ", "total_syn_agatston",  "total_alt_agatston"),
        ("Ref vs SyngoVia (raw) ", "total_syn_agatston",  "total_ref_agatston"),
        ("Alt vs Ref            ", "total_ref_agatston",  "total_alt_agatston"),
    ]

    rows_sum = []
    latex_snippets = []

    for title, ytrue_col, ypred_col in pairs:
        if ytrue_col not in df.columns or ypred_col not in df.columns:
            print(f"[WARN] skipping '{title}' – missing columns: {ytrue_col} or {ypred_col}")
            continue

        mask = df[ytrue_col].notna() & df[ypred_col].notna()
        y_true = df.loc[mask, ytrue_col]
        y_pred = df.loc[mask, ypred_col]

        # Categorize and compute confusion + stats
        cat_true = agat_category(y_true)
        cat_pred = agat_category(y_pred)
        stats = confusion_and_stats(cat_true, cat_pred)
        cm = stats["confusion"]   # expected 6x6
        n  = stats["n"]

        # Base filename for outputs
        fname = (title.lower()
                      .replace(" ", "_")
                      .replace("(", "").replace(")", "")
                      .replace("/", "_"))

        # Heatmap
        out_png = os.path.join(outdir, f"cm_{fname}.png")
        heatmap(cm, f"{title} (n={n})", out_png)

        # Normalize/clean to int matrix for CSV export
        cm_np = np.array(cm, dtype=float)
        cm_np = np.nan_to_num(cm_np, nan=0.0)
        cm_np[cm_np < 0] = 0
        cm_int = cm_np.round().astype(int)

        # Save CSV with explicit labels
        cm_df = pd.DataFrame(cm_int, index=CAT_LABELS, columns=CAT_LABELS)
        cm_csv = os.path.join(outdir, f"cm_{fname}.csv")
        cm_df.to_csv(cm_csv, index=True)
        print(f"[OK] Saved: {cm_csv}")

        # Per-category recall/specificity (one-vs-rest)
        total = cm_np.sum()
        per_rec, per_spec = [], []
        for i in range(len(CAT_LABELS)):
            TP = cm_np[i, i]
            FN = cm_np[i, :].sum() - TP
            FP = cm_np[:, i].sum() - TP
            TN = total - TP - FN - FP
            rec = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
            per_rec.append(rec)
            per_spec.append(spec)

        row_sum = {
            "comparison": title,
            "n": n,
            "accuracy_cat": stats.get("accuracy", np.nan),
            "kappa_w": stats.get("kappa_quadratic", np.nan),
        }
        for lbl, rec, spec in zip(CAT_LABELS, per_rec, per_spec):
            key = safe_key(lbl)  # e.g., 1_10, 401_1000, gt1000
            row_sum[f"rec_{key}"] = rec
            row_sum[f"spec_{key}"] = spec
        rows_sum.append(row_sum)

        # LaTeX confusion matrix (booktabs)
        ncols = len(CAT_LABELS)
        cols = 'l' + 'r' * ncols
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Confusion matrix of Agatston risk categories (0, 1--10, 11--100, 101--400, 401--1000, >1000): {title} (n={n}).}}")
        lines.append(f"\\label{{tab:cm_{fname}}}")
        lines.append(f"\\begin{{tabular}}{{{cols}}}")
        lines.append("\\toprule")
        lines.append(f" & \\multicolumn{{{ncols}}}{{c}}{{Predicted}} \\\\")
        lines.append(f"\\cmidrule(lr){{2-{ncols+1}}}")
        lines.append("True & " + " & ".join(esc_latex(x) for x in CAT_LABELS) + " \\\\")
        lines.append("\\midrule")
        for i in range(cm_int.shape[0]):
            row_vals = " & ".join(str(v) for v in cm_int[i, :].tolist())
            lines.append(f"{esc_latex(CAT_LABELS[i])} & {row_vals} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        latex_snippets.append("\n".join(lines))

    # Summary CSV
    df_sum = pd.DataFrame(rows_sum)
    df_sum.to_csv(os.path.join(outdir, "category_eval_summary.csv"), index=False)

    # All confusion matrices to a single .tex
    with open(os.path.join(outdir, "confusion_matrices.tex"), "w") as f:
        for snip in latex_snippets:
            f.write(snip + "\n\n")

    # Per-category metrics LaTeX table
    def fmt(x):
        return "---" if pd.isna(x) else f"{x:.3f}"

    rec_heads  = [f"Rec {lbl.replace('>', '$>$')}"  for lbl in CAT_LABELS]
    spec_heads = [f"Spec {lbl.replace('>', '$>$')}" for lbl in CAT_LABELS]
    num_cols = 2 + 2 + len(rec_heads) + len(spec_heads)  # comp,n,acc,kappa + 6 + 6

    lines = []
    lines.append("\\begin{table}[t]\n\\centering")
    lines.append("\\caption{Risk category agreement for total Agatston: overall accuracy, quadratic-weighted $\\kappa$, and per-category recall/specificity (one-vs-rest).}")
    lines.append("\\label{tab:agatston_cat_perclass_summary}")
    lines.append("\\begin{tabular}{l" + "c" * (num_cols) + "}")
    lines.append("\\toprule")
    header = ["Comparison", "$n$", "Acc.", "$\\kappa_w$"] + rec_heads + spec_heads
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for _, r in df_sum.iterrows():
        row_vals = [
            r['comparison'],
            str(int(r['n'])),
            fmt(r['accuracy_cat']),
            fmt(r['kappa_w'])
        ]
        for lbl in CAT_LABELS:
            key = safe_key(lbl)
            row_vals.append(fmt(r.get(f"rec_{key}", np.nan)))
        for lbl in CAT_LABELS:
            key = safe_key(lbl)
            row_vals.append(fmt(r.get(f"spec_{key}", np.nan)))
        lines.append(" & ".join(row_vals) + " \\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    with open(os.path.join(outdir, "category_eval_perclass_summary.tex"), "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] Saved: {os.path.join(outdir, 'category_eval_summary.csv')}")
    print(f"[OK] Saved: {os.path.join(outdir, 'confusion_matrices.tex')}")
    print(f"[OK] Saved: {os.path.join(outdir, 'category_eval_perclass_summary.tex')}")
    print(f"[OK] Confusion heatmaps in: {outdir}")


if __name__ == "__main__":
    # Example run (override defaults by setting env vars or editing the paths above):
    #   AGATSTON_CSV=/path/to/cac_master_agatston.csv OUTDIR=/path/to/cat_eval python eval_cac_categories.py
    main()
