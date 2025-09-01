# save as: eval_cac_numeric.py
import argparse, os, math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# ---------- helpers ----------

ARTERIES = ["lm_lad", "lcx", "rca", "total"]
SOURCES  = ["corr", "syn", "ref", "alt"]

def _exists(df: pd.DataFrame, col: str) -> bool:
    """Return True if column exists and has at least one non-NaN value."""
    return (col in df.columns) and (df[col].notna().sum() > 0)

def _pairwise_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute pairwise numeric metrics between two series:
    Pearson/Spearman, MAE, RMSE, mean bias, and sample size n.
    """
    s = pd.concat([y_true, y_pred], axis=1).dropna()
    if s.shape[0] < 3:
        return {"n": s.shape[0], "pearson": np.nan, "spearman": np.nan,
                "mae": np.nan, "rmse": np.nan, "bias": np.nan}
    t = s.iloc[:,0].astype(float)
    p = s.iloc[:,1].astype(float)
    pearson  = t.corr(p, method="pearson")
    spearman = t.corr(p, method="spearman")
    diff = p - t
    mae  = (diff.abs()).mean()
    rmse = math.sqrt((diff**2).mean())
    bias = diff.mean()
    return {"n": int(s.shape[0]), "pearson": float(pearson), "spearman": float(spearman),
            "mae": float(mae), "rmse": float(rmse), "bias": float(bias)}

def _bland_altman_data(y_true: pd.Series, y_pred: pd.Series) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Prepare arrays and LoA for a Bland–Altman plot."""
    s = pd.concat([y_true, y_pred], axis=1).dropna()
    a = s.mean(axis=1).values
    d = (s.iloc[:,1] - s.iloc[:,0]).values
    m = d.mean()
    sd = d.std(ddof=1)
    loa_low, loa_high = m - 1.96*sd, m + 1.96*sd
    return a, d, loa_low, loa_high

def _ensure_dir(p: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(p, exist_ok=True)

def _scan_pairs(df: pd.DataFrame, metric: str) -> List[Tuple[str,str,str,str]]:
    """
    Return a list of comparison pairs:
      (artery, comparator, col_true, col_pred)
    comparator ∈ {'ref_vs_corr','alt_vs_corr','ref_vs_syn','alt_vs_syn','alt_vs_ref'}
    """
    pairs = []
    for art in ARTERIES:
        cols = {src: f"{art}_{src}_{metric}" for src in SOURCES}
        # ref/alt vs corr
        if _exists(df, cols["corr"]):
            if _exists(df, cols["ref"]): pairs.append((art, "ref_vs_corr", cols["corr"], cols["ref"]))
            if _exists(df, cols["alt"]): pairs.append((art, "alt_vs_corr", cols["corr"], cols["alt"]))
        # ref/alt vs syn
        if _exists(df, cols["syn"]):
            if _exists(df, cols["ref"]): pairs.append((art, "ref_vs_syn", cols["syn"], cols["ref"]))
            if _exists(df, cols["alt"]): pairs.append((art, "alt_vs_syn", cols["syn"], cols["alt"]))
        # alt vs ref (mask source)
        if _exists(df, cols["alt"]) and _exists(df, cols["ref"]):
            pairs.append((art, "alt_vs_ref", cols["ref"], cols["alt"]))
    return pairs

def _plot_scatter(y_true: pd.Series, y_pred: pd.Series, title: str, out_png: str) -> None:
    """Save a scatter plot with y=x reference line."""
    s = pd.concat([y_true, y_pred], axis=1).dropna()
    if s.empty: return
    x = s.iloc[:,0].astype(float).values
    y = s.iloc[:,1].astype(float).values
    plt.figure()
    plt.scatter(x, y, alpha=0.7)
    lim = [0, max(x.max(), y.max())*1.05 if len(x) else 1]
    plt.plot(lim, lim)  # y=x
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Reference")
    plt.ylabel("Prediction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _plot_ba(y_true: pd.Series, y_pred: pd.Series, title: str, out_png: str) -> None:
    """Save a Bland–Altman plot with mean difference and 95% limits of agreement."""
    s = pd.concat([y_true, y_pred], axis=1).dropna()
    if s.empty: return
    a, d, loa_low, loa_high = _bland_altman_data(s.iloc[:,0], s.iloc[:,1])
    plt.figure()
    plt.scatter(a, d, alpha=0.7)
    plt.axhline(d.mean(), linestyle="--")
    plt.axhline(loa_low, linestyle=":")
    plt.axhline(loa_high, linestyle=":")
    plt.xlabel("Mean of methods")
    plt.ylabel("Difference (Pred - Ref)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ---------- main ----------
def evaluate(master_csv: str, metric: str, outdir: str, make_plots: bool=True) -> pd.DataFrame:
    """
    Evaluate numeric agreement for a given metric across arteries and method pairs.

    Expects a master CSV with columns like:
      lm_lad_corr_agatston, lm_lad_ref_agatston, lm_lad_alt_agatston, ...
      and similarly for _volume and for other arteries (lcx, rca, total).

    Args:
        master_csv: Path to the master table.
        metric: One of {'agatston', 'volume'}.
        outdir: Output directory for CSV and plots.
        make_plots: If True, produce scatter + Bland–Altman plots for n>=3.

    Returns:
        pandas.DataFrame with per-comparison metrics; also saved to CSV.
    """
    df = pd.read_csv(master_csv)
    _ensure_dir(outdir)

    results = []
    for art, cmp_name, col_true, col_pred in _scan_pairs(df, metric):
        met = _pairwise_metrics(df[col_true], df[col_pred])
        row = {"metric": metric, "artery": art, "comparison": cmp_name,
               "y_true": col_true, "y_pred": col_pred, **met}
        results.append(row)

        if make_plots and met["n"] and met["n"] >= 3:
            safe_name = f"{metric}_{art}_{cmp_name}".replace("/","-")
            _plot_scatter(df[col_true], df[col_pred],
                          f"{metric.upper()} {art} {cmp_name} (n={met['n']})",
                          os.path.join(outdir, f"{safe_name}_scatter.png"))
            _plot_ba(df[col_true], df[col_pred],
                     f"{metric.upper()} {art} {cmp_name} (Bland–Altman)",
                     os.path.join(outdir, f"{safe_name}_blandaltman.png"))

    res_df = pd.DataFrame(results).sort_values(["metric","artery","comparison"])
    res_df.to_csv(os.path.join(outdir, f"numeric_eval_{metric}.csv"), index=False)
    return res_df

if __name__ == "__main__":
    # Example local run — replace with your paths
    AGATSTON_CSV = "/path/to/cac_master_agatston.csv"
    VOLUME_CSV   = "/path/to/cac_master_volume.csv"
    OUTDIR       = "/path/to/numeric_eval"

    MAKE_PLOTS = True

    if AGATSTON_CSV and os.path.exists(AGATSTON_CSV):
        evaluate(AGATSTON_CSV, metric="agatston",
                 outdir=os.path.join(OUTDIR, "agatston"),
                 make_plots=MAKE_PLOTS)
    if VOLUME_CSV and os.path.exists(VOLUME_CSV):
        evaluate(VOLUME_CSV, metric="volume",
                 outdir=os.path.join(OUTDIR, "volume"),
                 make_plots=MAKE_PLOTS)
