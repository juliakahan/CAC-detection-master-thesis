import os
import pandas as pd
import re
from typing import List, Optional, Dict, Tuple

# -------------------- I/O: robust loader -------------------- #
def _read_tabular(path: str) -> pd.DataFrame:
    """
    Read CSV/TSV/Excel with automatic delimiter and encoding detection.
    Apple Numbers (.numbers) is not supported — export to CSV/XLSX first.

    Args:
        path: Path to the tabular file (.csv/.tsv/.xlsx/.xls).

    Returns:
        pandas.DataFrame

    Raises:
        ValueError: If the input is an Apple Numbers file.
        Exception:  If all autodetected delimiter/encoding attempts fail.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".numbers":
        raise ValueError(
            f"'{path}' is an Apple Numbers file. Please export to CSV/XLSX (File → Export To…) and pass that file."
        )
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)

    # CSV/TSV: try several delimiters and encodings
    seps = [",", ";", "\t", None]  # None + engine='python' => sniff
    encs = ["utf-8", "utf-8-sig", "cp1250", "latin1"]
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                if sep is None:
                    return pd.read_csv(path, delimiter=None, engine="python", encoding=enc)
                return pd.read_csv(path, delimiter=sep, encoding=enc)
            except Exception as e:
                last_err = e
                continue
    raise last_err

# -------------------- utils -------------------- #
def _clean_id_series(s: pd.Series) -> pd.Series:
    """Remove 'CAC_' prefix and trim whitespace."""
    return s.astype(str).str.strip().str.replace(r"^CAC_", "", regex=True)

def _find_cols(df: pd.DataFrame, patterns: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    """
    Find columns matching lists of regex patterns for logical keys.

    Args:
        df: Input dataframe.
        patterns: Mapping: logical_name -> list of regex patterns to try.

    Returns:
        Dict[str, Optional[str]]: logical_name -> actual column name (or None if not found).
    """
    cols = list(df.columns)
    out = {}
    for key, pats in patterns.items():
        found = None
        for p in pats:
            r = re.compile(p, re.IGNORECASE)
            for c in cols:
                if r.fullmatch(c) or r.search(c):
                    found = c
                    break
            if found:
                break
        out[key] = found
    return out

def _pick_expert_id_column(df: pd.DataFrame) -> str:
    """
    Determine the patient ID column in the expert file, with fallbacks:
      1) 'patient_id'
      2) typical names: patient, case, id, series_id, study_id, exam_id
      3) NAME + SURNAME → construct patient_id
      4) fallback: first column

    Returns:
        str: 'patient_id' (ensured to exist in df).
    """
    cols = [c for c in df.columns if isinstance(c, str)]
    if "patient_id" in df.columns:
        return "patient_id"

    patterns = [
        r"^patient[_ ]?id$", r"^patient$", r"^case$", r"^id$",
        r"^series[_ ]?id$", r"^study[_ ]?id$", r"^exam[_ ]?id$"
    ]
    for p in patterns:
        for c in cols:
            if re.match(p, c, flags=re.IGNORECASE):
                df["patient_id"] = df[c].astype(str)
                return "patient_id"

    if {"NAME", "SURNAME"}.issubset(set(df.columns)):
        df["patient_id"] = (
            df["NAME"].astype(str).str.strip() + "_" + df["SURNAME"].astype(str).str.strip()
        )
        return "patient_id"

    first_col = df.columns[0]
    df["patient_id"] = df[first_col].astype(str)
    return "patient_id"

# -------------------- METRICS selection -------------------- #
def _parse_metrics(metrics: str) -> tuple[bool, bool]:
    """
    Parse metrics selection string.

    Args:
        metrics: 'agatston' | 'volume' | 'both'

    Returns:
        (include_agatston, include_volume)
    """
    m = (metrics or "both").strip().lower()
    if m not in {"agatston", "volume", "both"}:
        raise ValueError("metrics must be one of: 'agatston', 'volume', 'both'")
    return (m in {"agatston", "both"}, m in {"volume", "both"})

# -------------------- extract_mine: by metrics -------------------- #
def extract_mine(df: pd.DataFrame, source_tag: str, metrics: str = "both") -> pd.DataFrame:
    """
    Extract our (model) metrics from a table into a normalized schema.

    Args:
        df: DataFrame with our results (columns may vary; regex-matched).
        source_tag: Identifier to suffix columns with ('ref'/'alt', etc.).
        metrics: 'agatston' | 'volume' | 'both'

    Returns:
        DataFrame with columns:
          - patient_id
          - [lm_lad_{tag}_agatston, lcx_{tag}_agatston, rca_{tag}_agatston, total_{tag}_agatston]*
          - [lm_lad_{tag}_volume,   lcx_{tag}_volume,   rca_{tag}_volume,   total_{tag}_volume]*
    """
    include_agatston, include_volume = _parse_metrics(metrics)
    out = pd.DataFrame({"patient_id": df["patient_id"]})

    # --- Agatston ---
    if include_agatston:
        patt_ag = {
            "lm_lad": [r"^agatston[_ ]?lm[_ ]?lad$", r"\blm[_ ]?lad\b.*agatston", r"\blm[_ ]?lad\b$"],
            "lcx":    [r"^agatston[_ ]?lcx$",       r"\blcx\b.*agatston",        r"\blcx\b$"],
            "rca":    [r"^agatston[_ ]?rca$",       r"\brca\b.*agatston",        r"\brca\b$"],
            "total":  [r"^agatston[_ ]?score$",     r"^total[_ ]?agatston$",     r"\bagatston[_ ]?total\b"],
        }
        m = _find_cols(df, patt_ag)

        total_calc = None
        if m["lm_lad"] and m["lcx"] and m["rca"]:
            total_calc = df[m["lm_lad"]] + df[m["lcx"]] + df[m["rca"]]

        if m["lm_lad"] is not None: out[f"lm_lad_{source_tag}_agatston"] = df[m["lm_lad"]].values
        if m["lcx"]    is not None: out[f"lcx_{source_tag}_agatston"]     = df[m["lcx"]].values
        if m["rca"]    is not None: out[f"rca_{source_tag}_agatston"]     = df[m["rca"]].values
        if total_calc is not None:
            out[f"total_{source_tag}_agatston"] = total_calc.values
        elif m["total"] is not None:
            out[f"total_{source_tag}_agatston"] = df[m["total"]].values

    # --- Volume (your naming: vol_mm3_* + volume_total) ---
    if include_volume:
        patt_vol = {
            "lm_lad_v": [r"^vol[_ ]?mm3[_ ]?lm[_ ]?lad$", r"\blm[_ ]?lad\b.*vol"],
            "lcx_v":    [r"^vol[_ ]?mm3[_ ]?lcx$",       r"\blcx\b.*vol"],
            "rca_v":    [r"^vol[_ ]?mm3[_ ]?rca$",       r"\brca\b.*vol"],
            "total_v":  [r"^volume[_ ]?total$", r"^vol[_ ]?total$", r"^total[_ ]?volume$"],
        }
        mv = _find_cols(df, patt_vol)

        if mv["lm_lad_v"] is not None: out[f"lm_lad_{source_tag}_volume"] = df[mv["lm_lad_v"]].values
        if mv["lcx_v"]    is not None: out[f"lcx_{source_tag}_volume"]     = df[mv["lcx_v"]].values
        if mv["rca_v"]    is not None: out[f"rca_{source_tag}_volume"]     = df[mv["rca_v"]].values

        if all(k in out.columns for k in [f"lm_lad_{source_tag}_volume", f"lcx_{source_tag}_volume", f"rca_{source_tag}_volume"]):
            out[f"total_{source_tag}_volume"] = (
                out[f"lm_lad_{source_tag}_volume"] + out[f"lcx_{source_tag}_volume"] + out[f"rca_{source_tag}_volume"]
            )
        elif mv["total_v"] is not None:
            out[f"total_{source_tag}_volume"] = df[mv["total_v"]].values

    return out


# -------------------- main function -------------------- #
def build_master_dataset(
    expert_csv: str,
    ref_csv: str,
    alt_csv: str,
    out_csv: str = None,
    id_map: Optional[Dict[str, str]] = None,
    metrics: str = "both"
) -> Tuple[str, pd.DataFrame]:
    """
    Build a unified master dataset combining expert readings and our model outputs.

    Steps:
      1) Load the three input tables (robustly).
      2) Normalize patient IDs (strip 'CAC_' prefix, trim, etc.).
      3) From expert: extract Agatston (corrected/synthetic), adjust totals to "total minus LM".
      4) From ref/alt: extract selected metrics (Agatston and/or volume).
      5) Merge on patient_id (outer join).
      6) Optionally restrict columns to requested metrics.
      7) Optional save to CSV.

    Args:
        expert_csv: Path to expert metrics table.
        ref_csv:    Path to our "ref" pipeline results.
        alt_csv:    Path to our "alt" pipeline results.
        out_csv:    Optional output CSV path for the master table.
        id_map:     Optional dict for ID remapping (currently unused; reserved).
        metrics:    'agatston' | 'volume' | 'both'

    Returns:
        (out_csv_path, master_dataframe)
    """
    include_agatston, include_volume = _parse_metrics(metrics)

    # 1) load
    expert = _read_tabular(expert_csv)
    ref    = _read_tabular(ref_csv)
    alt    = _read_tabular(alt_csv)

    # 2) IDs
    id_col_exp = _pick_expert_id_column(expert)
    expert["patient_id"] = _clean_id_series(expert["patient_id"])
    for df in (ref, alt):
        case_col = "case" if "case" in df.columns else ("patient_id" if "patient_id" in df.columns else None)
        if case_col is None:
            raise ValueError("In ref/alt files, neither 'case' nor 'patient_id' column was found.")
        df["patient_id"] = _clean_id_series(df[case_col])

    # 3) EXPERT → only if we compute Agatston
    master = pd.DataFrame({"patient_id": expert["patient_id"]})
    if include_agatston:
        patt_expert = {
            "lm_corr":     [r"^LM[_ ]?corr$", r"\blm\b.*corr"],
            "lad_corr":    [r"^LAD[_ ]?corr$", r"\blad\b.*corr"],
            "lcx_corr":    [r"^[LC]x[_ ]?corr$", r"\blcx\b.*corr", r"\bcx\b.*corr"],
            "rca_corr":    [r"^RCA[_ ]?corr$", r"\brca\b.*corr"],
            "total_corr":  [r"^Total[_ ]?corr$", r"\btotal\b.*corr"],
            "lm_syn":      [r"^LM[_ ]?syn$", r"\blm\b.*syn"],
            "lad_syn":     [r"^LAD[_ ]?syn$", r"\blad\b.*syn"],
            "lcx_syn":     [r"^[LC]x[_ ]?syn$", r"\blcx\b.*syn", r"\bcx\b.*syn"],
            "rca_syn":     [r"^RCA[_ ]?syn$", r"\brca\b.*syn"],
            "total_syn":   [r"^Total[_ ]?syn$", r"\btotal\b.*syn"],
        }
        exp_map = _find_cols(expert, patt_expert)
        for k, new_name in [
            ("lm_corr","lm_corr_agatston"), ("lad_corr","lad_corr_agatston"),
            ("lcx_corr","lcx_corr_agatston"), ("rca_corr","rca_corr_agatston"),
            ("total_corr","total_corr_agatston"),
            ("lm_syn","lm_syn_agatston"), ("lad_syn","lad_syn_agatston"),
            ("lcx_syn","lcx_syn_agatston"), ("rca_syn","rca_syn_agatston"),
            ("total_syn","total_syn_agatston"),
        ]:
            col = exp_map.get(k)
            if col is not None:
                master[new_name] = expert[col].values

        # 3a) totals minus LM (use original totals before any further adjustments)
        if all(c in master.columns for c in ["total_corr_agatston","lm_corr_agatston"]):
            master["total_corr_agatston_minus_lm"] = master["total_corr_agatston"] - master["lm_corr_agatston"]
        if all(c in master.columns for c in ["total_syn_agatston","lm_syn_agatston"]):
            master["total_syn_agatston_minus_lm"] = master["total_syn_agatston"] - master["lm_syn_agatston"]

        # 3b) adopt "total = total_minus_lm"
        # (skip totals that include mediastinum and avoid computing a separate "pure" total)
        if "total_corr_agatston_minus_lm" in master.columns:
            master["total_corr_agatston"] = master["total_corr_agatston_minus_lm"]
        if "total_syn_agatston_minus_lm" in master.columns:
            master["total_syn_agatston"] = master["total_syn_agatston_minus_lm"]

        # 3c) cleanup: drop helper cols and anything mediastinum-related
        cols_to_drop = [c for c in master.columns if c.endswith("_agatston_minus_lm")] + \
                       [c for c in master.columns if c.startswith("mediastinum_")]
        if cols_to_drop:
            master.drop(columns=[c for c in cols_to_drop if c in master.columns], inplace=True)

    # 4) our files: ref/alt — according to selected metrics
    mine_ref = extract_mine(ref, "ref", metrics=metrics)
    mine_alt = extract_mine(alt, "alt", metrics=metrics)

    # 5) merge
    master = (master
              .merge(mine_ref, on="patient_id", how="outer")
              .merge(mine_alt, on="patient_id", how="outer"))

    # 6) hard cut to requested metric set
    if metrics == "volume":
        keep = ["patient_id"] + [c for c in master.columns if c.endswith("_volume")]
        master = master[keep]
    elif metrics == "agatston":
        keep = ["patient_id"] + [c for c in master.columns if c.endswith("_agatston")]
        master = master[keep]
    # if "both": keep everything

    # 7) save
    if out_csv:
        master.to_csv(out_csv, index=False)
    return out_csv, master


if __name__ == "__main__":
    # Example local run — replace with your paths
    out_path, master_df = build_master_dataset(
        expert_csv="/path/to/expert_cac_metrics.csv",
        ref_csv="/path/to/cac_metrics_ref.csv",
        alt_csv="/path/to/cac_metrics_alt.csv",
        out_csv="/path/to/cac_master_agatston.csv",
        metrics="agatston"   # 'agatston' | 'volume' | 'both'
    )
    print("Saved master to:", out_path)
    print(master_df.head())
