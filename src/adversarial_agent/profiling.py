"""Profiling utilities for tabular datasets."""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np

def profile_table(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary metadata, column types, and missingness."""
    col_profiles = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        missing_rate = float(missing / max(1, len(df)))
        nunique = int(s.nunique(dropna=True))
        
        if pd.api.types.is_numeric_dtype(s):
            inferred = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            inferred = "datetime"
        else:
            inferred = "categorical" if nunique <= 20 else "text"
            
        col_profiles.append({
            "name": str(col),
            "inferred_type": inferred,
            "missing_count": missing,
            "missing_rate": missing_rate,
            "unique_count": nunique,
        })
        
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": col_profiles,
    }


def benjamini_hochberg_fdr(p_values: List[float]) -> List[float]:
    """Computes Benjamini-Hochberg False Discovery Rate (FDR) adjusted q-values.
    
    Guarantees that across m tested hypotheses, FDR <= q.
    q_(i) = min_{k >= i} (m / k * p_(k)), capped at 1.0.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    q_vals = [0.0] * m
    min_q = 1.0
    for rank_minus_1, (orig_idx, p) in reversed(list(enumerate(indexed))):
        rank = rank_minus_1 + 1
        q = min(1.0, (m / rank) * p)
        min_q = min(min_q, q)
        q_vals[orig_idx] = float(min_q)
    return q_vals

