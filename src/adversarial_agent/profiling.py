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
