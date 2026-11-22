"""Explorer Agent: Autonomous hypothesis and pattern discovery from tabular data."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class Claim:
    """An empirical hypothesis formulated by the Explorer Agent."""
    claim_id: str
    claim_type: str  # "correlation" | "group_difference" | "proportion_difference"
    source_variable: str
    target_variable: str
    statistic_name: str
    statistic_value: float
    p_value: float
    sample_size: int
    direction: str
    description: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


class ExplorerAgent:
    """Discovers bivariate relationships, group disparities, and potential predictive signals."""

    def __init__(self, significance_level: float = 0.05, min_sample_size: int = 15):
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size

    def explore(self, df: pd.DataFrame, target: Optional[str] = None) -> List[Claim]:
        """Explore tabular data and extract candidate empirical claims."""
        claims: List[Claim] = []
        counter = 1

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not self._is_identifier(c, df)]
        categorical_cols = [c for c in df.select_dtypes(include=["object", "string", "category", "bool"]).columns if not self._is_identifier(c, df)]

        # Target-driven exploration
        if target and target in df.columns:
            if target in numeric_cols:
                for col in numeric_cols:
                    if col == target:
                        continue
                    c = self._evaluate_correlation(df, col, target, f"CLM_{counter:03d}")
                    if c:
                        claims.append(c)
                        counter += 1
                for col in categorical_cols:
                    c = self._evaluate_group_difference(df, col, target, f"CLM_{counter:03d}")
                    if c:
                        claims.append(c)
                        counter += 1

            elif target in categorical_cols or (target in numeric_cols and df[target].nunique() <= 3):
                for col in numeric_cols:
                    if col == target:
                        continue
                    c = self._evaluate_binary_target_vs_numeric(df, col, target, f"CLM_{counter:03d}")
                    if c:
                        claims.append(c)
                        counter += 1
                for col in categorical_cols:
                    if col == target:
                        continue
                    c = self._evaluate_proportion_difference(df, col, target, f"CLM_{counter:03d}")
                    if c:
                        claims.append(c)
                        counter += 1

        # General pair exploration
        if len(claims) < 3:
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    c1, c2 = numeric_cols[i], numeric_cols[j]
                    if target and (c1 == target or c2 == target):
                        continue
                    c = self._evaluate_correlation(df, c1, c2, f"CLM_{counter:03d}")
                    if c and abs(c.statistic_value) >= 0.20:
                        claims.append(c)
                        counter += 1

            for cat in categorical_cols:
                for num in numeric_cols:
                    if target and (cat == target or num == target):
                        continue
                    c = self._evaluate_group_difference(df, cat, num, f"CLM_{counter:03d}")
                    if c and c.p_value < 0.05:
                        claims.append(c)
                        counter += 1

        claims.sort(key=lambda x: (x.p_value, -abs(x.statistic_value)))
        return claims

    def _is_identifier(self, col: str, df: pd.DataFrame) -> bool:
        low = str(col).lower()
        if "id" in low or low.endswith("_id") or low.startswith("id_"):
            if df[col].nunique() / max(1, len(df)) > 0.80:
                return True
        return False

    def _evaluate_correlation(self, df: pd.DataFrame, x_col: str, y_col: str, cid: str) -> Optional[Claim]:
        sub = df[[x_col, y_col]].dropna()
        if len(sub) < self.min_sample_size:
            return None
        r, p = stats.pearsonr(sub[x_col], sub[y_col])
        if np.isnan(r):
            return None
        direction = "positive" if r > 0 else "negative"
        desc = f"Observed global {direction} correlation between '{x_col}' and '{y_col}' (r = {r:.4f}, p = {p:.4e}, N = {len(sub)})."
        return Claim(
            claim_id=cid,
            claim_type="correlation",
            source_variable=x_col,
            target_variable=y_col,
            statistic_name="Pearson_r",
            statistic_value=float(r),
            p_value=float(p),
            sample_size=len(sub),
            direction=direction,
            description=desc,
            metadata={"mean_x": float(sub[x_col].mean()), "mean_y": float(sub[y_col].mean())},
        )

    def _evaluate_binary_target_vs_numeric(self, df: pd.DataFrame, num_col: str, target: str, cid: str) -> Optional[Claim]:
        sub = df[[num_col, target]].dropna()
        if len(sub) < self.min_sample_size:
            return None
        vals = sub[target].unique()
        if len(vals) != 2:
            return None
        g0 = sub[sub[target] == vals[0]][num_col]
        g1 = sub[sub[target] == vals[1]][num_col]
        if len(g0) < 5 or len(g1) < 5:
            return None
        
        # Safely convert target to binary 0/1 numeric codes (handles strings like 'Male'/'Female', 'Yes'/'No', etc.)
        target_series = pd.Series(pd.Categorical(sub[target]).codes, index=sub.index, dtype=float)
        num_series = pd.to_numeric(sub[num_col], errors='coerce')
        valid_mask = ~(target_series.isna() | num_series.isna())
        if valid_mask.sum() < self.min_sample_size:
            return None

        r, p = stats.pointbiserialr(target_series[valid_mask], num_series[valid_mask])
        if np.isnan(r):
            return None
        direction = "positive" if r > 0 else "negative"
        desc = (
            f"Observed relationship between '{num_col}' and binary outcome '{target}' "
            f"(r = {r:.4f}, p = {p:.4e}, N = {len(sub)})."
        )
        return Claim(
            claim_id=cid,
            claim_type="correlation",
            source_variable=num_col,
            target_variable=target,
            statistic_name="PointBiserial_r",
            statistic_value=float(r),
            p_value=float(p),
            sample_size=len(sub),
            direction=direction,
            description=desc,
            metadata={"mean_g0": float(g0.mean()), "mean_g1": float(g1.mean()), "target_values": [str(v) for v in vals]},
        )

    def _evaluate_group_difference(self, df: pd.DataFrame, cat_col: str, num_col: str, cid: str) -> Optional[Claim]:
        sub = df[[cat_col, num_col]].dropna()
        if len(sub) < self.min_sample_size:
            return None
        groups = [grp[num_col].values for _, grp in sub.groupby(cat_col) if len(grp) >= 5]
        if len(groups) < 2:
            return None
        f_stat, p = stats.f_oneway(*groups)
        if np.isnan(f_stat):
            return None
        means = sub.groupby(cat_col)[num_col].mean()
        max_grp = means.idxmax()
        min_grp = means.idxmin()
        diff = float(means.max() - means.min())
        desc = (
            f"Significant group disparity in '{num_col}' across '{cat_col}' "
            f"(ANOVA F = {f_stat:.2f}, p = {p:.4e}). Highest: '{max_grp}' ({means.max():.2f}) vs Lowest: '{min_grp}' ({means.min():.2f})."
        )
        return Claim(
            claim_id=cid,
            claim_type="group_difference",
            source_variable=cat_col,
            target_variable=num_col,
            statistic_name="ANOVA_F",
            statistic_value=float(f_stat),
            p_value=float(p),
            sample_size=len(sub),
            direction="difference",
            description=desc,
            metadata={"max_group": str(max_grp), "min_group": str(min_grp), "diff": diff},
        )

    def _evaluate_proportion_difference(self, df: pd.DataFrame, cat_col: str, target: str, cid: str) -> Optional[Claim]:
        sub = df[[cat_col, target]].dropna()
        if len(sub) < self.min_sample_size:
            return None
        ct = pd.crosstab(sub[cat_col], sub[target])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return None
        chi2, p, _, _ = stats.chi2_contingency(ct)
        if np.isnan(chi2):
            return None

        pos_val = sub[target].unique()[0]
        for candidate in [1, "1", True, "yes", "Yes", "admitted", "Admitted", "churn", "Churn"]:
            if candidate in sub[target].values:
                pos_val = candidate
                break

        rates = sub.groupby(cat_col)[target].apply(lambda s: (s == pos_val).mean())
        best_grp = rates.idxmax()
        worst_grp = rates.idxmin()
        diff = float(rates.max() - rates.min())

        desc = (
            f"Disparity in '{target}={pos_val}' rate across '{cat_col}' "
            f"(Chi2 = {chi2:.2f}, p = {p:.4e}). '{best_grp}' ({rates.max():.1%}) vs '{worst_grp}' ({rates.min():.1%})."
        )
        return Claim(
            claim_id=cid,
            claim_type="proportion_difference",
            source_variable=cat_col,
            target_variable=target,
            statistic_name="Chi2",
            statistic_value=float(chi2),
            p_value=float(p),
            sample_size=len(sub),
            direction="disparity",
            description=desc,
            metadata={
                "positive_value": str(pos_val),
                "best_group": str(best_grp),
                "worst_group": str(worst_grp),
                "best_rate": float(rates.max()),
                "worst_rate": float(rates.min()),
                "diff": diff,
            },
        )
