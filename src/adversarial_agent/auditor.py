"""Devil's Advocate (Auditor) Agent: Autonomous statistical falsification batteries."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from adversarial_agent.explorer import Claim


@dataclass
class SubgroupBreakdown:
    group_name: str
    sample_size: int
    subgroup_statistic: float
    p_value: float
    reverses_sign: bool
    description: str


@dataclass
class SimpsonsParadoxAudit:
    paradox_detected: bool
    confounder_column: Optional[str]
    global_statistic: float
    weighted_subgroup_statistic: float
    total_subgroups: int
    reversal_subgroups_count: int
    reversal_percentage: float
    breakdowns: List[SubgroupBreakdown]
    summary_verdict: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["breakdowns"] = [asdict(b) for b in self.breakdowns]
        return d


@dataclass
class ConfounderAudit:
    is_confounded: bool
    top_confounder: Optional[str]
    raw_correlation: float
    partial_correlation: Optional[float]
    attenuation_percentage: float
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StabilityAudit:
    is_stable: bool
    bootstrap_mean: float
    ci_95_low: float
    ci_95_high: float
    zero_crossing: bool
    trimmed_statistic: float
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditReport:
    """Full adversarial audit report for a single claim."""
    claim_id: str
    target_claim: Claim
    simpsons_audit: SimpsonsParadoxAudit
    confounder_audit: ConfounderAudit
    stability_audit: StabilityAudit
    leakage_detected: bool
    falsified: bool
    falsification_reason: str

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "target_claim": self.target_claim.to_dict(),
            "simpsons_audit": self.simpsons_audit.to_dict(),
            "confounder_audit": self.confounder_audit.to_dict(),
            "stability_audit": self.stability_audit.to_dict(),
            "leakage_detected": self.leakage_detected,
            "falsified": self.falsified,
            "falsification_reason": self.falsification_reason,
        }


class DevilsAdvocateAgent:
    """Adversarial stress tester searching for Simpson's Paradox, confounders, and instability."""

    def __init__(self, n_bootstrap: int = 150, min_subgroup_size: int = 8, random_seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.min_subgroup_size = min_subgroup_size
        self.rng = np.random.default_rng(random_seed)

    def audit_claim(self, claim: Claim, df: pd.DataFrame) -> AuditReport:
        """Run all falsification batteries against a candidate claim."""
        simpsons = self._audit_simpsons_paradox(claim, df)
        confounder = self._audit_confounders(claim, df)
        stability = self._audit_stability(claim, df)
        leakage = self._audit_leakage(claim, df)

        falsified = False
        reasons = []

        if simpsons.paradox_detected:
            falsified = True
            reasons.append(
                f"Simpson's Paradox: Trend flips sign when conditioned on '{simpsons.confounder_column}' "
                f"({simpsons.reversal_subgroups_count}/{simpsons.total_subgroups} subgroups reverse)."
            )

        if confounder.is_confounded:
            falsified = True
            reasons.append(
                f"Confounding: Effect collapses by {confounder.attenuation_percentage:.1f}% when controlling for '{confounder.top_confounder}'."
            )

        if not stability.is_stable:
            falsified = True
            reasons.append(
                f"Instability: Bootstrap 95% CI crosses zero [{stability.ci_95_low:.3f}, {stability.ci_95_high:.3f}]."
            )

        if leakage:
            falsified = True
            reasons.append("Target Leakage: Feature exhibits near-deterministic correlation with target.")

        summary_reason = " | ".join(reasons) if falsified else "Passed all adversarial falsification stress tests."

        return AuditReport(
            claim_id=claim.claim_id,
            target_claim=claim,
            simpsons_audit=simpsons,
            confounder_audit=confounder,
            stability_audit=stability,
            leakage_detected=leakage,
            falsified=falsified,
            falsification_reason=summary_reason,
        )

    def _audit_simpsons_paradox(self, claim: Claim, df: pd.DataFrame) -> SimpsonsParadoxAudit:
        source = claim.source_variable
        target = claim.target_variable

        ignore_cols = {source, target, "applicant_id", "customer_id", "id"}
        candidate_confounders = [
            c for c in df.columns
            if c not in ignore_cols and 2 <= df[c].nunique() <= 15
        ]

        if not candidate_confounders:
            numeric_candidates = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in ignore_cols and df[c].nunique() > 15
            ]
            for num_c in numeric_candidates[:2]:
                bin_name = f"{num_c}_bin"
                df = df.copy()
                df[bin_name] = pd.qcut(df[num_c], q=3, duplicates="drop").astype(str)
                candidate_confounders.append(bin_name)

        best_paradox: Optional[SimpsonsParadoxAudit] = None

        for conf_col in candidate_confounders:
            audit = self._test_single_confounder_reversal(claim, df, conf_col)
            if audit.paradox_detected:
                return audit
            if best_paradox is None or audit.reversal_percentage > best_paradox.reversal_percentage:
                best_paradox = audit

        if best_paradox:
            return best_paradox

        return SimpsonsParadoxAudit(
            paradox_detected=False,
            confounder_column=None,
            global_statistic=claim.statistic_value,
            weighted_subgroup_statistic=claim.statistic_value,
            total_subgroups=0,
            reversal_subgroups_count=0,
            reversal_percentage=0.0,
            breakdowns=[],
            summary_verdict="No subgroup reversals detected across candidate conditioning variables.",
        )

    def _test_single_confounder_reversal(self, claim: Claim, df: pd.DataFrame, conf_col: str) -> SimpsonsParadoxAudit:
        source = claim.source_variable
        target = claim.target_variable
        breakdowns: List[SubgroupBreakdown] = []
        is_global_positive = claim.statistic_value > 0

        groups = df[conf_col].dropna().unique()
        subgroup_stats = []
        subgroup_weights = []
        reversal_count = 0
        valid_groups = 0

        for grp in sorted(groups, key=str):
            sub = df[df[conf_col] == grp]
            if len(sub) < self.min_subgroup_size:
                continue

            if claim.claim_type == "correlation":
                sub_clean = sub[[source, target]].dropna()
                if len(sub_clean) < self.min_subgroup_size or sub_clean[source].nunique() < 2 or sub_clean[target].nunique() < 2:
                    continue
                r_sub, p_sub = stats.pearsonr(sub_clean[source].astype(float), sub_clean[target].astype(float))
                if np.isnan(r_sub):
                    continue

                valid_groups += 1
                subgroup_stats.append(r_sub)
                subgroup_weights.append(len(sub_clean))

                reverses = (r_sub > 0) != is_global_positive
                if reverses:
                    reversal_count += 1

                breakdowns.append(SubgroupBreakdown(
                    group_name=str(grp),
                    sample_size=len(sub_clean),
                    subgroup_statistic=float(r_sub),
                    p_value=float(p_sub),
                    reverses_sign=reverses,
                    description=f"Group '{grp}' (N={len(sub_clean)}): Pearson r = {r_sub:.4f} (p = {p_sub:.3e}) {'[REVERSED]' if reverses else ''}",
                ))

            elif claim.claim_type in ("group_difference", "proportion_difference"):
                sub_clean = sub[[source, target]].dropna()
                if len(sub_clean) < self.min_subgroup_size or sub_clean[source].nunique() < 2:
                    continue

                best_g = claim.metadata.get("best_group") or claim.metadata.get("max_group")
                worst_g = claim.metadata.get("worst_group") or claim.metadata.get("min_group")
                if not best_g or not worst_g:
                    continue

                if claim.claim_type == "proportion_difference":
                    pos_val = claim.metadata.get("positive_value", 1)
                    rates = sub_clean.groupby(source)[target].apply(lambda s: (s == pos_val).mean())
                    if best_g not in rates.index or worst_g not in rates.index:
                        continue
                    v_b, v_w = float(rates[best_g]), float(rates[worst_g])
                    sub_diff = v_b - v_w
                    desc_str = f"Rate diff ({best_g}: {v_b:.1%} vs {worst_g}: {v_w:.1%}) = {sub_diff:+.1%}"
                else:
                    means = sub_clean.groupby(source)[target].mean()
                    if best_g not in means.index or worst_g not in means.index:
                        continue
                    v_b, v_w = float(means[best_g]), float(means[worst_g])
                    sub_diff = v_b - v_w
                    desc_str = f"Mean diff ({best_g}: {v_b:.2f} vs {worst_g}: {v_w:.2f}) = {sub_diff:+.2f}"

                reverses = (sub_diff < 0)
                if reverses:
                    reversal_count += 1
                valid_groups += 1
                subgroup_stats.append(sub_diff)
                subgroup_weights.append(len(sub_clean))

                breakdowns.append(SubgroupBreakdown(
                    group_name=str(grp),
                    sample_size=len(sub_clean),
                    subgroup_statistic=sub_diff,
                    p_value=0.05,
                    reverses_sign=reverses,
                    description=f"Group '{grp}' (N={len(sub_clean)}): {desc_str} {'[REVERSED]' if reverses else ''}",
                ))

        if valid_groups == 0:
            return SimpsonsParadoxAudit(
                paradox_detected=False,
                confounder_column=conf_col,
                global_statistic=claim.statistic_value,
                weighted_subgroup_statistic=claim.statistic_value,
                total_subgroups=0,
                reversal_subgroups_count=0,
                reversal_percentage=0.0,
                breakdowns=[],
                summary_verdict=f"Insufficient data in subgroups of '{conf_col}' to evaluate.",
            )

        weighted_stat = float(np.average(subgroup_stats, weights=subgroup_weights))
        reversal_pct = float(reversal_count / valid_groups)
        weighted_reversed = (weighted_stat > 0) != is_global_positive if claim.claim_type == "correlation" else (weighted_stat < 0)
        paradox_detected = weighted_reversed or (reversal_pct >= 0.50)

        verdict = (
            f"SIMPSON'S PARADOX CONFIRMED: Stratifying by '{conf_col}' reverses the relationship in "
            f"{reversal_count}/{valid_groups} ({reversal_pct:.0%}) subgroups. "
            f"Global statistic was {claim.statistic_value:.4f}, but weighted within-group statistic is {weighted_stat:.4f}."
            if paradox_detected
            else f"No paradox under '{conf_col}': {valid_groups - reversal_count}/{valid_groups} subgroups align with global trend."
        )

        return SimpsonsParadoxAudit(
            paradox_detected=paradox_detected,
            confounder_column=conf_col,
            global_statistic=claim.statistic_value,
            weighted_subgroup_statistic=weighted_stat,
            total_subgroups=valid_groups,
            reversal_subgroups_count=reversal_count,
            reversal_percentage=reversal_pct,
            breakdowns=breakdowns,
            summary_verdict=verdict,
        )

    def _audit_confounders(self, claim: Claim, df: pd.DataFrame) -> ConfounderAudit:
        if claim.claim_type != "correlation":
            return ConfounderAudit(
                is_confounded=False,
                top_confounder=None,
                raw_correlation=claim.statistic_value,
                partial_correlation=claim.statistic_value,
                attenuation_percentage=0.0,
                description="Confounder partial correlation test applicable to continuous correlations only.",
            )

        source, target = claim.source_variable, claim.target_variable
        candidates = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in (source, target) and df[c].nunique() > 5
        ]

        if not candidates:
            return ConfounderAudit(
                is_confounded=False,
                top_confounder=None,
                raw_correlation=claim.statistic_value,
                partial_correlation=claim.statistic_value,
                attenuation_percentage=0.0,
                description="No additional numerical covariates available to test for confounding.",
            )

        r_xy = claim.statistic_value
        max_att = 0.0
        top_conf = None
        best_partial = r_xy

        for z in candidates:
            sub = df[[source, target, z]].dropna()
            if len(sub) < 20:
                continue
            try:
                r_xz, _ = stats.pearsonr(sub[source], sub[z])
                r_yz, _ = stats.pearsonr(sub[target], sub[z])
                denom = np.sqrt(max(1e-9, (1 - r_xz**2) * (1 - r_yz**2)))
                r_part = (r_xy - r_xz * r_yz) / denom
                att = (abs(r_xy) - abs(r_part)) / max(1e-6, abs(r_xy)) * 100.0
                if att > max_att:
                    max_att = att
                    top_conf = z
                    best_partial = r_part
            except Exception:
                continue

        is_conf = max_att >= 60.0 and top_conf is not None
        desc = (
            f"Correlation between '{source}' and '{target}' collapses from r = {r_xy:.4f} to partial r = {best_partial:.4f} "
            f"({max_att:.1f}% attenuation) when controlling for '{top_conf}'."
            if is_conf
            else f"Relationship is robust to linear confounding across tested covariates (max attenuation: {max_att:.1f}%)."
        )

        return ConfounderAudit(
            is_confounded=is_conf,
            top_confounder=top_conf,
            raw_correlation=r_xy,
            partial_correlation=float(best_partial),
            attenuation_percentage=float(max_att),
            description=desc,
        )

    def _audit_stability(self, claim: Claim, df: pd.DataFrame) -> StabilityAudit:
        source, target = claim.source_variable, claim.target_variable
        if claim.claim_type != "correlation":
            return StabilityAudit(
                is_stable=True,
                bootstrap_mean=claim.statistic_value,
                ci_95_low=claim.statistic_value,
                ci_95_high=claim.statistic_value,
                zero_crossing=False,
                trimmed_statistic=claim.statistic_value,
                description="Bootstrap stability check skipped for non-correlation claim.",
            )

        sub = df[[source, target]].dropna()
        n = len(sub)
        if n < 15:
            return StabilityAudit(
                is_stable=False,
                bootstrap_mean=0.0,
                ci_95_low=0.0,
                ci_95_high=0.0,
                zero_crossing=True,
                trimmed_statistic=0.0,
                description="Sample size too small for bootstrap estimation.",
            )

        x_vals, y_vals = sub[source].values, sub[target].values
        boot_stats = []
        for _ in range(self.n_bootstrap):
            idx = self.rng.integers(0, n, size=n)
            bx, by = x_vals[idx], y_vals[idx]
            if np.std(bx) > 1e-9 and np.std(by) > 1e-9:
                br, _ = stats.pearsonr(bx, by)
                if not np.isnan(br):
                    boot_stats.append(br)

        if not boot_stats:
            boot_stats = [claim.statistic_value]

        ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
        zero_crossing = (ci_low < 0 < ci_high)

        res = y_vals - (np.polyval(np.polyfit(x_vals, y_vals, 1), x_vals) if np.std(x_vals) > 0 else 0)
        non_outliers = np.abs(res) <= np.percentile(np.abs(res), 97.5)
        trim_r = stats.pearsonr(x_vals[non_outliers], y_vals[non_outliers])[0] if np.sum(non_outliers) > 10 else claim.statistic_value

        is_stable = (not zero_crossing) and (np.sign(trim_r) == np.sign(claim.statistic_value))
        desc = f"Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]. Trimmed r = {trim_r:.4f}."

        return StabilityAudit(
            is_stable=is_stable,
            bootstrap_mean=float(np.mean(boot_stats)),
            ci_95_low=float(ci_low),
            ci_95_high=float(ci_high),
            zero_crossing=zero_crossing,
            trimmed_statistic=float(trim_r),
            description=desc,
        )

    def _audit_leakage(self, claim: Claim, df: pd.DataFrame) -> bool:
        if claim.claim_type == "correlation" and abs(claim.statistic_value) >= 0.98:
            return True
        if claim.claim_type in ("group_difference", "proportion_difference"):
            diff = claim.metadata.get("diff", 0.0)
            if diff >= 0.99 or np.isinf(claim.statistic_value):
                return True
        return False
