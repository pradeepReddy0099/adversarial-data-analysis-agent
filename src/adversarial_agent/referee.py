"""Referee Agent: Adjudication and calibrated synthesis of adversarial debate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

from adversarial_agent.auditor import AuditReport
from adversarial_agent.explorer import Claim


@dataclass
class AdjudicatedFinding:
    """Final adjudicated verdict for a tested hypothesis."""
    finding_id: str
    verdict: str  # "ROBUST_VERIFIED" | "SIMPSONS_PARADOX_ALERT" | "SPURIOUS_OR_CONFOUNDED"
    verdict_badge: str
    claim: Claim
    audit: AuditReport
    confidence_score: float
    synthesis_summary: str
    actionable_takeaway: str
    classification_rationale: dict | None = None

    def to_dict(self) -> dict:
        return {
            "finding_id": self.finding_id,
            "verdict": self.verdict,
            "verdict_badge": self.verdict_badge,
            "claim": self.claim.to_dict(),
            "audit": self.audit.to_dict(),
            "confidence_score": self.confidence_score,
            "synthesis_summary": self.synthesis_summary,
            "actionable_takeaway": self.actionable_takeaway,
            "classification_rationale": self.classification_rationale or {},
        }


class RefereeAgent:
    """Impartial arbitrator evaluating claims against adversarial audits."""

    def adjudicate(self, claim: Claim, audit: AuditReport) -> AdjudicatedFinding:
        simpsons = audit.simpsons_audit
        confounder = audit.confounder_audit
        stability = audit.stability_audit

        if simpsons.paradox_detected:
            verdict = "SIMPSONS_PARADOX_ALERT"
            badge = "🟡 SIMPSON'S PARADOX"
            confidence = 0.25
            summary = (
                f"CRITICAL REVERSAL: The Explorer Agent identified a global trend ({claim.statistic_name} = {claim.statistic_value:.4f}), "
                f"but the Devil's Advocate proved that the trend completely REVERSES when conditioned on '{simpsons.confounder_column}'. "
                f"Specifically, {simpsons.reversal_subgroups_count} of {simpsons.total_subgroups} subgroups display the opposite direction. "
                f"The weighted within-group effect is {simpsons.weighted_subgroup_statistic:.4f}."
            )
            takeaway = (
                f"DO NOT act on the aggregate correlation between '{claim.source_variable}' and '{claim.target_variable}'. "
                f"The relationship is an artifact of composition differences across '{simpsons.confounder_column}'. "
                f"Interventions should be tailored within individual '{simpsons.confounder_column}' segments."
            )
            rationale = {
                "category_name": "Simpson's Paradox (Directional Inversion)",
                "why_assigned": (
                    f"Assigned because Battery 1 detected that the sign of the effect flips when the population is split "
                    f"by '{simpsons.confounder_column}'. While the aggregate population exhibits a global statistic of "
                    f"{claim.statistic_value:+.4f}, {simpsons.reversal_subgroups_count} out of {simpsons.total_subgroups} "
                    f"subgroups ({simpsons.reversal_percentage:.1f}%) exhibit the exact OPPOSITE direction, resulting in a "
                    f"weighted within-group effect of {simpsons.weighted_subgroup_statistic:+.4f}."
                ),
                "key_evidence": [
                    f"Global aggregate effect: {claim.statistic_name} = {claim.statistic_value:+.4f}",
                    f"Conditioning confounder: '{simpsons.confounder_column}'",
                    f"Subgroup reversals: {simpsons.reversal_subgroups_count}/{simpsons.total_subgroups} ({simpsons.reversal_percentage:.1f}%)",
                    f"Stratified weighted effect: {simpsons.weighted_subgroup_statistic:+.4f}",
                ],
                "decision_rule": "Rule: If >=50% of adequate subgroups or the weighted average reverses sign under stratification -> SIMPSON'S PARADOX ALERT.",
            }
            if simpsons.cmh_odds_ratio is not None:
                rationale["key_evidence"].append(f"Cochran-Mantel-Haenszel Common OR: {simpsons.cmh_odds_ratio:.3f} (p = {simpsons.cmh_p_value:.3e})")
            if simpsons.tarone_homogeneity_p is not None:
                rationale["key_evidence"].append(f"Tarone Homogeneity Test: p = {simpsons.tarone_homogeneity_p:.3e} (χ² = {simpsons.tarone_chi2:.2f})")
            if simpsons.e_value is not None:
                rationale["key_evidence"].append(f"Causal E-value: {simpsons.e_value:.2f}")

        elif confounder.is_confounded:
            verdict = "SPURIOUS_OR_CONFOUNDED"
            badge = "🔴 CONFOUNDED"
            confidence = 0.30
            summary = (
                f"CONFOUNDING DETECTED: The Explorer claimed a link between '{claim.source_variable}' and '{claim.target_variable}', "
                f"but this relationship collapses by {confounder.attenuation_percentage:.1f}% "
                f"(down to partial r = {confounder.partial_correlation:.4f}) when controlling for '{confounder.top_confounder}'."
            )
            takeaway = f"The correlation is confounded by '{confounder.top_confounder}'. Address '{confounder.top_confounder}' directly."
            rationale = {
                "category_name": "Confounded / Spurious Association",
                "why_assigned": (
                    f"Assigned because Battery 2 detected that controlling for '{confounder.top_confounder}' causes the "
                    f"apparent relationship to attenuate by {confounder.attenuation_percentage:.1f}% (exceeding the 60% threshold). "
                    f"This indicates '{claim.source_variable}' has little to no independent direct effect on '{claim.target_variable}'; "
                    f"the association was largely mediated or confounded by '{confounder.top_confounder}'."
                ),
                "key_evidence": [
                    f"Raw correlation: r = {confounder.raw_correlation:+.4f} (R² = {confounder.raw_r_squared:.3f})",
                    f"Controlling variable: '{confounder.top_confounder}'",
                    f"Partial correlation: partial r = {confounder.partial_correlation:+.4f} (partial R² = {confounder.partial_r_squared:.3f})",
                    f"Attenuation magnitude: {confounder.attenuation_percentage:.1f}% drop (Threshold: 60.0%)",
                ],
                "decision_rule": "Rule: If partial correlation attenuation >= 60% when controlling for a covariate -> SPURIOUS / CONFOUNDED.",
            }
            if confounder.e_value is not None:
                rationale["key_evidence"].append(f"E-value for unmeasured confounding: {confounder.e_value:.2f}")

        elif not stability.is_stable:
            verdict = "SPURIOUS_OR_CONFOUNDED"
            badge = "🔴 UNSTABLE / OUTLIER-DRIVEN"
            confidence = 0.35
            summary = (
                f"STATISTICAL INSTABILITY: The claim failed bootstrap stability testing. "
                f"The 95% confidence interval spans across zero [{stability.ci_95_low:.3f}, {stability.ci_95_high:.3f}]."
            )
            takeaway = "Treat this finding with skepticism; it is highly sensitive to sample noise."
            rationale = {
                "category_name": "Statistically Unstable (High Variance / Outlier Sensitive)",
                "why_assigned": (
                    f"Assigned because Battery 3 (non-parametric bootstrap resampling) proved that "
                    f"the finding's 95% empirical confidence interval spans across zero [{stability.ci_95_low:.3f}, {stability.ci_95_high:.3f}]. "
                    f"This proves the direction of effect cannot be distinguished from random noise at standard significance."
                ),
                "key_evidence": [
                    f"Bootstrap 95% Confidence Interval: [{stability.ci_95_low:.3f}, {stability.ci_95_high:.3f}]",
                    "Zero-crossing detected: True (interval includes 0.0)",
                    f"Bootstrap sample mean: {stability.bootstrap_mean:.4f}",
                ],
                "decision_rule": "Rule: If 95% bootstrap confidence interval crosses zero (includes 0.0) -> UNSTABLE / OUTLIER-DRIVEN.",
            }
            if stability.rank_discrepancy is not None:
                rationale["key_evidence"].append(f"Spearman rank discrepancy: Δr = {stability.rank_discrepancy:.3f}")
            if stability.max_leverage is not None:
                rationale["key_evidence"].append(f"Max leverage: {stability.max_leverage:.4f} ({stability.high_leverage_points} high-leverage points)")

        elif audit.leakage_detected:
            verdict = "SPURIOUS_OR_CONFOUNDED"
            badge = "🔴 TARGET LEAKAGE"
            confidence = 0.10
            summary = (
                f"TARGET LEAKAGE: '{claim.source_variable}' is near-deterministically correlated with '{claim.target_variable}' "
                f"({claim.statistic_value:.4f}), indicating identity overlap or data leakage."
            )
            takeaway = "Exclude this variable from production predictive pipelines."
            rationale = {
                "category_name": "Target Leakage / Data Identity Tautology",
                "why_assigned": (
                    f"Assigned because Battery 4 flagged near-deterministic collinearity (|statistic| >= 0.98). "
                    f"'{claim.source_variable}' is virtually an identical re-encoding of '{claim.target_variable}', "
                    f"which represents data pipeline leakage rather than an empirical scientific finding."
                ),
                "key_evidence": [
                    f"Observed collinearity: {claim.statistic_value:.4f} (Threshold: 0.9800)",
                    "Data integrity status: Tautological identity overlap",
                ],
                "decision_rule": "Rule: If correlation or categorical separation >= 0.98 -> TARGET LEAKAGE.",
            }

        else:
            verdict = "ROBUST_VERIFIED"
            badge = "🟢 ROBUST & VERIFIED"
            confidence = 0.92
            summary = (
                f"PASSED ADVERSARIAL AUDIT: The Explorer's claim ({claim.statistic_name} = {claim.statistic_value:.4f}, p = {claim.p_value:.4e}) "
                f"withstood all falsification attempts. Subgroups consistently exhibit the same direction, partial correlation remained strong, "
                f"and bootstrap resampling confirmed statistical stability."
            )
            takeaway = (
                f"High-confidence finding. The association between '{claim.source_variable}' and '{claim.target_variable}' "
                f"is statistically sound across subpopulations and robust to covariate adjustment."
            )
            rationale = {
                "category_name": "Robust & Statistically Verified Signal",
                "why_assigned": (
                    f"Assigned because the claim survived all 4 adversarial stress tests: "
                    f"(1) No Simpson's Paradox: subgroup trends consistently point in the same direction across candidate confounders. "
                    f"(2) No Confounding collapse: partial correlation remained strong under covariate adjustment. "
                    f"(3) High stability: bootstrap 95% confidence interval does not cross zero. "
                    f"(4) No leakage: feature is not a trivial tautology."
                ),
                "key_evidence": [
                    f"Observed effect size: {claim.statistic_name} = {claim.statistic_value:+.4f} (p = {claim.p_value:.4e})",
                    "Simpson's test: Passed (consistent direction across all tested strata)",
                    f"Confounder test: Passed (max attenuation {confounder.attenuation_percentage:.1f}% < 60% threshold)",
                    f"Bootstrap test: Passed (95% CI [{stability.ci_95_low:.3f}, {stability.ci_95_high:.3f}] does not cross zero)",
                ],
                "decision_rule": "Rule: Claim passes all 4 falsification batteries with no reversals, no collapse, and stable CI -> ROBUST & VERIFIED.",
            }
            if confounder.e_value is not None:
                rationale["key_evidence"].append(f"Causal sensitivity E-value: {confounder.e_value:.2f} ({confounder.e_value_interpretation})")
            if stability.spearman_rho is not None:
                rationale["key_evidence"].append(f"Monotonic Rank: Spearman ρ = {stability.spearman_rho:.3f} (Δr = {stability.rank_discrepancy:.3f})")

        return AdjudicatedFinding(
            finding_id=f"FINDING_{claim.claim_id}",
            verdict=verdict,
            verdict_badge=badge,
            claim=claim,
            audit=audit,
            confidence_score=confidence,
            synthesis_summary=summary,
            actionable_takeaway=takeaway,
            classification_rationale=rationale,
        )

    def adjudicate_all(self, claims: List[Claim], audits: List[AuditReport]) -> List[AdjudicatedFinding]:
        audit_map = {a.claim_id: a for a in audits}
        findings = []
        for claim in claims:
            if claim.claim_id in audit_map:
                findings.append(self.adjudicate(claim, audit_map[claim.claim_id]))
        return findings
