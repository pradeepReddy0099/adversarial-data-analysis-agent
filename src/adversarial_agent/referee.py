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

        elif not stability.is_stable:
            verdict = "SPURIOUS_OR_CONFOUNDED"
            badge = "🔴 UNSTABLE / OUTLIER-DRIVEN"
            confidence = 0.35
            summary = (
                f"STATISTICAL INSTABILITY: The claim failed bootstrap stability testing. "
                f"The 95% confidence interval spans across zero [{stability.ci_95_low:.3f}, {stability.ci_95_high:.3f}]."
            )
            takeaway = "Treat this finding with skepticism; it is highly sensitive to sample noise."

        elif audit.leakage_detected:
            verdict = "SPURIOUS_OR_CONFOUNDED"
            badge = "🔴 TARGET LEAKAGE"
            confidence = 0.10
            summary = (
                f"TARGET LEAKAGE: '{claim.source_variable}' is near-deterministically correlated with '{claim.target_variable}' "
                f"({claim.statistic_value:.4f}), indicating identity overlap or data leakage."
            )
            takeaway = "Exclude this variable from production predictive pipelines."

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

        return AdjudicatedFinding(
            finding_id=f"FINDING_{claim.claim_id}",
            verdict=verdict,
            verdict_badge=badge,
            claim=claim,
            audit=audit,
            confidence_score=confidence,
            synthesis_summary=summary,
            actionable_takeaway=takeaway,
        )

    def adjudicate_all(self, claims: List[Claim], audits: List[AuditReport]) -> List[AdjudicatedFinding]:
        audit_map = {a.claim_id: a for a in audits}
        findings = []
        for claim in claims:
            if claim.claim_id in audit_map:
                findings.append(self.adjudicate(claim, audit_map[claim.claim_id]))
        return findings
