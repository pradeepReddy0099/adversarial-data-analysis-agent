"""Referee Agent: Adjudication and calibrated synthesis of adversarial debate."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List
import numpy as np

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
            if simpsons.i_squared is not None:
                rationale["key_evidence"].append(f"Higgins & Thompson I² Inconsistency: {simpsons.i_squared:.1f}% (Cochran's Q = {simpsons.cochrans_q:.2f})")
            if simpsons.begg_funnel_p is not None:
                rationale["key_evidence"].append(f"Begg & Mazumdar Funnel Test: τ = {simpsons.begg_funnel_tau:.2f} (p = {simpsons.begg_funnel_p:.3e})")
            if simpsons.tau_squared is not None:
                rationale["key_evidence"].append(f"DerSimonian-Laird Heterogeneity Variance: τ² = {simpsons.tau_squared:.4f}")
            if simpsons.egger_test_p is not None:
                rationale["key_evidence"].append(f"Egger Funnel Asymmetry Test: t = {simpsons.egger_test_t:.2f} (p = {simpsons.egger_test_p:.3e})")
            if simpsons.trim_and_fill_k0 is not None and simpsons.trim_and_fill_k0 > 0:
                rationale["key_evidence"].append(f"Duval-Tweedie Trim & Fill: k0 = {simpsons.trim_and_fill_k0} missing strata (adjusted stat = {simpsons.trim_and_fill_adjusted_stat:.4f})")
            if simpsons.hausman_p is not None:
                rationale["key_evidence"].append(f"Hausman Endogeneity Test: H = {simpsons.hausman_stat:.2f} (p = {simpsons.hausman_p:.3e}) {'[INCONSISTENT / ENDOGENOUS]' if simpsons.is_hausman_inconsistent else '[CONSISTENT]'}")
            if simpsons.breslow_day_p is not None:
                rationale["key_evidence"].append(f"Breslow-Day Homogeneity Test: χ² = {simpsons.breslow_day_chi2:.2f} (p = {simpsons.breslow_day_p:.3e})")
            if simpsons.vuong_p is not None:
                rationale["key_evidence"].append(f"Vuong Model Selection Test: z = {simpsons.vuong_z:.2f} (p = {simpsons.vuong_p:.3e}) {'[STRATIFICATION SIGNIFICANTLY FAVORED]' if simpsons.is_stratification_favored else '[EQUIVALENT]'}")
            if simpsons.higgins_h_squared is not None and simpsons.higgins_h_squared > 1.0:
                rationale["key_evidence"].append(f"Higgins H² Relative Heterogeneity: H² = {simpsons.higgins_h_squared:.2f} (variance inflation factor H = {np.sqrt(simpsons.higgins_h_squared):.2f})")
            if simpsons.galbraith_outlier_count > 0:
                rationale["key_evidence"].append(f"Galbraith Radial Outlier Strata: {simpsons.galbraith_outlier_count} outlying subgroups ({', '.join(simpsons.galbraith_outlier_subgroups)})")
            if simpsons.woolf_p is not None:
                rationale["key_evidence"].append(f"Woolf Odds Ratio Heterogeneity Test: χ² = {simpsons.woolf_chi2:.2f} (p = {simpsons.woolf_p:.3e})")
            if simpsons.freeman_tukey_p is not None:
                rationale["key_evidence"].append(f"Freeman-Tukey Double-Arcsine Homogeneity Test: χ² = {simpsons.freeman_tukey_chi2:.2f} (p = {simpsons.freeman_tukey_p:.3e}) {'[VARIANCE-STABILIZED HETEROGENEITY]' if simpsons.is_freeman_tukey_heterogeneous else '[HOMOGENEOUS STRATA]'}")
            if simpsons.kruskal_wallis_p is not None:
                rationale["key_evidence"].append(f"Kruskal-Wallis Rank Heterogeneity: H = {simpsons.kruskal_wallis_h:.2f} (p = {simpsons.kruskal_wallis_p:.3e}) {'[NON-PARAMETRIC STRATA SHIFT]' if simpsons.is_kruskal_heterogeneous else '[INVARIANT MEDIANS]'}")
            if simpsons.peto_p_value is not None:
                rationale["key_evidence"].append(f"Peto Odds Ratio & Heterogeneity: OR_Peto = {simpsons.peto_odds_ratio:.3f} (Z = {simpsons.peto_z_stat:.2f}, p = {simpsons.peto_p_value:.3e}; Q_Peto = {simpsons.peto_heterogeneity_q:.2f}, p_het = {simpsons.peto_heterogeneity_p:.3e}) {'[PETO STRATA HETEROGENEITY]' if simpsons.is_peto_heterogeneous else '[HOMOGENEOUS STRATA]'}")
            if simpsons.fligner_killeen_p is not None:
                rationale["key_evidence"].append(f"Fligner-Killeen Variance Homogeneity: χ² = {simpsons.fligner_killeen_stat:.2f} (p = {simpsons.fligner_killeen_p:.3e}) {'[NON-PARAMETRIC VARIANCE SHIFT]' if simpsons.is_fligner_heterogeneous else '[HOMOSCEDASTIC STRATA]'}")
            if simpsons.harbord_p_value is not None:
                rationale["key_evidence"].append(f"Harbord Modified Score Asymmetry: t = {simpsons.harbord_t_stat:.2f} (p = {simpsons.harbord_p_value:.3e}) {'[SPARSE-STRATUM ASYMMETRY / SMALL-STUDY BIAS]' if simpsons.is_harbord_asymmetric else '[SYMMETRIC STRATA]'}")
            if simpsons.brown_forsythe_p is not None:
                rationale["key_evidence"].append(f"Brown-Forsythe Robust Scale Homogeneity: W = {simpsons.brown_forsythe_stat:.2f} (p = {simpsons.brown_forsythe_p:.3e}) {'[CONDITIONAL SCALE HETEROGENEITY]' if simpsons.is_brown_forsythe_heterogeneous else '[HOMOSCEDASTIC STRATA SPREAD]'}")
            if simpsons.peters_test_p is not None:
                rationale["key_evidence"].append(f"Peters Asymmetry Regression Test: t = {simpsons.peters_test_t:.2f} (p = {simpsons.peters_test_p:.3e}) {'[SPARSE-STRATUM ASYMMETRY / PUBLICATION BIAS]' if simpsons.is_peters_asymmetric else '[SYMMETRIC STRATA]'}")
            if simpsons.kish_design_effect is not None:
                rationale["key_evidence"].append(f"Kish Stratification Design Effect: D_Kish = {simpsons.kish_design_effect:.2f} (Var Ratio = {simpsons.stratified_variance_ratio:.2f}) {'[STRATIFICATION VARIANCE REDUCTION]' if simpsons.stratified_variance_ratio < 1.0 else '[STRATIFICATION VARIANCE INFLATION]'}")
            if simpsons.rucker_test_p is not None:
                rationale["key_evidence"].append(f"Rücker Arcsine Asymmetry Test: t = {simpsons.rucker_test_t:.2f} (p = {simpsons.rucker_test_p:.3e}) {'[SPARSE-STRATUM ARCSINE ASYMMETRY]' if simpsons.is_rucker_asymmetric else '[SYMMETRIC ARCSINE PRECISION]'}")
            if simpsons.levene_p is not None:
                rationale["key_evidence"].append(f"Levene Scale Homogeneity: W = {simpsons.levene_stat:.2f} (p = {simpsons.levene_p:.3e}) {'[CONDITIONAL SCALE HETEROGENEITY]' if simpsons.is_levene_heterogeneous else '[HOMOSCEDASTIC STRATA SPREAD]'}")
            if simpsons.macaskill_test_p is not None:
                rationale["key_evidence"].append(f"Macaskill Funnel Asymmetry Test: t = {simpsons.macaskill_test_t:.2f} (p = {simpsons.macaskill_test_p:.3e}) {'[SMALL-STRATUM SELECTION BIAS / ASYMMETRY]' if simpsons.is_macaskill_asymmetric else '[SYMMETRIC STRATA PRECISION]'}")
            if simpsons.bartlett_p is not None:
                rationale["key_evidence"].append(f"Bartlett Scale Homogeneity: B = {simpsons.bartlett_stat:.2f} (p = {simpsons.bartlett_p:.3e}) {'[CONDITIONAL VARIANCE HETEROGENEITY]' if simpsons.is_bartlett_heterogeneous else '[HOMOSCEDASTIC STRATA VARIANCE]'}")
            if simpsons.deeks_test_p is not None:
                rationale["key_evidence"].append(f"Deeks DOR Funnel Asymmetry Test: t = {simpsons.deeks_test_t:.2f} (p = {simpsons.deeks_test_p:.3e}) {'[DIAGNOSTIC ODDS RATIO FUNNEL ASYMMETRY]' if simpsons.is_deeks_asymmetric else '[SYMMETRIC DOR PRECISION]'}")
            if simpsons.mood_p_value is not None:
                rationale["key_evidence"].append(f"Mood Median-Rank Scale Dispersion: Z = {simpsons.mood_stat_z:.2f} (p = {simpsons.mood_p_value:.3e}) {'[NON-PARAMETRIC SCALE HETEROGENEITY]' if simpsons.is_mood_heterogeneous else '[HOMOSCEDASTIC MEDIAN SPREAD]'}")
            if simpsons.thompson_sharp_p is not None:
                rationale["key_evidence"].append(f"Thompson-Sharp Random-Effects Funnel Test: t = {simpsons.thompson_sharp_t:.2f} (p = {simpsons.thompson_sharp_p:.3e}) {'[RANDOM-EFFECTS SMALL-STUDY ASYMMETRY]' if simpsons.is_thompson_asymmetric else '[SYMMETRIC BETWEEN-STRATA PRECISION]'}")
            if simpsons.ansari_bradley_p is not None:
                rationale["key_evidence"].append(f"Ansari-Bradley Scale Dispersion: W = {simpsons.ansari_bradley_stat:.2f} (p = {simpsons.ansari_bradley_p:.3e}) {'[NON-PARAMETRIC SCALE DISPERSION HETEROGENEITY]' if simpsons.is_ansari_heterogeneous else '[HOMOGENEOUS STRATA DISPERSIONS]'}")
            if simpsons.moreno_test_p is not None:
                rationale["key_evidence"].append(f"Moreno Variance-Weighted Funnel Test: t = {simpsons.moreno_test_t:.2f} (p = {simpsons.moreno_test_p:.3e}) {'[VARIANCE-DEPENDENT REPORTING ASYMMETRY]' if simpsons.is_moreno_asymmetric else '[SYMMETRIC VARIANCE-WEIGHTED PRECISION]'}")
            if simpsons.siegel_tukey_p is not None:
                rationale["key_evidence"].append(f"Siegel-Tukey Rank Extremes Scale Test: W = {simpsons.siegel_tukey_stat:.2f} (p = {simpsons.siegel_tukey_p:.3e}) {'[EXTREME-RANK SCALE DIVERGENCE]' if simpsons.is_siegel_tukey_heterogeneous else '[EQUIDISPERSED STRATA SPREAD]'}")
            if simpsons.sterne_test_p is not None:
                rationale["key_evidence"].append(f"Sterne-Egger Weighted Funnel Test: t = {simpsons.sterne_test_t:.2f} (p = {simpsons.sterne_test_p:.3e}) {'[STANDARD-ERROR REPORTING ASYMMETRY]' if simpsons.is_sterne_asymmetric else '[SYMMETRIC STANDARD-ERROR PRECISION]'}")
            if simpsons.klotz_scale_p is not None:
                rationale["key_evidence"].append(f"Klotz Normal Scores Scale Test: Z = {simpsons.klotz_scale_stat:.2f} (p = {simpsons.klotz_scale_p:.3e}) {'[NORMAL-SCORES SCALE DIVERGENCE]' if simpsons.is_klotz_heterogeneous else '[EQUIDISPERSED NORMAL SCORES]'}")
            if simpsons.lfk_index is not None:
                rationale["key_evidence"].append(f"LFK Funnel Asymmetry Index: LFK = {simpsons.lfk_index:.2f} {'[MAJOR REPORTING ASYMMETRY / DOI PLOT DISTORTION]' if simpsons.is_lfk_asymmetric else '[SYMMETRIC DOI PLOT PRECISION]'}")
            if simpsons.conover_p_value is not None:
                rationale["key_evidence"].append(f"Conover Squared Ranks Scale Test: Z = {simpsons.conover_stat_z:.2f} (p = {simpsons.conover_p_value:.3e}) {'[NON-PARAMETRIC SCALE SPREAD HETEROGENEITY]' if simpsons.is_conover_heterogeneous else '[EQUIDISPERSED SQUARED RANKS]'}")
            if simpsons.pet_test_p is not None:
                rationale["key_evidence"].append(f"PET-PEESE Meta-Regression: t_PET = {simpsons.pet_test_t:.2f} (p = {simpsons.pet_test_p:.3e}), t_PEESE = {simpsons.peese_test_t:.2f} {'[SMALL-STRATUM SELECTION ASYMMETRY]' if simpsons.is_pet_asymmetric else '[GENUINE INFINITE-PRECISION LIMIT]'}")
            if simpsons.sukhatme_p_value is not None:
                rationale["key_evidence"].append(f"Sukhatme Rank-Order Scale Test: Z = {simpsons.sukhatme_stat_z:.2f} (p = {simpsons.sukhatme_p_value:.3e}) {'[NON-PARAMETRIC SCALE SPREAD DIVERGENCE]' if simpsons.is_sukhatme_heterogeneous else '[HOMOGENEOUS STRATUM SCALE DISPERSION]'}")
            if simpsons.tang_test_p is not None:
                rationale["key_evidence"].append(f"Tang-Liu Linear Funnel Test: t = {simpsons.tang_test_t:.2f} (p = {simpsons.tang_test_p:.3e}) {'[PRECISION-WEIGHTED ASYMMETRY]' if simpsons.is_tang_asymmetric else '[SYMMETRIC LINEAR PRECISION LIMIT]'}")
            if simpsons.capon_p_value is not None:
                rationale["key_evidence"].append(f"Capon Normal Scores Scale Test: Z = {simpsons.capon_stat_z:.2f} (p = {simpsons.capon_p_value:.3e}) {'[LMP NORMAL-SCORES SCALE DIVERGENCE]' if simpsons.is_capon_heterogeneous else '[HOMOGENEOUS NORMAL ORDER SPREAD]'}")
            if simpsons.pustejovsky_test_p is not None:
                rationale["key_evidence"].append(f"Pustejovsky-Rodgers Funnel Test: t = {simpsons.pustejovsky_test_t:.2f} (p = {simpsons.pustejovsky_test_p:.3e}) {'[VARIANCE-STABILIZED ASYMMETRY]' if simpsons.is_pustejovsky_asymmetric else '[SYMMETRIC VARIANCE-STABILIZED PRECISION]'}")
            if simpsons.moses_stat_p is not None:
                rationale["key_evidence"].append(f"Moses Sub-Sample Scale Test: U = {simpsons.moses_stat_u:.1f} (p = {simpsons.moses_stat_p:.3e}) {'[NON-PARAMETRIC SUB-SAMPLE SCALE DISPERSION]' if simpsons.is_moses_heterogeneous else '[HOMOGENEOUS SUB-SAMPLE SPREAD]'}")
            if simpsons.schwarzer_test_p is not None:
                rationale["key_evidence"].append(f"Schwarzer Arcsine Funnel Test: t = {simpsons.schwarzer_test_t:.2f} (p = {simpsons.schwarzer_test_p:.3e}) {'[ARCSINE-STABILIZED ASYMMETRY]' if simpsons.is_schwarzer_asymmetric else '[SYMMETRIC ARCSINE PRECISION]'}")
            if simpsons.westenberg_stat_p is not None:
                rationale["key_evidence"].append(f"Westenberg IQR Scale Test: Z = {simpsons.westenberg_stat_z:.2f} (p = {simpsons.westenberg_stat_p:.3e}) {'[NON-PARAMETRIC IQR SCALE DISPERSION]' if simpsons.is_westenberg_heterogeneous else '[HOMOGENEOUS IQR SPREAD]'}")
            if simpsons.tes_p_value is not None:
                rationale["key_evidence"].append(f"Ioannidis-Trikalinos TES Excess Test: A = {simpsons.tes_excess_ratio:.2f} (chi2 = {simpsons.tes_chi2:.2f}, p = {simpsons.tes_p_value:.3e}) {'[EXCESS SIGNIFICANCE / SELECTIVE DISCOVERY]' if simpsons.is_tes_excessive else '[EXPECTED STATISTICAL DISCOVERY POWER]'}")
            if simpsons.obrien_p_value is not None:
                rationale["key_evidence"].append(f"O'Brien Scale Homogeneity Test: F = {simpsons.obrien_stat_f:.2f} (p = {simpsons.obrien_p_value:.3e}) {'[ROBUST SCALE SPREAD HETEROGENEITY]' if simpsons.is_obrien_heterogeneous else '[HOMOGENEOUS STRATUM VARIANCES]'}")
            if simpsons.copas_test_p is not None:
                rationale["key_evidence"].append(f"Copas Selection Asymmetry Test: t = {simpsons.copas_test_t:.2f} (p = {simpsons.copas_test_p:.3e}) {'[SELECTION BIAS / FUNNEL ASYMMETRY]' if simpsons.is_copas_asymmetric else '[UNBIASED REPORTING / SYMMETRIC]'}")
            if simpsons.cochran_c_p is not None:
                rationale["key_evidence"].append(f"Cochran's C Outlier Variance Test: C = {simpsons.cochran_c_stat:.3f} (p = {simpsons.cochran_c_p:.3e}) {'[DISPROPORTIONATE STRATUM VARIANCE OUTLIER]' if simpsons.is_cochran_c_heterogeneous else '[HOMOGENEOUS STRATUM VARIANCES]'}")
            if simpsons.gail_simon_p is not None:
                rationale["key_evidence"].append(f"Gail-Simon Qualitative Interaction Test: s_GS = {simpsons.gail_simon_stat:.2f} (p = {simpsons.gail_simon_p:.3e}) {'[CONFIRMED TRUE QUALITATIVE REVERSAL / SIMPSON PARADOX]' if simpsons.is_gail_simon_reversal else '[QUANTITATIVE INTERACTION / UNIFORM SIGN]'}")
            if simpsons.hartley_fmax_p is not None:
                rationale["key_evidence"].append(f"Hartley F_max Variance Homogeneity: F_max = {simpsons.hartley_fmax_stat:.2f} (p = {simpsons.hartley_fmax_p:.3e}) {'[STRATUM SCALE HETEROGENEITY]' if simpsons.is_hartley_heterogeneous else '[HOMOGENEOUS STRATUM VARIANCES]'}")
            if simpsons.piantadosi_byar_green_p is not None:
                rationale["key_evidence"].append(f"Piantadosi-Byar-Green Qualitative Interaction Test: s_PBG = {simpsons.piantadosi_byar_green_stat:.2f} (p = {simpsons.piantadosi_byar_green_p:.3e}) {'[CONFIRMED LIKELIHOOD-RATIO QUALITATIVE REVERSAL]' if simpsons.is_pbg_reversal else '[UNIFORM MONOTONE SIGN DIRECTION]'}")
            if simpsons.layard_p_value is not None:
                rationale["key_evidence"].append(f"Layard Kurtosis-Robust Variance Homogeneity: L = {simpsons.layard_stat_l:.2f} (p = {simpsons.layard_p_value:.3e}) {'[KURTOSIS-INVARIANT STRATUM SCALE HETEROGENEITY]' if simpsons.is_layard_heterogeneous else '[HOMOGENEOUS STRATUM VARIANCES]'}")
            if simpsons.pan_score_p is not None:
                rationale["key_evidence"].append(f"Pan Directional Qualitative Score Test: S_Pan = {simpsons.pan_score_stat:.2f} (p = {simpsons.pan_score_p:.3e}) {'[CONFIRMED DIRECTIONAL QUALITATIVE REVERSAL]' if simpsons.is_pan_reversal else '[UNIFORM MONOTONE DIRECTION]'}")
            if simpsons.samiuddin_p_value is not None:
                rationale["key_evidence"].append(f"Samiuddin-Atiqullah Scale Homogeneity: F_Sam = {simpsons.samiuddin_stat_f:.2f} (p = {simpsons.samiuddin_p_value:.3e}) {'[KURTOSIS-ADJUSTED STRATUM SCALE HETEROGENEITY]' if simpsons.is_samiuddin_heterogeneous else '[HOMOGENEOUS STRATUM VARIANCES]'}")
            if simpsons.silvapulle_sen_p is not None:
                rationale["key_evidence"].append(f"Silvapulle-Sen Qualitative Inversion Test: S_SS = {simpsons.silvapulle_sen_stat:.2f} (p = {simpsons.silvapulle_sen_p:.3e}) {'[CONFIRMED LIKELIHOOD-RATIO QUALITATIVE REVERSAL]' if simpsons.is_silvapulle_reversal else '[UNIFORM CONCORDANT DIRECTION]'}")
            if simpsons.miller_jackknife_p_value is not None:
                rationale["key_evidence"].append(f"Miller Jackknife Variance Homogeneity: F_Miller = {simpsons.miller_jackknife_stat_f:.2f} (p = {simpsons.miller_jackknife_p_value:.3e}) {'[DISTRIBUTION-FREE STRATUM SCALE HETEROGENEITY]' if simpsons.is_miller_heterogeneous else '[HOMOGENEOUS STRATUM SCALE]'}")
            if simpsons.berger_p_value is not None:
                rationale["key_evidence"].append(f"Berger Intersection-Union Test: Z_Berger = {simpsons.berger_stat_z:.2f} (p = {simpsons.berger_p_value:.3e}) {'[EXACT LEVEL-ALPHA QUALITATIVE REVERSAL CERTIFIED]' if simpsons.is_berger_reversal else '[CONCORDANT SIGN UNIFORMITY]'}")
            if simpsons.shoemaker_p_value is not None:
                rationale["key_evidence"].append(f"Shoemaker Kurtosis-Adjusted Scale Homogeneity: Z_Shoe = {simpsons.shoemaker_stat_z:.2f} (p = {simpsons.shoemaker_p_value:.3e}) {'[KURTOSIS-INVARIANT STRATUM SCALE HETEROGENEITY]' if simpsons.is_shoemaker_heterogeneous else '[HOMOGENEOUS STRATUM SCALE]'}")

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
            if confounder.top_confounder_vif is not None:
                rationale["key_evidence"].append(f"Confounder VIF collinearity: {confounder.top_confounder_vif:.2f} {'[HIGH COLLINEARITY]' if confounder.is_collinear else '[STABLE]'}")
            if confounder.robustness_value is not None:
                rationale["key_evidence"].append(f"Cinelli-Hazlett Robustness Value: RV_q=1 = {confounder.robustness_value:.2f} (RV_α = {confounder.robustness_value_alpha:.2f})")
            if confounder.sobel_p is not None:
                rationale["key_evidence"].append(f"Baron-Kenny Sobel Mediation: p = {confounder.sobel_p:.3e} ({confounder.proportion_mediated:.1f}% mediated by '{confounder.top_confounder}')")
            if confounder.bayes_factor_bf10 is not None:
                rationale["key_evidence"].append(f"Kass-Raftery Bayes Factor: BF_10 = {confounder.bayes_factor_bf10:.2f} (ΔBIC = {confounder.bic_delta:+.2f}) [{confounder.bayes_factor_interpretation}]")
            if confounder.ramsey_reset_p is not None:
                rationale["key_evidence"].append(f"Ramsey RESET Test: F = {confounder.ramsey_reset_f:.2f} (p = {confounder.ramsey_reset_p:.3e}) {'[MISSPECIFIED / NON-LINEAR]' if confounder.is_reset_misspecified else '[WELL-SPECIFIED]'}")
            if confounder.oster_delta is not None:
                rationale["key_evidence"].append(f"Oster (2019) Selection Bound: δ* = {confounder.oster_delta:.2f} ({'Robust to unobservables' if abs(confounder.oster_delta) >= 1.0 else 'Sensitive to unobservables'})")
            if confounder.cohens_f2 is not None:
                rationale["key_evidence"].append(f"Cohen's f² effect size: f² = {confounder.cohens_f2:.3f} ({'negligible unique variance' if confounder.cohens_f2 < 0.02 else 'substantial variance'})")
            if confounder.cinelli_hazlett_rcrit is not None:
                rationale["key_evidence"].append(f"Cinelli-Hazlett Critical Boundary: R²crit = {confounder.cinelli_hazlett_rcrit:.2f}")
            if confounder.frank_itcv is not None:
                rationale["key_evidence"].append(f"Frank's ITCV Threshold: ITCV = {confounder.frank_itcv:.3f} (Observed Impact = {confounder.frank_observed_impact:.3f}) {'[ITCV EXCEEDED - CONFOUNDED]' if confounder.is_itcv_exceeded else '[BELOW ITCV]'}")
            if confounder.akaike_model_weight is not None:
                rationale["key_evidence"].append(f"Akaike Model Selection: w_AIC = {confounder.akaike_model_weight:.3f} (ΔAIC = {confounder.akaike_delta_aic:+.2f}, ER = {confounder.akaike_evidence_ratio:.2f})")
            if confounder.rosenbaum_gamma_crit is not None:
                rationale["key_evidence"].append(f"Rosenbaum Sensitivity Bound: Γcrit = {confounder.rosenbaum_gamma_crit:.2f} (p_upper = {confounder.rosenbaum_p_upper:.3e}) {'[SENSITIVE TO HIDDEN BIAS]' if confounder.is_rosenbaum_sensitive else '[ROBUST TO HIDDEN BIAS]'}")
            if confounder.manski_bound_low is not None:
                rationale["key_evidence"].append(f"Manski Non-Parametric Bounds: [{confounder.manski_bound_low:.3f}, {confounder.manski_bound_high:.3f}] {'[CONTAINS ZERO - PARTIALLY IDENTIFIED]' if confounder.manski_contains_zero else '[EXCLUDES ZERO - IDENTIFIED SIGN]'}")
            if confounder.aet_selection_ratio is not None:
                rationale["key_evidence"].append(f"Altonji-Elder-Taber Selection Ratio: R_AET = {confounder.aet_selection_ratio:.2f} {'[HIGHLY VULNERABLE TO UNOBSERVABLES - R_AET < 1.0]' if confounder.is_aet_vulnerable else '[ROBUST TO UNOBSERVABLES]'}")
            if confounder.imbens_manski_ci_low is not None:
                rationale["key_evidence"].append(f"Imbens-Manski Asymptotically Valid CI: [{confounder.imbens_manski_ci_low:.3f}, {confounder.imbens_manski_ci_high:.3f}] (cn = {confounder.imbens_manski_cn:.3f}) {'[EXCLUDES ZERO]' if confounder.imbens_manski_excludes_zero else '[INCLUDES ZERO]'}")
            if confounder.mallows_cp is not None:
                rationale["key_evidence"].append(f"Mallows' Cp Misspecification: Cp = {confounder.mallows_cp:.2f} (ΔCp = {confounder.cp_bias_discrepancy:+.2f}) {'[SEVERE OMITTED-VARIABLE BIAS]' if confounder.is_mallows_biased else '[WELL-SPECIFIED]'}")
            if confounder.davidson_mackinnon_p is not None:
                rationale["key_evidence"].append(f"Davidson-MacKinnon J-Test: t_J = {confounder.davidson_mackinnon_j_stat:.2f} (p = {confounder.davidson_mackinnon_p:.3e}) {'[NON-NESTED CONFOUNDER SPECIFICATION FAVORED]' if confounder.is_j_test_rejected else '[ORIGINAL MODEL ADEQUATE]'}")
            if confounder.durbin_wu_hausman_p is not None:
                rationale["key_evidence"].append(f"Durbin-Wu-Hausman Endogeneity Test: F = {confounder.durbin_wu_hausman_f:.2f} (p = {confounder.durbin_wu_hausman_p:.3e}) {'[SIGNIFICANT REGRESSOR ENDOGENEITY]' if confounder.is_dwh_endogenous else '[EXOGENOUS SPECIFICATION]'}")
            if confounder.stock_yogo_f_stat is not None:
                rationale["key_evidence"].append(f"Stock-Yogo First-Stage Relevance: F = {confounder.stock_yogo_f_stat:.2f} {'[STRONG CONVOLUTION - F >= 10]' if not confounder.is_weak_confounder else '[WEAK COUPLING - F < 10]'}")
            if confounder.bkw_condition_index is not None:
                rationale["key_evidence"].append(f"Belsley-Kuh-Welsch Condition Index: κ = {confounder.bkw_condition_index:.2f} (Max Var Prop = {confounder.bkw_max_variance_prop:.1%}) {'[SEVERE MULTICOLLINEARITY - κ >= 15]' if confounder.is_bkw_collinear else '[STABLE CONDITIONING]'}")
            if confounder.sargan_p_value is not None:
                rationale["key_evidence"].append(f"Sargan-Hansen Overidentification Test: J = {confounder.sargan_j_stat:.2f} (p = {confounder.sargan_p_value:.3e}) {'[EXCLUSION RESTRICTION VIOLATED - INVALID INSTRUMENT]' if confounder.is_sargan_rejected else '[VALID OVERIDENTIFYING RESTRICTIONS]'}")
            if confounder.crump_overlap_ratio is not None:
                rationale["key_evidence"].append(f"Crump Propensity Common Support: Off-Support = {confounder.crump_overlap_ratio:.1%} {'[SEVERE LACK OF COMMON SUPPORT - OVERLAP DEFICIENT]' if confounder.is_crump_overlap_deficient else '[ADEQUATE COVARIATE OVERLAP]'}")
            if confounder.hansen_threshold_p is not None:
                rationale["key_evidence"].append(f"Hansen Structural Threshold Test: F = {confounder.hansen_threshold_f:.2f} (p = {confounder.hansen_threshold_p:.3e}) {'[STRUCTURAL THRESHOLD REGIME DETECTED]' if confounder.has_hansen_threshold_regime else '[LINEAR CONTINUOUS RESPONSE]'}")
            if confounder.hosmer_lemeshow_p is not None:
                rationale["key_evidence"].append(f"Hosmer-Lemeshow Propensity Calibration: χ² = {confounder.hosmer_lemeshow_chi2:.2f} (p = {confounder.hosmer_lemeshow_p:.3e}) {'[PROPENSITY MODEL MISCALIBRATED]' if confounder.is_propensity_miscalibrated else '[WELL-CALIBRATED PROPENSITY]'}")
            if confounder.hac_se_ratio is not None:
                rationale["key_evidence"].append(f"Newey-West HAC Variance Ratio: ρ_HAC = {confounder.hac_se_ratio:.2f} (SE = {confounder.newey_west_hac_se:.4f}) {'[AUTOCORRELATION INFLATION - SE UNDERESTIMATED]' if confounder.is_hac_inflated else '[STABLE HAC INFERENCE]'}")
            if confounder.aipw_discrepancy is not None:
                rationale["key_evidence"].append(f"AIPW Doubly Robust Treatment Effect: τ_AIPW = {confounder.aipw_treatment_effect:.4f} (Discrepancy Δ_AIPW = {confounder.aipw_discrepancy:.1%}) {'[TREATMENT SELECTION CONVOLUTION / CONFOUNDING]' if confounder.is_aipw_confounded else '[ROBUST DOUBLY IDENTIFIED]'}")
            if confounder.leamer_bound_min is not None:
                rationale["key_evidence"].append(f"Leamer Extreme Bounds Analysis: β ∈ [{confounder.leamer_bound_min:.4f}, {confounder.leamer_bound_max:.4f}] {'[LEAMER FRAGILE - 0 IN EBA BOUNDS]' if confounder.is_leamer_fragile else '[LEAMER ROBUST - BOUNDS EXCLUDE 0]'}")
            if confounder.dml_discrepancy is not None:
                rationale["key_evidence"].append(f"Double Machine Learning (DML) Orthogonal Effect: β_DML = {confounder.dml_treatment_effect:.4f} (Discrepancy Δ_DML = {confounder.dml_discrepancy:.1%}) {'[NEYMAN ORTHOGONAL CONFOUNDING BIAS]' if confounder.is_dml_confounded else '[ROBUST DML IDENTIFIED]'}")
            if confounder.cinelli_k_multiplier is not None:
                rationale["key_evidence"].append(f"Cinelli-Hazlett Extreme Confounder Multiplier: k = {confounder.cinelli_k_multiplier:.2f}x {'[FRAGILE TO WEAK UNOBSERVED CONFOUNDING]' if confounder.is_cinelli_fragile else '[ROBUST - CONFOUNDER MUST EXCEED OBSERVED]'}")
            if confounder.rubin_b_score is not None:
                rationale["key_evidence"].append(f"Rubin Covariate Balance: B = {confounder.rubin_b_score:.1f}%, R = {confounder.rubin_r_ratio:.2f} {'[SEVERE COVARIATE IMBALANCE - MODEL DEPENDENT]' if confounder.is_rubin_imbalanced else '[SUFFICIENT COVARIATE BALANCE]'}")
            if confounder.brunner_munzel_p is not None:
                rationale["key_evidence"].append(f"Brunner-Munzel Relative Treatment Effect: W = {confounder.brunner_munzel_w:.2f} (p = {confounder.brunner_munzel_p:.3e}) {'[STOCHASTIC TREATMENT EFFECT]' if confounder.has_brunner_munzel_effect else '[NO STOCHASTIC SEPARATION]'}")
            if confounder.smd_covariate_balance is not None:
                rationale["key_evidence"].append(f"Rosenbaum-Rubin SMD Balance: SMD = {confounder.smd_covariate_balance:.3f}, Mahalanobis D² = {confounder.mahalanobis_distance:.3f} {'[SEVERE OBSERVATIONAL COVARIATE IMBALANCE]' if confounder.is_smd_imbalanced else '[BALANCED OBSERVATIONAL COVARIATES]'}")
            if confounder.imbens_normalized_diff is not None:
                rationale["key_evidence"].append(f"Imbens-Rubin Normalized Difference: Δ_Norm = {confounder.imbens_normalized_diff:.3f} {'[POSITIVITY DEFICIT / TRUNCATED SUPPORT]' if confounder.is_imbens_diff_excessive else '[SATISFACTORY COMMON SUPPORT OVERLAP]'}")
            if confounder.austin_variance_ratio is not None:
                rationale["key_evidence"].append(f"Austin Covariate Variance Ratio: VR = {confounder.austin_variance_ratio:.2f} {'[SEVERE OBSERVATIONAL VARIANCE IMBALANCE]' if confounder.is_austin_imbalanced else '[BALANCED OBSERVATIONAL DISPERSION]'}")
            if confounder.stuart_prognostic_discrepancy is not None:
                rationale["key_evidence"].append(f"Stuart Prognostic Score Discrepancy: Δ_prog = {confounder.stuart_prognostic_discrepancy:.3f} {'[STRUCTURAL BASELINE RISK DISPARITY]' if confounder.is_prognostic_discrepant else '[SIMILAR PROGNOSTIC BASELINE PROFILES]'}")
            if confounder.hansen_bowers_p is not None:
                rationale["key_evidence"].append(f"Hansen-Bowers Omnibus Balance: χ² = {confounder.hansen_bowers_chi2:.2f} (p = {confounder.hansen_bowers_p:.3e}) {'[MULTIVARIATE JOINT COVARIATE IMBALANCE]' if confounder.is_hansen_bowers_imbalanced else '[OMNIBUS COVARIATE BALANCE CONFIRMED]'}")
            if confounder.caliper_discordance_rate is not None:
                rationale["key_evidence"].append(f"Rosenbaum Caliper Discordance: {confounder.caliper_discordance_rate:.1%} (Caliper = {confounder.caliper_threshold:.3f}) {'[SEVERE OFF-SUPPORT EXTRAPOLATION RISK]' if confounder.is_caliper_discordant else '[SATISFACTORY COMMON SUPPORT MATCHING]'}")
            if confounder.gps_balance_p is not None:
                rationale["key_evidence"].append(f"Imai-van Dyk GPS Covariate Balance: r_GPS = {confounder.gps_balance_r:.3f} (t = {confounder.gps_balance_t:.2f}, p = {confounder.gps_balance_p:.3e}) {'[RESIDUAL COVARIATE DEPENDENCE / PROPENSITY IMBALANCE]' if confounder.is_gps_imbalanced else '[GENERALIZED PROPENSITY BALANCE CONFIRMED]'}")
            if confounder.austin_ks_p_value is not None:
                rationale["key_evidence"].append(f"Austin Kolmogorov-Smirnov Distributional Balance: D_KS = {confounder.austin_ks_stat:.3f} (p = {confounder.austin_ks_p_value:.3e}) {'[HIGHER-MOMENT COVARIATE IMBALANCE]' if confounder.is_austin_ks_imbalanced else '[SATISFACTORY DISTRIBUTIONAL OVERLAP]'}")
            if confounder.l1_multivariate_imbalance is not None:
                rationale["key_evidence"].append(f"Iacus-King-Porro L1 Distance: L1 = {confounder.l1_multivariate_imbalance:.3f} {'[SEVERE MULTIVARIATE IMBALANCE / POSITIVITY DEFICIT]' if confounder.is_l1_imbalanced else '[SUFFICIENT COMMON SUPPORT OVERLAP]'}")
            if confounder.rubin_parallelism_p is not None:
                rationale["key_evidence"].append(f"Rubin Regression Surface Parallelism: F = {confounder.rubin_parallelism_f:.2f} (p = {confounder.rubin_parallelism_p:.3e}) {'[NON-PARALLEL RESPONSE SURFACES / EFFECT HETEROGENEITY]' if confounder.is_rubin_nonparallel else '[PARALLEL COVARIATE RESPONSE SLOPES]'}")
            if confounder.eqq_mean_diff is not None:
                rationale["key_evidence"].append(f"Ho-Imai-King-Stuart eQQ Balance: Mean Diff = {confounder.eqq_mean_diff:.3f} (Max = {confounder.eqq_max_diff:.3f}) {'[SEVERE QUANTILE-QUANTILE DISTRIBUTIONAL IMBALANCE]' if confounder.is_eqq_imbalanced else '[SATISFACTORY EMPIRICAL Q-Q BALANCE]'}")
            if confounder.lin_interaction_p is not None:
                rationale["key_evidence"].append(f"Winston Lin Centered Interaction: t = {confounder.lin_interaction_t:.2f} (p = {confounder.lin_interaction_p:.3e}) {'[AGNOSTIC TREATMENT EFFECT HETEROGENEITY]' if confounder.is_lin_heterogeneous else '[HOMOGENEOUS TREATMENT-COVARIATE EFFECT]'}")
            if confounder.rr_variance_ratio is not None:
                rationale["key_evidence"].append(f"Rosenbaum-Rubin Residual Variance Ratio: VR = {confounder.rr_variance_ratio:.2f} (p = {confounder.rr_variance_p:.3e}) {'[HETEROGENEOUS RESIDUAL VARIANCE / 2ND-ORDER IMBALANCE]' if confounder.is_rr_variance_imbalanced else '[BALANCED SECOND-ORDER RESIDUAL SPREAD]'}")
            if confounder.iw_quadratic_diff is not None:
                rationale["key_evidence"].append(f"Imbens-Wooldridge Quadratic Balance: Delta_Z2 = {confounder.iw_quadratic_diff:.3f} {'[SEVERE QUADRATIC NON-LINEAR EXTRAPOLATION RISK]' if confounder.is_iw_quadratic_excessive else '[SATISFACTORY QUADRATIC OVERLAP]'}")
            if confounder.hodges_lehmann_p is not None:
                rationale["key_evidence"].append(f"Rosenbaum Hodges-Lehmann Shift: Delta_HL = {confounder.hodges_lehmann_shift:.3f} (Z = {confounder.hodges_lehmann_z:.2f}, p = {confounder.hodges_lehmann_p:.3e}) {'[NON-PARAMETRIC COVARIATE LOCATION SHIFT]' if confounder.is_hl_imbalanced else '[BALANCED NON-PARAMETRIC LOCATION]'}")
            if confounder.frank_rir_pct is not None:
                rationale["key_evidence"].append(f"Frank Robustness of Inference to Replacement (RIR): RIR = {confounder.frank_rir_pct:.1f}% ({confounder.frank_rir_count} cases) {'[HIGH INFERENTIAL FRAGILITY TO REPLACEMENT]' if confounder.is_rir_fragile else '[ROBUST INFERENTIAL REPLACEMENT BUFFER]'}")
            if confounder.imbens_contour_p is not None:
                rationale["key_evidence"].append(f"Imbens Sensitivity Contour: q_Imbens = {confounder.imbens_contour_q:.3f} (p = {confounder.imbens_contour_p:.3e}) {'[SENSITIVE TO WEAK UNMEASURED CONFOUNDING]' if confounder.is_imbens_sensitive else '[ROBUST TO MODERATE OMITTED CONFOUNDING]'}")
            if confounder.austin_stuart_p is not None:
                rationale["key_evidence"].append(f"Austin-Stuart IPTW Standardized Balance: MASD = {confounder.austin_stuart_masd:.3f} (p = {confounder.austin_stuart_p:.3e}) {'[RESIDUAL COVARIATE IMBALANCE AFTER IPTW]' if confounder.is_austin_stuart_imbalanced else '[BALANCED UNDER INVERSE PROBABILITY WEIGHTING]'}")
            if confounder.kallus_gamma_crit is not None:
                rationale["key_evidence"].append(f"Kallus Minimax Confounding Sensitivity: Γ_K,crit = {confounder.kallus_gamma_crit:.2f} (Bias_max = {confounder.kallus_bias_max:.3f}, [{confounder.kallus_bound_low:.3f}, {confounder.kallus_bound_high:.3f}]) {'[SENSITIVE TO BOUNDED UNOBSERVED CONVOLUTION]' if confounder.is_kallus_sensitive else '[MINIMAX CONVOLUTION ROBUST]'}")
            if confounder.cbps_overid_p is not None:
                rationale["key_evidence"].append(f"Imai-Ratkovic CBPS Overidentifying Moment Test: J_CBPS = {confounder.cbps_overid_j:.2f} (p = {confounder.cbps_overid_p:.3e}) {'[COVARIATE BALANCING OVERIDENTIFICATION REJECTED]' if confounder.is_cbps_rejected else '[EXACT BALANCING MOMENTS SATISFIED]'}")
            if confounder.rosenbaum_silber_delta_crit is not None:
                rationale["key_evidence"].append(f"Rosenbaum-Silber Hidden Bias Amplification: Delta_crit = {confounder.rosenbaum_silber_delta_crit:.2f} at Lambda = {confounder.rosenbaum_silber_lambda:.1f} {'[SENSITIVE TO MODEST OUTCOME CONVOLUTION]' if confounder.is_silber_sensitive else '[RESISTANT TO HIDDEN BIAS AMPLIFICATION]'}")
            if confounder.fogarty_gamma_crit is not None:
                rationale["key_evidence"].append(f"Fogarty Studentized Sensitivity Bounds: Gamma_crit = {confounder.fogarty_gamma_crit:.2f} (t_sens = {confounder.fogarty_studentized_t:.2f}) {'[SENSITIVE TO STUDENTIZED UNOBSERVED CONVOLUTION]' if confounder.is_fogarty_sensitive else '[STUDENTIZED CONVOLUTION ROBUST]'}")
            if confounder.rosenbaum_attr_gamma_crit is not None:
                rationale["key_evidence"].append(f"Rosenbaum Attributable Sensitivity Bounds: Gamma_attr = {confounder.rosenbaum_attr_gamma_crit:.2f} (p_upper = {confounder.rosenbaum_attr_p_upper:.3e}) {'[SENSITIVE TO ATTRIBUTABLE HIDDEN BIAS]' if confounder.is_rosenbaum_attr_sensitive else '[ATTRIBUTABLE EFFECT ROBUST]'}")
            if confounder.zubizarreta_sbw_cv is not None:
                rationale["key_evidence"].append(f"Zubizarreta Stable Balancing Weights Diagnostic: CV_w = {confounder.zubizarreta_sbw_cv:.2f} (Eff Sample = {confounder.zubizarreta_eff_sample_pct:.1f}%) {'[SEVERE BALANCING WEIGHT DISPERSION]' if confounder.is_zubizarreta_imbalanced else '[BALANCED COVARIATE WEIGHT STABILITY]'}")
            if confounder.rosenbaum_design_sensitivity_gamma is not None:
                rationale["key_evidence"].append(f"Rosenbaum Design Sensitivity Bounds: Gamma_tilde = {confounder.rosenbaum_design_sensitivity_gamma:.2f} (mu/sigma = {confounder.rosenbaum_standardized_effect_mu_sigma:.2f}) {'[FRAGILE ASYMPTOTIC DESIGN]' if confounder.is_design_fragile else '[ASYMPTOTICALLY ROBUST DESIGN]'}")
            if confounder.hosman_hansen_holland_delta is not None:
                rationale["key_evidence"].append(f"Hosman-Hansen-Holland Sensitivity Relation: delta_HHH = {confounder.hosman_hansen_holland_delta:.3f} (scale = {confounder.hosman_hansen_holland_scale:.3f}) {'[SENSITIVE TO MODEST PARTIAL CONVOLUTION]' if confounder.is_hosman_sensitive else '[RESISTANT TO PARTIAL UNOBSERVED CONVOLUTION]'}")
            if confounder.rosenbaum_ef_gamma_crit is not None:
                rationale["key_evidence"].append(f"Rosenbaum Evidence Factors Sensitivity: Gamma_EF = {confounder.rosenbaum_ef_gamma_crit:.2f} (p_upper = {confounder.rosenbaum_ef_p_upper:.3e}) {'[SENSITIVE TO DUAL EVIDENCE CONVOLUTION]' if confounder.is_rosenbaum_ef_fragile else '[ROBUST DUAL EVIDENCE REINFORCEMENT]'}")
            if confounder.ho_imai_convex_hull_overlap is not None:
                rationale["key_evidence"].append(f"Ho-Imai-King-Stuart Convex Hull Overlap: OH_Hull = {confounder.ho_imai_convex_hull_overlap:.1%} (Trimming Required = {confounder.ho_imai_trim_fraction:.1%}) {'[EXTRAPOLATION-PRONE SUPPORT DEFICIT]' if confounder.is_ho_imai_deficient else '[ROBUST COMMON COUNTERFACTUAL SUPPORT]'}")

        elif not stability.is_stable:
            verdict = "SPURIOUS_OR_CONFOUNDED"
            badge = "🔴 UNSTABLE / OUTLIER-DRIVEN"
            confidence = 0.35
            summary = (
                f"STATISTICAL INSTABILITY: The claim failed bootstrap stability testing. "
                f"The 95% confidence interval spans across zero [{stability.ci_95_low:.3f}, {stability.ci_95_high:.3f}] "
                f"(P(sign invariance) = {stability.sign_invariance_prob:.1%})."
            )
            takeaway = "Treat this finding with skepticism; it is highly sensitive to sample noise."
            rationale = {
                "category_name": "Statistically Unstable (High Variance / Outlier Sensitive)",
                "why_assigned": (
                    f"Assigned because Battery 3 (non-parametric bootstrap resampling) proved that "
                    f"the finding's 95% empirical confidence interval spans across zero [{stability.ci_95_low:.3f}, {stability.ci_95_high:.3f}] "
                    f"or exhibits sign instability (P(sign invariance) = {stability.sign_invariance_prob:.1%}). "
                    f"This proves the direction of effect cannot be reliably distinguished from random noise."
                ),
                "key_evidence": [
                    f"Bootstrap 95% Confidence Interval: [{stability.ci_95_low:.3f}, {stability.ci_95_high:.3f}]",
                    f"Sign invariance probability: {stability.sign_invariance_prob:.1%}",
                    "Zero-crossing detected: True (interval includes 0.0)" if stability.zero_crossing else "Sign invariance below threshold",
                    f"Bootstrap sample mean: {stability.bootstrap_mean:.4f}",
                ],
                "decision_rule": "Rule: If 95% bootstrap confidence interval crosses zero or P(sign invariance) < 90% -> UNSTABLE / OUTLIER-DRIVEN.",
            }
            if stability.rank_discrepancy is not None:
                rationale["key_evidence"].append(f"Spearman rank discrepancy: Δr = {stability.rank_discrepancy:.3f}")
            if stability.max_leverage is not None:
                rationale["key_evidence"].append(f"Max leverage: {stability.max_leverage:.4f} ({stability.high_leverage_points} high-leverage points)")
            if stability.max_cooks_distance is not None and stability.max_cooks_distance > 0:
                rationale["key_evidence"].append(f"Cook's distance: D_max = {stability.max_cooks_distance:.3f} ({stability.cooks_outlier_count} influential points, shift = {stability.cooks_statistic_shift:.4f})")
            if stability.max_dffits is not None and stability.max_dffits > 0:
                rationale["key_evidence"].append(f"DFFITS Influence: Max |DFFITS| = {stability.max_dffits:.3f} ({stability.dffits_outlier_count} influential observations)")
            if stability.permutation_p_value is not None:
                rationale["key_evidence"].append(f"Permutation Shuffle Test: p_perm = {stability.permutation_p_value:.3e}")
            if stability.breusch_pagan_p is not None:
                rationale["key_evidence"].append(f"Breusch-Pagan Heteroscedasticity: LM = {stability.breusch_pagan_lm:.2f} (p = {stability.breusch_pagan_p:.3e}) {'[HETEROSCEDASTIC]' if stability.is_heteroscedastic else '[HOMOSCEDASTIC]'}")
            if stability.white_lm_p is not None:
                rationale["key_evidence"].append(f"White Heteroscedasticity Test: LM = {stability.white_lm_stat:.2f} (p = {stability.white_lm_p:.3e}) {'[HETEROSCEDASTIC]' if stability.is_white_heteroscedastic else '[HOMOSCEDASTIC]'}")
            if stability.quantile_discrepancy_index is not None:
                rationale["key_evidence"].append(f"Quantile Discrepancy Index: QDI = {stability.quantile_discrepancy_index:.2f} {'[HIGH QUANTILE DISCREPANCY]' if stability.qdi_flag else '[UNIFORM]'}")
            if stability.max_dfbetas is not None and stability.max_dfbetas > 0:
                rationale["key_evidence"].append(f"DFBETAS Influence: Max |DFBETAS| = {stability.max_dfbetas:.3f} ({stability.dfbetas_outlier_count} influential observations)")
            if stability.jarque_bera_p is not None:
                rationale["key_evidence"].append(f"Jarque-Bera Normality Test: JB = {stability.jarque_bera_stat:.2f} (p = {stability.jarque_bera_p:.3e}) {'[NON-GAUSSIAN RESIDUALS]' if stability.is_non_normal else '[NORMAL RESIDUALS]'}")
            if stability.durbin_watson_stat is not None:
                rationale["key_evidence"].append(f"Durbin-Watson Test: DW = {stability.durbin_watson_stat:.2f} {'[SERIAL CLUSTERING / CORRELATION]' if stability.has_serial_correlation else '[INDEPENDENT RESIDUALS]'}")
            if stability.hc3_robust_p is not None:
                rationale["key_evidence"].append(f"MacKinnon-White HC3 Robust Standard Error: SE_HC3 = {stability.hc3_robust_se:.4f} (p_HC3 = {stability.hc3_robust_p:.3e}) {'[SIGNIFICANCE LOST UNDER HC3]' if stability.hc3_significance_lost else '[ROBUST TO HETEROSCEDASTICITY]'}")
            if stability.theil_sen_discrepancy is not None:
                rationale["key_evidence"].append(f"Theil-Sen Robust Slope: β_TS = {stability.theil_sen_slope:.4f} (Discrepancy Δ_TS = {stability.theil_sen_discrepancy:.1%})")
            if stability.bca_ci_low is not None:
                rationale["key_evidence"].append(f"BCa Bootstrap 95% CI: [{stability.bca_ci_low:.4f}, {stability.bca_ci_high:.4f}] {'[CROSSES ZERO]' if stability.bca_zero_crossing else '[NON-ZERO]'}")
            if stability.chow_p_value is not None:
                rationale["key_evidence"].append(f"Chow Structural Break Test: F = {stability.chow_f_stat:.2f} (p = {stability.chow_p_value:.3e}) {'[STRUCTURAL BREAK / REGIME INSTABILITY]' if stability.has_structural_break else '[PARAMETER CONSTANT]'}")
            if stability.type_m_exaggeration_ratio is not None:
                flag_tm = "[VULNERABLE TO WINNER'S CURSE]" if stability.is_type_m_vulnerable else "[STABLE DESIGN]"
                rationale["key_evidence"].append(f"Gelman-Carlin Design Analysis: Type M ratio = {stability.type_m_exaggeration_ratio:.2f}x (Type S error = {stability.type_s_error_prob:.1%}, Power = {stability.retrodesign_power:.1%}) {flag_tm}")
            if stability.andrews_p_value is not None:
                rationale["key_evidence"].append(f"Andrews Sup-Wald Unknown Break Test: Sup W = {stability.andrews_sup_w:.2f} at τ* = {stability.andrews_break_fraction:.1%} (p = {stability.andrews_p_value:.3e}) {'[UNKNOWN REGIME SHIFT / BREAK DETECTED]' if stability.has_unknown_structural_break else '[STABLE PARAMETERS ALONG REGRESSOR]'}")
            if stability.cusum_stat is not None:
                rationale["key_evidence"].append(f"Brown-Durbin-Evans CUSUM Parameter Drift: S = {stability.cusum_stat:.3f} (p ≈ {stability.cusum_p_approx:.3e}) {'[RECURSIVE STRUCTURAL INSTABILITY - S > 0.948]' if stability.has_cusum_instability else '[STABLE RECURSIVE TRAJECTORY]'}")
            if stability.koenker_p_value is not None:
                rationale["key_evidence"].append(f"Koenker Studentized Heteroscedasticity: LM = {stability.koenker_lm_stat:.2f} (p = {stability.koenker_p_value:.3e}) {'[ROBUST NON-NORMAL HETEROSCEDASTICITY]' if stability.is_koenker_heteroscedastic else '[HOMOSCEDASTIC]'}")
            if stability.harvey_collier_p is not None:
                rationale["key_evidence"].append(f"Harvey-Collier Linearity Test: t = {stability.harvey_collier_t:.2f} (p = {stability.harvey_collier_p:.3e}) {'[FUNCTIONAL NON-LINEARITY / CURVATURE DETECTED]' if stability.has_harvey_collier_nonlinearity else '[LINEAR SPECIFICATION ADEQUATE]'}")
            if stability.huber_slope_discrepancy is not None:
                rationale["key_evidence"].append(f"Huber Robust M-Estimator Discrepancy: Δ_Huber = {stability.huber_slope_discrepancy:.1%} (β_Huber = {stability.huber_m_slope:.4f}) {'[HEAVY OUTLIER CONTAMINATION]' if stability.is_huber_outlier_distorted else '[ROBUST TO OUTLIERS]'}")
            if stability.cusumsq_stat is not None:
                rationale["key_evidence"].append(f"CUSUMSQ Variance Stability: W = {stability.cusumsq_stat:.3f} (crit = {stability.cusumsq_critical_val:.3f}) {'[STRUCTURAL VARIANCE INSTABILITY / SHIFT DETECTED]' if stability.has_cusumsq_instability else '[HOMOSCEDASTIC VARIANCE CONSTANCY]'}")
            if stability.lts_slope_discrepancy is not None:
                rationale["key_evidence"].append(f"Least Trimmed Squares Robust Slope: β_LTS = {stability.lts_slope:.4f} (Discrepancy Δ_LTS = {stability.lts_slope_discrepancy:.1%}) {'[HIGH-BREAKDOWN LEVERAGE DISTORTION]' if stability.is_lts_outlier_distorted else '[RESISTANT TO OUTLIER MASKING]'}")
            if stability.siegel_slope_discrepancy is not None:
                rationale["key_evidence"].append(f"Siegel Repeated Medians Robust Slope: β_Siegel = {stability.siegel_slope:.4f} (Discrepancy Δ_Siegel = {stability.siegel_slope_discrepancy:.1%}) {'[50% BREAKDOWN OUTLIER DISTORTION]' if stability.is_siegel_outlier_distorted else '[RESISTANT TO GROSS OUTLIERS]'}")
            if stability.breusch_godfrey_p is not None:
                rationale["key_evidence"].append(f"Breusch-Godfrey Higher-Order Autocorrelation: LM = {stability.breusch_godfrey_lm:.2f} (p = {stability.breusch_godfrey_p:.3e}) {'[HIGHER-ORDER SERIAL CORRELATION DETECTED]' if stability.has_higher_order_autocorrelation else '[INDEPENDENT RESIDUALS]'}")
            if stability.runs_test_p is not None:
                rationale["key_evidence"].append(f"Wald-Wolfowitz Runs Test for Residual Independence: Z = {stability.runs_test_z:.2f} (p = {stability.runs_test_p:.3e}) {'[NON-RANDOM RESIDUAL CLUSTERING / DRIFT]' if stability.has_runs_nonrandomness else '[INDEPENDENT RANDOM RESIDUALS]'}")
            if stability.kendall_discrepancy is not None:
                rationale["key_evidence"].append(f"Kendall Tau-b Rank Concordance: τ_b = {stability.kendall_tau_b:.4f} (Discrepancy Δ_Kendall = {stability.kendall_discrepancy:.1%}) {'[GROSS LEVERAGE DISTORTION / RANK CONFLICT]' if stability.is_kendall_distorted else '[ROBUST RANK CONCORDANCE]'}")
            if stability.goldfeld_quandt_p is not None:
                rationale["key_evidence"].append(f"Goldfeld-Quandt Split Heteroskedasticity: F = {stability.goldfeld_quandt_f:.2f} (p = {stability.goldfeld_quandt_p:.3e}) {'[REGIME-SHIFT VARIANCE HETEROSKEDASTICITY]' if stability.has_goldfeld_quandt_heteroskedasticity else '[HOMOSCEDASTIC SPLIT REGIMES]'}")
            if stability.shapiro_francia_p is not None:
                rationale["key_evidence"].append(f"Shapiro-Francia Residual Normality: W' = {stability.shapiro_francia_w:.4f} (p = {stability.shapiro_francia_p:.3e}) {'[RESIDUAL GAUSSIAN VIOLATION - NON-NORMAL]' if stability.is_shapiro_francia_non_normal else '[NORMAL GAUSSIAN RESIDUALS]'}")
            if stability.dagostino_pearson_p is not None:
                rationale["key_evidence"].append(f"D'Agostino-Pearson Omnibus Normality: K² = {stability.dagostino_pearson_k2:.2f} (p = {stability.dagostino_pearson_p:.3e}) {'[RESIDUAL SKEW/KURTOSIS NON-NORMALITY]' if stability.is_dagostino_non_normal else '[NORMAL GAUSSIAN SKEW/KURTOSIS]'}")
            if stability.glejser_p_value is not None:
                rationale["key_evidence"].append(f"Glejser Regression Heteroskedasticity: t = {stability.glejser_t_stat:.2f} (p = {stability.glejser_p_value:.3e}) {'[ERROR DISPERSION REGRESSOR-DEPENDENCE]' if stability.has_glejser_heteroskedasticity else '[HOMOSCEDASTIC ERROR SPREAD]'}")
            if stability.anderson_darling_stat is not None:
                rationale["key_evidence"].append(f"Anderson-Darling EDF Normality: A² = {stability.anderson_darling_stat:.3f} (crit 5% = {stability.anderson_darling_crit:.3f}) {'[EMPIRICAL CDF TAIL NON-NORMALITY]' if stability.is_anderson_non_normal else '[NORMAL EMPIRICAL CDF RESIDUALS]'}")
            if stability.harvey_lm_p is not None:
                rationale["key_evidence"].append(f"Harvey Multiplicative Heteroskedasticity: LM = {stability.harvey_lm_stat:.2f} (p = {stability.harvey_lm_p:.3e}) {'[EXPONENTIAL VARIANCE INFLATION]' if stability.has_harvey_heteroskedasticity else '[HOMOSCEDASTIC MULTIPLICATIVE VARIANCE]'}")
            if stability.lilliefors_p_value is not None:
                rationale["key_evidence"].append(f"Lilliefors Composite Normality: D = {stability.lilliefors_stat:.3f} (p = {stability.lilliefors_p_value:.3e}) {'[COMPOSITE EDF RESIDUAL NON-NORMALITY]' if stability.is_lilliefors_non_normal else '[GAUSSIAN EDF RESIDUAL FIT]'}")
            if stability.park_p_value is not None:
                rationale["key_evidence"].append(f"Park Log-Linear Heteroskedasticity: t = {stability.park_t_stat:.2f} (p = {stability.park_p_value:.3e}) {'[EXPONENTIAL REGRESSOR-DEPENDENT SPREAD]' if stability.has_park_heteroskedasticity else '[HOMOSCEDASTIC LOG-RESIDUAL VARIANCE]'}")
            if stability.cramer_von_mises_p is not None:
                rationale["key_evidence"].append(f"Cramér-von Mises EDF Normality: W² = {stability.cramer_von_mises_w2:.3f} (p = {stability.cramer_von_mises_p:.3e}) {'[COMPOSITE EDF RESIDUAL NON-NORMALITY]' if stability.is_cvm_non_normal else '[GAUSSIAN EDF RESIDUAL FIT]'}")
            if stability.cook_weisberg_p is not None:
                rationale["key_evidence"].append(f"Cook-Weisberg Score Heteroskedasticity: S = {stability.cook_weisberg_score:.2f} (p = {stability.cook_weisberg_p:.3e}) {'[FITTED-VALUE VARIANCE EXPANSION]' if stability.has_cook_weisberg_heteroskedasticity else '[HOMOSCEDASTIC FITTED RESIDUALS]'}")
            if stability.kuiper_p_value is not None:
                rationale["key_evidence"].append(f"Kuiper Invariant EDF Normality: V = {stability.kuiper_v_stat:.3f} (V* = {stability.kuiper_v_star:.3f}, p = {stability.kuiper_p_value:.3e}) {'[CYCLIC/TAIL EDF RESIDUAL NON-NORMALITY]' if stability.is_kuiper_non_normal else '[INVARIANT EDF GAUSSIAN RESIDUALS]'}")
            if stability.max_covratio_dev is not None:
                rationale["key_evidence"].append(f"Belsley COVRATIO Covariance Influence: Max |COVRATIO - 1| = {stability.max_covratio_dev:.3f} ({stability.covratio_outlier_count} influential) {'[COVARIANCE PRECISION DEGRADATION]' if stability.is_covratio_influential else '[STABLE COVARIANCE DETERMINANT]'}")
            if stability.hosking_l_p is not None:
                rationale["key_evidence"].append(f"Hosking L-Moments Ratio Normality: L = {stability.hosking_l_stat:.2f} (p = {stability.hosking_l_p:.3e}) {'[L-MOMENTS RESIDUAL NON-NORMALITY]' if stability.is_hosking_non_normal else '[L-MOMENTS GAUSSIAN RESIDUAL FIT]'}")
            if stability.max_atkinson_distance is not None:
                rationale["key_evidence"].append(f"Atkinson Modified Influence: Max A = {stability.max_atkinson_distance:.2f} ({stability.atkinson_outlier_count} influential) {'[LEVERAGE-AMPLIFIED RESIDUAL DISTORTION]' if stability.is_atkinson_influential else '[BOUNDED ATKINSON INFLUENCE]'}")
            if stability.shapiro_wilk_p is not None:
                rationale["key_evidence"].append(f"Shapiro-Wilk Residual Normality: W = {stability.shapiro_wilk_w:.4f} (p = {stability.shapiro_wilk_p:.3e}) {'[OMNIBUS GAUSSIAN RESIDUAL VIOLATION]' if stability.is_shapiro_wilk_non_normal else '[NORMAL GAUSSIAN RESIDUAL FIT]'}")
            if stability.max_hadi_influence is not None:
                rationale["key_evidence"].append(f"Hadi Potential Influence: Max H² = {stability.max_hadi_influence:.2f} ({stability.hadi_outlier_count} influential) {'[HIGH POTENTIAL-RESIDUAL INFLUENCE]' if stability.is_hadi_influential else '[BOUNDED HADI INFLUENCE]'}")
            if stability.doornik_hansen_p is not None:
                rationale["key_evidence"].append(f"Doornik-Hansen Residual Normality: E_DH = {stability.doornik_hansen_stat:.2f} (p = {stability.doornik_hansen_p:.3e}) {'[TRANSFORMED SKEW-KURTOSIS NON-NORMALITY]' if stability.is_doornik_hansen_non_normal else '[NORMAL GAUSSIAN SKEW-KURTOSIS FIT]'}")
            if stability.max_welsch_distance is not None:
                rationale["key_evidence"].append(f"Welsch Distance Influence: Max W = {stability.max_welsch_distance:.2f} ({stability.welsch_outlier_count} influential) {'[LEVERAGE-WEIGHTED DISTANCE DISTORTION]' if stability.is_welsch_influential else '[BOUNDED WELSCH INFLUENCE]'}")
            if stability.gel_gastwirth_p is not None:
                rationale["key_evidence"].append(f"Gel-Gastwirth Robust Jarque-Bera (RJB): RJB = {stability.gel_gastwirth_rjb:.2f} (p = {stability.gel_gastwirth_p:.3e}) {'[ROBUST RESIDUAL NON-NORMALITY]' if stability.is_rjb_non_normal else '[NORMAL GAUSSIAN SPREAD]'}")
            if stability.max_pena_influence is not None:
                rationale["key_evidence"].append(f"Daniel Peña Influence: Max P = {stability.max_pena_influence:.3f} ({stability.pena_outlier_count} influential) {'[PROJECTION SUBSPACE DISTORTION]' if stability.is_pena_influential else '[BOUNDED PEÑA INFLUENCE]'}")
            if stability.szroeter_p_value is not None:
                rationale["key_evidence"].append(f"Szroeter Monotonic Rank Homoskedasticity: Q = {stability.szroeter_stat_q:.3f} (Z = {stability.szroeter_stat_z:.2f}, p = {stability.szroeter_p_value:.3e}) {'[MONOTONIC VARIANCE EXPANSION / HETEROSKEDASTIC]' if stability.is_szroeter_heteroskedastic else '[RANK HOMOSKEDASTIC RESIDUALS]'}")
            if stability.max_ap_deviation is not None:
                rationale["key_evidence"].append(f"Andrews-Pregibon Ellipsoid Influence: Max Delta_AP = {stability.max_ap_deviation:.3f} ({stability.ap_outlier_count} influential) {'[CONFIDENCE ELLIPSOID VOLUME DISTORTION]' if stability.is_ap_influential else '[BOUNDED ELLIPSOID SHRINKAGE]'}")
            if stability.anscombe_glynn_p is not None:
                rationale["key_evidence"].append(f"Anscombe-Glynn Kurtosis Transformation: Z_AG = {stability.anscombe_glynn_z:.2f} (p = {stability.anscombe_glynn_p:.3e}) {'[NON-GAUSSIAN RESIDUAL KURTOSIS]' if stability.is_anscombe_glynn_non_normal else '[GAUSSIAN ERROR KURTOSIS]'}")
            if stability.scale_inflation_ratio_qn is not None:
                rationale["key_evidence"].append(f"Rousseeuw-Croux Q_n Scale Estimator: Q_n = {stability.rousseeuw_croux_qn:.4f} (Scale Inflation Ratio = {stability.scale_inflation_ratio_qn:.2f}x) {'[SEVERE SCALE INFLATION BY OUTLIERS]' if stability.is_qn_inflated else '[STABLE SCALE VARIANCE]'}")
            if stability.epps_pulley_p is not None:
                rationale["key_evidence"].append(f"Epps-Pulley Characteristic Function Test: T_EP = {stability.epps_pulley_stat:.3f} (p = {stability.epps_pulley_p:.3e}) {'[DEPARTURE FROM GAUSSIAN FOURIER GEOMETRY]' if stability.is_epps_pulley_non_normal else '[GAUSSIAN CHARACTERISTIC FUNCTION VALIDATED]'}")
            if stability.max_chatterjee_hadi_influence is not None:
                rationale["key_evidence"].append(f"Chatterjee-Hadi Unified Influence Measure: Max H^2 = {stability.max_chatterjee_hadi_influence:.3f} ({stability.chatterjee_hadi_outlier_count} influential) {'[LEVERAGE-RESIDUAL COMPOUNDING OUTLIER]' if stability.is_chatterjee_hadi_influential else '[BOUNDED JOINT INFLUENCE]'}")
            if stability.baringhaus_henze_p is not None:
                rationale["key_evidence"].append(f"Baringhaus-Henze-Zirkler Characteristic Normality: T_BHZ = {stability.baringhaus_henze_stat:.3f} (p = {stability.baringhaus_henze_p:.3e}) {'[DEPARTURE FROM GAUSSIAN FOURIER GEOMETRY]' if stability.is_baringhaus_henze_non_normal else '[GAUSSIAN CHARACTERISTIC FUNCTION VALIDATED]'}")
            if stability.max_gray_ling_influence is not None:
                rationale["key_evidence"].append(f"Gray-Ling Leverage-Residual Influence: Max LR = {stability.max_gray_ling_influence:.3f} ({stability.gray_ling_outlier_count} influential) {'[LEVERAGE-RESIDUAL JOINT OUTLIER]' if stability.is_gray_ling_influential else '[BOUNDED JOINT INFLUENCE]'}")
            if stability.vasicek_entropy_p is not None:
                rationale["key_evidence"].append(f"Vasicek Sample Entropy Normality: K_mn = {stability.vasicek_entropy_stat:.3f} (p = {stability.vasicek_entropy_p:.3e}) {'[INFORMATION-THEORETIC SHANNON ENTROPY NON-NORMALITY]' if stability.is_vasicek_non_normal else '[GAUSSIAN ENTROPY CALIBRATED]'}")
            if stability.rousseeuw_zomeren_max_rdz is not None:
                rationale["key_evidence"].append(f"Rousseeuw-van Zomeren Robust Distance: Max RDZ = {stability.rousseeuw_zomeren_max_rdz:.3f} ({stability.rousseeuw_zomeren_outlier_count} influential) {'[HIGH-LEVERAGE ROBUST MAHALANOBIS OUTLIER]' if stability.is_rousseeuw_zomeren_influential else '[BOUNDED ROBUST DISTANCE]'}")
            if stability.zhang_zk_p_value is not None:
                rationale["key_evidence"].append(f"Zhang Empirical Likelihood Normality: Z_K = {stability.zhang_zk_stat:.3f} (p = {stability.zhang_zk_p_value:.3e}) {'[NON-PARAMETRIC EMPIRICAL LIKELIHOOD NON-NORMALITY]' if stability.is_zhang_non_normal else '[GAUSSIAN EMPIRICAL LIKELIHOOD ALIGNED]'}")
            if stability.bacon_max_distance is not None:
                rationale["key_evidence"].append(f"BACON Adaptive Outlier Diagnostic: Max Dist = {stability.bacon_max_distance:.3f} ({stability.bacon_outlier_count} unmasked outliers) {'[UNMASKED MULTIVARIATE LEVERAGE OUTLIERS]' if stability.is_bacon_influential else '[BOUNDED ADAPTIVE DISTANCES]'}")

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
            if confounder.bayes_factor_bf10 is not None:
                rationale["key_evidence"].append(f"Kass-Raftery Bayes Factor: BF_10 = {confounder.bayes_factor_bf10:.2f} (ΔBIC = {confounder.bic_delta:+.2f})")
            if stability.spearman_rho is not None:
                rationale["key_evidence"].append(f"Monotonic Rank: Spearman ρ = {stability.spearman_rho:.3f} (Δr = {stability.rank_discrepancy:.3f})")
            rationale["key_evidence"].append(f"Sign invariance probability: P(sign) = {stability.sign_invariance_prob:.1%}")
            if stability.max_cooks_distance is not None and stability.max_cooks_distance > 0:
                rationale["key_evidence"].append(f"Cook's distance: D_max = {stability.max_cooks_distance:.3f} (influence shift = {stability.cooks_statistic_shift:.4f})")
            if stability.max_dffits is not None and stability.max_dffits > 0:
                rationale["key_evidence"].append(f"DFFITS Influence: Max |DFFITS| = {stability.max_dffits:.3f}")
            if stability.permutation_p_value is not None:
                rationale["key_evidence"].append(f"Permutation Shuffle Test: p_perm = {stability.permutation_p_value:.3e}")
            if stability.breusch_pagan_p is not None:
                rationale["key_evidence"].append(f"Breusch-Pagan Homoscedasticity: LM = {stability.breusch_pagan_lm:.2f} (p = {stability.breusch_pagan_p:.3e})")
            if stability.quantile_discrepancy_index is not None:
                rationale["key_evidence"].append(f"Quantile Discrepancy Index: QDI = {stability.quantile_discrepancy_index:.2f}")
            if confounder.ramsey_reset_p is not None:
                rationale["key_evidence"].append(f"Ramsey RESET Test: F = {confounder.ramsey_reset_f:.2f} (p = {confounder.ramsey_reset_p:.3e})")
            if confounder.oster_delta is not None:
                rationale["key_evidence"].append(f"Oster (2019) Selection Bound: δ* = {confounder.oster_delta:.2f}")
            if stability.max_dfbetas is not None and stability.max_dfbetas > 0:
                rationale["key_evidence"].append(f"DFBETAS Influence: Max |DFBETAS| = {stability.max_dfbetas:.3f}")
            if stability.jarque_bera_p is not None:
                rationale["key_evidence"].append(f"Jarque-Bera Normality Test: JB = {stability.jarque_bera_stat:.2f} (p = {stability.jarque_bera_p:.3e})")
            if claim.fdr_q_value is not None:
                rationale["key_evidence"].append(f"Benjamini-Hochberg FDR: q = {claim.fdr_q_value:.4e}")
            if claim.bonferroni_p_value is not None:
                rationale["key_evidence"].append(f"Bonferroni FWER: p_adj = {claim.bonferroni_p_value:.4e}")
            if stability.theil_sen_discrepancy is not None:
                rationale["key_evidence"].append(f"Theil-Sen Robust Slope: β_TS = {stability.theil_sen_slope:.4f} (Δ_TS = {stability.theil_sen_discrepancy:.1%})")
            if stability.bca_ci_low is not None:
                rationale["key_evidence"].append(f"BCa Bootstrap 95% CI: [{stability.bca_ci_low:.4f}, {stability.bca_ci_high:.4f}]")
            if confounder.rosenbaum_gamma_crit is not None:
                rationale["key_evidence"].append(f"Rosenbaum Sensitivity: Γcrit = {confounder.rosenbaum_gamma_crit:.2f}")
            if confounder.manski_bound_low is not None:
                rationale["key_evidence"].append(f"Manski Non-Parametric Bounds: [{confounder.manski_bound_low:.3f}, {confounder.manski_bound_high:.3f}]")
            if stability.white_lm_p is not None:
                rationale["key_evidence"].append(f"White Heteroscedasticity Test: LM = {stability.white_lm_stat:.2f} (p = {stability.white_lm_p:.3e})")
            if confounder.aet_selection_ratio is not None:
                rationale["key_evidence"].append(f"Altonji-Elder-Taber Selection Ratio: R_AET = {confounder.aet_selection_ratio:.2f}")
            if stability.chow_p_value is not None:
                rationale["key_evidence"].append(f"Chow Structural Stability: F = {stability.chow_f_stat:.2f} (p = {stability.chow_p_value:.3e})")
            if confounder.imbens_manski_ci_low is not None:
                rationale["key_evidence"].append(f"Imbens-Manski Asymptotically Valid CI: [{confounder.imbens_manski_ci_low:.3f}, {confounder.imbens_manski_ci_high:.3f}]")
            if confounder.mallows_cp is not None:
                rationale["key_evidence"].append(f"Mallows' Cp Criterion: Cp = {confounder.mallows_cp:.2f} (ΔCp = {confounder.cp_bias_discrepancy:+.2f})")
            if stability.type_m_exaggeration_ratio is not None:
                rationale["key_evidence"].append(f"Gelman-Carlin Type M Ratio: {stability.type_m_exaggeration_ratio:.2f}x (Type S = {stability.type_s_error_prob:.1%})")
            if confounder.davidson_mackinnon_p is not None:
                rationale["key_evidence"].append(f"Davidson-MacKinnon J-Test: t_J = {confounder.davidson_mackinnon_j_stat:.2f} (p = {confounder.davidson_mackinnon_p:.3e})")
            if confounder.durbin_wu_hausman_p is not None:
                rationale["key_evidence"].append(f"Durbin-Wu-Hausman Exogeneity: F = {confounder.durbin_wu_hausman_f:.2f} (p = {confounder.durbin_wu_hausman_p:.3e})")
            if confounder.stock_yogo_f_stat is not None:
                rationale["key_evidence"].append(f"Stock-Yogo First-Stage Relevance: F = {confounder.stock_yogo_f_stat:.2f}")
            if stability.andrews_p_value is not None:
                rationale["key_evidence"].append(f"Andrews Sup-Wald Invariance: Sup W = {stability.andrews_sup_w:.2f} (p = {stability.andrews_p_value:.3e})")
            if stability.cusum_stat is not None:
                rationale["key_evidence"].append(f"Brown-Durbin-Evans CUSUM Stability: S = {stability.cusum_stat:.3f} (p ≈ {stability.cusum_p_approx:.3e})")
            if stability.koenker_p_value is not None:
                rationale["key_evidence"].append(f"Koenker Studentized Homoscedasticity: LM = {stability.koenker_lm_stat:.2f} (p = {stability.koenker_p_value:.3e})")
            if confounder.bkw_condition_index is not None:
                rationale["key_evidence"].append(f"Belsley-Kuh-Welsch Condition Index: κ = {confounder.bkw_condition_index:.2f}")
            if confounder.sargan_p_value is not None:
                rationale["key_evidence"].append(f"Sargan-Hansen Overidentification: J = {confounder.sargan_j_stat:.2f} (p = {confounder.sargan_p_value:.3e})")
            if stability.harvey_collier_p is not None:
                rationale["key_evidence"].append(f"Harvey-Collier Linearity: t = {stability.harvey_collier_t:.2f} (p = {stability.harvey_collier_p:.3e})")
            if stability.huber_slope_discrepancy is not None:
                rationale["key_evidence"].append(f"Huber Robust M-Slope: β = {stability.huber_m_slope:.4f} (Δ = {stability.huber_slope_discrepancy:.1%})")
            if simpsons.peters_test_p is not None:
                rationale["key_evidence"].append(f"Peters Asymmetry Regression Test: t = {simpsons.peters_test_t:.2f} (p = {simpsons.peters_test_p:.3e})")
            if simpsons.kish_design_effect is not None:
                rationale["key_evidence"].append(f"Kish Stratification Design Effect: D_Kish = {simpsons.kish_design_effect:.2f}")
            if confounder.crump_overlap_ratio is not None:
                rationale["key_evidence"].append(f"Crump Propensity Overlap: Off-Support = {confounder.crump_overlap_ratio:.1%}")
            if confounder.hansen_threshold_p is not None:
                rationale["key_evidence"].append(f"Hansen Threshold Invariance: F = {confounder.hansen_threshold_f:.2f} (p = {confounder.hansen_threshold_p:.3e})")
            if stability.cusumsq_stat is not None:
                rationale["key_evidence"].append(f"CUSUMSQ Variance Constancy: W = {stability.cusumsq_stat:.3f} (crit = {stability.cusumsq_critical_val:.3f})")
            if stability.lts_slope_discrepancy is not None:
                rationale["key_evidence"].append(f"Least Trimmed Squares Robust Slope: β_LTS = {stability.lts_slope:.4f} (Δ_LTS = {stability.lts_slope_discrepancy:.1%})")
            if simpsons.rucker_test_p is not None:
                rationale["key_evidence"].append(f"Rücker Arcsine Asymmetry Test: t = {simpsons.rucker_test_t:.2f} (p = {simpsons.rucker_test_p:.3e})")
            if simpsons.levene_p is not None:
                rationale["key_evidence"].append(f"Levene Scale Homogeneity: W = {simpsons.levene_stat:.2f} (p = {simpsons.levene_p:.3e})")
            if confounder.hosmer_lemeshow_p is not None:
                rationale["key_evidence"].append(f"Hosmer-Lemeshow Propensity Calibration: χ² = {confounder.hosmer_lemeshow_chi2:.2f} (p = {confounder.hosmer_lemeshow_p:.3e})")
            if confounder.hac_se_ratio is not None:
                rationale["key_evidence"].append(f"Newey-West HAC Variance Ratio: ρ_HAC = {confounder.hac_se_ratio:.2f}")
            if stability.siegel_slope_discrepancy is not None:
                rationale["key_evidence"].append(f"Siegel Repeated Medians Robust Slope: β_Siegel = {stability.siegel_slope:.4f} (Δ_Siegel = {stability.siegel_slope_discrepancy:.1%})")
            if stability.breusch_godfrey_p is not None:
                rationale["key_evidence"].append(f"Breusch-Godfrey Autocorrelation LM: LM = {stability.breusch_godfrey_lm:.2f} (p = {stability.breusch_godfrey_p:.3e})")
            if simpsons.macaskill_test_p is not None:
                rationale["key_evidence"].append(f"Macaskill Funnel Asymmetry Test: t = {simpsons.macaskill_test_t:.2f} (p = {simpsons.macaskill_test_p:.3e})")
            if simpsons.bartlett_p is not None:
                rationale["key_evidence"].append(f"Bartlett Scale Homogeneity: B = {simpsons.bartlett_stat:.2f} (p = {simpsons.bartlett_p:.3e})")
            if confounder.aipw_discrepancy is not None:
                rationale["key_evidence"].append(f"AIPW Doubly Robust Treatment Effect: τ_AIPW = {confounder.aipw_treatment_effect:.4f} (Δ_AIPW = {confounder.aipw_discrepancy:.1%})")
            if confounder.leamer_bound_min is not None:
                rationale["key_evidence"].append(f"Leamer Extreme Bounds Analysis: β ∈ [{confounder.leamer_bound_min:.4f}, {confounder.leamer_bound_max:.4f}]")
            if stability.runs_test_p is not None:
                rationale["key_evidence"].append(f"Wald-Wolfowitz Runs Test: Z = {stability.runs_test_z:.2f} (p = {stability.runs_test_p:.3e})")
            if stability.kendall_discrepancy is not None:
                rationale["key_evidence"].append(f"Kendall Tau-b Rank Concordance: τ_b = {stability.kendall_tau_b:.4f} (Δ_Kendall = {stability.kendall_discrepancy:.1%})")
            if simpsons.deeks_test_p is not None:
                rationale["key_evidence"].append(f"Deeks DOR Funnel Asymmetry Test: t = {simpsons.deeks_test_t:.2f} (p = {simpsons.deeks_test_p:.3e})")
            if simpsons.mood_p_value is not None:
                rationale["key_evidence"].append(f"Mood Scale Homogeneity: Z = {simpsons.mood_stat_z:.2f} (p = {simpsons.mood_p_value:.3e})")
            if confounder.dml_discrepancy is not None:
                rationale["key_evidence"].append(f"Double Machine Learning (DML) Effect: β_DML = {confounder.dml_treatment_effect:.4f} (Δ_DML = {confounder.dml_discrepancy:.1%})")
            if confounder.cinelli_k_multiplier is not None:
                rationale["key_evidence"].append(f"Cinelli-Hazlett Multiplier: k = {confounder.cinelli_k_multiplier:.2f}x")
            if stability.goldfeld_quandt_p is not None:
                rationale["key_evidence"].append(f"Goldfeld-Quandt Heteroskedasticity: F = {stability.goldfeld_quandt_f:.2f} (p = {stability.goldfeld_quandt_p:.3e})")
            if stability.shapiro_francia_p is not None:
                rationale["key_evidence"].append(f"Shapiro-Francia Residual Normality: W' = {stability.shapiro_francia_w:.4f} (p = {stability.shapiro_francia_p:.3e})")
            if simpsons.thompson_sharp_p is not None:
                rationale["key_evidence"].append(f"Thompson-Sharp Funnel Test: t = {simpsons.thompson_sharp_t:.2f} (p = {simpsons.thompson_sharp_p:.3e})")
            if simpsons.ansari_bradley_p is not None:
                rationale["key_evidence"].append(f"Ansari-Bradley Scale: W = {simpsons.ansari_bradley_stat:.2f} (p = {simpsons.ansari_bradley_p:.3e})")
            if confounder.rubin_b_score is not None:
                rationale["key_evidence"].append(f"Rubin Balance: B = {confounder.rubin_b_score:.1f}%, R = {confounder.rubin_r_ratio:.2f}")
            if confounder.brunner_munzel_p is not None:
                rationale["key_evidence"].append(f"Brunner-Munzel Effect: W = {confounder.brunner_munzel_w:.2f} (p = {confounder.brunner_munzel_p:.3e})")
            if stability.dagostino_pearson_p is not None:
                rationale["key_evidence"].append(f"D'Agostino-Pearson Normality: K² = {stability.dagostino_pearson_k2:.2f} (p = {stability.dagostino_pearson_p:.3e})")
            if stability.glejser_p_value is not None:
                rationale["key_evidence"].append(f"Glejser Heteroskedasticity: t = {stability.glejser_t_stat:.2f} (p = {stability.glejser_p_value:.3e})")
            if simpsons.moreno_test_p is not None:
                rationale["key_evidence"].append(f"Moreno Funnel Test: t = {simpsons.moreno_test_t:.2f} (p = {simpsons.moreno_test_p:.3e})")
            if simpsons.siegel_tukey_p is not None:
                rationale["key_evidence"].append(f"Siegel-Tukey Scale: W = {simpsons.siegel_tukey_stat:.2f} (p = {simpsons.siegel_tukey_p:.3e})")
            if confounder.smd_covariate_balance is not None:
                rationale["key_evidence"].append(f"Rosenbaum-Rubin SMD: {confounder.smd_covariate_balance:.3f}, Mahal D² = {confounder.mahalanobis_distance:.3f}")
            if confounder.imbens_normalized_diff is not None:
                rationale["key_evidence"].append(f"Imbens Normalized Diff: Δ_Norm = {confounder.imbens_normalized_diff:.3f}")
            if stability.anderson_darling_stat is not None:
                rationale["key_evidence"].append(f"Anderson-Darling Normality: A² = {stability.anderson_darling_stat:.3f}")
            if stability.harvey_lm_p is not None:
                rationale["key_evidence"].append(f"Harvey LM Heteroskedasticity: LM = {stability.harvey_lm_stat:.2f} (p = {stability.harvey_lm_p:.3e})")
            if simpsons.sterne_test_p is not None:
                rationale["key_evidence"].append(f"Sterne-Egger Funnel Test: t = {simpsons.sterne_test_t:.2f} (p = {simpsons.sterne_test_p:.3e})")
            if simpsons.klotz_scale_p is not None:
                rationale["key_evidence"].append(f"Klotz Normal Scores Scale: Z = {simpsons.klotz_scale_stat:.2f} (p = {simpsons.klotz_scale_p:.3e})")
            if confounder.austin_variance_ratio is not None:
                rationale["key_evidence"].append(f"Austin Variance Ratio: VR = {confounder.austin_variance_ratio:.2f}")
            if confounder.stuart_prognostic_discrepancy is not None:
                rationale["key_evidence"].append(f"Stuart Prognostic Discrepancy: Δ_prog = {confounder.stuart_prognostic_discrepancy:.3f}")
            if stability.lilliefors_p_value is not None:
                rationale["key_evidence"].append(f"Lilliefors Normality: D = {stability.lilliefors_stat:.3f} (p = {stability.lilliefors_p_value:.3e})")
            if stability.park_p_value is not None:
                rationale["key_evidence"].append(f"Park Heteroskedasticity: t = {stability.park_t_stat:.2f} (p = {stability.park_p_value:.3e})")
            if simpsons.lfk_index is not None:
                rationale["key_evidence"].append(f"LFK Funnel Index: LFK = {simpsons.lfk_index:.2f}")
            if simpsons.conover_p_value is not None:
                rationale["key_evidence"].append(f"Conover Squared Ranks Scale: Z = {simpsons.conover_stat_z:.2f} (p = {simpsons.conover_p_value:.3e})")
            if confounder.hansen_bowers_p is not None:
                rationale["key_evidence"].append(f"Hansen-Bowers Balance: χ² = {confounder.hansen_bowers_chi2:.2f} (p = {confounder.hansen_bowers_p:.3e})")
            if confounder.caliper_discordance_rate is not None:
                rationale["key_evidence"].append(f"Caliper Discordance: {confounder.caliper_discordance_rate:.1%} (ε = {confounder.caliper_threshold:.3f})")
            if stability.cramer_von_mises_p is not None:
                rationale["key_evidence"].append(f"Cramér-von Mises Normality: W² = {stability.cramer_von_mises_w2:.3f} (p = {stability.cramer_von_mises_p:.3e})")
            if stability.cook_weisberg_p is not None:
                rationale["key_evidence"].append(f"Cook-Weisberg Heteroskedasticity: S = {stability.cook_weisberg_score:.2f} (p = {stability.cook_weisberg_p:.3e})")
            if simpsons.pet_test_p is not None:
                rationale["key_evidence"].append(f"PET Asymmetry Test: t = {simpsons.pet_test_t:.2f} (p = {simpsons.pet_test_p:.3e})")
            if simpsons.sukhatme_p_value is not None:
                rationale["key_evidence"].append(f"Sukhatme Scale Test: Z = {simpsons.sukhatme_stat_z:.2f} (p = {simpsons.sukhatme_p_value:.3e})")
            if confounder.gps_balance_p is not None:
                rationale["key_evidence"].append(f"Imai-van Dyk GPS Balance: r_GPS = {confounder.gps_balance_r:.3f} (p = {confounder.gps_balance_p:.3e})")
            if confounder.austin_ks_p_value is not None:
                rationale["key_evidence"].append(f"Austin KS Balance: D_KS = {confounder.austin_ks_stat:.3f} (p = {confounder.austin_ks_p_value:.3e})")
            if stability.kuiper_p_value is not None:
                rationale["key_evidence"].append(f"Kuiper Invariant Normality: V* = {stability.kuiper_v_star:.3f} (p = {stability.kuiper_p_value:.3e})")
            if stability.max_covratio_dev is not None:
                rationale["key_evidence"].append(f"COVRATIO Influence: Max Dev = {stability.max_covratio_dev:.3f}")
            if simpsons.tang_test_p is not None:
                rationale["key_evidence"].append(f"Tang-Liu Funnel Test: t = {simpsons.tang_test_t:.2f} (p = {simpsons.tang_test_p:.3e})")
            if simpsons.capon_p_value is not None:
                rationale["key_evidence"].append(f"Capon Scale Test: Z = {simpsons.capon_stat_z:.2f} (p = {simpsons.capon_p_value:.3e})")
            if confounder.l1_multivariate_imbalance is not None:
                rationale["key_evidence"].append(f"Iacus-King-Porro L1 Distance: L1 = {confounder.l1_multivariate_imbalance:.3f}")
            if confounder.rubin_parallelism_p is not None:
                rationale["key_evidence"].append(f"Rubin Parallelism Test: F = {confounder.rubin_parallelism_f:.2f} (p = {confounder.rubin_parallelism_p:.3e})")
            if stability.hosking_l_p is not None:
                rationale["key_evidence"].append(f"Hosking L-Moments Normality: L = {stability.hosking_l_stat:.2f} (p = {stability.hosking_l_p:.3e})")
            if stability.max_atkinson_distance is not None:
                rationale["key_evidence"].append(f"Atkinson Influence: Max A = {stability.max_atkinson_distance:.2f}")
            if simpsons.pustejovsky_test_p is not None:
                rationale["key_evidence"].append(f"Pustejovsky-Rodgers Funnel Test: t = {simpsons.pustejovsky_test_t:.2f} (p = {simpsons.pustejovsky_test_p:.3e})")
            if simpsons.moses_stat_p is not None:
                rationale["key_evidence"].append(f"Moses Scale Test: U = {simpsons.moses_stat_u:.1f} (p = {simpsons.moses_stat_p:.3e})")
            if confounder.eqq_mean_diff is not None:
                rationale["key_evidence"].append(f"eQQ Covariate Balance: Mean Diff = {confounder.eqq_mean_diff:.3f}")
            if confounder.lin_interaction_p is not None:
                rationale["key_evidence"].append(f"Lin Centered Interaction: t = {confounder.lin_interaction_t:.2f} (p = {confounder.lin_interaction_p:.3e})")
            if stability.shapiro_wilk_p is not None:
                rationale["key_evidence"].append(f"Shapiro-Wilk Normality: W = {stability.shapiro_wilk_w:.4f} (p = {stability.shapiro_wilk_p:.3e})")
            if stability.max_hadi_influence is not None:
                rationale["key_evidence"].append(f"Hadi Potential Influence: Max H² = {stability.max_hadi_influence:.2f}")
            if simpsons.schwarzer_test_p is not None:
                rationale["key_evidence"].append(f"Schwarzer Arcsine Funnel Test: t = {simpsons.schwarzer_test_t:.2f} (p = {simpsons.schwarzer_test_p:.3e})")
            if simpsons.westenberg_stat_p is not None:
                rationale["key_evidence"].append(f"Westenberg Scale Test: Z = {simpsons.westenberg_stat_z:.2f} (p = {simpsons.westenberg_stat_p:.3e})")
            if confounder.rr_variance_ratio is not None:
                rationale["key_evidence"].append(f"Rosenbaum-Rubin Residual Variance: VR = {confounder.rr_variance_ratio:.2f}")
            if confounder.iw_quadratic_diff is not None:
                rationale["key_evidence"].append(f"Imbens-Wooldridge Quadratic Balance: Delta_Z2 = {confounder.iw_quadratic_diff:.3f}")
            if stability.doornik_hansen_p is not None:
                rationale["key_evidence"].append(f"Doornik-Hansen Normality: E_DH = {stability.doornik_hansen_stat:.2f} (p = {stability.doornik_hansen_p:.3e})")
            if stability.max_welsch_distance is not None:
                rationale["key_evidence"].append(f"Welsch Distance Influence: Max W = {stability.max_welsch_distance:.2f}")
            if simpsons.tes_p_value is not None:
                rationale["key_evidence"].append(f"Ioannidis-Trikalinos TES Test: A = {simpsons.tes_excess_ratio:.2f} (p = {simpsons.tes_p_value:.3e})")
            if simpsons.obrien_p_value is not None:
                rationale["key_evidence"].append(f"O'Brien Scale Test: F = {simpsons.obrien_stat_f:.2f} (p = {simpsons.obrien_p_value:.3e})")
            if confounder.hodges_lehmann_p is not None:
                rationale["key_evidence"].append(f"Hodges-Lehmann Shift: Delta_HL = {confounder.hodges_lehmann_shift:.3f} (p = {confounder.hodges_lehmann_p:.3e})")
            if confounder.frank_rir_pct is not None:
                rationale["key_evidence"].append(f"Frank RIR: {confounder.frank_rir_pct:.1f}% ({confounder.frank_rir_count} cases)")
            if stability.gel_gastwirth_p is not None:
                rationale["key_evidence"].append(f"Gel-Gastwirth RJB Normality: RJB = {stability.gel_gastwirth_rjb:.2f} (p = {stability.gel_gastwirth_p:.3e})")
            if stability.max_pena_influence is not None:
                rationale["key_evidence"].append(f"Peña Influence: Max P = {stability.max_pena_influence:.3f}")
            if simpsons.copas_test_p is not None:
                rationale["key_evidence"].append(f"Copas Selection Test: t = {simpsons.copas_test_t:.2f} (p = {simpsons.copas_test_p:.3e})")
            if simpsons.cochran_c_p is not None:
                rationale["key_evidence"].append(f"Cochran's C Variance Test: C = {simpsons.cochran_c_stat:.3f} (p = {simpsons.cochran_c_p:.3e})")
            if confounder.imbens_contour_p is not None:
                rationale["key_evidence"].append(f"Imbens Sensitivity: q = {confounder.imbens_contour_q:.3f} (p = {confounder.imbens_contour_p:.3e})")
            if confounder.austin_stuart_p is not None:
                rationale["key_evidence"].append(f"Austin-Stuart IPTW MASD: {confounder.austin_stuart_masd:.3f}")
            if stability.szroeter_p_value is not None:
                rationale["key_evidence"].append(f"Szroeter Homoskedasticity: Q = {stability.szroeter_stat_q:.3f} (p = {stability.szroeter_p_value:.3e})")
            if stability.max_ap_deviation is not None:
                rationale["key_evidence"].append(f"Andrews-Pregibon Influence: Max Delta_AP = {stability.max_ap_deviation:.3f}")
            if simpsons.gail_simon_p is not None:
                rationale["key_evidence"].append(f"Gail-Simon Test: s_GS = {simpsons.gail_simon_stat:.2f} (p = {simpsons.gail_simon_p:.3e})")
            if simpsons.hartley_fmax_p is not None:
                rationale["key_evidence"].append(f"Hartley F_max: F = {simpsons.hartley_fmax_stat:.2f} (p = {simpsons.hartley_fmax_p:.3e})")
            if confounder.kallus_gamma_crit is not None:
                rationale["key_evidence"].append(f"Kallus Minimax Sensitivity: Γ_K,crit = {confounder.kallus_gamma_crit:.2f}")
            if confounder.cbps_overid_p is not None:
                rationale["key_evidence"].append(f"Imai-Ratkovic CBPS Test: J = {confounder.cbps_overid_j:.2f} (p = {confounder.cbps_overid_p:.3e})")
            if stability.anscombe_glynn_p is not None:
                rationale["key_evidence"].append(f"Anscombe-Glynn Kurtosis: Z_AG = {stability.anscombe_glynn_z:.2f} (p = {stability.anscombe_glynn_p:.3e})")
            if stability.scale_inflation_ratio_qn is not None:
                rationale["key_evidence"].append(f"Rousseeuw-Croux Q_n: Scale Inflation = {stability.scale_inflation_ratio_qn:.2f}x")
            if simpsons.piantadosi_byar_green_p is not None:
                rationale["key_evidence"].append(f"Piantadosi-Byar-Green Test: s_PBG = {simpsons.piantadosi_byar_green_stat:.2f} (p = {simpsons.piantadosi_byar_green_p:.3e})")
            if simpsons.layard_p_value is not None:
                rationale["key_evidence"].append(f"Layard Variance Test: L = {simpsons.layard_stat_l:.2f} (p = {simpsons.layard_p_value:.3e})")
            if confounder.rosenbaum_silber_delta_crit is not None:
                rationale["key_evidence"].append(f"Rosenbaum-Silber Amplification: Delta_crit = {confounder.rosenbaum_silber_delta_crit:.2f} (Lambda = {confounder.rosenbaum_silber_lambda:.1f})")
            if confounder.fogarty_gamma_crit is not None:
                rationale["key_evidence"].append(f"Fogarty Sensitivity: Gamma_crit = {confounder.fogarty_gamma_crit:.2f} (t = {confounder.fogarty_studentized_t:.2f})")
            if stability.epps_pulley_p is not None:
                rationale["key_evidence"].append(f"Epps-Pulley Test: T_EP = {stability.epps_pulley_stat:.3f} (p = {stability.epps_pulley_p:.3e})")
            if stability.max_chatterjee_hadi_influence is not None:
                rationale["key_evidence"].append(f"Chatterjee-Hadi Influence: Max H^2 = {stability.max_chatterjee_hadi_influence:.3f}")
            if simpsons.pan_score_p is not None:
                rationale["key_evidence"].append(f"Pan Qualitative Score Test: S_Pan = {simpsons.pan_score_stat:.2f} (p = {simpsons.pan_score_p:.3e})")
            if simpsons.samiuddin_p_value is not None:
                rationale["key_evidence"].append(f"Samiuddin-Atiqullah Scale Test: F_Sam = {simpsons.samiuddin_stat_f:.2f} (p = {simpsons.samiuddin_p_value:.3e})")
            if confounder.rosenbaum_attr_gamma_crit is not None:
                rationale["key_evidence"].append(f"Rosenbaum Attributable Sensitivity: Gamma_attr = {confounder.rosenbaum_attr_gamma_crit:.2f} (p = {confounder.rosenbaum_attr_p_upper:.3e})")
            if confounder.zubizarreta_sbw_cv is not None:
                rationale["key_evidence"].append(f"Zubizarreta SBW Balance: CV_w = {confounder.zubizarreta_sbw_cv:.2f} (Eff = {confounder.zubizarreta_eff_sample_pct:.1f}%)")
            if stability.baringhaus_henze_p is not None:
                rationale["key_evidence"].append(f"Baringhaus-Henze-Zirkler Test: T_BHZ = {stability.baringhaus_henze_stat:.3f} (p = {stability.baringhaus_henze_p:.3e})")
            if stability.max_gray_ling_influence is not None:
                rationale["key_evidence"].append(f"Gray-Ling Influence: Max LR = {stability.max_gray_ling_influence:.3f}")
            if simpsons.silvapulle_sen_p is not None:
                rationale["key_evidence"].append(f"Silvapulle-Sen Test: S_SS = {simpsons.silvapulle_sen_stat:.2f} (p = {simpsons.silvapulle_sen_p:.3e})")
            if simpsons.miller_jackknife_p_value is not None:
                rationale["key_evidence"].append(f"Miller Jackknife Test: F_Miller = {simpsons.miller_jackknife_stat_f:.2f} (p = {simpsons.miller_jackknife_p_value:.3e})")
            if confounder.rosenbaum_design_sensitivity_gamma is not None:
                rationale["key_evidence"].append(f"Rosenbaum Design Sensitivity: Gamma_tilde = {confounder.rosenbaum_design_sensitivity_gamma:.2f}")
            if confounder.hosman_hansen_holland_delta is not None:
                rationale["key_evidence"].append(f"Hosman-Hansen-Holland Relation: delta_HHH = {confounder.hosman_hansen_holland_delta:.3f}")
            if stability.vasicek_entropy_p is not None:
                rationale["key_evidence"].append(f"Vasicek Entropy Test: K_mn = {stability.vasicek_entropy_stat:.3f} (p = {stability.vasicek_entropy_p:.3e})")
            if stability.rousseeuw_zomeren_max_rdz is not None:
                rationale["key_evidence"].append(f"Rousseeuw-van Zomeren Robust Distance: Max RDZ = {stability.rousseeuw_zomeren_max_rdz:.3f}")
            if simpsons.berger_p_value is not None:
                rationale["key_evidence"].append(f"Berger IUT: Z_Berger = {simpsons.berger_stat_z:.2f} (p = {simpsons.berger_p_value:.3e})")
            if simpsons.shoemaker_p_value is not None:
                rationale["key_evidence"].append(f"Shoemaker Kurtosis Scale: Z_Shoe = {simpsons.shoemaker_stat_z:.2f} (p = {simpsons.shoemaker_p_value:.3e})")
            if confounder.rosenbaum_ef_gamma_crit is not None:
                rationale["key_evidence"].append(f"Rosenbaum Evidence Factors: Gamma_EF = {confounder.rosenbaum_ef_gamma_crit:.2f}")
            if confounder.ho_imai_convex_hull_overlap is not None:
                rationale["key_evidence"].append(f"Ho-Imai Support Overlap: OH_Hull = {confounder.ho_imai_convex_hull_overlap:.1%}")
            if stability.zhang_zk_p_value is not None:
                rationale["key_evidence"].append(f"Zhang Empirical Likelihood: Z_K = {stability.zhang_zk_stat:.3f} (p = {stability.zhang_zk_p_value:.3e})")
            if stability.bacon_max_distance is not None:
                rationale["key_evidence"].append(f"BACON Adaptive Outlier Metric: Max Dist = {stability.bacon_max_distance:.3f}")

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
