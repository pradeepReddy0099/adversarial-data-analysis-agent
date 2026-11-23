"""Devil's Advocate (Auditor) Agent: Autonomous statistical falsification batteries."""
from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import optimize, stats

from adversarial_agent.explorer import Claim
from adversarial_agent.profiling import benjamini_hochberg_fdr


@dataclass
class SubgroupBreakdown:
    group_name: str
    sample_size: int
    subgroup_statistic: float
    p_value: float
    reverses_sign: bool
    description: str
    effect_size: Optional[float] = None


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
    cmh_odds_ratio: Optional[float] = None
    cmh_p_value: Optional[float] = None
    cmh_chi2: Optional[float] = None
    cohens_d: Optional[float] = None
    is_odds_ratio_reversal: bool = False
    tarone_homogeneity_p: Optional[float] = None
    tarone_chi2: Optional[float] = None
    e_value: Optional[float] = None
    e_value_interpretation: Optional[str] = None
    i_squared: Optional[float] = None
    cochrans_q: Optional[float] = None
    begg_funnel_tau: Optional[float] = None
    begg_funnel_p: Optional[float] = None
    tau_squared: Optional[float] = None
    egger_test_t: Optional[float] = None
    egger_test_p: Optional[float] = None
    trim_and_fill_k0: Optional[int] = None
    trim_and_fill_adjusted_stat: Optional[float] = None
    hausman_stat: Optional[float] = None
    hausman_p: Optional[float] = None
    is_hausman_inconsistent: bool = False
    breslow_day_chi2: Optional[float] = None
    breslow_day_p: Optional[float] = None
    vuong_z: Optional[float] = None
    vuong_p: Optional[float] = None
    is_stratification_favored: bool = False
    higgins_h_squared: Optional[float] = None
    galbraith_outlier_count: int = 0
    galbraith_outlier_subgroups: List[str] = field(default_factory=list)
    woolf_chi2: Optional[float] = None
    woolf_p: Optional[float] = None
    is_woolf_heterogeneous: bool = False
    freeman_tukey_chi2: Optional[float] = None
    freeman_tukey_p: Optional[float] = None
    is_freeman_tukey_heterogeneous: bool = False
    kruskal_wallis_h: Optional[float] = None
    kruskal_wallis_p: Optional[float] = None
    is_kruskal_heterogeneous: bool = False
    peto_odds_ratio: Optional[float] = None
    peto_z_stat: Optional[float] = None
    peto_p_value: Optional[float] = None
    peto_heterogeneity_q: Optional[float] = None
    peto_heterogeneity_p: Optional[float] = None
    is_peto_heterogeneous: bool = False
    fligner_killeen_stat: Optional[float] = None
    fligner_killeen_p: Optional[float] = None
    is_fligner_heterogeneous: bool = False
    harbord_t_stat: Optional[float] = None
    harbord_p_value: Optional[float] = None
    is_harbord_asymmetric: bool = False
    brown_forsythe_stat: Optional[float] = None
    brown_forsythe_p: Optional[float] = None
    is_brown_forsythe_heterogeneous: bool = False
    peters_test_t: Optional[float] = None
    peters_test_p: Optional[float] = None
    is_peters_asymmetric: bool = False
    kish_design_effect: Optional[float] = None
    stratified_variance_ratio: Optional[float] = None
    rucker_test_t: Optional[float] = None
    rucker_test_p: Optional[float] = None
    is_rucker_asymmetric: bool = False
    levene_stat: Optional[float] = None
    levene_p: Optional[float] = None
    is_levene_heterogeneous: bool = False
    macaskill_test_t: Optional[float] = None
    macaskill_test_p: Optional[float] = None
    is_macaskill_asymmetric: bool = False
    bartlett_stat: Optional[float] = None
    bartlett_p: Optional[float] = None
    is_bartlett_heterogeneous: bool = False
    deeks_test_t: Optional[float] = None
    deeks_test_p: Optional[float] = None
    is_deeks_asymmetric: bool = False
    mood_stat_z: Optional[float] = None
    mood_p_value: Optional[float] = None
    is_mood_heterogeneous: bool = False
    thompson_sharp_t: Optional[float] = None
    thompson_sharp_p: Optional[float] = None
    is_thompson_asymmetric: bool = False
    ansari_bradley_stat: Optional[float] = None
    ansari_bradley_p: Optional[float] = None
    is_ansari_heterogeneous: bool = False
    moreno_test_t: Optional[float] = None
    moreno_test_p: Optional[float] = None
    is_moreno_asymmetric: bool = False
    siegel_tukey_stat: Optional[float] = None
    siegel_tukey_p: Optional[float] = None
    is_siegel_tukey_heterogeneous: bool = False
    sterne_test_t: Optional[float] = None
    sterne_test_p: Optional[float] = None
    is_sterne_asymmetric: bool = False
    klotz_scale_stat: Optional[float] = None
    klotz_scale_p: Optional[float] = None
    is_klotz_heterogeneous: bool = False
    lfk_index: Optional[float] = None
    is_lfk_asymmetric: bool = False
    conover_stat_z: Optional[float] = None
    conover_p_value: Optional[float] = None
    is_conover_heterogeneous: bool = False
    pet_test_t: Optional[float] = None
    pet_test_p: Optional[float] = None
    peese_test_t: Optional[float] = None
    peese_test_p: Optional[float] = None
    is_pet_asymmetric: bool = False
    sukhatme_stat_z: Optional[float] = None
    sukhatme_p_value: Optional[float] = None
    is_sukhatme_heterogeneous: bool = False
    tang_test_t: Optional[float] = None
    tang_test_p: Optional[float] = None
    is_tang_asymmetric: bool = False
    capon_stat_z: Optional[float] = None
    capon_p_value: Optional[float] = None
    is_capon_heterogeneous: bool = False
    pustejovsky_test_t: Optional[float] = None
    pustejovsky_test_p: Optional[float] = None
    is_pustejovsky_asymmetric: bool = False
    moses_stat_u: Optional[float] = None
    moses_stat_p: Optional[float] = None
    is_moses_heterogeneous: bool = False

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
    raw_r_squared: float = 0.0
    partial_r_squared: float = 0.0
    variance_explained_loss_pct: float = 0.0
    e_value: Optional[float] = None
    e_value_interpretation: Optional[str] = None
    top_confounder_vif: Optional[float] = None
    is_collinear: bool = False
    robustness_value: Optional[float] = None
    robustness_value_alpha: Optional[float] = None
    sobel_z: Optional[float] = None
    sobel_p: Optional[float] = None
    proportion_mediated: Optional[float] = None
    bayes_factor_bf10: Optional[float] = None
    bic_delta: Optional[float] = None
    bayes_factor_interpretation: Optional[str] = None
    ramsey_reset_f: Optional[float] = None
    ramsey_reset_p: Optional[float] = None
    is_reset_misspecified: bool = False
    oster_delta: Optional[float] = None
    cohens_f2: Optional[float] = None
    cinelli_hazlett_rcrit: Optional[float] = None
    frank_itcv: Optional[float] = None
    frank_observed_impact: Optional[float] = None
    is_itcv_exceeded: bool = False
    akaike_delta_aic: Optional[float] = None
    akaike_evidence_ratio: Optional[float] = None
    akaike_model_weight: Optional[float] = None
    rosenbaum_gamma_crit: Optional[float] = None
    rosenbaum_p_upper: Optional[float] = None
    is_rosenbaum_sensitive: bool = False
    manski_bound_low: Optional[float] = None
    manski_bound_high: Optional[float] = None
    manski_contains_zero: bool = False
    aet_selection_ratio: Optional[float] = None
    is_aet_vulnerable: bool = False
    imbens_manski_ci_low: Optional[float] = None
    imbens_manski_ci_high: Optional[float] = None
    imbens_manski_cn: Optional[float] = None
    imbens_manski_excludes_zero: bool = False
    mallows_cp: Optional[float] = None
    cp_bias_discrepancy: Optional[float] = None
    is_mallows_biased: bool = False
    davidson_mackinnon_j_stat: Optional[float] = None
    davidson_mackinnon_p: Optional[float] = None
    is_j_test_rejected: bool = False
    durbin_wu_hausman_f: Optional[float] = None
    durbin_wu_hausman_p: Optional[float] = None
    is_dwh_endogenous: bool = False
    stock_yogo_f_stat: Optional[float] = None
    is_weak_confounder: bool = False
    bkw_condition_index: Optional[float] = None
    bkw_max_variance_prop: Optional[float] = None
    is_bkw_collinear: bool = False
    sargan_j_stat: Optional[float] = None
    sargan_p_value: Optional[float] = None
    is_sargan_rejected: bool = False
    crump_overlap_ratio: Optional[float] = None
    is_crump_overlap_deficient: bool = False
    hansen_threshold_f: Optional[float] = None
    hansen_threshold_p: Optional[float] = None
    has_hansen_threshold_regime: bool = False
    hosmer_lemeshow_chi2: Optional[float] = None
    hosmer_lemeshow_p: Optional[float] = None
    is_propensity_miscalibrated: bool = False
    newey_west_hac_se: Optional[float] = None
    hac_se_ratio: Optional[float] = None
    is_hac_inflated: bool = False
    aipw_treatment_effect: Optional[float] = None
    aipw_discrepancy: Optional[float] = None
    is_aipw_confounded: bool = False
    leamer_bound_min: Optional[float] = None
    leamer_bound_max: Optional[float] = None
    is_leamer_fragile: bool = False
    dml_treatment_effect: Optional[float] = None
    dml_discrepancy: Optional[float] = None
    is_dml_confounded: bool = False
    cinelli_k_multiplier: Optional[float] = None
    is_cinelli_fragile: bool = False
    rubin_b_score: Optional[float] = None
    rubin_r_ratio: Optional[float] = None
    is_rubin_imbalanced: bool = False
    brunner_munzel_w: Optional[float] = None
    brunner_munzel_p: Optional[float] = None
    has_brunner_munzel_effect: bool = False
    smd_covariate_balance: Optional[float] = None
    mahalanobis_distance: Optional[float] = None
    is_smd_imbalanced: bool = False
    imbens_normalized_diff: Optional[float] = None
    is_imbens_diff_excessive: bool = False
    austin_variance_ratio: Optional[float] = None
    is_austin_imbalanced: bool = False
    stuart_prognostic_discrepancy: Optional[float] = None
    is_prognostic_discrepant: bool = False
    hansen_bowers_chi2: Optional[float] = None
    hansen_bowers_p: Optional[float] = None
    is_hansen_bowers_imbalanced: bool = False
    caliper_discordance_rate: Optional[float] = None
    caliper_threshold: Optional[float] = None
    is_caliper_discordant: bool = False
    gps_balance_r: Optional[float] = None
    gps_balance_t: Optional[float] = None
    gps_balance_p: Optional[float] = None
    is_gps_imbalanced: bool = False
    austin_ks_stat: Optional[float] = None
    austin_ks_p_value: Optional[float] = None
    is_austin_ks_imbalanced: bool = False
    l1_multivariate_imbalance: Optional[float] = None
    is_l1_imbalanced: bool = False
    rubin_parallelism_f: Optional[float] = None
    rubin_parallelism_p: Optional[float] = None
    is_rubin_nonparallel: bool = False
    eqq_mean_diff: Optional[float] = None
    eqq_max_diff: Optional[float] = None
    is_eqq_imbalanced: bool = False
    lin_interaction_t: Optional[float] = None
    lin_interaction_p: Optional[float] = None
    is_lin_heterogeneous: bool = False

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
    bootstrap_std_err: float = 0.0
    execution_time_ms: float = 0.0
    spearman_rho: Optional[float] = None
    rank_discrepancy: Optional[float] = None
    max_leverage: Optional[float] = None
    high_leverage_points: int = 0
    sign_invariance_prob: float = 1.0
    max_cooks_distance: Optional[float] = None
    cooks_outlier_count: int = 0
    cooks_trimmed_statistic: Optional[float] = None
    cooks_statistic_shift: Optional[float] = None
    max_dffits: Optional[float] = None
    dffits_outlier_count: int = 0
    permutation_p_value: Optional[float] = None
    breusch_pagan_lm: Optional[float] = None
    breusch_pagan_p: Optional[float] = None
    is_heteroscedastic: bool = False
    quantile_discrepancy_index: Optional[float] = None
    qdi_flag: bool = False
    max_dfbetas: Optional[float] = None
    dfbetas_outlier_count: int = 0
    jarque_bera_stat: Optional[float] = None
    jarque_bera_p: Optional[float] = None
    is_non_normal: bool = False
    durbin_watson_stat: Optional[float] = None
    has_serial_correlation: bool = False
    hc3_robust_se: Optional[float] = None
    hc3_robust_p: Optional[float] = None
    hc3_significance_lost: bool = False
    theil_sen_slope: Optional[float] = None
    theil_sen_discrepancy: Optional[float] = None
    bca_ci_low: Optional[float] = None
    bca_ci_high: Optional[float] = None
    bca_zero_crossing: bool = False
    white_lm_stat: Optional[float] = None
    white_lm_p: Optional[float] = None
    is_white_heteroscedastic: bool = False
    chow_f_stat: Optional[float] = None
    chow_p_value: Optional[float] = None
    has_structural_break: bool = False
    type_s_error_prob: Optional[float] = None
    type_m_exaggeration_ratio: Optional[float] = None
    retrodesign_power: Optional[float] = None
    is_type_m_vulnerable: bool = False
    andrews_sup_w: Optional[float] = None
    andrews_break_fraction: Optional[float] = None
    andrews_p_value: Optional[float] = None
    has_unknown_structural_break: bool = False
    cusum_stat: Optional[float] = None
    cusum_p_approx: Optional[float] = None
    has_cusum_instability: bool = False
    koenker_lm_stat: Optional[float] = None
    koenker_p_value: Optional[float] = None
    is_koenker_heteroscedastic: bool = False
    harvey_collier_t: Optional[float] = None
    harvey_collier_p: Optional[float] = None
    has_harvey_collier_nonlinearity: bool = False
    huber_m_slope: Optional[float] = None
    huber_slope_discrepancy: Optional[float] = None
    is_huber_outlier_distorted: bool = False
    cusumsq_stat: Optional[float] = None
    cusumsq_critical_val: Optional[float] = None
    has_cusumsq_instability: bool = False
    lts_slope: Optional[float] = None
    lts_slope_discrepancy: Optional[float] = None
    is_lts_outlier_distorted: bool = False
    siegel_slope: Optional[float] = None
    siegel_slope_discrepancy: Optional[float] = None
    is_siegel_outlier_distorted: bool = False
    breusch_godfrey_lm: Optional[float] = None
    breusch_godfrey_p: Optional[float] = None
    has_higher_order_autocorrelation: bool = False
    runs_test_z: Optional[float] = None
    runs_test_p: Optional[float] = None
    has_runs_nonrandomness: bool = False
    kendall_tau_b: Optional[float] = None
    kendall_discrepancy: Optional[float] = None
    is_kendall_distorted: bool = False
    goldfeld_quandt_f: Optional[float] = None
    goldfeld_quandt_p: Optional[float] = None
    has_goldfeld_quandt_heteroskedasticity: bool = False
    shapiro_francia_w: Optional[float] = None
    shapiro_francia_p: Optional[float] = None
    is_shapiro_francia_non_normal: bool = False
    dagostino_pearson_k2: Optional[float] = None
    dagostino_pearson_p: Optional[float] = None
    is_dagostino_non_normal: bool = False
    glejser_t_stat: Optional[float] = None
    glejser_p_value: Optional[float] = None
    has_glejser_heteroskedasticity: bool = False
    anderson_darling_stat: Optional[float] = None
    anderson_darling_crit: Optional[float] = None
    is_anderson_non_normal: bool = False
    harvey_lm_stat: Optional[float] = None
    harvey_lm_p: Optional[float] = None
    has_harvey_heteroskedasticity: bool = False
    lilliefors_stat: Optional[float] = None
    lilliefors_p_value: Optional[float] = None
    is_lilliefors_non_normal: bool = False
    park_t_stat: Optional[float] = None
    park_p_value: Optional[float] = None
    has_park_heteroskedasticity: bool = False
    cramer_von_mises_w2: Optional[float] = None
    cramer_von_mises_p: Optional[float] = None
    is_cvm_non_normal: bool = False
    cook_weisberg_score: Optional[float] = None
    cook_weisberg_p: Optional[float] = None
    has_cook_weisberg_heteroskedasticity: bool = False
    kuiper_v_stat: Optional[float] = None
    kuiper_v_star: Optional[float] = None
    kuiper_p_value: Optional[float] = None
    is_kuiper_non_normal: bool = False
    max_covratio_dev: Optional[float] = None
    covratio_outlier_count: int = 0
    is_covratio_influential: bool = False
    hosking_l_stat: Optional[float] = None
    hosking_l_p: Optional[float] = None
    is_hosking_non_normal: bool = False
    max_atkinson_distance: Optional[float] = None
    atkinson_outlier_count: int = 0
    is_atkinson_influential: bool = False
    shapiro_wilk_w: Optional[float] = None
    shapiro_wilk_p: Optional[float] = None
    is_shapiro_wilk_non_normal: bool = False
    max_hadi_influence: Optional[float] = None
    hadi_outlier_count: int = 0
    is_hadi_influential: bool = False

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


def calculate_e_value(rr: float) -> tuple[float, str]:
    """Calculates the VanderWeele-Ding E-value for unmeasured confounding.
    
    E-value = RR + sqrt(RR * (RR - 1)) for RR >= 1.
    Quantifies the minimum strength of association an unmeasured confounder must have
    with both the exposure and outcome to fully explain away the observed association.
    """
    if rr <= 0 or np.isnan(rr):
        return 1.0, "Undefined (non-positive or NaN effect)"
    rr_star = rr if rr >= 1.0 else 1.0 / rr
    e_val = float(rr_star + np.sqrt(max(0.0, rr_star * (rr_star - 1.0))))
    if e_val < 1.3:
        interp = f"High sensitivity (E-value = {e_val:.2f}): A weak unmeasured confounder (RR >= {e_val:.2f}) could explain away the effect."
    elif e_val < 2.5:
        interp = f"Moderate sensitivity (E-value = {e_val:.2f}): An unmeasured confounder with RR >= {e_val:.2f} with both exposure and outcome would be needed to nullify this association."
    else:
        interp = f"Strong robustness (E-value = {e_val:.2f}): An unmeasured confounder would need an association of RR >= {e_val:.2f} with both exposure and outcome to fully explain away this finding."
    return e_val, interp


def breslow_day_homogeneity_test(
    tables_2x2: List[tuple[float, float, float, float]], common_or: float
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Calculates both classic Breslow-Day (1980) and Tarone's (1985) test for odds ratio homogeneity.
    
    H0: The odds ratio is homogeneous across all K strata.
    Rejection (p < 0.05) indicates effect modification (interaction) across strata.
    Returns (bd_stat, bd_p, tarone_stat, tarone_p).
    """
    K = len(tables_2x2)
    if K < 2 or common_or <= 0 or np.isnan(common_or):
        return None, None, None, None
    
    bd_stat = 0.0
    sum_diff = 0.0
    sum_var = 0.0
    valid_k = 0

    for a, b, c, d in tables_2x2:
        n1 = a + b
        n2 = c + d
        m1 = a + c
        m2 = b + d
        N = n1 + n2
        if N <= 0 or n1 <= 0 or n2 <= 0 or m1 <= 0 or m2 <= 0:
            continue
        
        A_q = common_or - 1.0
        B_q = -(common_or * (n1 + m1) + (N - n1 - m1)) if abs(A_q) > 1e-7 else -N
        C_q = common_or * n1 * m1
        
        if abs(A_q) < 1e-7:
            mu = (n1 * m1) / N
        else:
            disc = max(0.0, B_q**2 - 4 * A_q * C_q)
            low_bound = max(0.0, n1 + m1 - N)
            high_bound = min(n1, m1)
            r1 = (-B_q - np.sqrt(disc)) / (2 * A_q)
            r2 = (-B_q + np.sqrt(disc)) / (2 * A_q)
            if low_bound - 1e-5 <= r1 <= high_bound + 1e-5:
                mu = r1
            elif low_bound - 1e-5 <= r2 <= high_bound + 1e-5:
                mu = r2
            else:
                mu = (n1 * m1) / N
        
        d1 = mu
        d2 = n1 - mu
        d3 = m1 - mu
        d4 = N - n1 - m1 + mu
        if d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0:
            var_mu = 1.0 / (1.0 / d1 + 1.0 / d2 + 1.0 / d3 + 1.0 / d4)
            bd_stat += ((a - mu) ** 2) / max(1e-9, var_mu)
            sum_diff += (a - mu)
            sum_var += var_mu
            valid_k += 1

    if valid_k < 2 or sum_var <= 1e-9:
        return None, None, None, None
    
    bd_stat_val = float(max(0.0, bd_stat))
    bd_p = float(stats.chi2.sf(bd_stat_val, df=valid_k - 1))
    tarone_stat = float(max(0.0, bd_stat - (sum_diff ** 2) / sum_var))
    tarone_p = float(stats.chi2.sf(tarone_stat, df=valid_k - 1))
    return bd_stat_val, bd_p, tarone_stat, tarone_p


def tarone_homogeneity_test(tables_2x2: List[tuple[float, float, float, float]], common_or: float) -> tuple[Optional[float], Optional[float]]:
    """Calculates Tarone's modification of the Breslow-Day test for odds ratio homogeneity."""
    _, _, tarone_stat, tarone_p = breslow_day_homogeneity_test(tables_2x2, common_or)
    return tarone_stat, tarone_p


def calculate_cinelli_hazlett_rv(partial_r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Calculates Cinelli & Hazlett (2020) Robustness Values (RV_q=1 and RV_alpha).
    
    RV_q=1: Minimum strength of association (partial R2) an unmeasured confounder must have
    with both exposure and outcome to reduce the estimated effect to zero.
    RV_alpha: Minimum strength of association to nullify statistical significance at level alpha.
    """
    if n <= 3 or np.isnan(partial_r):
        return 0.0, 0.0
    
    r_abs = min(0.9999, abs(partial_r))
    df = max(1, n - 3)
    f = r_abs / np.sqrt(max(1e-9, 1.0 - r_abs ** 2))
    
    # RV for point estimate nullification (q = 1)
    rv_point = float(r_abs)
    
    # RV for statistical significance nullification
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    f_crit = t_crit / np.sqrt(df)
    f_alpha = f - f_crit
    
    if f_alpha <= 0.0:
        rv_alpha = 0.0
    else:
        rv_alpha = float(f_alpha / np.sqrt(f_alpha ** 2 + 1.0))
        
    return rv_point, rv_alpha


def sobel_mediation_test(r_xy: float, r_xz: float, r_yz: float, n: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculates the Sobel test for mediation vs. confounding.
    
    Tests whether the indirect pathway X -> Z -> Y is statistically significant:
    z_sobel = (a * b) / sqrt(b^2 * SE_a^2 + a^2 * SE_b^2).
    Returns (z_statistic, p_value, proportion_mediated_pct).
    """
    if n < 10 or abs(r_xz) >= 0.999 or np.isnan(r_xy) or np.isnan(r_xz) or np.isnan(r_yz):
        return None, None, None
        
    a = r_xz
    se_a = np.sqrt(max(1e-9, (1.0 - r_xz ** 2) / max(1, n - 2)))
    
    denom_b = max(1e-9, 1.0 - r_xz ** 2)
    b = (r_yz - r_xy * r_xz) / denom_b
    
    r2_yz_x = min(0.999, max(0.0, (r_xy ** 2 + r_yz ** 2 - 2.0 * r_xy * r_yz * r_xz) / denom_b))
    se_b = np.sqrt(max(1e-9, (1.0 - r2_yz_x) / (max(1, n - 3) * denom_b)))
    
    indirect = a * b
    se_sobel = np.sqrt(b ** 2 * se_a ** 2 + a ** 2 * se_b ** 2)
    
    if se_sobel <= 1e-9:
        return None, None, None
        
    z_stat = float(indirect / se_sobel)
    p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
    
    prop_med = float((indirect / r_xy) * 100.0) if abs(r_xy) > 1e-6 else 0.0
    return z_stat, p_val, prop_med


def higgins_i_squared_heterogeneity(subgroup_stats: List[float], subgroup_weights: List[int], is_correlation: bool = True) -> tuple[Optional[float], Optional[float]]:
    """Calculates Cochran's Q and Higgins & Thompson I^2 inconsistency index across strata.
    
    I^2 = max(0, (Q - df) / Q) * 100%. Quantifies the percentage of total variation across
    subgroups due to true heterogeneity rather than sampling error.
    """
    K = len(subgroup_stats)
    if K < 2:
        return None, None
        
    df = K - 1
    stats_arr = np.array(subgroup_stats, dtype=float)
    weights_arr = np.array(subgroup_weights, dtype=float)
    
    if is_correlation:
        clipped = np.clip(stats_arr, -0.999, 0.999)
        z_vals = 0.5 * np.log((1.0 + clipped) / (1.0 - clipped))
        w = np.maximum(1.0, weights_arr - 3.0)
        mean_z = np.average(z_vals, weights=w)
        q_stat = float(np.sum(w * (z_vals - mean_z) ** 2))
    else:
        w = weights_arr
        mean_stat = np.average(stats_arr, weights=w)
        norm_var = max(1e-6, float(np.var(stats_arr, ddof=1))) if K > 2 else 1.0
        q_stat = float(np.sum(w * (stats_arr - mean_stat) ** 2) / (np.mean(w) * norm_var) * df)
        
    i_sq = max(0.0, float((q_stat - df) / max(1e-9, q_stat) * 100.0))
    return q_stat, i_sq


def begg_funnel_asymmetry_test(subgroup_stats: List[float], subgroup_weights: List[int]) -> tuple[Optional[float], Optional[float]]:
    """Calculates Begg & Mazumdar (1994) rank correlation test for funnel plot asymmetry across strata.
    
    Evaluates whether subgroup effect sizes systematically correlate with subgroup sample sizes or standard errors.
    Tau != 0 with p < 0.05 indicates small-subgroup effect distortion or selective reporting.
    """
    K = len(subgroup_stats)
    if K < 3:
        return None, None
    se_arr = 1.0 / np.sqrt(np.maximum(1.0, np.array(subgroup_weights, dtype=float) - 3.0))
    tau, p_val = stats.kendalltau(subgroup_stats, se_arr)
    if np.isnan(tau) or np.isnan(p_val):
        return None, None
    return float(tau), float(p_val)


def calculate_bayes_factor_bic(r_xy: float, r_xz: float, r_yz: float, n: int) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Calculates Kass & Raftery (1995) Bayes Factor BF_10 and Delta BIC between unconditioned and confounder models.
    
    M0 (bivariate claim): Y ~ X (2 params: intercept, slope)
    M1 (confounded model): Y ~ X + Z (3 params: intercept, slope X, slope Z)
    Delta BIC = BIC_0 - BIC_1.
    BF_10 = exp(Delta BIC / 2).
    """
    if n < 10 or abs(r_xz) >= 0.999 or np.isnan(r_xy) or np.isnan(r_xz) or np.isnan(r_yz):
        return None, None, None
        
    denom = max(1e-9, 1.0 - r_xz ** 2)
    r2_full = min(0.9999, max(0.0, (r_xy ** 2 + r_yz ** 2 - 2.0 * r_xy * r_yz * r_xz) / denom))
    r2_null = min(0.9999, max(0.0, r_xy ** 2))
    
    rss_null = max(1e-9, 1.0 - r2_null)
    rss_full = max(1e-9, 1.0 - r2_full)
    
    bic_0 = n * np.log(rss_null) + 2.0 * np.log(n)
    bic_1 = n * np.log(rss_full) + 3.0 * np.log(n)
    
    delta_bic = float(bic_0 - bic_1)
    clipped_half_db = min(50.0, max(-50.0, delta_bic / 2.0))
    bf10 = float(np.exp(clipped_half_db))
    
    if bf10 < 1.0:
        interp = f"Negative evidence for confounding (BF10 = {bf10:.2f} < 1.0, favors bivariate claim)"
    elif bf10 < 3.2:
        interp = f"Weak evidence for confounding (BF10 = {bf10:.2f})"
    elif bf10 < 10.0:
        interp = f"Positive evidence for confounding (BF10 = {bf10:.2f})"
    elif bf10 < 100.0:
        interp = f"Strong evidence for confounding (BF10 = {bf10:.2f})"
    else:
        interp = f"Decisive evidence for confounding (BF10 = {bf10:.2f} >= 100.0, confounder dominates)"
        
    return bf10, delta_bic, interp


def breusch_pagan_test(x: np.ndarray, residuals: np.ndarray) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Breusch-Pagan Lagrange Multiplier test for heteroscedasticity.
    
    H0: Homoscedastic error variance across predictor levels.
    Rejection (p < 0.05) indicates heteroscedasticity, invalidating standard OLS standard errors.
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9:
        return None, None, False
        
    e2 = residuals ** 2
    e2_mean = np.mean(e2)
    if e2_mean < 1e-12:
        return 0.0, 1.0, False
        
    x_c = x - np.mean(x)
    x_var = np.sum(x_c ** 2)
    if x_var < 1e-9:
        return None, None, False
        
    slope_e2 = np.sum(x_c * (e2 - e2_mean)) / x_var
    fit_e2 = e2_mean + slope_e2 * x_c
    ss_reg = np.sum((fit_e2 - e2_mean) ** 2)
    ss_tot = np.sum((e2 - e2_mean) ** 2)
    
    r2_e2 = ss_reg / max(1e-12, ss_tot)
    lm_stat = float(n * r2_e2)
    p_val = float(stats.chi2.sf(lm_stat, df=1))
    is_hetero = bool(p_val < 0.05)
    return lm_stat, p_val, is_hetero


def calculate_quantile_discrepancy(x: np.ndarray, y: np.ndarray) -> tuple[Optional[float], bool]:
    """Computes the Quantile Discrepancy Index (QDI) across the 25th, 50th, and 75th percentiles.
    
    QDI = |beta_0.75 - beta_0.25| / max(1e-4, |beta_0.50|).
    Detects asymmetric distributional response or heavy-tailed skew across quantiles.
    """
    n = len(x)
    if n < 20 or np.std(x) < 1e-9:
        return None, False
        
    b1_ols = float(np.cov(x, y)[0, 1] / max(1e-9, np.var(x)))
    b0_ols = float(np.mean(y) - b1_ols * np.mean(x))
    
    def q_loss(params, q):
        u = y - (params[0] + params[1] * x)
        return np.sum(np.where(u >= 0, q * u, (q - 1.0) * u))
        
    try:
        r25 = optimize.minimize(q_loss, [b0_ols, b1_ols], args=(0.25,), method='Nelder-Mead', options={'maxiter': 80})
        r50 = optimize.minimize(q_loss, [b0_ols, b1_ols], args=(0.50,), method='Nelder-Mead', options={'maxiter': 80})
        r75 = optimize.minimize(q_loss, [b0_ols, b1_ols], args=(0.75,), method='Nelder-Mead', options={'maxiter': 80})
        
        b25 = r25.x[1]
        b50 = r50.x[1]
        b75 = r75.x[1]
        
        qdi = float(abs(b75 - b25) / max(1e-4, abs(b50)))
        flag = bool(qdi > 0.50)
        return qdi, flag
    except Exception:
        return None, False


def dersimonian_laird_tau_squared(
    subgroup_stats: List[float],
    subgroup_weights: List[int],
    cochrans_q: Optional[float],
    is_correlation: bool = True
) -> Optional[float]:
    """Calculates DerSimonian & Laird (1986) between-stratum variance tau^2.
    
    tau^2 = max(0, (Q - (K - 1)) / (sum(w) - sum(w^2) / sum(w)))
    Quantifies absolute excess between-stratum variance beyond random sampling noise.
    """
    K = len(subgroup_stats)
    if K < 2 or cochrans_q is None:
        return None
    weights_arr = np.array(subgroup_weights, dtype=float)
    if is_correlation:
        w = np.maximum(1.0, weights_arr - 3.0)
    else:
        w = np.maximum(1.0, weights_arr)
    sum_w = float(np.sum(w))
    sum_w_sq = float(np.sum(w ** 2))
    denom = sum_w - (sum_w_sq / max(1e-9, sum_w))
    if denom <= 1e-9:
        return 0.0
    df = K - 1
    tau_sq = max(0.0, float((cochrans_q - df) / denom))
    return float(tau_sq)


def eggers_funnel_regression_test(
    subgroup_stats: List[float],
    subgroup_weights: List[int],
    is_correlation: bool = True
) -> tuple[Optional[float], Optional[float]]:
    """Calculates Egger et al. (1997) linear regression test for funnel plot asymmetry.
    
    Regresses standardized effect (z_k = theta_k / SE_k) on precision (prec_k = 1 / SE_k):
    z_k = alpha + beta * prec_k + epsilon_k.
    H0: alpha = 0 (symmetric funnel). An intercept alpha != 0 with p < 0.05 indicates
    small-subgroup effect distortion or selection asymmetry.
    """
    K = len(subgroup_stats)
    if K < 3:
        return None, None
    weights_arr = np.array(subgroup_weights, dtype=float)
    if is_correlation:
        clipped = np.clip(np.array(subgroup_stats, dtype=float), -0.999, 0.999)
        theta = 0.5 * np.log((1.0 + clipped) / (1.0 - clipped))
        se = 1.0 / np.sqrt(np.maximum(1.0, weights_arr - 3.0))
    else:
        theta = np.array(subgroup_stats, dtype=float)
        se = 1.0 / np.sqrt(np.maximum(1.0, weights_arr))
    
    prec = 1.0 / np.maximum(1e-9, se)
    z = theta / np.maximum(1e-9, se)
    
    p_prec = prec - np.mean(prec)
    var_prec = np.sum(p_prec ** 2)
    if var_prec < 1e-9:
        return None, None
    
    b_slope = np.sum(p_prec * (z - np.mean(z))) / var_prec
    alpha = np.mean(z) - b_slope * np.mean(prec)
    
    residuals = z - (alpha + b_slope * prec)
    df_res = max(1, K - 2)
    mse = np.sum(residuals ** 2) / df_res
    se_alpha = np.sqrt(max(1e-12, mse * (1.0 / K + (np.mean(prec) ** 2) / var_prec)))
    
    t_stat = float(alpha / max(1e-9, se_alpha))
    p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_res))
    return float(t_stat), float(p_val)


def ramsey_reset_test(x: np.ndarray, z: np.ndarray, y: np.ndarray) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Ramsey's RESET test for functional form misspecification and omitted non-linear confounding.
    
    Restricted model: y ~ 1 + x + z
    Unrestricted model: y ~ 1 + x + z + y_hat^2 + y_hat^3
    H0: gamma_1 = gamma_2 = 0 (correct linear specification).
    Rejection (p < 0.05) indicates non-linear confounder bias or omitted polynomial effects.
    """
    n = len(y)
    if n < 20:
        return None, None, False
    try:
        X_rest = np.column_stack([np.ones(n), x, z])
        beta_rest, _, _, _ = np.linalg.lstsq(X_rest, y, rcond=None)
        y_hat = X_rest @ beta_rest
        ssr_rest = float(np.sum((y - y_hat) ** 2))
        
        y_hat2 = y_hat ** 2
        y_hat3 = y_hat ** 3
        X_unrest = np.column_stack([np.ones(n), x, z, y_hat2, y_hat3])
        beta_unrest, _, _, _ = np.linalg.lstsq(X_unrest, y, rcond=None)
        y_hat_unrest = X_unrest @ beta_unrest
        ssr_unrest = float(np.sum((y - y_hat_unrest) ** 2))
        
        df_num = 2
        df_denom = max(1, n - 5)
        diff_ssr = max(0.0, ssr_rest - ssr_unrest)
        f_stat = float((diff_ssr / df_num) / max(1e-9, ssr_unrest / df_denom))
        p_val = float(stats.f.sf(f_stat, df_num, df_denom))
        is_misspecified = bool(p_val < 0.05)
        return float(f_stat), float(p_val), is_misspecified
    except Exception:
        return None, None, False


def calculate_oster_delta(r_xy: float, r_xz: float, r_yz: float, n: int) -> Optional[float]:
    """Calculates Emily Oster's (2019) coefficient stability bound delta* for selection on unobservables.
    
    Quantifies the proportional degree of selection on unobservables relative to observables
    required to produce an effect of zero. Under R_max = min(1.0, 1.3 * R_controlled^2):
    delta* = beta_controlled * (R_max - R_uncontrolled) / ((beta_uncontrolled - beta_controlled) * (R_controlled - R_uncontrolled)).
    |delta*| >= 1.0 indicates robustness (unobservables must be stronger than observables to overturn).
    """
    if n < 15 or abs(r_xz) >= 0.999 or np.isnan(r_xy) or np.isnan(r_xz) or np.isnan(r_yz):
        return None
    denom = max(1e-9, 1.0 - r_xz ** 2)
    beta_0 = r_xy
    r2_0 = min(0.999, r_xy ** 2)
    
    beta_1 = (r_xy - r_xz * r_yz) / denom
    r2_1 = min(0.999, max(0.0, (r_xy ** 2 + r_yz ** 2 - 2.0 * r_xy * r_yz * r_xz) / denom))
    
    diff_beta = beta_0 - beta_1
    if abs(diff_beta) < 1e-6:
        return 99.9
        
    r_max = min(1.0, 1.3 * r2_1)
    delta_denom = diff_beta * max(1e-6, r2_1 - r2_0)
    if abs(delta_denom) < 1e-9:
        return 99.9
        
    delta_star = float((beta_1 * (r_max - r2_0)) / delta_denom)
    return round(delta_star, 2)


def jarque_bera_normality_test(residuals: np.ndarray) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Jarque-Bera goodness-of-fit test for residual normality.
    
    JB = (n / 6) * (S^2 + (K - 3)^2 / 4) ~ chi^2(2)
    H0: Residuals are normally distributed.
    Rejection (p < 0.05) indicates heavy-tailed distribution or skewness violating Gauss-Markov assumptions.
    """
    n = len(residuals)
    if n < 15 or np.std(residuals) < 1e-9:
        return None, None, False
    try:
        jb_stat, p_val = stats.jarque_bera(residuals)
        is_non_norm = bool(p_val < 0.05)
        return float(jb_stat), float(p_val), is_non_norm
    except Exception:
        return None, None, False


def duval_tweedie_trim_and_fill(
    subgroup_stats: List[float],
    subgroup_weights: List[int],
    is_correlation: bool = True,
) -> tuple[int, float]:
    """Calculates Duval & Tweedie (2000) Trim and Fill algorithm for publication / stratum selection asymmetry.
    
    Estimates the number of suppressed/missing strata (k0) on the funnel side with fewer observations,
    and returns (k0, adjusted_pooled_statistic).
    """
    K = len(subgroup_stats)
    if K < 3:
        avg_stat = float(np.average(subgroup_stats, weights=subgroup_weights)) if K > 0 else 0.0
        return 0, avg_stat

    weights_arr = np.array(subgroup_weights, dtype=float)
    if is_correlation:
        clipped = np.clip(np.array(subgroup_stats, dtype=float), -0.999, 0.999)
        theta = 0.5 * np.log((1.0 + clipped) / (1.0 - clipped))
        w = np.maximum(1.0, weights_arr - 3.0)
    else:
        theta = np.array(subgroup_stats, dtype=float)
        w = np.maximum(1.0, weights_arr)

    theta_curr = float(np.average(theta, weights=w))
    k0 = 0

    for _ in range(10):
        diffs = theta - theta_curr
        abs_diffs = np.abs(diffs)
        ranks = stats.rankdata(abs_diffs)
        
        pos_rank_sum = float(np.sum(ranks[diffs > 0]))
        neg_rank_sum = float(np.sum(ranks[diffs < 0]))
        
        t_stat = max(pos_rank_sum, neg_rank_sum)
        l0_est = max(0, int(round((4.0 * t_stat - K * (K + 1.0)) / max(1.0, 2.0 * K - 1.0))))
        
        if l0_est == k0:
            break
        k0 = min(K, l0_est)
        
        if k0 > 0:
            if pos_rank_sum > neg_rank_sum:
                keep_idx = np.argsort(theta)[: (K - k0)]
            else:
                keep_idx = np.argsort(theta)[k0:]
            if len(keep_idx) > 0:
                theta_curr = float(np.average(theta[keep_idx], weights=w[keep_idx]))
            else:
                break
        else:
            break

    if k0 > 0:
        if pos_rank_sum > neg_rank_sum:
            trimmed_theta = np.sort(theta)[-k0:]
            trimmed_w = np.sort(w)[-k0:]
        else:
            trimmed_theta = np.sort(theta)[:k0]
            trimmed_w = np.sort(w)[:k0]
        imputed_theta = 2.0 * theta_curr - trimmed_theta
        all_theta = np.concatenate([theta, imputed_theta])
        all_w = np.concatenate([w, trimmed_w])
        adjusted_theta = float(np.average(all_theta, weights=all_w))
    else:
        adjusted_theta = theta_curr

    if is_correlation:
        adjusted_stat = float(np.tanh(adjusted_theta))
    else:
        adjusted_stat = float(adjusted_theta)

    return int(k0), float(adjusted_stat)


def hausman_specification_test(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    confounder_col: str,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Hausman's (1978) specification test for group-level endogeneity and stratification consistency.
    
    H0: Difference in coefficients between Within-Stratum Fixed Effects (FE) and Pooled OLS is not systematic.
    Rejection (p < 0.05) indicates that subgroup-level heterogeneity causes endogeneity,
    proving that pooled estimates are inconsistent and confirming Simpson's Paradox / confounding.
    """
    sub = df[[source_col, target_col, confounder_col]].dropna()
    if len(sub) < 20 or sub[confounder_col].nunique() < 2:
        return None, None, False

    if pd.api.types.is_numeric_dtype(sub[source_col]):
        x = pd.to_numeric(sub[source_col], errors='coerce')
    else:
        x = pd.Series(pd.Categorical(sub[source_col]).codes, index=sub.index, dtype=float)

    if pd.api.types.is_numeric_dtype(sub[target_col]):
        y = pd.to_numeric(sub[target_col], errors='coerce')
    else:
        y = pd.Series(pd.Categorical(sub[target_col]).codes, index=sub.index, dtype=float)

    valid_m = ~(x.isna() | y.isna())
    if valid_m.sum() < 20:
        return None, None, False

    x_arr = x[valid_m].to_numpy(dtype=float)
    y_arr = y[valid_m].to_numpy(dtype=float)
    groups = sub[confounder_col][valid_m].to_numpy()
    n = len(x_arr)

    # 1. Pooled OLS
    x_c = x_arr - np.mean(x_arr)
    y_c = y_arr - np.mean(y_arr)
    var_x = np.sum(x_c ** 2)
    if var_x < 1e-9:
        return None, None, False

    b_pooled = float(np.sum(x_c * y_c) / var_x)
    res_pooled = y_c - b_pooled * x_c
    s2_pooled = float(np.sum(res_pooled ** 2) / max(1, n - 2))
    var_b_pooled = s2_pooled / var_x

    # 2. Fixed Effects (Within-stratum mean centering)
    x_fe = np.empty_like(x_arr)
    y_fe = np.empty_like(y_arr)
    uniq_groups = np.unique(groups)
    k_groups = len(uniq_groups)

    for g in uniq_groups:
        mask = (groups == g)
        x_fe[mask] = x_arr[mask] - np.mean(x_arr[mask])
        y_fe[mask] = y_arr[mask] - np.mean(y_arr[mask])

    var_x_fe = np.sum(x_fe ** 2)
    if var_x_fe < 1e-9:
        return None, None, False

    b_fe = float(np.sum(x_fe * y_fe) / var_x_fe)
    res_fe = y_fe - b_fe * x_fe
    df_fe = max(1, n - k_groups - 1)
    s2_fe = float(np.sum(res_fe ** 2) / df_fe)
    var_b_fe = s2_fe / var_x_fe

    diff_b = b_fe - b_pooled
    var_diff = abs(var_b_fe - var_b_pooled)
    if var_diff < 1e-12:
        return 0.0, 1.0, False

    h_stat = float((diff_b ** 2) / var_diff)
    p_val = float(stats.chi2.sf(h_stat, df=1))
    is_inconsistent = bool(p_val < 0.05)

    return h_stat, p_val, is_inconsistent


def calculate_cohens_f2(r2_full: float, r2_confounder: float) -> float:
    """Calculates Cohen's f^2 effect size for exposure conditional on confounder.
    
    f^2 = (R^2_full - R^2_confounder) / (1 - R^2_full)
    f^2 < 0.02: negligible unique contribution; >= 0.02: small; >= 0.15: medium; >= 0.35: large.
    """
    denom = max(1e-9, 1.0 - min(0.9999, r2_full))
    num = max(0.0, r2_full - r2_confounder)
    return float(num / denom)


def calculate_cinelli_hazlett_rcrit(partial_r: float) -> float:
    """Calculates Cinelli & Hazlett (2020) critical confounder boundary R2_crit.
    
    The symmetric explanatory power an unmeasured confounder must share with BOTH
    treatment and outcome (R2_Y~Z|X = R2_X~Z = R2_crit) to completely explain away the effect:
    R2_crit = |partial_r| / (1 + |partial_r|).
    """
    r_abs = min(0.9999, abs(partial_r))
    return float(r_abs / (1.0 + r_abs))


def durbin_watson_test(residuals: np.ndarray) -> tuple[Optional[float], bool]:
    """Calculates the Durbin-Watson statistic for first-order serial correlation in residuals.
    
    DW = sum((e_t - e_{t-1})^2) / sum(e_t^2).
    DW ~ 2.0: no autocorrelation. DW < 1.5 indicates significant positive serial clustering.
    """
    n = len(residuals)
    if n < 10:
        return None, False
    diff = np.diff(residuals)
    ss_diff = float(np.sum(diff ** 2))
    ss_tot = float(np.sum(residuals ** 2))
    if ss_tot < 1e-12:
        return 2.0, False
    dw = float(ss_diff / ss_tot)
    has_serial = bool(dw < 1.5 or dw > 2.5)
    return dw, has_serial


def mackinnon_white_hc3_test(
    x: np.ndarray,
    y: np.ndarray,
    residuals: np.ndarray,
    h_ii: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates MacKinnon & White (1985) HC3 jackknife-weighted heteroscedasticity-consistent test.
    
    Var_HC3(b1) = sum((x_i - x_bar)^2 * (e_i / (1 - h_ii))^2) / (sum((x_i - x_bar)^2))^2.
    Returns (se_hc3, p_hc3, significance_lost).
    """
    n = len(x)
    if n < 15:
        return None, None, False
    x_c = x - np.mean(x)
    x_ss = float(np.sum(x_c ** 2))
    if x_ss < 1e-9:
        return None, None, False

    denom_h = np.maximum(1e-4, 1.0 - h_ii)
    e_hc3 = residuals / denom_h
    var_hc3 = float(np.sum((x_c ** 2) * (e_hc3 ** 2)) / (x_ss ** 2))
    se_hc3 = float(np.sqrt(max(1e-12, var_hc3)))

    b1 = float(np.sum(x_c * (y - np.mean(y))) / x_ss)
    t_hc3 = float(b1 / max(1e-9, se_hc3))
    p_hc3 = float(2.0 * stats.t.sf(abs(t_hc3), df=max(1, n - 2)))

    # Evaluate if naive OLS was p < 0.05 but HC3 is p >= 0.05
    naive_se = np.sqrt(np.sum(residuals ** 2) / max(1, n - 2) / x_ss)
    t_naive = b1 / max(1e-9, naive_se)
    p_naive = 2.0 * stats.t.sf(abs(t_naive), df=max(1, n - 2))
    significance_lost = bool(p_naive < 0.05 and p_hc3 >= 0.05)

    return se_hc3, p_hc3, significance_lost


def vuong_model_selection_test(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    confounder_col: str,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Vuong's (1989) likelihood ratio test for non-nested model selection.
    
    Compares the pooled bivariate OLS model (M0: Y ~ X) against the stratified subgroup model
    (M1: Y ~ X within strata of confounder_col).
    H0: Both models are equally close to the true data-generating process in Kullback-Leibler divergence.
    Rejection (z > 1.96, p < 0.05) demonstrates that the stratified specification strictly dominates
    the crude pooled claim, providing formal information-theoretic proof of Simpson's Paradox / subgroup heterogeneity.
    """
    sub = df[[source_col, target_col, confounder_col]].dropna()
    if len(sub) < 20 or sub[confounder_col].nunique() < 2:
        return None, None, False

    if pd.api.types.is_numeric_dtype(sub[source_col]):
        x = pd.to_numeric(sub[source_col], errors='coerce')
    else:
        x = pd.Series(pd.Categorical(sub[source_col]).codes, index=sub.index, dtype=float)

    if pd.api.types.is_numeric_dtype(sub[target_col]):
        y = pd.to_numeric(sub[target_col], errors='coerce')
    else:
        y = pd.Series(pd.Categorical(sub[target_col]).codes, index=sub.index, dtype=float)

    valid_m = ~(x.isna() | y.isna())
    if valid_m.sum() < 20:
        return None, None, False

    x_arr = x[valid_m].to_numpy(dtype=float)
    y_arr = y[valid_m].to_numpy(dtype=float)
    groups = sub[confounder_col][valid_m].to_numpy()
    n = len(x_arr)

    var_x = np.var(x_arr)
    if var_x < 1e-9:
        return None, None, False

    # M0: Pooled model
    b1_pool = float(np.cov(x_arr, y_arr)[0, 1] / var_x)
    b0_pool = float(np.mean(y_arr) - b1_pool * np.mean(x_arr))
    y_hat0 = b0_pool + b1_pool * x_arr
    e0 = y_arr - y_hat0
    s2_0 = max(1e-12, float(np.mean(e0 ** 2)))
    l0 = -0.5 * np.log(2.0 * np.pi * s2_0) - (e0 ** 2) / (2.0 * s2_0)

    # M1: Stratified model
    y_hat1 = np.empty_like(y_arr)
    for grp in np.unique(groups):
        m = (groups == grp)
        if np.sum(m) >= 2 and np.var(x_arr[m]) > 1e-9:
            b1_g = float(np.cov(x_arr[m], y_arr[m])[0, 1] / np.var(x_arr[m]))
            b0_g = float(np.mean(y_arr[m]) - b1_g * np.mean(x_arr[m]))
            y_hat1[m] = b0_g + b1_g * x_arr[m]
        else:
            y_hat1[m] = np.mean(y_arr[m])

    e1 = y_arr - y_hat1
    s2_1 = max(1e-12, float(np.mean(e1 ** 2)))
    l1 = -0.5 * np.log(2.0 * np.pi * s2_1) - (e1 ** 2) / (2.0 * s2_1)

    m_diff = l1 - l0
    omega = float(np.std(m_diff, ddof=1))
    if omega < 1e-9:
        return 0.0, 1.0, False

    z_vuong = float(np.sqrt(n) * np.mean(m_diff) / omega)
    p_vuong = float(2.0 * stats.norm.sf(abs(z_vuong)))
    is_strat_favored = bool(z_vuong > 1.96 and p_vuong < 0.05)

    return z_vuong, p_vuong, is_strat_favored


def calculate_frank_itcv(
    r_xy: float,
    r_xz: float,
    r_yz: float,
    n: int,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Kenneth Frank's (2000) Impact Threshold for a Confounding Variable (ITCV).
    
    ITCV = (|r_xy| - r_crit) / (1 - r_crit)
    where r_crit = t_crit / sqrt(t_crit^2 + n - 2) is the critical correlation threshold for significance.
    Observed impact of Z = |r_xz * r_yz|.
    If observed impact >= ITCV, the confounder invalidates the empirical inference.
    """
    if n < 10 or np.isnan(r_xy) or np.isnan(r_xz) or np.isnan(r_yz):
        return None, None, False
    try:
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=max(1, n - 2)))
        r_crit = float(t_crit / np.sqrt(t_crit ** 2 + max(1, n - 2)))
        r_abs = min(0.9999, abs(r_xy))
        if r_abs <= r_crit:
            itcv = 0.0
        else:
            itcv = float((r_abs - r_crit) / max(1e-6, 1.0 - r_crit))
        
        impact_obs = float(abs(r_xz * r_yz))
        is_exceeded = bool(impact_obs >= itcv)
        return itcv, impact_obs, is_exceeded
    except Exception:
        return None, None, False


def calculate_akaike_weights(
    r_xy: float,
    r_xz: float,
    r_yz: float,
    n: int,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculates small-sample AICc Delta, Evidence Ratio, and Akaike Model Weight (Burnham & Anderson 2002).
    
    M0 (uncontrolled): Y ~ X (k=2)
    M1 (controlled): Y ~ X + Z (k=3)
    Returns (delta_aic, evidence_ratio, model_weight_adjusted).
    """
    if n < 12 or abs(r_xz) >= 0.999 or np.isnan(r_xy) or np.isnan(r_xz) or np.isnan(r_yz):
        return None, None, None
    try:
        denom = max(1e-9, 1.0 - r_xz ** 2)
        r2_full = min(0.9999, max(0.0, (r_xy ** 2 + r_yz ** 2 - 2.0 * r_xy * r_yz * r_xz) / denom))
        r2_null = min(0.9999, r_xy ** 2)

        rss0 = max(1e-9, 1.0 - r2_null)
        rss1 = max(1e-9, 1.0 - r2_full)

        # AICc with Hurvich & Tsai correction
        aic0 = n * np.log(rss0) + 2.0 * 2 + (2.0 * 2 * 3) / max(1, n - 3)
        aic1 = n * np.log(rss1) + 2.0 * 3 + (2.0 * 3 * 4) / max(1, n - 4)

        delta_aic = float(aic0 - aic1)
        clipped_exp = min(50.0, max(-50.0, 0.5 * delta_aic))
        evidence_ratio = float(np.exp(clipped_exp))
        model_weight = float(evidence_ratio / (1.0 + evidence_ratio))

        return delta_aic, evidence_ratio, model_weight
    except Exception:
        return None, None, None


def calculate_theil_sen_discrepancy(x: np.ndarray, y: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Calculates Theil-Sen (1950) robust median slope and discrepancy relative to OLS slope.
    
    Theil-Sen has a 29.3% breakdown point, resisting catastrophic outlier distortion.
    Discrepancy = |beta_OLS - beta_TS| / max(1e-4, |beta_TS|).
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9:
        return None, None
    try:
        b1_ols = float(np.cov(x, y)[0, 1] / max(1e-9, np.var(x)))
        ts_res = stats.theilslopes(y, x)
        b1_ts = float(ts_res.slope)
        disc = float(abs(b1_ols - b1_ts) / max(1e-4, abs(b1_ts)))
        return b1_ts, disc
    except Exception:
        return None, None


def calculate_bca_bootstrap_ci(
    r_boots: np.ndarray,
    r_obs: float,
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Calculates Efron & Tibshirani (1993) Bias-Corrected and Accelerated (BCa) Bootstrap 95% Confidence Interval.
    
    Uses vectorized jackknife leave-one-out calculations to estimate acceleration parameter a,
    and quantile proportion to estimate bias correction z0.
    """
    n = len(x)
    valid_boots = r_boots[~np.isnan(r_boots)]
    if len(valid_boots) < 10 or n < 10:
        return float(np.percentile(valid_boots, 2.5)), float(np.percentile(valid_boots, 97.5)), False

    try:
        # 1. Bias correction parameter z0
        prop_less = np.clip(np.mean(valid_boots < r_obs), 1e-4, 1.0 - 1e-4)
        z0 = float(stats.norm.ppf(prop_less))

        # 2. Acceleration parameter a via vectorized leave-one-out jackknife
        x_sum, y_sum = float(np.sum(x)), float(np.sum(y))
        x2_sum, y2_sum = float(np.sum(x ** 2)), float(np.sum(y ** 2))
        xy_sum = float(np.sum(x * y))

        x_loo_sum = x_sum - x
        y_loo_sum = y_sum - y
        xy_loo = xy_sum - (x * y)
        x2_loo = x2_sum - (x ** 2)
        y2_loo = y2_sum - (y ** 2)
        n_loo = n - 1

        cov_loo = xy_loo - (x_loo_sum * y_loo_sum) / n_loo
        var_x_loo = np.maximum(1e-12, x2_loo - (x_loo_sum ** 2) / n_loo)
        var_y_loo = np.maximum(1e-12, y2_loo - (y_loo_sum ** 2) / n_loo)
        r_jack = cov_loo / np.sqrt(var_x_loo * var_y_loo)

        r_bar = float(np.mean(r_jack))
        diff = r_bar - r_jack
        a_num = float(np.sum(diff ** 3))
        a_denom = float(6.0 * (np.sum(diff ** 2) ** 1.5))
        a = float(a_num / max(1e-12, a_denom)) if a_denom > 1e-12 else 0.0

        # 3. Adjusted quantiles
        z_a1 = float(stats.norm.ppf(alpha / 2.0))
        z_a2 = float(stats.norm.ppf(1.0 - alpha / 2.0))

        denom1 = max(1e-4, 1.0 - a * (z0 + z_a1))
        denom2 = max(1e-4, 1.0 - a * (z0 + z_a2))

        pct1 = float(stats.norm.cdf(z0 + (z0 + z_a1) / denom1))
        pct2 = float(stats.norm.cdf(z0 + (z0 + z_a2) / denom2))

        bca_low = float(np.percentile(valid_boots, 100.0 * np.clip(pct1, 0.005, 0.995)))
        bca_high = float(np.percentile(valid_boots, 100.0 * np.clip(pct2, 0.005, 0.995)))
        zero_cross = bool(bca_low <= 0.0 <= bca_high)

        return bca_low, bca_high, zero_cross
    except Exception:
        ci_l = float(np.percentile(valid_boots, 2.5))
        ci_h = float(np.percentile(valid_boots, 97.5))
        return ci_l, ci_h, bool(ci_l <= 0.0 <= ci_h)


def calculate_rosenbaum_bounds(
    r_xy: float,
    n: int,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul Rosenbaum's (1987, 2002) sensitivity bounds (Gamma) for unmeasured confounding.
    
    Quantifies the minimum odds ratio difference in treatment/exposure assignment (Gamma >= 1.0)
    that an unmeasured confounder must produce to explain away the observed statistical significance.
    Under hidden bias Gamma, the critical threshold Gamma_crit is the value at which the upper-bound p-value exceeds alpha.
    Gamma_crit < 1.30 indicates high vulnerability to hidden bias; Gamma_crit >= 2.0 indicates strong robustness.
    """
    if n < 10 or np.isnan(r_xy) or abs(r_xy) >= 0.9999:
        return None, None, False
    try:
        t_obs = float(abs(r_xy) * np.sqrt(max(1, n - 2)) / np.sqrt(max(1e-9, 1.0 - r_xy ** 2)))
        z_obs = t_obs
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        
        delta = (z_obs - z_crit) / np.sqrt(n)
        if delta <= 0.0:
            gamma_crit = 1.0
        elif delta >= 0.98:
            gamma_crit = 99.9
        else:
            gamma_crit = float((1.0 + delta) / (1.0 - delta))
        
        gamma_crit = float(min(99.9, max(1.0, gamma_crit)))
        
        # Calculate upper bound p-value at benchmark Gamma = 1.5
        gamma_test = 1.5
        shift = ((gamma_test - 1.0) / (gamma_test + 1.0)) * np.sqrt(n)
        z_upper = z_obs - shift
        p_upper = float(stats.norm.sf(z_upper))
        is_sensitive = bool(gamma_crit < 1.30)
        
        return gamma_crit, p_upper, is_sensitive
    except Exception:
        return None, None, False


def white_heteroscedasticity_test(
    x: np.ndarray,
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Halbert White's (1980) general Lagrange Multiplier (LM) test for heteroscedasticity.
    
    Regresses squared residuals e_i^2 on [1, x_i, x_i^2] without cross-terms:
    e_i^2 = alpha_0 + alpha_1 * x_i + alpha_2 * x_i^2 + u_i.
    LM = n * R^2 ~ chi^2(2).
    H0: Homoscedasticity. Rejection (p < 0.05) detects general non-linear variance heteroscedasticity,
    justifying MacKinnon-White HC3 robust standard errors.
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9:
        return None, None, False
    try:
        e2 = residuals ** 2
        e2_mean = float(np.mean(e2))
        ss_tot = float(np.sum((e2 - e2_mean) ** 2))
        if ss_tot < 1e-12:
            return 0.0, 1.0, False
            
        x_mat = np.column_stack([np.ones(n), x, x ** 2])
        beta, _, _, _ = np.linalg.lstsq(x_mat, e2, rcond=None)
        fit = x_mat @ beta
        ss_reg = float(np.sum((fit - e2_mean) ** 2))
        
        r2 = min(0.9999, max(0.0, ss_reg / ss_tot))
        lm_stat = float(n * r2)
        p_val = float(stats.chi2.sf(lm_stat, df=2))
        is_hetero = bool(p_val < 0.05)
        return lm_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def calculate_manski_bounds(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Charles Manski's (1990, 2003) non-parametric partial identification bounds.
    
    Evaluates worst-case bounds on the treatment/exposure effect under arbitrary non-random selection,
    without imposing parametric distributions or exclusion restrictions.
    Binarizes treatment at the median of x (D = 1 if x > median(x)) and computes sharp identification
    limits [Delta_low, Delta_high] over the empirical support [y_min, y_max].
    If 0 in [Delta_low, Delta_high], the sign of the effect is non-parametrically indeterminate.
    """
    n = len(x)
    if n < 20 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        x_med = float(np.median(x))
        d = (x > x_med).astype(float)
        n1 = int(np.sum(d == 1.0))
        n0 = int(np.sum(d == 0.0))
        if n1 < 5 or n0 < 5:
            return None, None, False
            
        p1 = float(n1 / n)
        p0 = float(n0 / n)
        y1 = y[d == 1.0]
        y0 = y[d == 0.0]
        
        e_y1 = float(np.mean(y1))
        e_y0 = float(np.mean(y0))
        y_min = float(np.percentile(y, 1.0))
        y_max = float(np.percentile(y, 99.0))
        
        delta_low = float(p1 * e_y1 + p0 * y_min - (p1 * y_max + p0 * e_y0))
        delta_high = float(p1 * e_y1 + p0 * y_max - (p1 * y_min + p0 * e_y0))
        contains_zero = bool(delta_low <= 0.0 <= delta_high)
        
        return delta_low, delta_high, contains_zero
    except Exception:
        return None, None, False


def calculate_altonji_elder_taber_ratio(
    r_xy: float,
    r_xz: float,
    r_yz: float,
) -> tuple[Optional[float], bool]:
    """Calculates Altonji, Elder, and Taber (2005) selection ratio (R_AET) on unobservables.
    
    Evaluates how much stronger selection on unmeasured confounders would have to be
    relative to selection on observed confounders to attribute the entire observed effect to selection bias:
    R_AET = |beta_full| / |beta_restricted - beta_full|
    
    Where in standardized terms:
      beta_restricted = r_xy
      beta_full = (r_xy - r_xz * r_yz) / (1 - r_xz^2)
    
    If R_AET < 1.0, selection on unobservables would only need to be smaller than selection
    on the observed confounder to explain away the effect (highly vulnerable to unmeasured confounding).
    If R_AET >= 3.0, unobservables would need to be 3x more explanatory than the top confounder (robust).
    """
    if abs(r_xz) >= 0.999 or abs(r_xy) >= 0.999:
        return None, False
    try:
        denom = abs(r_xz * (r_yz - r_xy * r_xz))
        num = abs(r_xy - r_xz * r_yz)
        if denom < 1e-7:
            # Confounder has virtually zero impact on beta; unobservables need infinite selection
            return 99.9, False
        r_aet = float(num / denom)
        r_aet = min(99.9, max(0.0, r_aet))
        is_vulnerable = bool(r_aet < 1.0)
        return r_aet, is_vulnerable
    except Exception:
        return None, False


def chow_structural_break_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Gregory Chow's (1960) F-test for structural breaks and parameter constancy.
    
    Tests H0: beta_1 = beta_2 across partitioned subsamples split at the median of x.
    F_Chow = ((RSS_pooled - (RSS_1 + RSS_2)) / k) / ((RSS_1 + RSS_2) / (n - 2k)) ~ F(k, n - 2k)
    where k = 2 (intercept + slope).
    Rejection (p < 0.05) detects a structural break or regime instability, demonstrating that
    the claimed relationship fails parameter invariance and is non-stationary.
    """
    n = len(x)
    if n < 20:
        return None, None, False
    try:
        x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
        sxx = float(np.sum((x - x_mean) ** 2))
        sxy = float(np.sum((x - x_mean) * (y - y_mean)))
        if sxx < 1e-12:
            return None, None, False
        b1 = sxy / sxx
        b0 = y_mean - b1 * x_mean
        rss_pooled = float(np.sum((y - (b0 + b1 * x)) ** 2))
        
        med = float(np.median(x))
        m1, m2 = (x <= med), (x > med)
        n1, n2 = int(np.sum(m1)), int(np.sum(m2))
        if n1 < 5 or n2 < 5:
            return None, None, False
            
        def _quick_rss(xs: np.ndarray, ys: np.ndarray) -> float:
            xm, ym = float(np.mean(xs)), float(np.mean(ys))
            sx = float(np.sum((xs - xm) ** 2))
            if sx < 1e-12:
                return float(np.sum((ys - ym) ** 2))
            b = float(np.sum((xs - xm) * (ys - ym)) / sx)
            a = ym - b * xm
            return float(np.sum((ys - (a + b * xs)) ** 2))
            
        rss1 = _quick_rss(x[m1], y[m1])
        rss2 = _quick_rss(x[m2], y[m2])
        rss_u = rss1 + rss2
        
        k = 2
        dfn = k
        dfd = n - 2 * k
        if dfd <= 0 or rss_u < 1e-12:
            return 0.0, 1.0, False
            
        num = max(0.0, (rss_pooled - rss_u) / dfn)
        den = max(1e-12, rss_u / dfd)
        f_stat = float(num / den)
        p_val = float(stats.f.sf(f_stat, dfn, dfd))
        has_break = bool(p_val < 0.05)
        return f_stat, p_val, has_break
    except Exception:
        return None, None, False


def calculate_higgins_h_and_galbraith_outliers(
    subgroup_stats: List[float],
    subgroup_weights: List[int],
    subgroup_names: List[str],
    cochrans_q: Optional[float] = None,
) -> tuple[Optional[float], int, List[str]]:
    """Calculates Higgins & Thompson (2002) H^2 relative excess heterogeneity index
    and detects Galbraith (1988) radial plot outlier strata.
    
    H^2 = max(1.0, Q / (k - 1)), measuring the multiplicative inflation of total variance
    relative to within-subgroup sampling variance. H^2 > 1.5 indicates substantive heterogeneity.
    
    Galbraith radial standardized residuals:
    z_i = (theta_i - theta_pooled) / SE_i
    Subgroups with |z_i| >= 2.0 represent statistically significant outlier strata exerting
    anomalous leverage on the aggregate estimate.
    """
    k = len(subgroup_stats)
    if k < 2:
        return 1.0, 0, []
    try:
        if cochrans_q is None:
            w = np.array(subgroup_weights, dtype=float)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 1.0, 0, []
            pooled = float(np.sum(w * np.array(subgroup_stats)) / w_sum)
            cochrans_q = float(np.sum(w * (np.array(subgroup_stats) - pooled) ** 2))
            
        h2 = max(1.0, float(cochrans_q / max(1, k - 1)))
        
        # Standard errors for each subgroup based on sample size
        se = np.array([1.0 / np.sqrt(max(4, n - 3)) for n in subgroup_weights], dtype=float)
        prec = 1.0 / np.maximum(1e-6, se)
        pooled_theta = float(np.sum(prec ** 2 * np.array(subgroup_stats)) / np.sum(prec ** 2))
        z_scores = (np.array(subgroup_stats) - pooled_theta) / np.maximum(1e-6, se)
        
        outliers = []
        for i, z in enumerate(z_scores):
            if abs(z) >= 2.0:
                name = subgroup_names[i] if i < len(subgroup_names) else f"Strata_{i+1}"
                outliers.append(str(name))
                
        return h2, len(outliers), outliers
    except Exception:
        return 1.0, 0, []


def woolf_stratified_heterogeneity_test(
    tables_2x2: List[tuple[float, float, float, float]],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Woolf's (1955) stratified odds ratio heterogeneity test
    with Haldane (1956) and Anscombe (1956) 0.5 continuity corrections.
    
    For each stratum k in {1, ..., K} with table (a_k, b_k, c_k, d_k):
      ln(OR_k) = ln( (a_k + 0.5)(d_k + 0.5) / ((b_k + 0.5)(c_k + 0.5)) )
      v_k = 1/(a_k + 0.5) + 1/(b_k + 0.5) + 1/(c_k + 0.5) + 1/(d_k + 0.5)
      w_k = 1 / v_k
    Pooled log odds ratio:
      ln(OR_bar) = sum(w_k * ln(OR_k)) / sum(w_k)
    Woolf test statistic:
      chi2_Woolf = sum(w_k * (ln(OR_k) - ln(OR_bar))^2) ~ chi2(K - 1)
    
    Unlike asymptotic Breslow-Day tests, the Haldane-Anscombe correction guarantees
    finite, non-zero variance even in the presence of sparse or empty cells.
    Rejection (p < 0.05) certifies statistically significant effect modification across strata.
    """
    k = len(tables_2x2)
    if k < 2:
        return None, None, False
    try:
        ln_ors = []
        weights = []
        for a, b, c, d in tables_2x2:
            a_c = float(a) + 0.5
            b_c = float(b) + 0.5
            c_c = float(c) + 0.5
            d_c = float(d) + 0.5
            lor = float(np.log((a_c * d_c) / (b_c * c_c)))
            var = float(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)
            if var < 1e-12:
                continue
            w = float(1.0 / var)
            ln_ors.append(lor)
            weights.append(w)
        
        if len(ln_ors) < 2:
            return None, None, False
        
        w_arr = np.array(weights, dtype=float)
        lor_arr = np.array(ln_ors, dtype=float)
        w_sum = float(np.sum(w_arr))
        if w_sum < 1e-12:
            return None, None, False
        
        pooled_lor = float(np.sum(w_arr * lor_arr) / w_sum)
        chi2_val = float(np.sum(w_arr * (lor_arr - pooled_lor) ** 2))
        df = len(ln_ors) - 1
        p_val = float(stats.chi2.sf(chi2_val, df=df))
        is_hetero = bool(p_val < 0.05)
        return chi2_val, p_val, is_hetero
    except Exception:
        return None, None, False


def calculate_imbens_manski_ci(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Imbens & Manski (2004) asymptotically valid confidence intervals
    for partially identified Average Treatment Effects under arbitrary non-random selection.
    
    Extends Manski's (1990) point identification bounds [Delta_low, Delta_high] by accounting
    for boundary estimator sampling variability. Solves for critical value c_n in [z_{1-alpha}, z_{1-alpha/2}]
    satisfying:
      Phi(c_n + sqrt(n) * (Delta_high - Delta_low) / sigma_max) - Phi(-c_n) = 1 - alpha
    
    Yields asymptotically valid coverage:
      CI_{1-alpha} = [Delta_low - c_n * sigma_low / sqrt(n), Delta_high + c_n * sigma_high / sqrt(n)]
    Certifies whether zero is strictly excluded from the entire identification region.
    """
    n = len(x)
    if n < 20 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, None, False
    try:
        x_med = float(np.median(x))
        d = (x > x_med).astype(float)
        n1 = int(np.sum(d == 1.0))
        n0 = int(np.sum(d == 0.0))
        if n1 < 5 or n0 < 5:
            return None, None, None, False
        
        p1 = float(n1 / n)
        p0 = float(n0 / n)
        y1 = y[d == 1.0]
        y0 = y[d == 0.0]
        
        e_y1 = float(np.mean(y1))
        e_y0 = float(np.mean(y0))
        s1_sq = float(np.var(y1, ddof=1)) if len(y1) > 1 else 0.0
        s0_sq = float(np.var(y0, ddof=1)) if len(y0) > 1 else 0.0
        
        y_min = float(np.percentile(y, 1.0))
        y_max = float(np.percentile(y, 99.0))
        
        delta_low = float(p1 * e_y1 + p0 * y_min - (p1 * y_max + p0 * e_y0))
        delta_high = float(p1 * e_y1 + p0 * y_max - (p1 * y_min + p0 * e_y0))
        
        sigma_low = float(np.sqrt(max(1e-8, s1_sq / p1 + s0_sq / p0)))
        sigma_high = float(np.sqrt(max(1e-8, s1_sq / p1 + s0_sq / p0)))
        sigma_max = max(sigma_low, sigma_high, 1e-6)
        
        bound_width = max(0.0, delta_high - delta_low)
        delta_n = float(np.sqrt(n) * bound_width / sigma_max)
        
        z_one_sided = float(stats.norm.ppf(1.0 - alpha))
        z_two_sided = float(stats.norm.ppf(1.0 - alpha / 2.0))
        
        def _obj(c: float) -> float:
            return float(stats.norm.cdf(c + delta_n) - stats.norm.cdf(-c) - (1.0 - alpha))
        
        if delta_n >= 20.0:
            c_n = z_one_sided
        elif delta_n <= 1e-6:
            c_n = z_two_sided
        else:
            try:
                c_n = float(optimize.brentq(_obj, z_one_sided - 0.05, z_two_sided + 0.05))
            except Exception:
                c_n = z_two_sided
        
        ci_low = float(delta_low - c_n * sigma_low / np.sqrt(n))
        ci_high = float(delta_high + c_n * sigma_high / np.sqrt(n))
        excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
        
        return ci_low, ci_high, c_n, excludes_zero
    except Exception:
        return None, None, None, False


def calculate_mallows_cp(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Colin Mallows' (1973) Cp model misspecification and selection criterion.
    
    Evaluates whether omitting candidate confounder z from the model y ~ x introduces
    substantial specification bias and variance inflation.
    Restricted model: y ~ 1 + x (p = 2)
    Full model: y ~ 1 + x + z (p_full = 3)
      Cp = (RSS_rest / MSE_full) - (n - 2 * p)
      Discrepancy: Delta Cp = Cp - p
    
    If Delta Cp > 2.0, the restricted model suffers from severe omitted-variable bias,
    demonstrating that omitting confounder z misspecifies the true data-generating process.
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        x_mat = np.column_stack([np.ones(n), x])
        beta_rest, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
        res_rest = y - x_mat @ beta_rest
        rss_rest = float(np.sum(res_rest ** 2))
        
        full_mat = np.column_stack([np.ones(n), x, z])
        beta_full, _, _, _ = np.linalg.lstsq(full_mat, y, rcond=None)
        res_full = y - full_mat @ beta_full
        rss_full = float(np.sum(res_full ** 2))
        
        p = 2
        p_full = 3
        df_full = n - p_full
        if df_full <= 0 or rss_full < 1e-12:
            return None, None, False
            
        mse_full = rss_full / df_full
        cp = float((rss_rest / mse_full) - (n - 2 * p))
        discrepancy = float(cp - p)
        is_biased = bool(discrepancy > 2.0)
        
        return cp, discrepancy, is_biased
    except Exception:
        return None, None, False


def calculate_gelman_carlin_design_analysis(
    estimate: float,
    se: float,
    prior_effect: Optional[float] = None,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Gelman & Carlin (2014) design analysis: Type S (Sign) error probability,
    Type M (Exaggeration / Magnitude) error ratio, and retrodesign power.
    
    In exploratory and observational discovery, statistical significance (p < 0.05) often
    leads to dramatic effect exaggeration ('Winner's Curse') and wrong-sign conclusions
    under low statistical power.
    Given standard error se and hypothesized true effect theta_0:
      c = z_{1-alpha/2} * se
      Power = 1 - beta = Phi((theta_0 - c)/se) + Phi((-theta_0 - c)/se)
      Type S = Phi((-theta_0 - c)/se) / (1 - beta)
      Type M = E[|theta_hat| | |theta_hat| > c] / theta_0
    
    Flags is_vulnerable if Type M ratio > 1.50 or Type S error > 0.05.
    """
    if se <= 1e-9 or abs(estimate) < 1e-9:
        return None, None, None, False
    try:
        theta_0 = abs(prior_effect) if prior_effect is not None else max(0.05, abs(estimate) / 2.0)
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        c = z_crit * se
        
        term1 = (theta_0 - c) / se
        term2 = (-theta_0 - c) / se
        power = float(stats.norm.cdf(term1) + stats.norm.cdf(term2))
        power = min(1.0, max(1e-12, power))
        
        type_s = float(stats.norm.cdf(term2) / power)
        type_s = min(1.0, max(0.0, type_s))
        
        part_pos = theta_0 * stats.norm.cdf(term1) + se * stats.norm.pdf(term1)
        part_neg = -theta_0 * stats.norm.cdf(term2) + se * stats.norm.pdf(term2)
        e_abs_given_reject = float((part_pos + part_neg) / power)
        
        type_m = float(e_abs_given_reject / theta_0)
        type_m = max(1.0, min(99.9, type_m))
        
        is_vulnerable = bool(type_m > 1.50 or type_s > 0.05)
        return type_s, type_m, power, is_vulnerable
    except Exception:
        return None, None, None, False


def freeman_tukey_double_arcsine_test(
    tables_2x2: List[tuple[float, float, float, float]],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Freeman & Tukey's (1950) variance-stabilizing double-arcsine
    transformation and cross-stratum homogeneity test for stratified binomial data.
    
    For each stratum k in {1, ..., K} with counts (a_k, b_k, c_k, d_k),
    let successes x_k = a_k and sample size n_k = a_k + b_k (e.g. treatment arm):
      theta_k = 0.5 * (arcsin(sqrt(x_k / (n_k + 1))) + arcsin(sqrt((x_k + 1) / (n_k + 1))))
      Var(theta_k) = 1 / (4 * n_k + 2)
      w_k = 4 * n_k + 2
    Pooled double-arcsine transformed rate:
      theta_bar = sum(w_k * theta_k) / sum(w_k)
    Freeman-Tukey test statistic:
      chi2_FT = sum(w_k * (theta_k - theta_bar)^2) ~ chi2(K - 1)
      
    Stabilizes binomial sampling variance across strata independent of p, resolving
    the severe distortion and failure of naive chi-squared tests near 0 or 1.
    Rejection (p < 0.05) certifies statistically significant variance-stabilized heterogeneity.
    """
    k = len(tables_2x2)
    if k < 2:
        return None, None, False
    try:
        thetas = []
        weights = []
        for a, b, c, d in tables_2x2:
            n_grp = float(a + b)
            if n_grp <= 0:
                continue
            x = float(a)
            t_k = 0.5 * (np.arcsin(np.sqrt(x / (n_grp + 1.0))) + np.arcsin(np.sqrt((x + 1.0) / (n_grp + 1.0))))
            w_k = 4.0 * n_grp + 2.0
            thetas.append(float(t_k))
            weights.append(float(w_k))
            
        if len(thetas) < 2:
            return None, None, False
            
        t_arr = np.array(thetas, dtype=float)
        w_arr = np.array(weights, dtype=float)
        w_sum = float(np.sum(w_arr))
        if w_sum <= 0:
            return None, None, False
            
        theta_bar = float(np.sum(w_arr * t_arr) / w_sum)
        chi2_val = float(np.sum(w_arr * (t_arr - theta_bar) ** 2))
        df = len(thetas) - 1
        p_val = float(stats.chi2.sf(chi2_val, df=df))
        is_hetero = bool(p_val < 0.05)
        return chi2_val, p_val, is_hetero
    except Exception:
        return None, None, False


def kruskal_wallis_strata_test(
    subgroup_samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Kruskal & Wallis (1952) non-parametric rank-sum H-test across strata.
    
    Evaluates whether the distribution medians of the target variable differ across K >= 2
    subgroups without assuming normality or homoscedasticity.
    Rejection (p < 0.05) certifies non-parametric distributional heterogeneity across strata.
    """
    valid_samples = [s for s in subgroup_samples if len(s) >= 3 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        stat_h, p_val = stats.kruskal(*valid_samples)
        h_val = float(stat_h)
        p_val = float(p_val)
        is_hetero = bool(p_val < 0.05)
        return h_val, p_val, is_hetero
    except Exception:
        return None, None, False


def davidson_mackinnon_j_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Davidson & MacKinnon's (1981) J-test for non-nested model specification.
    
    Tests whether the bivariate model M_1: Y = a_0 + a_1 * X is correctly specified
    against the alternative confounder model M_2: Y = b_0 + b_1 * Z.
    Procedure:
      1. Fit M_2: Y ~ 1 + Z, obtaining predicted values Y_hat_M2.
      2. Fit augmented model: Y ~ 1 + X + theta * Y_hat_M2.
      3. Test H_0: theta = 0 via t-statistic.
      
    If |t_J| >= 1.96 (p < 0.05), M_1 is rejected: the candidate confounder Z captures
    statistically significant predictive information that X cannot account for.
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        # Step 1: Fit M2 (Y ~ 1 + Z)
        z_mat = np.column_stack([np.ones(n), z])
        beta_z, _, _, _ = np.linalg.lstsq(z_mat, y, rcond=None)
        y_hat_m2 = z_mat @ beta_z
        
        # Step 2: Fit augmented model Y ~ 1 + X + Y_hat_M2
        aug_mat = np.column_stack([np.ones(n), x, y_hat_m2])
        beta_aug, _, _, _ = np.linalg.lstsq(aug_mat, y, rcond=None)
        res_aug = y - aug_mat @ beta_aug
        rss_aug = float(np.sum(res_aug ** 2))
        
        df_e = n - 3
        if df_e <= 0 or rss_aug < 1e-12:
            return None, None, False
            
        mse_aug = rss_aug / df_e
        cov_matrix = mse_aug * np.linalg.pinv(aug_mat.T @ aug_mat)
        se_theta = float(np.sqrt(max(1e-12, cov_matrix[2, 2])))
        
        theta_val = float(beta_aug[2])
        t_stat = float(theta_val / se_theta)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        is_rejected = bool(p_val < 0.05)
        
        return t_stat, p_val, is_rejected
    except Exception:
        return None, None, False


def andrews_sup_wald_test(
    x: np.ndarray,
    y: np.ndarray,
    trim: float = 0.15,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Donald Andrews' (1993) Supremum Wald test (Sup W) for structural breaks
    at an unknown change point with Hansen (1997) asymptotic distribution approximation.
    
    Sorts observations by regressor x and evaluates parameter instability across all candidate
    split points tau in [trim, 1 - trim] (k = 2 parameters: intercept and slope).
      W(tau) = [ (RSS_pooled - (RSS_1 + RSS_2)) / k ] / [ (RSS_1 + RSS_2) / (n - 2*k) ] * k
      Sup W = max_{tau} W(tau)
      
    Asymptotic 5% critical value for k=2 with 15% trimming is ~11.72 (1% is ~15.20).
    Rejection (p < 0.05) proves the relationship suffers from an unmodelled structural change
    or regime shift along x without requiring prior knowledge of the break location.
    """
    n = len(x)
    if n < 30 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, None, False
    try:
        # Sort by x
        idx = np.argsort(x)
        x_s = x[idx]
        y_s = y[idx]
        
        # Pooled RSS
        mat_pooled = np.column_stack([np.ones(n), x_s])
        beta_p, _, _, _ = np.linalg.lstsq(mat_pooled, y_s, rcond=None)
        rss_p = float(np.sum((y_s - mat_pooled @ beta_p) ** 2))
        
        k = 2
        start_idx = max(k + 2, int(np.floor(trim * n)))
        end_idx = min(n - (k + 2), int(np.ceil((1.0 - trim) * n)))
        if end_idx <= start_idx:
            return None, None, None, False
            
        step = max(1, (end_idx - start_idx) // 50)  # Evaluate up to 50 candidate break points
        candidate_indices = list(range(start_idx, end_idx, step))
        
        max_wald = -1.0
        best_idx = candidate_indices[0]
        
        for i in candidate_indices:
            x1, y1 = x_s[:i], y_s[:i]
            x2, y2 = x_s[i:], y_s[i:]
            
            if np.std(x1) < 1e-9 or np.std(x2) < 1e-9:
                continue
                
            mat1 = np.column_stack([np.ones(i), x1])
            mat2 = np.column_stack([np.ones(n - i), x2])
            
            b1, _, _, _ = np.linalg.lstsq(mat1, y1, rcond=None)
            b2, _, _, _ = np.linalg.lstsq(mat2, y2, rcond=None)
            
            rss1 = float(np.sum((y1 - mat1 @ b1) ** 2))
            rss2 = float(np.sum((y2 - mat2 @ b2) ** 2))
            rss_sum = rss1 + rss2
            
            df_denom = n - 2 * k
            if df_denom <= 0 or rss_sum < 1e-12:
                continue
                
            f_stat = ((rss_p - rss_sum) / k) / (rss_sum / df_denom)
            wald_stat = f_stat * k  # Wald statistic with k=2 restrictions
            
            if wald_stat > max_wald:
                max_wald = wald_stat
                best_idx = i
                
        if max_wald < 0:
            return None, None, None, False
            
        break_fraction = float(best_idx / n)
        
        # Hansen (1997) asymptotic p-value approximation for k=2, 15% trimming
        # Critical values: 10% = 10.15, 5% = 11.72, 1% = 15.20
        if max_wald >= 15.20:
            p_val = float(0.01 * np.exp(-0.4 * (max_wald - 15.20)))
        elif max_wald >= 11.72:
            p_val = float(0.01 + 0.04 * (15.20 - max_wald) / (15.20 - 11.72))
        elif max_wald >= 10.15:
            p_val = float(0.05 + 0.05 * (11.72 - max_wald) / (11.72 - 10.15))
        else:
            p_val = float(0.10 + 0.90 * max(0.0, (10.15 - max_wald) / 10.15))
        p_val = min(1.0, max(1e-8, p_val))
        
        has_break = bool(p_val < 0.05)
        return float(max_wald), float(break_fraction), p_val, has_break
    except Exception:
        return None, None, None, False


def peto_stratified_odds_ratio_test(
    tables_2x2: List[tuple[float, float, float, float]],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Peto et al.'s (1977) assumption-free one-step odds ratio estimator
    and cross-stratum heterogeneity Q-test for stratified 2x2 contingency tables.

    For each stratum k with cells (a_k, b_k, c_k, d_k), marginals n_1k = a_k + b_k,
    n_2k = c_k + d_k, m_1k = a_k + c_k, m_2k = b_k + d_k, N_k = n_1k + n_2k:
      Expected events: E_k = (n_1k * m_1k) / N_k
      Hypergeometric variance: V_k = (n_1k * n_2k * m_1k * m_2k) / (N_k^2 * (N_k - 1))
      Peto ln(OR) = sum(a_k - E_k) / sum(V_k), OR_Peto = exp(ln(OR))
      Peto test statistic: Z_Peto = sum(a_k - E_k) / sqrt(sum(V_k)) ~ N(0, 1)
      Peto heterogeneity: Q_Peto = sum((a_k - E_k)^2 / V_k) - (sum(a_k - E_k))^2 / sum(V_k) ~ chi2(K - 1)

    Peto's method provides superior finite-sample properties and robust power in the presence
    of sparse event counts or highly unbalanced exposure allocations.
    Rejection (p < 0.05) proves statistically significant stratum odds ratio heterogeneity.
    """
    if len(tables_2x2) < 2:
        return None, None, None, None, None, False
    try:
        o_minus_e = []
        v_list = []
        for a, b, c, d in tables_2x2:
            n1 = float(a + b)
            n2 = float(c + d)
            m1 = float(a + c)
            m2 = float(b + d)
            n_tot = n1 + n2
            if n_tot <= 1 or n1 <= 0 or n2 <= 0 or m1 <= 0 or m2 <= 0:
                continue
            e_k = (n1 * m1) / n_tot
            v_k = (n1 * n2 * m1 * m2) / ((n_tot ** 2) * (n_tot - 1.0))
            if v_k > 1e-12:
                o_minus_e.append(float(a - e_k))
                v_list.append(float(v_k))

        if len(v_list) < 2:
            return None, None, None, None, None, False

        sum_ome = float(np.sum(o_minus_e))
        sum_v = float(np.sum(v_list))
        if sum_v <= 1e-12:
            return None, None, None, None, None, False

        ln_or = sum_ome / sum_v
        peto_or = float(np.exp(np.clip(ln_or, -20.0, 20.0)))
        z_stat = float(sum_ome / np.sqrt(sum_v))
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))

        # Peto Heterogeneity Q
        q_stat = float(np.sum((np.array(o_minus_e) ** 2) / np.array(v_list)) - (sum_ome ** 2) / sum_v)
        q_stat = max(0.0, q_stat)
        df_q = len(v_list) - 1
        q_p = float(stats.chi2.sf(q_stat, df=df_q))
        is_hetero = bool(q_p < 0.05)

        return peto_or, z_stat, p_val, q_stat, q_p, is_hetero
    except Exception:
        return None, None, None, None, None, False


def fligner_killeen_homogeneity_test(
    subgroup_samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates the Fligner & Killeen (1976) non-parametric test for homogeneity of variances.

    Scores ranked absolute deviations from subgroup medians |Y_kj - median(Y_k)| using
    standard normal quantiles a_N(i) = Phi^{-1}((1 + i / (N + 1)) / 2).
    Offers robust error rate control against departures from Gaussian normality.
    Rejection (p < 0.05) demonstrates that target variance scales diverge significantly across strata.
    """
    valid_samples = [s for s in subgroup_samples if len(s) >= 4 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        stat_val, p_val = stats.fligner(*valid_samples)
        f_stat = float(stat_val)
        p_val = float(p_val)
        is_hetero = bool(p_val < 0.05)
        return f_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def durbin_wu_hausman_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates the Durbin (1954), Wu (1973), and Hausman (1978) specification test
    for regressor endogeneity and omitted variable bias.

    Evaluates whether the candidate confounder Z induces statistically significant endogeneity
    into regressor X when omitted from the bivariate specification Y ~ 1 + X.
    Procedure:
      1. First-stage auxiliary regression: X = pi_0 + pi_1 * Z + v.
         Extract estimated residuals v_hat.
      2. Augmented outcome regression: Y = beta_0 + beta_1 * X + delta * v_hat + u.
      3. Under H_0: delta = 0 (X is exogenous w.r.t. Z; omitting Z does not bias beta_1).
         Under H_1: delta != 0 (X is endogenous; bivariate OLS suffers severe omitted variable bias).

    Test statistic:
      F_DWH = (delta_hat / SE(delta_hat))^2 ~ F(1, n - 3).
    Rejection (p < 0.05) confirms statistically significant endogeneity / omitted variable bias.
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        # Step 1: Auxiliary regression X ~ 1 + Z
        z_mat = np.column_stack([np.ones(n), z])
        beta_z, _, _, _ = np.linalg.lstsq(z_mat, x, rcond=None)
        v_hat = x - z_mat @ beta_z

        if np.std(v_hat) < 1e-9:
            return None, None, False

        # Step 2: Augmented regression Y ~ 1 + X + v_hat
        aug_mat = np.column_stack([np.ones(n), x, v_hat])
        beta_aug, _, _, _ = np.linalg.lstsq(aug_mat, y, rcond=None)
        res_aug = y - aug_mat @ beta_aug
        rss_aug = float(np.sum(res_aug ** 2))

        df_e = n - 3
        if df_e <= 0 or rss_aug < 1e-12:
            return None, None, False

        mse_aug = rss_aug / df_e
        cov_matrix = mse_aug * np.linalg.pinv(aug_mat.T @ aug_mat)
        se_delta = float(np.sqrt(max(1e-12, cov_matrix[2, 2])))

        delta_hat = float(beta_aug[2])
        t_stat = delta_hat / se_delta
        f_stat = float(t_stat ** 2)
        p_val = float(stats.f.sf(f_stat, dfn=1, dfd=df_e))
        is_endogenous = bool(p_val < 0.05)

        return f_stat, p_val, is_endogenous
    except Exception:
        return None, None, False


def stock_yogo_first_stage_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], bool]:
    """Calculates the Stock & Yogo (2005) first-stage relevance F-statistic between regressor X and confounder Z.

    F_first_stage = (R^2_{XZ} / 1) / ((1 - R^2_{XZ}) / (n - 2)).
    Evaluates whether the coupling between candidate confounder Z and regressor X satisfies
    the standard rule-of-thumb threshold (F >= 10.0) certifying strong first-stage identification
    (bias < 10% of maximal bias). If F < 10.0, the confounder exhibits weak explanatory coupling.
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, False
    try:
        r = float(np.corrcoef(x, z)[0, 1])
        r2 = min(0.999999, max(0.0, r ** 2))
        f_stat = float((r2 / 1.0) / ((1.0 - r2) / (n - 2)))
        is_weak = bool(f_stat < 10.0)
        return f_stat, is_weak
    except Exception:
        return None, False


def brown_durbin_evans_cusum_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates the Brown, Durbin, & Evans (1975) Cumulative Sum of Recursive Residuals (CUSUM) test.

    Evaluates parameter constancy across sorted regressor values (or sequential ordering).
    Recursively updates least-squares estimates for r = k+1, ..., n (k=2) via rank-one Sherman-Morrison
    inversions:
      Standardized recursive residual:
        w_r = (y_r - x_r' beta_{r-1}) / sqrt(1 + x_r' (X_{r-1}' X_{r-1})^{-1} x_r)
      Standardized cumulative sum:
        W_t = (1 / sigma_hat) * sum_{r=k+1}^t w_r, where sigma_hat^2 = (1 / (n - k)) * sum w_r^2
      Boundary test ratio:
        S = max_{k+1 <= t <= n} |W_t| / [ sqrt(n - k) * (1 + 2 * (t - k) / (n - k)) ]

    The 5% asymptotic critical value is a = 0.948. If S > 0.948, the recursive path breaches
    the 95% confidence boundary (p < 0.05), proving parameter instability and cumulative structural drift.
    """
    n = len(x)
    if n < 25 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        idx = np.argsort(x)
        x_s = x[idx]
        y_s = y[idx]

        k = 2
        x_mat = np.column_stack([np.ones(n), x_s])
        
        # Determine minimal base window init_m where design matrix achieves full rank
        init_m = max(k + 2, 5)
        while init_m < n // 2 and np.linalg.matrix_rank(x_mat[:init_m]) < k:
            init_m += 1

        if np.linalg.matrix_rank(x_mat[:init_m]) < k:
            return None, None, False

        x_init = x_mat[:init_m]
        y_init = y_s[:init_m]
        a_inv = np.linalg.pinv(x_init.T @ x_init)
        xty = x_init.T @ y_init
        beta = a_inv @ xty

        w_list = []
        for r in range(init_m, n):
            xr = x_mat[r]
            yr = y_s[r]
            f_r = float(1.0 + xr @ a_inv @ xr)
            if f_r < 1e-12:
                f_r = 1e-12
            res_r = float(yr - xr @ beta)
            w_r = res_r / np.sqrt(f_r)
            w_list.append(w_r)

            # Sherman-Morrison rank-1 update
            a_xr = a_inv @ xr
            a_inv = a_inv - np.outer(a_xr, a_xr) / f_r
            xty = xty + yr * xr
            beta = a_inv @ xty

        w_arr = np.array(w_list, dtype=float)
        m = len(w_arr)
        if m < 5:
            return None, None, False

        sigma_sq = float(np.sum(w_arr ** 2) / m)
        if sigma_sq <= 1e-12:
            return None, None, False

        sigma_hat = np.sqrt(sigma_sq)
        w_cum = np.cumsum(w_arr) / sigma_hat

        t_vec = np.arange(init_m + 1, n + 1, dtype=float)
        boundary_denom = np.sqrt(n - k) * (1.0 + 2.0 * (t_vec - k) / float(n - k))
        s_ratios = np.abs(w_cum) / boundary_denom
        s_stat = float(np.max(s_ratios))

        # Asymptotic p-value approximation based on boundary crossing distribution
        if s_stat >= 1.143:
            p_val = float(0.01 * np.exp(-3.0 * (s_stat - 1.143)))
        elif s_stat >= 0.948:
            p_val = float(0.01 + 0.04 * (1.143 - s_stat) / (1.143 - 0.948))
        elif s_stat >= 0.850:
            p_val = float(0.05 + 0.05 * (0.948 - s_stat) / (0.948 - 0.850))
        else:
            p_val = float(0.10 + 0.90 * max(0.0, (0.850 - s_stat) / 0.850))
        p_val = min(1.0, max(1e-8, p_val))
        has_instability = bool(s_stat > 0.948)

        return s_stat, p_val, has_instability
    except Exception:
        return None, None, False


def koenker_studentized_breusch_pagan_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates the Koenker (1981) studentized Breusch-Pagan test for heteroscedasticity.

    Standard Breusch-Pagan assumes Gaussian error distributions. Koenker demonstrated that
    excess kurtosis severely inflates the type I error rate.
    The studentized LM statistic replaces 2*sigma^4 with the empirical sample variance
    of squared residuals:
      S_{e^2}^2 = (1 / n) * sum (e_i^2 - bar(e^2))^2
      LM_Koenker = n * R_{e^2}^2 = sum (hat{e}_i^2 - bar(e^2))^2 / S_{e^2}^2 ~ chi2(1).

    Rejection (p < 0.05) proves heteroscedasticity that is robust to leptokurtic and fat-tailed residuals.
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(mat, y, rcond=None)
        residuals = y - mat @ beta
        u = residuals ** 2
        u_bar = float(np.mean(u))
        var_u = float(np.mean((u - u_bar) ** 2))

        if var_u < 1e-12:
            return None, None, False

        beta_u, _, _, _ = np.linalg.lstsq(mat, u, rcond=None)
        u_hat = mat @ beta_u
        ess_u = float(np.sum((u_hat - u_bar) ** 2))

        lm_stat = float(ess_u / var_u)
        p_val = float(stats.chi2.sf(lm_stat, df=1))
        is_hetero = bool(p_val < 0.05)

        return lm_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def harbord_modified_score_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates the Harbord, Egger, & Sterne (2006) modified score test for funnel plot asymmetry.

    Unlike classical Egger regression on log odds ratios, Harbord's test replaces log-OR and standard errors
    with the hypergeometric score (a_k - E_k) / sqrt(V_k) and precision proxy sqrt(V_k), eliminating the
    false-positive rate inflation driven by correlation between effect size and standard error under H_0.
    Regresses standardized score Z_k = (a_k - E_k) / sqrt(V_k) on precision proxy S_k = sqrt(V_k):
      Z_k = alpha + beta * S_k + epsilon_k.
    Test statistic:
      t_Harbord = alpha_hat / SE(alpha_hat) ~ t(K - 2).
    A statistically significant intercept (p < 0.05) proves funnel plot asymmetry / small-stratum bias.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        scores = []
        precisions = []
        for t in tables_2x2:
            if isinstance(t, (tuple, list)):
                a, b, c, d = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            else:
                a, b = float(t[0, 0]), float(t[0, 1])
                c, d = float(t[1, 0]), float(t[1, 1])
            n1 = a + b
            n0 = c + d
            m1 = a + c
            m0 = b + d
            N = n1 + n0
            if N > 2 and n1 > 0 and n0 > 0 and m1 > 0 and m0 > 0:
                E_k = (n1 * m1) / N
                V_k = (n1 * n0 * m1 * m0) / (N ** 2 * (N - 1))
                if V_k > 1e-9:
                    scores.append((a - E_k) / np.sqrt(V_k))
                    precisions.append(np.sqrt(V_k))

        k_valid = len(scores)
        if k_valid < 3:
            return None, None, False

        X_mat = np.column_stack([np.ones(k_valid), precisions])
        y_vec = np.array(scores, dtype=float)
        beta, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        res = y_vec - X_mat @ beta
        df_e = k_valid - 2
        if df_e <= 0:
            return None, None, False

        mse = float(np.sum(res ** 2) / df_e)
        cov = mse * np.linalg.pinv(X_mat.T @ X_mat)
        se_alpha = float(np.sqrt(max(1e-12, cov[0, 0])))
        if se_alpha < 1e-12:
            return None, None, False

        t_stat = float(beta[0] / se_alpha)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        is_asym = bool(p_val < 0.05)
        return t_stat, p_val, is_asym
    except Exception:
        return None, None, False


def brown_forsythe_homogeneity_test(
    subgroup_samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates the Brown & Forsythe (1974) robust scale homogeneity test across strata.

    Extends Levene's test by taking absolute deviations from subgroup medians rather than means:
      z_{ki} = |Y_{ki} - median(Y_k)|
    Performs one-way ANOVA on median deviations z_{ki}.
    Provides optimal empirical test size and power without assuming normality or homoscedasticity.
    Rejection (p < 0.05) proves conditional variance heterogeneity across conditioning strata.
    """
    valid_samples = [s for s in subgroup_samples if len(s) >= 4 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        stat_val, p_val = stats.levene(*valid_samples, center="median")
        f_stat = float(stat_val)
        p_val = float(p_val)
        is_hetero = bool(p_val < 0.05)
        return f_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def belsley_kuh_welsch_collinearity_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates the Belsley, Kuh, & Welsch (1980) Collinearity Condition Index (kappa_BKW)
    and variance decomposition proportions for design matrix [1, X, Z].

    Column-equilibrates design matrix X_* = [1, X, Z] / ||column||_2, computes Singular Value
    Decomposition X_* = U Sigma V', and evaluates condition indices kappa_j = mu_max / mu_j.
    Computes parameter variance decomposition proportions pi_{jk}.
    Flags severe collinearity if kappa >= 30.0 (or moderate >= 15.0) and high variance proportion (> 0.50).
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x, z])
        norms = np.linalg.norm(X_mat, axis=0)
        norms[norms < 1e-9] = 1.0
        X_scaled = X_mat / norms

        _, s, vh = np.linalg.svd(X_scaled, full_matrices=False)
        s_max = float(np.max(s))
        s_min = float(np.min(s))
        if s_min < 1e-12:
            return 999.0, 1.0, True

        cond_indices = s_max / s
        max_cond_index = float(np.max(cond_indices))

        v = vh.T
        phi = (v ** 2) / (s ** 2)[None, :]
        phi_k = np.sum(phi, axis=1, keepdims=True)
        phi_k[phi_k < 1e-12] = 1e-12
        pi_mat = phi / phi_k

        max_idx = int(np.argmax(cond_indices))
        max_var_prop = float(np.max(pi_mat[:, max_idx]))
        is_collinear = bool(max_cond_index >= 15.0 and max_var_prop > 0.50)

        return max_cond_index, max_var_prop, is_collinear
    except Exception:
        return None, None, False


def sargan_overidentifying_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates the Sargan (1958) & Hansen (1982) Overidentifying Restrictions Test (J_Sargan).

    Evaluates whether candidate instrumental/confounder specifications satisfy orthogonality
    with structural disturbances. Uses Z and Z^2 as instruments for endogenous regressor X.
    Procedure:
      1. First stage: X = gamma_0 + gamma_1 * Z + gamma_2 * Z^2 + v -> fitted X_hat.
      2. Second stage: Y = beta_0 + beta_1 * X_hat + u_2sls.
      3. Structural residuals: u_hat = Y - (beta_0 + beta_1 * X).
      4. Auxiliary regression: u_hat = alpha_0 + alpha_1 * Z + alpha_2 * Z^2 + e -> R^2_{u_hat}.
    Test statistic:
      J_Sargan = n * R^2_{u_hat} ~ chi^2(L - K), where L=2 instruments, K=1 regressor -> df=1.
    Rejection (p < 0.05) proves instruments are invalid / exclusion restrictions are violated.
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        z_std = (z - np.mean(z)) / max(1e-6, np.std(z))
        Z_mat = np.column_stack([np.ones(n), z_std, z_std ** 2])
        if np.linalg.matrix_rank(Z_mat) < 3:
            return None, None, False

        # First stage: X ~ Z_mat
        gamma, _, _, _ = np.linalg.lstsq(Z_mat, x, rcond=None)
        x_hat = Z_mat @ gamma
        if np.std(x_hat) < 1e-9:
            return None, None, False

        # Second stage: Y ~ 1 + X_hat
        X_2sls = np.column_stack([np.ones(n), x_hat])
        beta_2sls, _, _, _ = np.linalg.lstsq(X_2sls, y, rcond=None)

        # Structural residuals
        u_hat = y - (beta_2sls[0] + beta_2sls[1] * x)
        tss_u = float(np.sum((u_hat - np.mean(u_hat)) ** 2))
        if tss_u < 1e-12:
            return None, None, False

        # Auxiliary regression: u_hat ~ Z_mat
        aux_gamma, _, _, _ = np.linalg.lstsq(Z_mat, u_hat, rcond=None)
        u_pred = Z_mat @ aux_gamma
        ess_u = float(np.sum((u_pred - np.mean(u_hat)) ** 2))
        r2_u = min(0.999999, max(0.0, ess_u / tss_u))

        j_stat = float(n * r2_u)
        p_val = float(stats.chi2.sf(j_stat, df=1))
        is_rejected = bool(p_val < 0.05)
        return j_stat, p_val, is_rejected
    except Exception:
        return None, None, False


def harvey_collier_linearity_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates the Harvey & Collier (1977) recursive residual test for functional linearity.

    Sorts observations along regressor X and computes standardized recursive residuals w_t.
    Under H_0 (correct linear specification), w_t ~ iid N(0, sigma^2). Under non-linear curvature
    (concavity, convexity, or unmodeled polynomials), w_t exhibits a non-zero mean.
    Test statistic:
      t_HC = (mean(w) / (s_w / sqrt(m))) ~ t(m - 1).
    Rejection (p < 0.05) detects significant structural non-linearity / functional form misspecification.
    """
    n = len(x)
    if n < 25 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        idx = np.argsort(x)
        x_s = x[idx]
        y_s = y[idx]

        k = 2
        x_mat = np.column_stack([np.ones(n), x_s])
        init_m = max(k + 2, 5)
        while init_m < n // 2 and np.linalg.matrix_rank(x_mat[:init_m]) < k:
            init_m += 1

        if np.linalg.matrix_rank(x_mat[:init_m]) < k:
            return None, None, False

        x_init = x_mat[:init_m]
        y_init = y_s[:init_m]
        a_inv = np.linalg.pinv(x_init.T @ x_init)
        xty = x_init.T @ y_init
        beta = a_inv @ xty

        w_list = []
        for r in range(init_m, n):
            xr = x_mat[r]
            yr = y_s[r]
            f_r = float(1.0 + xr @ a_inv @ xr)
            if f_r < 1e-12:
                f_r = 1e-12
            res_r = float(yr - xr @ beta)
            w_r = res_r / np.sqrt(f_r)
            w_list.append(w_r)

            a_xr = a_inv @ xr
            a_inv = a_inv - np.outer(a_xr, a_xr) / f_r
            xty = xty + yr * xr
            beta = a_inv @ xty

        w_arr = np.array(w_list, dtype=float)
        m = len(w_arr)
        if m < 5:
            return None, None, False

        w_mean = float(np.mean(w_arr))
        w_std = float(np.std(w_arr, ddof=1))
        if w_std < 1e-12:
            return None, None, False

        se_w = w_std / np.sqrt(m)
        t_stat = float(w_mean / se_w)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=m - 1))
        has_nonlinearity = bool(p_val < 0.05)

        return t_stat, p_val, has_nonlinearity
    except Exception:
        return None, None, False


def huber_m_estimator_test(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 30,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Huber's (1964) robust M-estimator regression slope and discrepancy vs OLS.

    Applies Iteratively Reweighted Least Squares (IRLS) using Huber's psi function with tuning
    constant c = 1.345 * sigma_MAD (95% Gaussian asymptotic efficiency).
    Discrepancy: Delta_Huber = |beta_OLS - beta_Huber| / max(1e-6, |beta_OLS|).
    Flags outlier contamination if discrepancy > 0.20 (20% slope shift).
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta_ols, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        ols_slope = float(beta_ols[1])

        beta = beta_ols.copy()
        for _ in range(max_iter):
            residuals = y - X_mat @ beta
            mad = float(stats.median_abs_deviation(residuals, scale="normal"))
            if mad < 1e-7:
                break
            c = 1.345 * mad
            weights = np.clip(c / np.maximum(np.abs(residuals), 1e-9), 0.0, 1.0)
            W_sqrt = np.sqrt(weights)[:, None]
            beta_new, _, _, _ = np.linalg.lstsq(X_mat * W_sqrt, y * W_sqrt.squeeze(), rcond=None)
            if np.max(np.abs(beta_new - beta)) < 1e-6:
                beta = beta_new
                break
            beta = beta_new

        huber_slope = float(beta[1])
        denom = max(1e-6, abs(ols_slope))
        discrepancy = float(abs(ols_slope - huber_slope) / denom)
        is_distorted = bool(discrepancy > 0.20)

        return huber_slope, discrepancy, is_distorted
    except Exception:
        return None, None, False


def peters_regression_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Peters et al. (2006) regression test for funnel plot asymmetry in 2x2xK tables.

    Regresses unstandardized log-odds ratio y_k = ln(OR_k) on inverse stratum sample size 1 / n_k
    weighted by effective sample size w_k = (n_1k * n_0k) / (n_1k + n_0k):
      ln(OR_k) = alpha + beta * (1 / n_k) + epsilon_k.
    Test statistic:
      t_Peters = alpha_hat / SE(alpha_hat) ~ t(K - 2).
    A statistically significant intercept (p < 0.05) proves funnel plot asymmetry without
    the spurious mathematical correlation between log-OR and standard error in Egger's test.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        log_ors = []
        inv_n = []
        weights = []
        for t in tables_2x2:
            if isinstance(t, (tuple, list)):
                a, b, c, d = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            else:
                a, b = float(t[0, 0]), float(t[0, 1])
                c, d = float(t[1, 0]), float(t[1, 1])
            n1 = a + b
            n0 = c + d
            N = n1 + n0
            if N > 4 and n1 > 0 and n0 > 0:
                a_c = a + 0.5 if (a == 0 or b == 0 or c == 0 or d == 0) else a
                b_c = b + 0.5 if (a == 0 or b == 0 or c == 0 or d == 0) else b
                c_c = c + 0.5 if (a == 0 or b == 0 or c == 0 or d == 0) else c
                d_c = d + 0.5 if (a == 0 or b == 0 or c == 0 or d == 0) else d
                
                lor = np.log((a_c * d_c) / (b_c * c_c))
                w = (n1 * n0) / N
                log_ors.append(lor)
                inv_n.append(1.0 / N)
                weights.append(w)

        k_valid = len(log_ors)
        if k_valid < 3:
            return None, None, False

        y_vec = np.array(log_ors, dtype=float)
        x_vec = np.array(inv_n, dtype=float)
        w_vec = np.array(weights, dtype=float)
        W_diag = np.sqrt(w_vec)[:, None]

        X_mat = np.column_stack([np.ones(k_valid), x_vec])
        X_w = X_mat * W_diag
        y_w = y_vec * W_diag.squeeze()

        beta, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
        res = y_vec - X_mat @ beta
        df_e = k_valid - 2
        if df_e <= 0:
            return None, None, False

        s2 = float(np.sum(w_vec * (res ** 2)) / df_e)
        cov = s2 * np.linalg.pinv(X_w.T @ X_w)
        se_alpha = float(np.sqrt(max(1e-12, cov[0, 0])))
        if se_alpha < 1e-12:
            return None, None, False

        t_stat = float(beta[0] / se_alpha)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        is_asym = bool(p_val < 0.05)
        return t_stat, p_val, is_asym
    except Exception:
        return None, None, False


def calculate_kish_design_effect(
    subgroup_sizes: List[int],
    subgroup_vars: List[float],
    crude_var: float,
) -> tuple[Optional[float], Optional[float]]:
    """Calculates Leslie Kish's (1965) Stratification Design Effect (D_Kish)
    and the Stratified Variance Ratio (D_eff).

    D_Kish = 1 + CV(n_k)^2 = 1 + Var(n_k) / mean(n_k)^2 = K * sum(n_k^2) / (sum(n_k))^2.
    Stratified Variance Ratio: D_eff = Var_strat(theta_hat) / Var_crude(theta_hat)
      where Var_strat = sum( (n_k / N)^2 * Var(theta_k) ).
    When D_eff < 1.0, conditioning on subgroups reduces estimation variance.
    When D_eff > 1.0, unequal stratum weighting inflates sampling variance.
    """
    if len(subgroup_sizes) < 2 or sum(subgroup_sizes) == 0:
        return None, None
    try:
        n_arr = np.array(subgroup_sizes, dtype=float)
        k = len(n_arr)
        n_total = float(np.sum(n_arr))
        if n_total <= 0:
            return None, None
        
        d_kish = float((k * np.sum(n_arr ** 2)) / (n_total ** 2))

        v_arr = np.array(subgroup_vars, dtype=float)
        w_arr = n_arr / n_total
        strat_var = float(np.sum((w_arr ** 2) * v_arr))
        
        if crude_var > 1e-9:
            d_eff = float(strat_var / crude_var)
        else:
            d_eff = 1.0

        return d_kish, d_eff
    except Exception:
        return None, None


def crump_propensity_overlap_test(
    x: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.10,
) -> tuple[Optional[float], bool]:
    """Calculates the Crump, Hotz, Imbens, & Mitnik (2009) Propensity Score Overlap Ratio.

    Dichotomizes regressor X at median to evaluate binary exposure assignment vs confounder Z.
    Fits logistic propensity score e(z_i) = P(X_i = 1 | z_i) via Newton-Raphson / IRLS.
    Crump et al. prove optimal covariate overlap requires alpha <= e(z_i) <= 1 - alpha (alpha = 0.10).
    Evaluates fraction of units violating common support:
      Overlap_Deficiency_Ratio = (1 / n) * sum( I(e(z_i) < alpha or e(z_i) > 1 - alpha) ).
    Flags lack of overlap if ratio > 0.20 (more than 20% outside common support).
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, False
    try:
        med_x = float(np.median(x))
        d_treat = (x > med_x).astype(float)
        if np.mean(d_treat) == 0.0 or np.mean(d_treat) == 1.0:
            return None, False

        z_norm = (z - np.mean(z)) / max(1e-9, float(np.std(z)))
        X_mat = np.column_stack([np.ones(n), z_norm])
        
        beta = np.zeros(2)
        for _ in range(20):
            p = 1.0 / (1.0 + np.exp(-np.clip(X_mat @ beta, -20.0, 20.0)))
            w = np.clip(p * (1.0 - p), 1e-5, 0.25)
            grad = X_mat.T @ (d_treat - p)
            hess = X_mat.T @ (X_mat * w[:, None])
            try:
                step = np.linalg.solve(hess + 1e-4 * np.eye(2), grad)
                beta += step
                if np.max(np.abs(step)) < 1e-5:
                    break
            except Exception:
                break

        propensities = 1.0 / (1.0 + np.exp(-np.clip(X_mat @ beta, -20.0, 20.0)))
        off_support = np.sum((propensities < alpha) | (propensities > (1.0 - alpha)))
        overlap_ratio = float(off_support / n)
        is_deficient = bool(overlap_ratio > 0.20)
        return overlap_ratio, is_deficient
    except Exception:
        return None, False


def hansen_threshold_regime_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    trim_pct: float = 0.15,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Bruce Hansen's (2000) Structural Threshold Non-Linearity F-Test.

    Searches for sample-splitting threshold gamma along candidate confounder Z:
      Y_i = beta_1 * X_i * I(Z_i <= gamma) + beta_2 * X_i * I(Z_i > gamma) + epsilon_i.
    Computes supremum F-statistic:
      F_n(gamma) = (RSS_0 - RSS_1(gamma)) / sigma^2_1(gamma)
      F_Hansen = sup_{gamma in Gamma} F_n(gamma).
    Computes Davies / Hansen supremum chi-square asymptotic p-value.
    Rejection (p < 0.05) detects significant structural regime-switching along confounder Z.
    """
    n = len(x)
    if n < 30 or np.std(x) < 1e-9 or np.std(z) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X0 = np.column_stack([np.ones(n), x])
        b0, _, _, _ = np.linalg.lstsq(X0, y, rcond=None)
        rss0 = float(np.sum((y - X0 @ b0) ** 2))

        q_low, q_high = np.quantile(z, [trim_pct, 1.0 - trim_pct])
        candidate_gammas = np.linspace(q_low, q_high, num=min(30, n // 3))
        
        max_f = 0.0
        for gamma in candidate_gammas:
            d1 = (z <= gamma).astype(float)
            d2 = (z > gamma).astype(float)
            if np.sum(d1) < 8 or np.sum(d2) < 8:
                continue
            X1 = np.column_stack([d1, d1 * x, d2, d2 * x])
            if np.linalg.matrix_rank(X1) < 4:
                continue
            b1, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
            rss1 = float(np.sum((y - X1 @ b1) ** 2))
            df_e = n - 4
            s2 = rss1 / max(1, df_e)
            if s2 > 1e-12:
                f_val = (rss0 - rss1) / s2
                if f_val > max_f:
                    max_f = f_val

        if max_f <= 0.0:
            return 0.0, 1.0, False

        p_val = float(np.exp(-0.5 * max_f) * (1.0 + 0.5 * max_f))
        p_val = float(np.clip(p_val, 0.0, 1.0))
        has_regime = bool(p_val < 0.05)

        return float(max_f), p_val, has_regime
    except Exception:
        return None, None, False


def brown_durbin_evans_cusumsq_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates the Brown, Durbin, & Evans (1975) / Ploberger & Krämer (1989) CUSUM of Squares Test.

    Evaluates variance constancy and structural stability along ordered regressor X.
    Uses recursive standardized residuals w_t to construct normalized cumulative sum of squares:
      s_t = sum_{j=k+1}^t w_j^2 / sum_{j=k+1}^n w_j^2.
    Test statistic:
      W_CUSUMSQ = max_{k < t <= n} |s_t - (t - k) / (n - k)|.
    Evaluated against Durbin's 5% critical boundary: c_0.05 = 1.358 / sqrt(n - k).
    Flags structural variance instability / parameter shift if W_CUSUMSQ > c_0.05.
    """
    n = len(x)
    if n < 25 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        idx = np.argsort(x)
        x_s = x[idx]
        y_s = y[idx]

        k = 2
        x_mat = np.column_stack([np.ones(n), x_s])
        init_m = max(k + 2, 5)
        while init_m < n // 2 and np.linalg.matrix_rank(x_mat[:init_m]) < k:
            init_m += 1

        if np.linalg.matrix_rank(x_mat[:init_m]) < k:
            return None, None, False

        x_init = x_mat[:init_m]
        y_init = y_s[:init_m]
        a_inv = np.linalg.pinv(x_init.T @ x_init)
        xty = x_init.T @ y_init
        beta = a_inv @ xty

        w_sq_list = []
        for r in range(init_m, n):
            xr = x_mat[r]
            yr = y_s[r]
            f_r = float(1.0 + xr @ a_inv @ xr)
            if f_r < 1e-12:
                f_r = 1e-12
            res_r = float(yr - xr @ beta)
            w_r = res_r / np.sqrt(f_r)
            w_sq_list.append(w_r ** 2)

            a_xr = a_inv @ xr
            a_inv = a_inv - np.outer(a_xr, a_xr) / f_r
            xty = xty + yr * xr
            beta = a_inv @ xty

        w_sq = np.array(w_sq_list, dtype=float)
        m = len(w_sq)
        if m < 5:
            return None, None, False

        total_sum_w2 = float(np.sum(w_sq))
        if total_sum_w2 < 1e-12:
            return None, None, False

        cum_w2 = np.cumsum(w_sq) / total_sum_w2
        t_fractions = np.arange(1, m + 1) / float(m)
        devs = np.abs(cum_w2 - t_fractions)
        max_dev = float(np.max(devs))

        c_crit = float(1.358 / np.sqrt(m))
        has_instability = bool(max_dev > c_crit)

        return max_dev, c_crit, has_instability
    except Exception:
        return None, None, False


def least_trimmed_squares_test(
    x: np.ndarray,
    y: np.ndarray,
    coverage_pct: float = 0.75,
    max_iter: int = 25,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Peter Rousseeuw's (1984) Least Trimmed Squares (LTS) High-Breakdown Slope Discrepancy.

    Minimizes the sum of the lowest h = floor(coverage_pct * n) squared residuals via FAST-LTS C-steps:
      beta_LTS = argmin sum_{i=1}^h (r^2)_{(i)}.
    Achieves high breakdown point (25% with coverage 0.75) resilient against clustered leverage points.
    Evaluates slope discrepancy vs classical OLS:
      Delta_LTS = |beta_OLS - beta_LTS| / max(1e-6, |beta_OLS|).
    Flags outlier contamination if discrepancy > 0.20 (20% slope shift).
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        b_ols, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        ols_slope = float(b_ols[1])

        h = int(np.floor(coverage_pct * n))
        h = max(5, min(h, n - 2))

        best_beta = b_ols.copy()
        residuals_init = (y - X_mat @ best_beta) ** 2
        idx_h = np.argsort(residuals_init)[:h]
        best_obj = float(np.sum(residuals_init[idx_h]))

        beta_curr = best_beta.copy()
        for _ in range(max_iter):
            res_sq = (y - X_mat @ beta_curr) ** 2
            h_indices = np.argsort(res_sq)[:h]
            obj = float(np.sum(res_sq[h_indices]))
            
            X_sub = X_mat[h_indices]
            y_sub = y[h_indices]
            if np.linalg.matrix_rank(X_sub) < 2:
                break
            beta_next, _, _, _ = np.linalg.lstsq(X_sub, y_sub, rcond=None)
            if obj < best_obj:
                best_obj = obj
                best_beta = beta_curr.copy()
            if np.max(np.abs(beta_next - beta_curr)) < 1e-6:
                break
            beta_curr = beta_next

        lts_slope = float(best_beta[1])
        denom = max(1e-6, abs(ols_slope))
        discrepancy = float(abs(ols_slope - lts_slope) / denom)
        is_distorted = bool(discrepancy > 0.20)

        return lts_slope, discrepancy, is_distorted
    except Exception:
        return None, None, False


def rucker_arcsine_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Gerta Rücker, Guido Schwarzer, & James Carpenter's (2008) Arcsine Test
    for Funnel Plot Asymmetry in Binary 2x2xK Stratified Contingency Tables.

    Applies the angular arcsine transformation y_k = arcsin(sqrt(p_1k)) - arcsin(sqrt(p_0k))
    where event rates are stabilized with variance sigma_k^2 = 1 / (4 * n_1k) + 1 / (4 * n_0k).
    Because the asymptotic variance is strictly independent of the underlying effect under H_0,
    regressing standardized effect z_k = y_k / sigma_k on precision x_k = 1 / sigma_k:
      z_k = alpha + beta * x_k + epsilon_k
    yields an orthogonalized asymmetry test:
      t_Rucker = alpha_hat / SE(alpha_hat) ~ t(K - 2).
    A statistically significant intercept (p < 0.05) detects small-stratum asymmetry without
    the false-positive Type I error inflation of standard Egger regression under sparse cells.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        y_trans = []
        precisions = []
        for t in tables_2x2:
            if isinstance(t, (tuple, list)):
                a, b, c, d = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            else:
                a, b = float(t[0, 0]), float(t[0, 1])
                c, d = float(t[1, 0]), float(t[1, 1])
            n1 = a + b
            n0 = c + d
            if n1 > 0 and n0 > 0:
                p1 = (a + 0.5) / (n1 + 1.0)
                p0 = (c + 0.5) / (n0 + 1.0)
                p1 = min(max(p1, 1e-4), 1.0 - 1e-4)
                p0 = min(max(p0, 1e-4), 1.0 - 1e-4)
                y_k = float(np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p0)))
                sigma_k2 = float(1.0 / (4.0 * n1) + 1.0 / (4.0 * n0))
                sigma_k = float(np.sqrt(sigma_k2))
                if sigma_k > 1e-9:
                    z_k = y_k / sigma_k
                    prec_k = 1.0 / sigma_k
                    y_trans.append(z_k)
                    precisions.append(prec_k)

        k_valid = len(y_trans)
        if k_valid < 3:
            return None, None, False

        y_vec = np.array(y_trans, dtype=float)
        x_vec = np.array(precisions, dtype=float)

        X_mat = np.column_stack([np.ones(k_valid), x_vec])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        res = y_vec - X_mat @ beta
        df_e = k_valid - 2
        if df_e <= 0:
            return None, None, False

        s2 = float(np.sum(res ** 2) / df_e)
        cov = s2 * np.linalg.pinv(X_mat.T @ X_mat)
        se_alpha = float(np.sqrt(max(1e-12, cov[0, 0])))
        if se_alpha < 1e-12:
            return None, None, False

        t_stat = float(beta[0] / se_alpha)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        is_asym = bool(p_val < 0.05)
        return t_stat, p_val, is_asym
    except Exception:
        return None, None, False


def levene_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Howard Levene's (1960) Classical Mean-Centered Scale Homogeneity Test.

    Evaluates whether conditional target scale dispersion varies across conditioning strata:
      z_{ki} = |Y_{ki} - mean(Y_k)|
      W_Levene = [ sum n_k (zbar_{k.} - zbar_{..})^2 / (K - 1) ] / [ sum sum (z_{ki} - zbar_{k.})^2 / (N - K) ]
    distributed as F(K - 1, N - K).
    Rejection (p < 0.05) detects significant conditional heteroscedasticity across strata.
    """
    valid_samples = [s for s in samples if len(s) >= 3 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        stat_val, p_val = stats.levene(*valid_samples, center="mean")
        if np.isnan(stat_val) or np.isnan(p_val):
            return None, None, False
        return float(stat_val), float(p_val), bool(p_val < 0.05)
    except Exception:
        return None, None, False


def hosmer_lemeshow_test(
    x: np.ndarray,
    z: np.ndarray,
    n_bins: int = 10,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates David Hosmer & Stanley Lemeshow's (1980) Goodness-of-Fit Calibration Test
    for Propensity Confounder Models.

    Dichotomizes focal regressor X at its median to construct pseudo-treatment assignments D = I(X > median(X)).
    Estimates logistic propensity score pi_i = P(D = 1 | z_i) via IRLS / Newton-Raphson.
    Partitions predictions into G quantiles and computes Pearson-type goodness-of-fit:
      chi2_HL = sum_{g=1}^G [ (O_g - N_g * pbar_g)^2 / (N_g * pbar_g * (1 - pbar_g)) ] ~ chi^2(G - 2).
    Rejection (p < 0.05) diagnoses severe propensity model miscalibration, demonstrating that
    observational confounder conditioning is distorted by misspecified exposure weights.
    """
    n = len(x)
    if n < 30 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        d_treat = (x > med_x).astype(float)
        if np.mean(d_treat) == 0.0 or np.mean(d_treat) == 1.0:
            return None, None, False

        z_norm = (z - np.mean(z)) / max(1e-9, float(np.std(z)))
        X_mat = np.column_stack([np.ones(n), z_norm])

        beta = np.zeros(2)
        for _ in range(20):
            p = 1.0 / (1.0 + np.exp(-np.clip(X_mat @ beta, -20.0, 20.0)))
            w = np.clip(p * (1.0 - p), 1e-5, 0.25)
            grad = X_mat.T @ (d_treat - p)
            hess = X_mat.T @ (X_mat * w[:, None])
            try:
                step = np.linalg.solve(hess + 1e-4 * np.eye(2), grad)
                beta += step
                if np.max(np.abs(step)) < 1e-5:
                    break
            except Exception:
                break

        propensities = 1.0 / (1.0 + np.exp(-np.clip(X_mat @ beta, -20.0, 20.0)))
        
        g = min(n_bins, max(3, n // 10))
        quantiles = np.quantile(propensities, np.linspace(0.0, 1.0, g + 1))
        quantiles[0] -= 1e-6
        quantiles[-1] += 1e-6

        hl_stat = 0.0
        df_hl = g - 2
        if df_hl <= 0:
            return None, None, False

        for k in range(g):
            bin_mask = (propensities > quantiles[k]) & (propensities <= quantiles[k + 1])
            n_k = np.sum(bin_mask)
            if n_k == 0:
                continue
            o_k = float(np.sum(d_treat[bin_mask]))
            pbar_k = float(np.mean(propensities[bin_mask]))
            var_k = n_k * pbar_k * (1.0 - pbar_k)
            e_k = n_k * pbar_k
            if var_k > 1e-6:
                hl_stat += float(((o_k - e_k) ** 2) / var_k)

        p_val = float(stats.chi2.sf(hl_stat, df=df_hl))
        is_miscal = bool(p_val < 0.05)
        return float(hl_stat), p_val, is_miscal
    except Exception:
        return None, None, False


def newey_west_hac_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Whitney Newey & Kenneth West's (1987) Heteroscedasticity and Autocorrelation
    Consistent (HAC) Variance Ratio.

    Fits regression Y = beta_0 + beta_1 * X + beta_2 * Z + epsilon.
    Constructs Bartlett kernel autocovariance matrix across bandwidth L = floor(4 * (n / 100)^(2/9)):
      Omega_hat = Gamma_0 + sum_{l=1}^L [ 1 - l / (L + 1) ] * (Gamma_l + Gamma_l^T).
    Computes HAC Standard Error for focal regressor X and evaluates inflation ratio:
      rho_HAC = SE_HAC(beta_1) / SE_OLS(beta_1).
    Flags severe inference distortion if rho_HAC > 1.25 (standard errors underestimated by > 20%).
    """
    n = len(x)
    if n < 20 or np.std(x) < 1e-9 or np.std(z) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x, z])
        if np.linalg.matrix_rank(X_mat) < 3:
            return None, None, False

        beta_hat, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        residuals = y - X_mat @ beta_hat
        df_e = n - 3
        s2 = float(np.sum(residuals ** 2) / max(1, df_e))

        xtx_inv = np.linalg.pinv(X_mat.T @ X_mat)
        v_ols = s2 * xtx_inv
        se_ols = float(np.sqrt(max(1e-12, v_ols[1, 1])))
        if se_ols < 1e-12:
            return None, None, False

        L = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
        u = residuals[:, None] * X_mat

        gamma_0 = (u.T @ u) / float(n)
        omega = gamma_0.copy()
        for l in range(1, L + 1):
            w_l = 1.0 - float(l) / float(L + 1)
            gamma_l = (u[l:].T @ u[:-l]) / float(n)
            omega += w_l * (gamma_l + gamma_l.T)

        v_hac = float(n) * (xtx_inv @ omega @ xtx_inv)
        se_hac = float(np.sqrt(max(1e-12, v_hac[1, 1])))
        ratio = float(se_hac / se_ols)
        is_inflated = bool(ratio > 1.25)
        return se_hac, ratio, is_inflated
    except Exception:
        return None, None, False


def siegel_repeated_medians_test(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int = 500,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Andrew Siegel's (1982) Repeated Medians High-Breakdown Robust Regression Discrepancy.

    Computes the 50% breakdown point non-parametric estimator:
      beta_Siegel = median_i { median_{j != i} [ (y_j - y_i) / (x_j - x_i) ] }.
    Resistant to arbitrary gross contamination of up to 50% of the sample.
    Evaluates relative slope discrepancy vs classical OLS:
      Delta_Siegel = |beta_OLS - beta_Siegel| / max(1e-6, |beta_OLS|).
    Flags outlier contamination if discrepancy > 0.20 (20% slope shift).
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        b_ols, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        ols_slope = float(b_ols[1])

        if n > max_samples:
            idx = np.random.default_rng(42).choice(n, size=max_samples, replace=False)
            x_sub = x[idx]
            y_sub = y[idx]
        else:
            x_sub = x
            y_sub = y

        dx = x_sub[:, None] - x_sub[None, :]
        dy = y_sub[:, None] - y_sub[None, :]
        valid_mask = np.abs(dx) > 1e-9
        np.fill_diagonal(valid_mask, False)

        pairwise_slopes = np.where(valid_mask, dy / np.where(valid_mask, dx, 1.0), np.nan)
        inner_medians = np.nanmedian(pairwise_slopes, axis=1)
        valid_inner = inner_medians[~np.isnan(inner_medians)]
        if len(valid_inner) == 0:
            return None, None, False

        siegel_slope = float(np.median(valid_inner))
        denom = max(1e-6, abs(ols_slope))
        discrepancy = float(abs(ols_slope - siegel_slope) / denom)
        is_distorted = bool(discrepancy > 0.20)
        return siegel_slope, discrepancy, is_distorted
    except Exception:
        return None, None, False


def breusch_godfrey_serial_correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    lags: int = 2,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Trevor Breusch (1978) & Leslie Godfrey's (1978) Higher-Order Serial Correlation LM Test.

    Evaluates AR(p) serial correlation in residuals along sorted regressor X:
      e_t = alpha_0 + alpha_1 * X_t + sum_{j=1}^p rho_j * e_{t-j} + v_t.
    Computes Lagrange Multiplier statistic:
      LM_BG = (n - lags) * R^2_aux ~ chi^2(p).
    Rejection (p < 0.05) detects higher-order autocorrelation that invalidates naive standard errors.
    """
    n = len(x)
    if n < 20 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        idx = np.argsort(x)
        x_s = x[idx]
        y_s = y[idx]

        X_mat = np.column_stack([np.ones(n), x_s])
        beta_hat, _, _, _ = np.linalg.lstsq(X_mat, y_s, rcond=None)
        residuals = y_s - X_mat @ beta_hat

        p = min(lags, max(1, n // 10))
        n_eff = n - p
        if n_eff <= p + 2:
            return None, None, False

        y_aux = residuals[p:]
        x_aux_cols = [np.ones(n_eff), x_s[p:]]
        for j in range(1, p + 1):
            x_aux_cols.append(residuals[p - j : n - j])
        X_aux = np.column_stack(x_aux_cols)

        b_aux, _, _, _ = np.linalg.lstsq(X_aux, y_aux, rcond=None)
        y_pred = X_aux @ b_aux
        ss_tot = float(np.sum((y_aux - np.mean(y_aux)) ** 2))
        ss_res = float(np.sum((y_aux - y_pred) ** 2))
        if ss_tot > 1e-12:
            r2_aux = max(0.0, 1.0 - ss_res / ss_tot)
        else:
            r2_aux = 0.0

        lm_stat = float(n_eff * r2_aux)
        p_val = float(stats.chi2.sf(lm_stat, df=p))
        has_autocorr = bool(p_val < 0.05)
        return lm_stat, p_val, has_autocorr
    except Exception:
        return None, None, False


def macaskill_regression_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Peter Macaskill, Stewart Walter, & Les Irwig's (2001) Sample-Size Weighted
    Regression Test for Funnel Plot Asymmetry in Stratified 2x2xK Contingency Tables.

    Regresses stratum log-odds ratio theta_k = ln(OR_k) on inverse stratum sample size 1 / N_k,
    weighted by inverse variance w_k = 1 / Var(theta_k) using Haldane-Anscombe 0.5 correction:
      sqrt(w_k) * theta_k = alpha * sqrt(w_k) + beta * (sqrt(w_k) / N_k) + epsilon_k*
    Macaskill test statistic:
      t_Macaskill = beta_hat / SE(beta_hat) ~ t(K - 2).
    Rejection (p < 0.05) detects small-study / stratum-size selection bias without the
    false-positive rate inflation of standard unweighted Egger regression.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        y_vec = []
        inv_n_vec = []
        w_vec = []
        for t in tables_2x2:
            if isinstance(t, (tuple, list)):
                if len(t) == 4 and not isinstance(t[0], (tuple, list)):
                    a, b, c, d = float(t[0]), float(t[1]), float(t[2]), float(t[3])
                else:
                    a, b = float(t[0][0]), float(t[0][1])
                    c, d = float(t[1][0]), float(t[1][1])
            else:
                a, b = float(t[0, 0]), float(t[0, 1])
                c, d = float(t[1, 0]), float(t[1, 1])
            n_tot = a + b + c + d
            if n_tot > 0 and (a + b) > 0 and (c + d) > 0:
                a_c = a + 0.5
                b_c = b + 0.5
                c_c = c + 0.5
                d_c = d + 0.5
                ln_or = float(np.log((a_c * d_c) / (b_c * c_c)))
                var_ln_or = float(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)
                if var_ln_or > 1e-9:
                    w = 1.0 / var_ln_or
                    y_vec.append(ln_or)
                    inv_n_vec.append(1.0 / float(n_tot))
                    w_vec.append(w)

        k_valid = len(y_vec)
        if k_valid < 3:
            return None, None, False

        y = np.array(y_vec, dtype=float)
        x = np.array(inv_n_vec, dtype=float)
        w = np.array(w_vec, dtype=float)
        if float(np.std(x)) < 1e-9:
            return None, None, False

        sw = np.sqrt(w)
        X_mat = np.column_stack([sw, sw * x])
        Y_trans = sw * y
        beta, _, _, _ = np.linalg.lstsq(X_mat, Y_trans, rcond=None)
        res = Y_trans - X_mat @ beta
        df_e = k_valid - 2
        if df_e <= 0:
            return None, None, False

        s2 = float(np.sum(res ** 2) / df_e)
        cov = s2 * np.linalg.pinv(X_mat.T @ X_mat)
        se_slope = float(np.sqrt(max(1e-12, cov[1, 1])))
        if se_slope < 1e-12:
            return None, None, False

        t_stat = float(beta[1] / se_slope)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        is_asym = bool(p_val < 0.05)
        return t_stat, p_val, is_asym
    except Exception:
        return None, None, False


def bartlett_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Maurice Bartlett's (1937) Likelihood Ratio Test for Homogeneity of Variances.

    Evaluates whether conditional target scale variances are constant across K conditioning strata:
      B = [ (N - K) * ln(s_p^2) - sum_{k=1}^K (n_k - 1) * ln(s_k^2) ] / [ 1 + (sum 1/(n_k-1) - 1/(N-K)) / (3*(K-1)) ]
    distributed asymptotically as chi^2(K - 1).
    Rejection (p < 0.05) detects significant variance heterogeneity across strata.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 3 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        k = len(valid_samples)
        n_vec = np.array([len(s) for s in valid_samples], dtype=float)
        s2_vec = np.array([float(np.var(s, ddof=1)) for s in valid_samples], dtype=float)
        if np.any(s2_vec <= 1e-12):
            return None, None, False
        N = float(np.sum(n_vec))
        df_total = N - float(k)
        if df_total <= 0:
            return None, None, False
        s2_pooled = float(np.sum((n_vec - 1.0) * s2_vec) / df_total)
        if s2_pooled <= 1e-12:
            return None, None, False
        num = df_total * np.log(s2_pooled) - np.sum((n_vec - 1.0) * np.log(s2_vec))
        c = 1.0 + (np.sum(1.0 / (n_vec - 1.0)) - 1.0 / df_total) / (3.0 * (k - 1.0))
        b_stat = float(num / c)
        if b_stat < 0.0:
            b_stat = 0.0
        p_val = float(stats.chi2.sf(b_stat, df=k - 1))
        is_hetero = bool(p_val < 0.05)
        return b_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def aipw_doubly_robust_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Robins-Rotnitzky-Zhao (1994) & Guido Imbens' (2004) Augmented Inverse
    Probability Weighted (AIPW) Doubly Robust Average Treatment Effect Discrepancy.

    Dichotomizes focal regressor X at its median to construct pseudo-treatment assignments D = I(X > median(X)).
    Estimates propensity scores e_hat(z_i) = P(D = 1 | z_i) via logistic regression (clipped to [0.02, 0.98]).
    Estimates outcome models mu_1(z) = E[Y | D = 1, z] and mu_0(z) = E[Y | D = 0, z] via OLS.
    Computes doubly robust score:
      Gamma_i = mu_1(z_i) - mu_0(z_i) + D_i * (y_i - mu_1(z_i)) / e_hat(z_i) - (1 - D_i) * (y_i - mu_0(z_i)) / (1 - e_hat(z_i))
      tau_AIPW = mean(Gamma_i).
    Evaluates relative discrepancy against naive mean difference tau_raw = mean(Y_{D=1}) - mean(Y_{D=0}):
      Delta_AIPW = |tau_raw - tau_AIPW| / max(1e-6, |tau_raw|).
    Flags severe selection confounding if Delta_AIPW > 0.25 (25% divergence).
    """
    n = len(x)
    if n < 30 or np.std(x) < 1e-9 or np.std(z) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        d_treat = (x > med_x).astype(float)
        n1 = int(np.sum(d_treat == 1.0))
        n0 = int(np.sum(d_treat == 0.0))
        if n1 < 5 or n0 < 5:
            return None, None, False

        tau_raw = float(np.mean(y[d_treat == 1.0]) - np.mean(y[d_treat == 0.0]))

        z_norm = (z - np.mean(z)) / max(1e-9, float(np.std(z)))
        Z_mat = np.column_stack([np.ones(n), z_norm])

        # Logistic propensity
        beta_p = np.zeros(2)
        for _ in range(15):
            p = 1.0 / (1.0 + np.exp(-np.clip(Z_mat @ beta_p, -20.0, 20.0)))
            w = np.clip(p * (1.0 - p), 1e-4, 0.25)
            grad = Z_mat.T @ (d_treat - p)
            hess = Z_mat.T @ (Z_mat * w[:, None])
            try:
                step = np.linalg.solve(hess + 1e-4 * np.eye(2), grad)
                beta_p += step
                if np.max(np.abs(step)) < 1e-5:
                    break
            except Exception:
                break
        e_hat = np.clip(1.0 / (1.0 + np.exp(-np.clip(Z_mat @ beta_p, -20.0, 20.0))), 0.02, 0.98)

        # Outcome models
        b1, _, _, _ = np.linalg.lstsq(Z_mat[d_treat == 1.0], y[d_treat == 1.0], rcond=None)
        b0, _, _, _ = np.linalg.lstsq(Z_mat[d_treat == 0.0], y[d_treat == 0.0], rcond=None)
        mu1 = Z_mat @ b1
        mu0 = Z_mat @ b0

        gamma = (mu1 - mu0) + (d_treat * (y - mu1)) / e_hat - ((1.0 - d_treat) * (y - mu0)) / (1.0 - e_hat)
        tau_aipw = float(np.mean(gamma))
        denom = max(1e-6, abs(tau_raw))
        discrepancy = float(abs(tau_raw - tau_aipw) / denom)
        is_confounded = bool(discrepancy > 0.25)
        return tau_aipw, discrepancy, is_confounded
    except Exception:
        return None, None, False


def leamer_extreme_bounds_test(
    df: pd.DataFrame,
    source: str,
    target: str,
    candidate_confounders: List[str],
    max_candidates: int = 4,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Edward Leamer's (1983) Extreme Bounds Analysis (EBA) on Focal Regression Slope.

    Evaluates the stability of focal slope beta_X across all subsets S of candidate confounders:
      Y = beta_0 + beta_X * X + gamma_S^T * Z_S + epsilon.
    Identifies extreme bounds:
      beta_min = min_S beta_X(S),  beta_max = max_S beta_X(S).
    If 0 in [beta_min, beta_max], Leamer declares the claim FRAGILE (the sign of the finding can be
    overturned by changing control covariate conditioning). If [beta_min, beta_max] excludes zero,
    the finding is declared ROBUST.
    """
    conf_list = [c for c in candidate_confounders if c not in (source, target)][:max_candidates]
    cols = [source, target] + conf_list
    sub = df[cols].dropna()
    n = len(sub)
    if n < 20 or sub[source].std() < 1e-9 or sub[target].std() < 1e-9:
        return None, None, False
    try:
        y = sub[target].to_numpy(dtype=float)
        x = sub[source].to_numpy(dtype=float)

        betas = []
        # Base bivariate model
        X_base = np.column_stack([np.ones(n), x])
        b_base, _, _, _ = np.linalg.lstsq(X_base, y, rcond=None)
        betas.append(float(b_base[1]))

        # Evaluate across subsets
        for r in range(1, len(conf_list) + 1):
            for combo in itertools.combinations(conf_list, r):
                Z = sub[list(combo)].to_numpy(dtype=float)
                X_full = np.column_stack([np.ones(n), x, Z])
                if np.linalg.matrix_rank(X_full) == X_full.shape[1]:
                    b_c, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
                    betas.append(float(b_c[1]))

        b_min = float(min(betas))
        b_max = float(max(betas))
        is_fragile = bool(b_min <= 0.0 <= b_max)
        return b_min, b_max, is_fragile
    except Exception:
        return None, None, False


def wald_wolfowitz_runs_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Abraham Wald & Jacob Wolfowitz's (1940) Non-Parametric Runs Test
    for Residual Independence along Regressor Ordering.

    Evaluates the sequence of regression residual signs along sorted X:
      s_i = sign(e_i) in {-1, +1}.
    Counts total runs R against expectation E[R] and variance Var(R) under the null of independence:
      mu_R = 2 * n_+ * n_- / N + 1
      sigma_R^2 = 2 * n_+ * n_- * (2 * n_+ * n_- - N) / (N^2 * (N - 1))
      Z_runs = (R - mu_R) / sigma_R ~ N(0, 1).
    Rejection (|Z_runs| >= 1.96, p < 0.05) detects significant residual clustering or non-random drift.
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        idx = np.argsort(x)
        x_s = x[idx]
        y_s = y[idx]
        X_mat = np.column_stack([np.ones(n), x_s])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y_s, rcond=None)
        res = y_s - X_mat @ beta

        signs = np.where(res >= 0.0, 1, -1)
        n_pos = int(np.sum(signs == 1))
        n_neg = int(np.sum(signs == -1))
        if n_pos < 3 or n_neg < 3:
            return None, None, False

        runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
        n_tot = float(n_pos + n_neg)
        mu_r = (2.0 * n_pos * n_neg) / n_tot + 1.0
        var_r = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n_tot)) / (n_tot ** 2 * (n_tot - 1.0))
        if var_r < 1e-12:
            return None, None, False

        z_stat = float((runs - mu_r) / np.sqrt(var_r))
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
        has_nonrandom = bool(p_val < 0.05)
        return z_stat, p_val, has_nonrandom
    except Exception:
        return None, None, False


def kendall_tau_b_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Maurice Kendall's (1938) Tau-b Rank Concordance Discrepancy.

    Evaluates bivariate ordinal concordance with exact tie adjustments:
      tau_b = (P - Q) / sqrt((P + Q + T_x) * (P + Q + T_y)).
    Computes discrepancy relative to Pearson linear correlation:
      Delta_Kendall = |r_OLS - tau_b| / max(1e-6, |r_OLS|).
    Flags outlier/leverage distortion if Delta_Kendall > 0.35 (35% discrepancy).
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        r_ols, _ = stats.pearsonr(x, y)
        res = stats.kendalltau(x, y)
        tau_val = float(res.statistic)
        if np.isnan(tau_val):
            return None, None, False
        denom = max(1e-6, abs(float(r_ols)))
        discrepancy = float(abs(float(r_ols) - tau_val) / denom)
        is_distorted = bool(discrepancy > 0.35)
        return tau_val, discrepancy, is_distorted
    except Exception:
        return None, None, False


def deeks_funnel_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Jon Deeks, Peter Macaskill, & Les Irwig's (2005) Diagnostic Odds Ratio (DOR)
    Regression Test for Funnel Plot Asymmetry in Stratified 2x2xK Tables.

    Regresses stratum log-diagnostic odds ratio theta_k = ln(DOR_k) on 1 / sqrt(ESS_k),
    where effective sample size ESS_k = 4 * (a+b) * (c+d) / (a+b+c+d), weighted by ESS_k:
      sqrt(ESS_k) * theta_k = alpha * sqrt(ESS_k) + beta + epsilon_k*
    Test statistic:
      t_Deeks = beta_hat / SE(beta_hat) ~ t(K - 2).
    Rejection (p < 0.05) detects significant stratified publication or selection asymmetry.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        ln_dor_vec = []
        inv_sqrt_ess_vec = []
        ess_vec = []
        for t in tables_2x2:
            if isinstance(t, (tuple, list)):
                if len(t) == 4 and not isinstance(t[0], (tuple, list)):
                    a, b, c, d = float(t[0]), float(t[1]), float(t[2]), float(t[3])
                else:
                    a, b = float(t[0][0]), float(t[0][1])
                    c, d = float(t[1][0]), float(t[1][1])
            else:
                a, b = float(t[0, 0]), float(t[0, 1])
                c, d = float(t[1, 0]), float(t[1, 1])
            n_tot = a + b + c + d
            n1 = a + b
            n2 = c + d
            if n_tot > 0 and n1 > 0 and n2 > 0:
                a_c = a + 0.5
                b_c = b + 0.5
                c_c = c + 0.5
                d_c = d + 0.5
                ln_dor = float(np.log((a_c * d_c) / (b_c * c_c)))
                ess = float((4.0 * n1 * n2) / n_tot)
                if ess > 1e-6:
                    ln_dor_vec.append(ln_dor)
                    inv_sqrt_ess_vec.append(1.0 / np.sqrt(ess))
                    ess_vec.append(ess)

        k_valid = len(ln_dor_vec)
        if k_valid < 3:
            return None, None, False

        y = np.array(ln_dor_vec, dtype=float)
        x = np.array(inv_sqrt_ess_vec, dtype=float)
        w = np.array(ess_vec, dtype=float)

        sw = np.sqrt(w)
        X_mat = np.column_stack([sw, sw * x])
        Y_trans = sw * y
        beta, _, _, _ = np.linalg.lstsq(X_mat, Y_trans, rcond=None)
        res = Y_trans - X_mat @ beta
        df_e = k_valid - 2
        if df_e <= 0:
            return None, None, False

        s2 = float(np.sum(res ** 2) / df_e)
        cov = s2 * np.linalg.pinv(X_mat.T @ X_mat)
        se_slope = float(np.sqrt(max(1e-12, cov[1, 1])))
        if se_slope < 1e-12:
            return None, None, False

        t_stat = float(beta[1] / se_slope)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        is_asym = bool(p_val < 0.05)
        return t_stat, p_val, is_asym
    except Exception:
        return None, None, False


def mood_dispersion_scale_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Alexander M. Mood's (1954) Non-Parametric Median-Rank Dispersion Scale Test.

    Evaluates whether subgroup dispersions differ without requiring Gaussianity or symmetry.
    For two conditioning strata of sizes n1 and n2 (N = n1 + n2), ranks pooled observations:
      M = sum_{i in sample 1} (R_i - (N + 1) / 2)^2
      E[M] = n1 * (N^2 - 1) / 12
      Var(M) = n1 * n2 * (N + 1) * (N^2 - 4) / 180
      Z_Mood = (M - E[M]) / sqrt(Var(M)) ~ N(0, 1).
    Rejection (|Z_Mood| >= 1.96, p < 0.05) detects significant scale heterogeneity across strata.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        s1 = valid_samples[0]
        s2 = valid_samples[1]
        n1 = len(s1)
        n2 = len(s2)
        N = n1 + n2
        if N < 8:
            return None, None, False

        pooled = np.concatenate([s1, s2])
        ranks = stats.rankdata(pooled)
        ranks_s1 = ranks[:n1]

        mid = (N + 1.0) / 2.0
        m_stat = float(np.sum((ranks_s1 - mid) ** 2))
        e_m = float(n1 * (N ** 2 - 1.0) / 12.0)
        var_m = float(n1 * n2 * (N + 1.0) * (N ** 2 - 4.0) / 180.0)
        if var_m < 1e-12:
            return None, None, False

        z_stat = float((m_stat - e_m) / np.sqrt(var_m))
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
        is_hetero = bool(p_val < 0.05)
        return z_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def double_machine_learning_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, & Robins (2018)
    Double Machine Learning (DML) / Neyman-Orthogonalized Treatment Effect Discrepancy.

    Constructs Neyman-orthogonal partialing-out residuals:
      x_tilde = X - E[X | Z]
      y_tilde = Y - E[Y | Z]
    Computes orthogonalized causal slope:
      beta_DML = <x_tilde, y_tilde> / ||x_tilde||^2
    Evaluates relative discrepancy against naive bivariate slope beta_raw = Cov(X, Y) / Var(X):
      Delta_DML = |beta_raw - beta_DML| / max(1e-6, |beta_raw|)
    Flags observational confounding when Delta_DML > 0.25 (25% divergence).
    """
    n = len(x)
    if n < 20 or np.std(x) < 1e-9 or np.std(z) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        Z_mat = np.column_stack([np.ones(n), z])
        # Residualize X on Z
        b_x, _, _, _ = np.linalg.lstsq(Z_mat, x, rcond=None)
        x_tilde = x - Z_mat @ b_x
        # Residualize Y on Z
        b_y, _, _, _ = np.linalg.lstsq(Z_mat, y, rcond=None)
        y_tilde = y - Z_mat @ b_y

        var_x_tilde = float(np.sum(x_tilde ** 2))
        if var_x_tilde < 1e-12:
            return None, None, False

        beta_dml = float(np.sum(x_tilde * y_tilde) / var_x_tilde)

        # Naive bivariate slope
        var_x = float(np.var(x))
        if var_x < 1e-12:
            return None, None, False
        cov_xy = float(np.cov(x, y)[0, 1])
        beta_raw = float(cov_xy / var_x)

        denom = max(1e-6, abs(beta_raw))
        discrepancy = float(abs(beta_raw - beta_dml) / denom)
        is_confounded = bool(discrepancy > 0.25)
        return beta_dml, discrepancy, is_confounded
    except Exception:
        return None, None, False


def cinelli_extreme_confounder_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    rv_q: Optional[float],
) -> tuple[Optional[float], bool]:
    """Calculates Carlos Cinelli & Chad Hazlett's (2020) Extreme Confounder Benchmark Multiplier.

    Evaluates how many multiples (k) of the top observed confounder's explanatory power
    would be required to completely overturn the estimated treatment relationship:
      Pi = R^2_{X ~ Z} * R^2_{Y ~ Z | X}
      k_Cinelli = RV_q^2 / max(1e-6, Pi)
    If k_Cinelli < 1.0, an unobserved confounder WEAKER than the observed covariate Z
    can completely wipe out the effect (declared FRAGILE). If k_Cinelli >= 1.0, unobserved
    confounders must be stronger than observed controls (declared ROBUST).
    """
    n = len(x)
    if rv_q is None or n < 20 or np.std(x) < 1e-9 or np.std(z) < 1e-9 or np.std(y) < 1e-9:
        return None, False
    try:
        # Bivariate R^2 of X on Z
        r_xz, _ = stats.pearsonr(x, z)
        r2_xz = float(r_xz ** 2)

        # Partial R^2 of Y on Z given X:
        # Residualize Y on X and Z on X
        X_mat = np.column_stack([np.ones(n), x])
        b_yx, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res_yx = y - X_mat @ b_yx

        b_zx, _, _, _ = np.linalg.lstsq(X_mat, z, rcond=None)
        res_zx = z - X_mat @ b_zx

        if np.std(res_yx) < 1e-9 or np.std(res_zx) < 1e-9:
            return None, False

        r_part, _ = stats.pearsonr(res_yx, res_zx)
        r2_yz_given_x = float(r_part ** 2)

        product = max(1e-6, r2_xz * r2_yz_given_x)
        k_mult = float((rv_q ** 2) / product)
        is_fragile = bool(k_mult < 1.0)
        return k_mult, is_fragile
    except Exception:
        return None, False


def goldfeld_quandt_heteroskedasticity_test(
    x: np.ndarray,
    y: np.ndarray,
    omit_fraction: float = 0.20,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Stephen M. Goldfeld & Richard E. Quandt's (1965) Split-Regime Parametric Variance Ratio Test.

    Sorts sample along regressor X and omits central c = floor(omit_fraction * N) observations.
    Fits OLS regressions on lower regime (n1) and upper regime (n2) sub-samples:
      F_GQ = max(MSE_2 / MSE_1, MSE_1 / MSE_2) ~ F(df2, df1).
    Rejection (two-sided p < 0.05) detects systematic regressor-dependent regime heteroskedasticity.
    """
    n = len(x)
    if n < 30 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        idx = np.argsort(x)
        x_s = x[idx]
        y_s = y[idx]

        c = int(np.floor(omit_fraction * n))
        n1 = int((n - c) // 2)
        n2 = n1
        if n1 < 6:
            return None, None, False

        # Lower regime
        x1, y1 = x_s[:n1], y_s[:n1]
        X1 = np.column_stack([np.ones(n1), x1])
        b1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
        res1 = y1 - X1 @ b1
        df1 = n1 - 2
        mse1 = float(np.sum(res1 ** 2) / df1)

        # Upper regime
        x2, y2 = x_s[n - n2:], y_s[n - n2:]
        X2 = np.column_stack([np.ones(n2), x2])
        b2, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
        res2 = y2 - X2 @ b2
        df2 = n2 - 2
        mse2 = float(np.sum(res2 ** 2) / df2)

        if mse1 < 1e-12 or mse2 < 1e-12:
            return None, None, False

        if mse2 >= mse1:
            f_stat = float(mse2 / mse1)
            p_val = float(2.0 * min(1.0, stats.f.sf(f_stat, df2, df1)))
        else:
            f_stat = float(mse1 / mse2)
            p_val = float(2.0 * min(1.0, stats.f.sf(f_stat, df1, df2)))

        has_hetero = bool(p_val < 0.05)
        return f_stat, p_val, has_hetero
    except Exception:
        return None, None, False


def shapiro_francia_normality_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates S. S. Shapiro & R. S. Francia's (1972) Normality Test for Regression Residuals.

    Evaluates squared correlation between ordered OLS residuals e_(i) and standard normal order scores m_i:
      m_i = Phi^{-1}((i - 3/8) / (N + 1/4))
      W' = (sum m_i e_(i))^2 / [ (sum m_i^2) * sum (e_i - e_bar)^2 ]
    Computes Royston's (1983) normalizing transformation for W' ~ N(0, 1).
    Rejection (p < 0.05) diagnoses significant residual non-Gaussianity, invalidating standard OLS inference.
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        res_sorted = np.sort(res)
        res_centered = res - np.mean(res)
        ss_res = float(np.sum(res_centered ** 2))
        if ss_res < 1e-12:
            return None, None, False

        i_vec = np.arange(1, n + 1, dtype=float)
        p_ranks = (i_vec - 0.375) / (n + 0.25)
        m_vec = stats.norm.ppf(p_ranks)
        ss_m = float(np.sum(m_vec ** 2))
        if ss_m < 1e-12:
            return None, None, False

        num = float(np.sum(m_vec * res_sorted)) ** 2
        w_prime = float(num / (ss_m * ss_res))
        w_prime = min(1.0, max(0.001, w_prime))

        # Royston's normalizing transformation for W'
        log_n = np.log(n)
        nu = np.log(max(1e-9, 1.0 - w_prime))
        mu = -0.375 + 1.082 * log_n - 0.0886 * (log_n ** 2)
        sigma = np.exp(-0.485 - 0.048 * log_n)

        z_sf = float((nu - mu) / sigma)
        p_val = float(stats.norm.sf(z_sf))
        is_non_normal = bool(p_val < 0.05)
        return w_prime, p_val, is_non_normal
    except Exception:
        return None, None, False


def thompson_sharp_funnel_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates S. G. Thompson & S. J. Sharp's (1999) Random-Effects Weighted Meta-Regression Funnel Asymmetry Test.

    Incorporates DerSimonian-Laird between-study variance tau^2_DL into random-effects weights:
      w_k* = 1 / (v_k + tau^2_DL)
    Regresses log odds ratio theta_k on standard error SE_k:
      theta_k = alpha + beta * SE_k + epsilon_k*
    Test statistic:
      t_TS = beta_hat / SE(beta_hat) ~ t(K - 2).
    Rejection (p < 0.05) detects significant small-study publication or reporting asymmetry
    while properly conditioning on between-stratum random-effects heterogeneity.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        theta_vec = []
        var_vec = []
        for t in tables_2x2:
            if isinstance(t, (tuple, list)):
                if len(t) == 4 and not isinstance(t[0], (tuple, list)):
                    a, b, c, d = float(t[0]), float(t[1]), float(t[2]), float(t[3])
                else:
                    a, b = float(t[0][0]), float(t[0][1])
                    c, d = float(t[1][0]), float(t[1][1])
            else:
                a, b = float(t[0, 0]), float(t[0, 1])
                c, d = float(t[1, 0]), float(t[1, 1])
            if a + b > 0 and c + d > 0 and a + c > 0 and b + d > 0:
                a_c = a + 0.5
                b_c = b + 0.5
                c_c = c + 0.5
                d_c = d + 0.5
                theta = float(np.log((a_c * d_c) / (b_c * c_c)))
                v = float(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)
                theta_vec.append(theta)
                var_vec.append(v)
        k_val = len(theta_vec)
        if k_val < 3:
            return None, None, False

        y = np.array(theta_vec, dtype=float)
        v = np.array(var_vec, dtype=float)
        se = np.sqrt(v)
        w_fe = 1.0 / v
        W = float(np.sum(w_fe))
        pooled_fe = float(np.sum(w_fe * y) / W)
        Q = float(np.sum(w_fe * (y - pooled_fe) ** 2))
        df_q = k_val - 1
        denom_tau = W - float(np.sum(w_fe ** 2)) / W
        tau2 = max(0.0, float((Q - df_q) / max(1e-9, denom_tau)))
        w_re = 1.0 / (v + tau2)
        sw = np.sqrt(w_re)
        X_mat = np.column_stack([sw, sw * se])
        Y_trans = sw * y
        beta, _, _, _ = np.linalg.lstsq(X_mat, Y_trans, rcond=None)
        res = Y_trans - X_mat @ beta
        df_e = k_val - 2
        if df_e <= 0:
            return None, None, False
        s2 = float(np.sum(res ** 2) / df_e)
        cov = s2 * np.linalg.pinv(X_mat.T @ X_mat)
        se_slope = float(np.sqrt(max(1e-12, cov[1, 1])))
        if se_slope < 1e-12:
            return None, None, False
        t_stat = float(beta[1] / se_slope)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        is_asym = bool(p_val < 0.05)
        return t_stat, p_val, is_asym
    except Exception:
        return None, None, False


def ansari_bradley_scale_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates C. E. Ansari & R. A. Bradley's (1960) Non-Parametric Rank-Sum Scale Dispersion Test.

    Evaluates equality of scale/dispersion parameters across conditioning strata without normality assumptions.
    Compares the two primary conditioning strata samples:
      W_AB = sum_{i in group 1} s_i, where s_i are folded triangular ranks.
    Rejection (two-sided p < 0.05) certifies statistically significant scale heterogeneity.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        s1 = valid_samples[0]
        s2 = valid_samples[1]
        res = stats.ansari(s1, s2)
        stat_w = float(res.statistic)
        p_val = float(res.pvalue)
        is_hetero = bool(p_val < 0.05)
        return stat_w, p_val, is_hetero
    except Exception:
        return None, None, False


def rubin_covariate_balance_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Donald B. Rubin's (2001) Causal Covariate Balance Diagnostics (Rubin's B and R).

    Binarizes exposure X at its median into treated and comparison groups.
    Estimates the linear propensity index q_i = E[T | Z] and evaluates:
      Rubin's B = (|mean(q_1) - mean(q_0)| / sqrt((s_1^2 + s_0^2)/2)) * 100
      Rubin's R = s_1^2 / s_0^2
    Rubin's rule of thumb recommends B < 25% for balance and R in [0.5, 2.0] for variance equity.
    Observational comparisons with B >= 25% or R not in [0.5, 2.0] suffer from severe model dependence.
    """
    n = len(x)
    if n < 20 or np.std(x) < 1e-9 or np.std(z) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        treated = (x > med_x)
        if np.sum(treated) < 5 or np.sum(~treated) < 5:
            return None, None, False

        Z_mat = np.column_stack([np.ones(n), z])
        t_float = treated.astype(float)
        b_prop, _, _, _ = np.linalg.lstsq(Z_mat, t_float, rcond=None)
        p_score = Z_mat @ b_prop

        p1 = p_score[treated]
        p0 = p_score[~treated]
        var1 = float(np.var(p1, ddof=1))
        var0 = float(np.var(p0, ddof=1))
        if var1 < 1e-12 or var0 < 1e-12:
            return None, None, False

        mean_diff = abs(float(np.mean(p1) - np.mean(p0)))
        pooled_sd = np.sqrt((var1 + var0) / 2.0)
        rubin_b = float((mean_diff / max(1e-9, pooled_sd)) * 100.0)
        rubin_r = float(var1 / var0)

        is_imbalanced = bool(rubin_b >= 25.0 or rubin_r < 0.5 or rubin_r > 2.0)
        return rubin_b, rubin_r, is_imbalanced
    except Exception:
        return None, None, False


def brunner_munzel_relative_effect_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Edgar Brunner & Ullrich Munzel's (2000) Generalized Relative Treatment Effect.

    Solves the non-parametric Behrens-Fisher problem for stochastic superiority without equal variances:
      P(Y_low < Y_high) + 0.5 * P(Y_low == Y_high) = 0.5
    Test statistic W_BM follows Brunner-Munzel's t-distribution with Satterthwaite-type degrees of freedom.
    Significant difference (p < 0.05) confirms stochastic separation between exposure tiers.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        y_high = y[x > med_x]
        y_low = y[x <= med_x]
        if len(y_high) < 4 or len(y_low) < 4:
            return None, None, False

        res = stats.brunnermunzel(y_high, y_low)
        w_stat = float(res.statistic)
        p_val = float(res.pvalue)
        has_effect = bool(p_val < 0.05)
        return w_stat, p_val, has_effect
    except Exception:
        return None, None, False


def dagostino_pearson_normality_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Ralph D'Agostino & E. S. Pearson's (1973) Omnibus Normality Test for Regression Residuals.

    Combines D'Agostino skewness Z_1(s) and Anscombe-Glynn kurtosis Z_2(k) transformations:
      K^2 = Z_1(s)^2 + Z_2(k)^2 ~ chi^2(2).
    Rejection (p < 0.05) indicates departure from residual Gaussianity due to excessive skew or kurtosis.
    """
    n = len(x)
    if n < 20 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta
        if np.std(res) < 1e-9:
            return None, None, False

        k2, p_val = stats.normaltest(res)
        k2_stat = float(k2)
        p_float = float(p_val)
        is_non_norm = bool(p_float < 0.05)
        return k2_stat, p_float, is_non_norm
    except Exception:
        return None, None, False


def glejser_heteroskedasticity_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Herbert Glejser's (1969) Regression Test for Heteroskedasticity.

    Regresses absolute OLS residuals |e_i| on regressor X_i:
      |e_i| = gamma_0 + gamma_1 * X_i + nu_i
    Evaluates t_Glejser = gamma_1 / SE(gamma_1) ~ t(N - 2).
    Rejection (two-sided p < 0.05) diagnoses scale dispersion varying systematically with X magnitude.
    """
    n = len(x)
    if n < 15 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta
        abs_res = np.abs(res)

        b_g, _, _, _ = np.linalg.lstsq(X_mat, abs_res, rcond=None)
        res_g = abs_res - X_mat @ b_g
        df_e = n - 2
        if df_e <= 0:
            return None, None, False

        s2 = float(np.sum(res_g ** 2) / df_e)
        cov = s2 * np.linalg.pinv(X_mat.T @ X_mat)
        se_slope = float(np.sqrt(max(1e-12, cov[1, 1])))
        if se_slope < 1e-12:
            return None, None, False

        t_stat = float(b_g[1] / se_slope)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        has_hetero = bool(p_val < 0.05)
        return t_stat, p_val, has_hetero
    except Exception:
        return None, None, False


def moreno_funnel_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Moreno et al.'s (2009) Variance-Weighted Regression Test for Funnel Plot Asymmetry.

    Regresses standardized effect sizes (theta_k / sqrt(v_k)) on 1 / sqrt(v_k) and sqrt(v_k):
      theta_k / sqrt(v_k) = alpha * (1 / sqrt(v_k)) + beta * sqrt(v_k) + epsilon_k
    Evaluates t_Moreno = beta / SE(beta) ~ t(K - 2).
    Testing H_0: beta = 0 provides a variance-dependent asymmetry test invariant to effect size.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        theta_vec = []
        var_vec = []
        for t in tables_2x2:
            if isinstance(t, (tuple, list)):
                if len(t) == 4 and not isinstance(t[0], (tuple, list)):
                    a, b, c, d = float(t[0]), float(t[1]), float(t[2]), float(t[3])
                else:
                    a, b = float(t[0][0]), float(t[0][1])
                    c, d = float(t[1][0]), float(t[1][1])
            else:
                a, b = float(t[0, 0]), float(t[0, 1])
                c, d = float(t[1, 0]), float(t[1, 1])
            if a + b > 0 and c + d > 0 and a + c > 0 and b + d > 0:
                a_c = a + 0.5
                b_c = b + 0.5
                c_c = c + 0.5
                d_c = d + 0.5
                theta = float(np.log((a_c * d_c) / (b_c * c_c)))
                v = float(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)
                theta_vec.append(theta)
                var_vec.append(v)
        k_val = len(theta_vec)
        if k_val < 3:
            return None, None, False

        y = np.array(theta_vec, dtype=float)
        v = np.array(var_vec, dtype=float)
        se = np.sqrt(v)
        y_trans = y / se
        x1 = 1.0 / se
        x2 = se
        X_mat = np.column_stack([x1, x2])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y_trans, rcond=None)
        res = y_trans - X_mat @ beta
        df_e = k_val - 2
        if df_e <= 0:
            return None, None, False
        s2 = float(np.sum(res ** 2) / df_e)
        cov = s2 * np.linalg.pinv(X_mat.T @ X_mat)
        se_beta = float(np.sqrt(max(1e-12, cov[1, 1])))
        if se_beta < 1e-12:
            return None, None, False
        t_stat = float(beta[1] / se_beta)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        is_asym = bool(p_val < 0.05)
        return t_stat, p_val, is_asym
    except Exception:
        return None, None, False


def siegel_tukey_scale_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Sidney Siegel & John W. Tukey's (1960) Non-Parametric Rank Extremes Scale Test.

    Evaluates relative dispersion/scale between two primary conditioning strata without normality assumptions.
    Assigns alternating extreme ranks from low to high:
      r(1)=1, r(N)=2, r(N-1)=3, r(2)=4, r(3)=5, r(N-2)=6, r(N-3)=7, ...
    Test statistic:
      W_ST = sum_{i in group 1} r_i, E[W] = n1(N + 1)/2, Var(W) = n1*n2*(N + 1)/12.
    Rejection (|Z| >= 1.96, p < 0.05) certifies statistically significant scale/spread divergence.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        s1 = valid_samples[0]
        s2 = valid_samples[1]
        n1 = len(s1)
        n2 = len(s2)
        N = n1 + n2

        combined = np.concatenate([s1, s2])
        group_id = np.concatenate([np.ones(n1, dtype=int), np.zeros(n2, dtype=int)])
        order = np.argsort(combined)
        sorted_groups = group_id[order]

        ranks = np.zeros(N, dtype=float)
        left = 0
        right = N - 1
        curr_rank = 1
        turn = 0
        while left <= right:
            if turn == 0:
                ranks[left] = curr_rank
                left += 1
                curr_rank += 1
                turn = 1
            elif turn == 1:
                ranks[right] = curr_rank
                right -= 1
                curr_rank += 1
                if left <= right:
                    ranks[right] = curr_rank
                    right -= 1
                    curr_rank += 1
                turn = 2
            elif turn == 2:
                ranks[left] = curr_rank
                left += 1
                curr_rank += 1
                if left <= right:
                    ranks[left] = curr_rank
                    left += 1
                    curr_rank += 1
                turn = 1

        w_st = float(np.sum(ranks[sorted_groups == 1]))
        e_w = float(n1 * (N + 1) / 2.0)
        var_w = float(n1 * n2 * (N + 1) / 12.0)
        if var_w < 1e-12:
            return None, None, False
        z_stat = float((w_st - e_w) / np.sqrt(var_w))
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
        is_hetero = bool(p_val < 0.05)
        return w_st, p_val, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_rubin_smd_balance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Rosenbaum & Rubin's (1985) Standardized Mean Difference (SMD) & Mahalanobis Distance.

    Binarizes exposure X at its median into exposure cohorts (D = 1 vs D = 0).
    Evaluates standardized mean difference:
      SMD = |mean(Z_1) - mean(Z_0)| / sqrt((s_1^2 + s_0^2)/2)
    and squared Mahalanobis distance D^2 = (mu_1 - mu_0)^T Sigma_pooled^{-1} (mu_1 - mu_0).
    Covariate balance is flagged as compromised if SMD > 0.10 or Mahalanobis D^2 > 0.25.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        treated = (x > med_x)
        if np.sum(treated) < 4 or np.sum(~treated) < 4:
            return None, None, False

        z1 = z[treated]
        z0 = z[~treated]
        var1 = float(np.var(z1, ddof=1))
        var0 = float(np.var(z0, ddof=1))
        if var1 < 1e-12 or var0 < 1e-12:
            return None, None, False

        mean_diff = abs(float(np.mean(z1) - np.mean(z0)))
        pooled_sd = np.sqrt((var1 + var0) / 2.0)
        smd = float(mean_diff / max(1e-9, pooled_sd))

        X_feat1 = np.column_stack([x[treated], z1])
        X_feat0 = np.column_stack([x[~treated], z0])
        diff_mean = np.mean(X_feat1, axis=0) - np.mean(X_feat0, axis=0)
        cov1 = np.cov(X_feat1, rowvar=False)
        cov0 = np.cov(X_feat0, rowvar=False)
        cov_pooled = (cov1 * (len(z1) - 1) + cov0 * (len(z0) - 1)) / max(1, n - 2)
        inv_cov = np.linalg.pinv(cov_pooled)
        d2_mahal = float(diff_mean.T @ inv_cov @ diff_mean)

        is_imbalanced = bool(smd > 0.10 or d2_mahal > 0.25)
        return smd, d2_mahal, is_imbalanced
    except Exception:
        return None, None, False


def imbens_rubin_normalized_difference_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], bool]:
    """Calculates Guido Imbens & Donald Rubin's (2015) Normalized Difference Metric (Delta_Norm).

    Evaluates scale-invariant covariate overlap between exposure strata:
      Delta_Norm = |mean(Z_1) - mean(Z_0)| / sqrt(0.5 * (s_1^2 + s_0^2))
    Unlike standard t-statistics, Delta_Norm does not scale with sample size N.
    Values exceeding 0.25 indicate severe lack of common support and propensity overlap.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, False
    try:
        med_x = float(np.median(x))
        treated = (x > med_x)
        if np.sum(treated) < 4 or np.sum(~treated) < 4:
            return None, False

        z1 = z[treated]
        z0 = z[~treated]
        var1 = float(np.var(z1, ddof=1))
        var0 = float(np.var(z0, ddof=1))
        if var1 < 1e-12 or var0 < 1e-12:
            return None, False

        mean_diff = abs(float(np.mean(z1) - np.mean(z0)))
        denom = np.sqrt(0.5 * (var1 + var0))
        delta_norm = float(mean_diff / max(1e-9, denom))
        is_excessive = bool(delta_norm > 0.25)
        return delta_norm, is_excessive
    except Exception:
        return None, False


def anderson_darling_normality_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Anderson & Darling's (1954) Empirical Distribution Function Residual Normality Test.

    Evaluates whether standardized OLS residuals depart from Gaussianity across the entire distribution,
    giving substantial weight to heavy tails:
      A^2 = -n - (1/n) sum_{i=1}^n (2i - 1) [ln Phi(z_(i)) + ln(1 - Phi(z_(n+1-i)))]
    Compares against the 5% significance critical value (approx 0.752 with Stephens adjustment).
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta
        if np.std(res) < 1e-9:
            return None, None, False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            res_norm = stats.anderson(res, dist='norm')
        a2_stat = float(res_norm.statistic)
        crit_val_5pct = float(res_norm.critical_values[2])
        is_non_normal = bool(a2_stat > crit_val_5pct)
        return a2_stat, crit_val_5pct, is_non_normal
    except Exception:
        return None, None, False


def harvey_multiplicative_heteroskedasticity_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Andrew C. Harvey's (1976) Multiplicative Error Variance Heteroskedasticity LM Test.

    Models disturbance variance as sigma_i^2 = sigma^2 * exp(gamma * x_i).
    Evaluates score vector g_i = (e_i^2 / sigma_hat^2) - 1 in auxiliary regression on X:
      LM = 0.5 * g^T Z (Z^T Z)^{-1} Z^T g ~ chi^2(1).
    Rejection (p < 0.05) diagnoses exponential variance inflation invalidating homoskedastic standard errors.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta
        s2 = float(np.mean(res ** 2))
        if s2 < 1e-12:
            return None, None, False

        g_vec = (res ** 2) / s2 - 1.0
        b_aux, _, _, _ = np.linalg.lstsq(X_mat, g_vec, rcond=None)
        g_pred = X_mat @ b_aux
        lm_stat = float(0.5 * np.sum(g_pred ** 2))
        p_val = float(stats.chi2.sf(lm_stat, df=1))
        has_hetero = bool(p_val < 0.05)
        return lm_stat, p_val, has_hetero
    except Exception:
        return None, None, False


def sterne_funnel_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Sterne & Egger's (2000) Inverse-Variance Weighted Funnel Asymmetry Regression Test.

    Regresses standardized stratum effect sizes theta_k / SE_k against precision 1 / SE_k and constant 1:
      theta_k / SE_k = alpha_0 * (1 / SE_k) + alpha_1 + epsilon_k
    where weights w_k = 1 / v_k = 1 / SE_k^2 are implicitly incorporated by OLS in transformed coordinates.
    Evaluates t_Sterne = alpha_hat_1 / SE(alpha_hat_1) ~ t(K - 2).
    Testing H_0: alpha_1 = 0 identifies reporting/publication asymmetry where effect sizes correlate with standard error.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        theta_vec = []
        var_vec = []
        for t in tables_2x2:
            if isinstance(t, (tuple, list)):
                if len(t) == 4 and not isinstance(t[0], (tuple, list)):
                    a, b, c, d = float(t[0]), float(t[1]), float(t[2]), float(t[3])
                else:
                    a, b = float(t[0][0]), float(t[0][1])
                    c, d = float(t[1][0]), float(t[1][1])
            else:
                a, b = float(t[0, 0]), float(t[0, 1])
                c, d = float(t[1, 0]), float(t[1, 1])
            if a + b > 0 and c + d > 0 and a + c > 0 and b + d > 0:
                a_c = a + 0.5
                b_c = b + 0.5
                c_c = c + 0.5
                d_c = d + 0.5
                theta = float(np.log((a_c * d_c) / (b_c * c_c)))
                v = float(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)
                theta_vec.append(theta)
                var_vec.append(v)
        k_val = len(theta_vec)
        if k_val < 3:
            return None, None, False

        y = np.array(theta_vec, dtype=float)
        v = np.array(var_vec, dtype=float)
        se = np.sqrt(v)
        y_trans = y / se
        x1 = 1.0 / se
        x2 = np.ones(k_val, dtype=float)
        X_mat = np.column_stack([x1, x2])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y_trans, rcond=None)
        res = y_trans - X_mat @ beta
        df_e = k_val - 2
        if df_e <= 0:
            return None, None, False
        s2 = float(np.sum(res ** 2) / df_e)
        cov = s2 * np.linalg.pinv(X_mat.T @ X_mat)
        se_beta1 = float(np.sqrt(max(1e-12, cov[1, 1])))
        if se_beta1 < 1e-12:
            return None, None, False
        t_stat = float(beta[1] / se_beta1)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        is_asym = bool(p_val < 0.05 and abs(t_stat) >= 1.96)
        return t_stat, p_val, is_asym
    except Exception:
        return None, None, False


def klotz_normal_scores_scale_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Jerome Klotz's (1962) Non-Parametric Normal Scores Rank Scale Dispersion Test.

    Evaluates relative dispersion/scale between two conditioning strata without assuming normality.
    Assigns squared standard normal quantile inverse scores to pooled ranks:
      a_i = [Phi^{-1}(R_i / (N + 1))]^2
    The test statistic is K = sum_{i in stratum 1} a_i.
    Under H_0: E[K] = n1 * mean(a), Var(K) = [n1 * n2 / (N * (N - 1))] * sum((a_i - mean(a))^2).
    Standardized Z = (K - E[K]) / sqrt(Var(K)) ~ N(0, 1).
    Rejection (p < 0.05) detects significant scale heterogeneity across strata.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        s1 = valid_samples[0]
        s2 = valid_samples[1]
        n1 = len(s1)
        n2 = len(s2)
        N = n1 + n2

        combined = np.concatenate([s1, s2])
        ranks = stats.rankdata(combined)
        u = ranks / (N + 1.0)
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        scores = stats.norm.ppf(u) ** 2

        k_stat = float(np.sum(scores[:n1]))
        mean_score = float(np.mean(scores))
        e_k = float(n1 * mean_score)
        var_k = float((n1 * n2) / (N * (N - 1.0)) * np.sum((scores - mean_score) ** 2))
        if var_k < 1e-12:
            return None, None, False
        z_stat = float((k_stat - e_k) / np.sqrt(var_k))
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
        is_hetero = bool(p_val < 0.05)
        return z_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def austin_covariate_variance_ratio_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], bool]:
    """Calculates Peter C. Austin's (2009) Covariate Variance Ratio Balance Metric (VR_Austin).

    Binarizes exposure X at its median into treated (D = 1) and control (D = 0) cohorts.
    Evaluates the ratio of sample variances of covariate Z across exposure groups:
      VR = s_1^2 / s_0^2
    Satisfactory balance requires VR in [0.80, 1.25] (relaxed to [0.50, 2.00] for observational data).
    Values VR < 0.50 or VR > 2.00 indicate severe covariate dispersion imbalance, risking model extrapolation.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, False
    try:
        med_x = float(np.median(x))
        treated = (x > med_x)
        if np.sum(treated) < 4 or np.sum(~treated) < 4:
            return None, False

        z1 = z[treated]
        z0 = z[~treated]
        var1 = float(np.var(z1, ddof=1))
        var0 = float(np.var(z0, ddof=1))
        if var1 < 1e-12 or var0 < 1e-12:
            return None, False

        vr = float(var1 / var0)
        is_imbal = bool(vr < 0.50 or vr > 2.00)
        return vr, is_imbal
    except Exception:
        return None, False


def stuart_prognostic_score_discrepancy_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], bool]:
    """Calculates Elizabeth A. Stuart's (2010) Prognostic Score Discrepancy Metric (Delta_prog).

    Models baseline outcome prognosis Psi(Z) = E[Y | Z, D=0] by regressing Y on Z in the control cohort.
    Projects predicted baseline potential outcomes across both treated and control units:
      Delta_prog = |mean(Psi_1) - mean(Psi_0)| / SD(Psi_0)
    Quantifies structural baseline outcome risk divergence between treatment arms.
    Values Delta_prog > 0.25 indicate severe prognostic disparity guaranteeing residual confounding.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9 or np.std(y) < 1e-9:
        return None, False
    try:
        med_x = float(np.median(x))
        treated = (x > med_x)
        if np.sum(treated) < 4 or np.sum(~treated) < 4:
            return None, False

        z_ctrl = z[~treated]
        y_ctrl = y[~treated]
        if np.std(z_ctrl) < 1e-9:
            return None, False

        Z_ctrl_mat = np.column_stack([np.ones(len(z_ctrl)), z_ctrl])
        b_prog, _, _, _ = np.linalg.lstsq(Z_ctrl_mat, y_ctrl, rcond=None)

        Z_all_mat = np.column_stack([np.ones(n), z])
        psi_all = Z_all_mat @ b_prog
        psi_1 = psi_all[treated]
        psi_0 = psi_all[~treated]

        sd_0 = float(np.std(psi_0, ddof=1))
        if sd_0 < 1e-9:
            return None, False

        mean_diff = abs(float(np.mean(psi_1) - np.mean(psi_0)))
        delta_prog = float(mean_diff / sd_0)
        is_discrepant = bool(delta_prog > 0.25)
        return delta_prog, is_discrepant
    except Exception:
        return None, False


def lilliefors_normality_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Hubert W. Lilliefors' (1967) Composite Normality Goodness-of-Fit Test.

    Evaluates whether standardized OLS residuals depart from Gaussianity when mean and variance are estimated:
      D = max_{1 <= i <= N} max( |i/N - Phi(z_(i))|, |Phi(z_(i)) - (i-1)/N| )
    Evaluates exact critical values and p-values using Dallal & Wilkinson (1986) / Stephens (1974) formulations:
      Crit 5% = 0.886 / sqrt(N).
    Rejection (p < 0.05) detects empirical distribution function departure from normality.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta
        s_e = float(np.std(res, ddof=1))
        if s_e < 1e-12:
            return None, None, False

        z_sorted = np.sort((res - np.mean(res)) / s_e)
        i_vec = np.arange(1, n + 1, dtype=float)
        cdf_vals = stats.norm.cdf(z_sorted)
        d_plus = np.max(i_vec / n - cdf_vals)
        d_minus = np.max(cdf_vals - (i_vec - 1.0) / n)
        d_stat = float(max(d_plus, d_minus))

        crit_5pct = float(0.886 / np.sqrt(n))
        n_adj = n + 2.78019
        term1 = -7.01256 * (d_stat ** 2) * n_adj
        term2 = 2.99587 * d_stat * np.sqrt(n_adj)
        term3 = -0.122119 + 0.974598 / np.sqrt(n) + 1.67997 / n
        log_p = term1 + term2 + term3
        log_p = min(0.0, max(-35.0, log_p))
        p_val = float(np.exp(log_p))
        p_val = min(1.0, max(1e-15, p_val))
        is_non_normal = bool(p_val < 0.05 or d_stat > crit_5pct)
        return d_stat, p_val, is_non_normal
    except Exception:
        return None, None, False


def park_loglinear_heteroskedasticity_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Richard E. Park's (1966) Log-Linear Residual Variance Heteroskedasticity Test.

    Models heteroskedastic error variance as a log-linear function of regressor X:
      ln(e_i^2) = gamma_0 + gamma_1 * x_i + v_i
    Evaluates t_Park = gamma_hat_1 / SE(gamma_hat_1) ~ t(N - 2).
    Rejection (|t| >= 1.96, p < 0.05) diagnoses multiplicative / exponential variance expansion.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta
        res_sq = np.maximum(res ** 2, 1e-12)
        log_e2 = np.log(res_sq)

        gamma, _, _, _ = np.linalg.lstsq(X_mat, log_e2, rcond=None)
        v_res = log_e2 - X_mat @ gamma
        df_e = n - 2
        if df_e <= 0:
            return None, None, False
        s2_v = float(np.sum(v_res ** 2) / df_e)
        cov = s2_v * np.linalg.pinv(X_mat.T @ X_mat)
        se_gamma1 = float(np.sqrt(max(1e-12, cov[1, 1])))
        if se_gamma1 < 1e-12:
            return None, None, False

        t_stat = float(gamma[1] / se_gamma1)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        has_hetero = bool(p_val < 0.05 and abs(t_stat) >= 1.96)
        return t_stat, p_val, has_hetero
    except Exception:
        return None, None, False


def lfk_index_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], bool]:
    """Calculates Luis Furuya-Kanamori & Suhail A. R. Doi's (2018) LFK Index of Funnel Plot Asymmetry.

    Evaluates publication and selection bias across binary stratified contingency strata on the Doi plot.
    Converts stratum log-odds ratios into standardized Z-scores relative to the precision-weighted pooled effect.
    Computes empirical area discrepancy between right (A_R) and left (A_L) halves of the distribution:
      LFK = 5 * (A_R - A_L) / (A_R + A_L)
    Interpretation:
      |LFK| <= 1.0: No asymmetry (symmetric precision)
      1.0 < |LFK| <= 2.0: Minor asymmetry
      |LFK| > 2.0: Major asymmetry (reporting bias / small-stratum distortion)
    """
    if len(tables_2x2) < 3:
        return None, False
    try:
        theta_vec = []
        var_vec = []
        for t in tables_2x2:
            if isinstance(t, (tuple, list)):
                if len(t) == 4 and not isinstance(t[0], (tuple, list)):
                    a, b, c, d = float(t[0]), float(t[1]), float(t[2]), float(t[3])
                else:
                    a, b = float(t[0][0]), float(t[0][1])
                    c, d = float(t[1][0]), float(t[1][1])
            else:
                a, b = float(t[0, 0]), float(t[0, 1])
                c, d = float(t[1, 0]), float(t[1, 1])
            if a + b > 0 and c + d > 0 and a + c > 0 and b + d > 0:
                a_c = a + 0.5
                b_c = b + 0.5
                c_c = c + 0.5
                d_c = d + 0.5
                theta = float(np.log((a_c * d_c) / (b_c * c_c)))
                v = float(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)
                theta_vec.append(theta)
                var_vec.append(v)
        k_val = len(theta_vec)
        if k_val < 3:
            return None, False

        y = np.array(theta_vec, dtype=float)
        v = np.array(var_vec, dtype=float)
        w = 1.0 / v
        pooled_theta = float(np.sum(w * y) / np.sum(w))
        z_scores = (y - pooled_theta) / np.sqrt(v)

        right_mask = (z_scores > 0)
        left_mask = (z_scores < 0)
        a_r = float(np.sum(z_scores[right_mask] * np.sqrt(w[right_mask]))) if np.any(right_mask) else 0.0
        a_l = float(np.sum(np.abs(z_scores[left_mask]) * np.sqrt(w[left_mask]))) if np.any(left_mask) else 0.0

        denom = a_r + a_l
        if denom < 1e-9:
            return 0.0, False
        lfk_val = float(5.0 * (a_r - a_l) / denom)
        is_asym = bool(abs(lfk_val) > 2.0)
        return lfk_val, is_asym
    except Exception:
        return None, False


def conover_squared_ranks_scale_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates W. J. Conover & Ronald L. Iman's (1981) Squared Ranks Scale Dispersion Test.

    Evaluates relative dispersion/scale between two conditioning strata without assuming normality.
    Computes absolute deviations from group medians: D_ij = |X_ij - median(X_j)|.
    Ranks pooled absolute deviations across both strata: R_i = rank(D_i).
    Assigns squared ranks: S_i = R_i^2.
    Test statistic T = sum_{i in stratum 1} S_i.
    Under H_0: E[T] = n1 * mean(S), Var(T) = [n1 * n2 / (N * (N - 1))] * sum((S_i - mean(S))^2).
    Standardized Z = (T - E[T]) / sqrt(Var(T)) ~ N(0, 1).
    Rejection (p < 0.05) detects significant non-parametric scale spread heterogeneity.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        s1 = valid_samples[0]
        s2 = valid_samples[1]
        n1 = len(s1)
        n2 = len(s2)
        N = n1 + n2

        dev1 = np.abs(s1 - np.median(s1))
        dev2 = np.abs(s2 - np.median(s2))
        combined_dev = np.concatenate([dev1, dev2])
        ranks = stats.rankdata(combined_dev)
        sq_ranks = ranks ** 2

        t_stat = float(np.sum(sq_ranks[:n1]))
        mean_sq = float(np.mean(sq_ranks))
        e_t = float(n1 * mean_sq)
        var_t = float((n1 * n2) / (N * (N - 1.0)) * np.sum((sq_ranks - mean_sq) ** 2))
        if var_t < 1e-12:
            return None, None, False
        z_stat = float((t_stat - e_t) / np.sqrt(var_t))
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
        is_hetero = bool(p_val < 0.05)
        return z_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def hansen_bowers_balance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Ben B. Hansen & Jake Bowers' (2008) Omnibus Covariate Balance Test.

    Binarizes exposure X at its median into treated (D = 1) and control (D = 0) cohorts.
    Evaluates standardized mean difference vector across primary and higher-order covariate moments [Z, Z^2].
    Computes omnibus quadratic form chi2 = d^T [Var(d)]^{-1} d ~ chi2(p).
    Rejection (p < 0.05) certifies multivariate covariate balance failure under the unconfoundedness model.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        treated = (x > med_x)
        n1 = int(np.sum(treated))
        n0 = int(np.sum(~treated))
        if n1 < 6 or n0 < 6:
            return None, None, False

        z_norm = (z - np.mean(z)) / max(1e-9, float(np.std(z)))
        W_mat = np.column_stack([z_norm, z_norm ** 2])
        p_dim = W_mat.shape[1]

        w1 = W_mat[treated]
        w0 = W_mat[~treated]
        mean_diff = np.mean(w1, axis=0) - np.mean(w0, axis=0)

        cov1 = np.cov(w1, rowvar=False)
        cov0 = np.cov(w0, rowvar=False)
        pooled_cov = ((n1 - 1.0) * cov1 + (n0 - 1.0) * cov0) / (n1 + n0 - 2.0)
        var_d = (1.0 / n1 + 1.0 / n0) * pooled_cov

        inv_var_d = np.linalg.pinv(var_d)
        chi2_val = float(mean_diff.T @ inv_var_d @ mean_diff)
        chi2_val = max(0.0, chi2_val)
        p_val = float(stats.chi2.sf(chi2_val, df=p_dim))
        is_imbal = bool(p_val < 0.05)
        return chi2_val, p_val, is_imbal
    except Exception:
        return None, None, False


def rosenbaum_rubin_caliper_discordance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum & Donald B. Rubin's (1985) Propensity Score Caliper Discordance Metric.

    Binarizes exposure X at its median into treated (D = 1) and control (D = 0) cohorts.
    Fits logistic propensity model P(D = 1 | Z) and evaluates logit propensity scores ell = ln(e / (1 - e)).
    Applies Austin (2011) / Rosenbaum optimal caliper radius epsilon = 0.20 * SD(ell).
    Quantifies fraction of treated units lacking any control counterpart within epsilon:
      delta_caliper = (1 / n1) * sum_{i in treated} I(min_{j in control} |ell_i - ell_j| > epsilon).
    Values delta_caliper > 0.15 indicate severe common support discordance and extrapolation risk.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        treated = (x > med_x).astype(float)
        n1 = int(np.sum(treated == 1.0))
        n0 = int(np.sum(treated == 0.0))
        if n1 < 6 or n0 < 6:
            return None, None, False

        z_std = (z - np.mean(z)) / max(1e-9, float(np.std(z)))
        Z_mat = np.column_stack([np.ones(n), z_std])

        beta = np.zeros(2)
        for _ in range(15):
            p_hat = 1.0 / (1.0 + np.exp(-np.clip(Z_mat @ beta, -25.0, 25.0)))
            p_hat = np.clip(p_hat, 1e-5, 1.0 - 1e-5)
            W_diag = p_hat * (1.0 - p_hat)
            grad = Z_mat.T @ (treated - p_hat)
            Hess = Z_mat.T @ (Z_mat * W_diag[:, None])
            try:
                step = np.linalg.solve(Hess, grad)
                beta += step
                if np.max(np.abs(step)) < 1e-5:
                    break
            except Exception:
                break

        eta = Z_mat @ beta
        sd_eta = float(np.std(eta))
        if sd_eta < 1e-9:
            return 0.0, 0.0, False

        caliper_radius = float(0.20 * sd_eta)
        eta_1 = eta[treated == 1.0]
        eta_0 = eta[treated == 0.0]

        diff_matrix = np.abs(eta_1[:, None] - eta_0[None, :])
        min_dists = np.min(diff_matrix, axis=1)
        discordant_count = int(np.sum(min_dists > caliper_radius))
        discordance_rate = float(discordant_count / n1)
        is_discordant = bool(discordance_rate > 0.15)
        return discordance_rate, caliper_radius, is_discordant
    except Exception:
        return None, None, False


def cramer_von_mises_normality_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Harald Cramér (1928) & Richard von Mises' (1928) EDF Residual Normality Test.

    Evaluates integrated quadratic deviation between empirical residual CDF and theoretical normal CDF:
      W^2 = 1 / (12 * N) + sum_{i=1}^N [Phi(z_(i)) - (2i - 1) / (2N)]^2
    Applies Stephens' (1974) finite-sample adjustment for estimated mean and variance:
      W^{*2} = W^2 * (1 + 0.5 / N).
    Critical value at alpha = 0.05 is 0.126.
    Rejection (W^{*2} > 0.126, p < 0.05) detects empirical residual departure from Gaussianity.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta
        s_e = float(np.std(res, ddof=1))
        if s_e < 1e-12:
            return None, None, False

        z_sorted = np.sort((res - np.mean(res)) / s_e)
        i_vec = np.arange(1, n + 1, dtype=float)
        cdf_vals = stats.norm.cdf(z_sorted)

        diffs = cdf_vals - (2.0 * i_vec - 1.0) / (2.0 * n)
        w2_raw = float(1.0 / (12.0 * n) + np.sum(diffs ** 2))
        w2_adj = float(w2_raw * (1.0 + 0.5 / n))

        if w2_adj <= 0.0275:
            p_val = 1.0 - 0.5 * np.exp(-1.0 * (0.0275 - w2_adj) * 40.0)
        else:
            p_val = float(np.exp(0.505 - 5.926 * w2_adj - 6.995 * (w2_adj ** 2)))
        p_val = min(1.0, max(1e-15, p_val))
        is_non_normal = bool(p_val < 0.05 or w2_adj > 0.126)
        return w2_adj, p_val, is_non_normal
    except Exception:
        return None, None, False


def cook_weisberg_heteroskedasticity_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates R. Dennis Cook & Sanford Weisberg's (1983) Fitted-Value Heteroskedasticity Score Test.

    Models error variance as an exponential function of fitted values: sigma_i^2 = sigma^2 * exp(lambda * y_hat_i).
    Evaluates scaled squared residuals u_i = e_i^2 / sigma_hat^2.
    Computes score statistic:
      S_CW = 0.5 * [sum((u_i - 1) * (y_hat_i - mean(y_hat)))]^2 / sum((y_hat_i - mean(y_hat))^2) ~ chi2(1).
    Rejection (S_CW >= 3.841, p < 0.05) detects fitted-value variance scaling.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        y_hat = X_mat @ beta
        res = y - y_hat
        sigma2 = float(np.mean(res ** 2))
        if sigma2 < 1e-12:
            return None, None, False

        u_i = (res ** 2) / sigma2
        y_hat_dev = y_hat - np.mean(y_hat)
        denom = float(np.sum(y_hat_dev ** 2))
        if denom < 1e-12:
            return None, None, False

        numer = float(np.sum((u_i - 1.0) * y_hat_dev))
        score_cw = float(0.5 * (numer ** 2) / denom)
        p_val = float(stats.chi2.sf(score_cw, df=1))
        has_hetero = bool(score_cw >= 3.841 or p_val < 0.05)
        return score_cw, p_val, has_hetero
    except Exception:
        return None, None, False


def pet_peese_meta_regression_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Stanley & Doucouliagos (2012, 2014) PET-PEESE Meta-Regression Falsification Test.

    Evaluates whether genuine empirical effects survive when standard error approaches zero (infinite precision).
    Computes stratum log-odds ratios and standard errors via Haldane-Anscombe continuity corrections:
      theta_k = ln((a_k + 0.5)(d_k + 0.5) / ((b_k + 0.5)(c_k + 0.5)))
      v_k = 1 / (a_k + 0.5) + 1 / (b_k + 0.5) + 1 / (c_k + 0.5) + 1 / (d_k + 0.5),  SE_k = sqrt(v_k).
    1. Precision-Effect Test (PET):
       WLS regression theta_k = alpha_0 + beta_1 * SE_k + e_k with weights w_k = 1 / v_k.
       Transformed: theta_k / SE_k = alpha_0 * (1 / SE_k) + beta_1 + e_k^*.
       t_PET = beta_1_hat / SE(beta_1_hat) tests asymmetry H_0: beta_1 = 0 (df = K - 2).
    2. Precision-Effect Estimate with Standard Error (PEESE):
       WLS regression theta_k = alpha_0 + beta_1 * SE_k^2 + e_k.
       Transformed: theta_k / SE_k = alpha_0 * (1 / SE_k) + beta_1 * SE_k + e_k^*.
       alpha_0_hat is the bias-corrected limit estimate as precision approaches infinity.
    Rejection (pet_p < 0.05 or |t_PET| > 2.0) flags small-stratum publication/reporting bias.
    """
    if len(tables_2x2) < 3:
        return None, None, None, None, False
    try:
        theta_list = []
        se_list = []
        for tab in tables_2x2:
            if isinstance(tab, (tuple, list)):
                a, b, c, d = float(tab[0]), float(tab[1]), float(tab[2]), float(tab[3])
            else:
                a, b = float(tab[0, 0]), float(tab[0, 1])
                c, d = float(tab[1, 0]), float(tab[1, 1])
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
            lor = float(np.log((a * d) / (b * c)))
            v = float(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
            theta_list.append(lor)
            se_list.append(np.sqrt(v))

        y = np.array(theta_list, dtype=float)
        se = np.array(se_list, dtype=float)
        K = len(y)
        if K < 3 or np.any(se < 1e-9):
            return None, None, None, None, False

        # PET: y / se = alpha_0 * (1 / se) + beta_1
        y_trans = y / se
        X_pet = np.column_stack([1.0 / se, np.ones(K)])
        beta_pet, _, _, _ = np.linalg.lstsq(X_pet, y_trans, rcond=None)
        residuals_pet = y_trans - X_pet @ beta_pet
        mse_pet = float(np.sum(residuals_pet ** 2) / max(1, K - 2))
        cov_pet = mse_pet * np.linalg.pinv(X_pet.T @ X_pet)
        se_beta1_pet = float(np.sqrt(max(1e-12, cov_pet[1, 1])))
        t_pet = float(beta_pet[1] / se_beta1_pet)
        p_pet = float(2.0 * stats.t.sf(abs(t_pet), df=K - 2))

        # PEESE: y / se = alpha_0 * (1 / se) + beta_1 * se
        X_peese = np.column_stack([1.0 / se, se])
        beta_peese, _, _, _ = np.linalg.lstsq(X_peese, y_trans, rcond=None)
        residuals_peese = y_trans - X_peese @ beta_peese
        mse_peese = float(np.sum(residuals_peese ** 2) / max(1, K - 2))
        cov_peese = mse_peese * np.linalg.pinv(X_peese.T @ X_peese)
        se_alpha0_peese = float(np.sqrt(max(1e-12, cov_peese[0, 0])))
        t_peese = float(beta_peese[0] / se_alpha0_peese)
        p_peese = float(2.0 * stats.t.sf(abs(t_peese), df=K - 2))

        is_asym = bool(p_pet < 0.05 or abs(t_pet) > 2.0)
        return t_pet, p_pet, t_peese, p_peese, is_asym
    except Exception:
        return None, None, None, None, False


def sukhatme_scale_dispersion_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates B. V. Sukhatme's (1957, 1958) Non-Parametric Rank-Order Scale Dispersion Test.

    Evaluates scale/variance equality between two independent conditioning strata with differing medians:
      Let u_i = X_i - median(X), v_j = Y_j - median(Y).
      Sukhatme's test statistic T counts co-occurring quadrants:
        T = sum_{i=1}^m sum_{j=1}^n [I(u_i > 0 and v_j > 0) + I(u_i < 0 and v_j < 0)].
      Under H_0: sigma_X = sigma_Y:
        E[T] = (m * n) / 4,  Var(T) = [m * n * (m + n + 1)] / 48.
        Z_Sukh = (T - E[T]) / sqrt(Var(T)) ~ N(0, 1).
    Rejection (|Z_Sukh| >= 1.96, p < 0.05) detects significant scale dispersion heterogeneity between cohorts.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        s1 = valid_samples[0]
        s2 = valid_samples[1]
        m = len(s1)
        n = len(s2)

        u = s1 - np.median(s1)
        v = s2 - np.median(s2)

        u_pos = float(np.sum(u > 0))
        u_neg = float(np.sum(u < 0))
        v_pos = float(np.sum(v > 0))
        v_neg = float(np.sum(v < 0))

        t_stat = float(u_pos * v_pos + u_neg * v_neg)
        e_t = float(m * n / 4.0)
        var_t = float((m * n * (m + n + 1.0)) / 48.0)
        if var_t < 1e-12:
            return None, None, False

        z_stat = float((t_stat - e_t) / np.sqrt(var_t))
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
        is_hetero = bool(p_val < 0.05)
        return z_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def imai_van_dyk_gps_balance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Kosuke Imai & David A. van Dyk's (2004) Generalized Propensity Score Covariate Balance Test.

    Evaluates whether conditioning on continuous generalized propensity scores eliminates residual covariate dependence.
    Fits quadratic GPS model e_hat(Z) = E[X | Z] = alpha_0 + alpha_1 * Z + alpha_2 * Z^2.
    Evaluates residual exposure X_tilde = X - e_hat(Z).
    Calculates Pearson correlation r_GPS = Corr(X_tilde, Z) and t-statistic:
      t_GPS = r_GPS * sqrt(n - 2) / sqrt(1 - r_GPS^2) ~ t(n - 2).
    A significant residual correlation (p < 0.05 and |r_GPS| > 0.10) indicates that propensity adjustment
    fails to balance the candidate confounder.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, None, False
    try:
        z_norm = (z - np.mean(z)) / max(1e-9, float(np.std(z)))
        Z_mat = np.column_stack([np.ones(n), z_norm, z_norm ** 2])
        beta, _, _, _ = np.linalg.lstsq(Z_mat, x, rcond=None)
        x_tilde = x - Z_mat @ beta

        if np.std(x_tilde) < 1e-9:
            return 0.0, 0.0, 1.0, False

        r_gps, _ = stats.pearsonr(x_tilde, z)
        r_gps = float(r_gps)
        if abs(r_gps) >= 0.9999:
            return r_gps, 99.0, 1e-15, True

        t_gps = float(r_gps * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1.0 - r_gps ** 2)))
        p_gps = float(2.0 * stats.t.sf(abs(t_gps), df=n - 2))
        is_imbal = bool(p_gps < 0.05 and abs(r_gps) > 0.10)
        return r_gps, t_gps, p_gps, is_imbal
    except Exception:
        return None, None, None, False


def austin_ks_covariate_balance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Peter C. Austin's (2008, 2011) Kolmogorov-Smirnov Distributional Covariate Balance Test.

    Evaluates complete distributional balance of candidate confounder Z across exposure cohorts.
    Dichotomizes exposure X at its median into treated (D = 1) and control (D = 0) cohorts.
    Computes two-sample Kolmogorov-Smirnov statistic D_KS = sup_z |F_1(z) - F_0(z)|.
    Rejection (D_KS > 0.10 and p < 0.05) certifies distributional imbalance across higher-order moments.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        treated = (x > med_x)
        z1 = z[treated]
        z0 = z[~treated]
        if len(z1) < 6 or len(z0) < 6:
            return None, None, False

        res = stats.ks_2samp(z1, z0)
        ks_stat = float(res.statistic)
        ks_p = float(res.pvalue)
        is_imbal = bool(ks_stat > 0.10 and ks_p < 0.05)
        return ks_stat, ks_p, is_imbal
    except Exception:
        return None, None, False


def kuiper_normality_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Nicolaas H. Kuiper's (1960) Invariant EDF Residual Normality Test.

    Evaluates supremum empirical distribution function discrepancies in both directions:
      D^+ = max_{1 <= i <= n} (i / n - Phi(z_(i)))
      D^- = max_{1 <= i <= n} (Phi(z_(i)) - (i - 1) / n)
      V = D^+ + D^-.
    Applies Stephens' (1970) finite-sample modified statistic:
      V^* = V * (sqrt(n) + 0.155 + 0.24 / sqrt(n)).
    Critical value at alpha = 0.05 is 1.747.
    Rejection (V^* > 1.747, p < 0.05) detects empirical residual departure from Gaussianity with equal
    sensitivity in both tails and the median.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta
        s_e = float(np.std(res, ddof=1))
        if s_e < 1e-12:
            return None, None, None, False

        z_sorted = np.sort((res - np.mean(res)) / s_e)
        i_vec = np.arange(1, n + 1, dtype=float)
        cdf_vals = stats.norm.cdf(z_sorted)

        d_plus = float(np.max(i_vec / n - cdf_vals))
        d_minus = float(np.max(cdf_vals - (i_vec - 1.0) / n))
        v_stat = float(d_plus + d_minus)
        v_star = float(v_stat * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)))

        # Stephens (1970) approximation formula for Kuiper p-value
        k_terms = np.arange(1, 11, dtype=float)
        p_val = float(2.0 * np.sum((4.0 * (k_terms ** 2) * (v_star ** 2) - 1.0) * np.exp(-2.0 * (k_terms ** 2) * (v_star ** 2))))
        p_val = min(1.0, max(1e-15, p_val))
        is_non_normal = bool(v_star > 1.747 or p_val < 0.05)
        return v_stat, v_star, p_val, is_non_normal
    except Exception:
        return None, None, None, False


def covratio_influence_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], int, bool]:
    """Calculates David A. Belsley, Edwin Kuh, & Roy E. Welsch's (1980) COVRATIO Covariance Influence Diagnostic.

    Evaluates observation influence on the determinant of the estimated parameter covariance matrix:
      COVRATIO_i = (1 / (1 - h_ii)) * [(n - p - e_i^{*2}) / (n - p - 1)]^p
    where h_ii is leverage and e_i^* is internally studentized residual.
    Cutoff threshold: |COVRATIO_i - 1| > 3 * p / n.
    Values deviating significantly identify observations exerting disproportionate leverage on estimator precision.
    """
    n = len(x)
    p = 2
    if n <= p + 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, 0, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        H = X_mat @ np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T
        h_diag = np.clip(np.diag(H), 1e-6, 1.0 - 1e-6)

        mse = float(np.sum(res ** 2) / (n - p))
        if mse < 1e-12:
            return 0.0, 0, False

        # Internally studentized residuals squared
        e_star_sq = (res ** 2) / (mse * (1.0 - h_diag))
        e_star_sq = np.clip(e_star_sq, 0.0, float(n - p - 1.01))

        # COVRATIO
        ratio_term = (n - p - e_star_sq) / max(1e-6, float(n - p - 1.0))
        covratio = (1.0 / (1.0 - h_diag)) * (ratio_term ** p)

        threshold = 3.0 * p / n
        deviations = np.abs(covratio - 1.0)
        max_dev = float(np.max(deviations))
        outlier_count = int(np.sum(deviations > threshold))
        is_influential = bool(outlier_count > 0 and max_dev > threshold)
        return max_dev, outlier_count, is_influential
    except Exception:
        return None, 0, False


def tang_funnel_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Tang & Liu's (2000) Linear Regression Funnel Asymmetry Test.

    Evaluates funnel asymmetry in stratified 2x2 contingency tables by regressing
    standardized effect size z_k = theta_k / SE_k on stratum precision sqrt(n_k):
      z_k = alpha + beta * sqrt(n_k) + epsilon_k.
    Under H_0: beta = 0 (symmetry and absence of small-study sample-size distortion).
    t_Tang = beta_hat / SE(beta_hat) ~ t(K - 2).
    Rejection (p < 0.05 or |t_Tang| >= 1.96) flags significant sample-size distortion across strata.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        z_list = []
        prec_list = []
        for tab in tables_2x2:
            if isinstance(tab, (tuple, list)):
                if len(tab) == 4:
                    a, b, c, d = float(tab[0]), float(tab[1]), float(tab[2]), float(tab[3])
                else:
                    tab = np.array(tab, dtype=float)
                    a, b = float(tab[0, 0]), float(tab[0, 1])
                    c, d = float(tab[1, 0]), float(tab[1, 1])
            elif isinstance(tab, np.ndarray) and tab.shape == (2, 2):
                a, b = float(tab[0, 0]), float(tab[0, 1])
                c, d = float(tab[1, 0]), float(tab[1, 1])
            else:
                arr = np.array(tab, dtype=float).ravel()
                a, b, c, d = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
            # Haldane-Anscombe continuity correction
            a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            theta = np.log((a_c * d_c) / (b_c * c_c))
            v_k = 1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c
            se_k = np.sqrt(v_k)
            n_k = a + b + c + d
            z_k = theta / se_k
            z_list.append(z_k)
            prec_list.append(np.sqrt(max(1.0, n_k)))

        k = len(z_list)
        if k < 3:
            return None, None, False

        z_arr = np.array(z_list, dtype=float)
        p_arr = np.array(prec_list, dtype=float)

        if np.std(p_arr) < 1e-9:
            return 0.0, 1.0, False

        X_mat = np.column_stack([np.ones(k), p_arr])
        beta, _, _, _ = np.linalg.lstsq(X_mat, z_arr, rcond=None)
        res = z_arr - X_mat @ beta

        df_e = max(1, k - 2)
        s_sq = float(np.sum(res ** 2) / df_e)
        cov_mat = s_sq * np.linalg.pinv(X_mat.T @ X_mat)
        se_slope = np.sqrt(max(1e-12, cov_mat[1, 1]))

        t_tang = float(beta[1] / se_slope)
        p_val = float(2.0 * stats.t.sf(abs(t_tang), df=df_e))
        is_asym = bool(p_val < 0.05)
        return t_tang, p_val, is_asym
    except Exception:
        return None, None, False


def capon_scale_dispersion_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Capon's (1961) Expected Normal Order Statistics Scale Dispersion Test.

    Evaluates non-parametric scale dispersion between two conditioning strata.
    Centers observations by their stratum medians u_i = s_1i - median(s_1), v_j = s_2j - median(s_2).
    Ranks the combined absolute deviations |u_i|, |v_j| (N = n_1 + n_2).
    Assigns normal order score weights using Blom's (1958) formula:
      a(i) = [Phi^{-1}((r_i - 3/8) / (N + 1/4))]^2.
    Test statistic C = sum_{i in s_1} a(i).
    Under H_0: sigma_1 = sigma_2, E[C] = n_1 * a_bar, Var(C) = [n_1 * n_2 / (N(N-1))] * sum (a(i) - a_bar)^2.
    Z_Capon = (C - E[C]) / sqrt(Var(C)) ~ N(0, 1).
    Rejection (|Z_Capon| >= 1.96, p < 0.05) certifies non-parametric scale dispersion heterogeneity.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        s1 = valid_samples[0]
        s2 = valid_samples[1]
        n1 = len(s1)
        n2 = len(s2)
        N = n1 + n2

        u = np.abs(s1 - np.median(s1))
        v = np.abs(s2 - np.median(s2))
        combined = np.concatenate([u, v])

        ranks = stats.rankdata(combined)
        q = np.clip((ranks - 0.375) / (N + 0.25), 1e-6, 1.0 - 1e-6)
        norm_scores = stats.norm.ppf(q) ** 2

        c_stat = float(np.sum(norm_scores[:n1]))
        a_bar = float(np.mean(norm_scores))
        var_a = float(np.sum((norm_scores - a_bar) ** 2))
        var_c = (n1 * n2 / (N * (N - 1.0))) * var_a

        if var_c < 1e-12:
            return 0.0, 1.0, False

        z_capon = float((c_stat - n1 * a_bar) / np.sqrt(var_c))
        p_val = float(2.0 * stats.norm.sf(abs(z_capon)))
        is_hetero = bool(p_val < 0.05)
        return z_capon, p_val, is_hetero
    except Exception:
        return None, None, False


def l1_multivariate_imbalance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], bool]:
    """Calculates Stefano M. Iacus, Gary King, & Giuseppe Porro's (2011, 2012) L1 Multivariate Imbalance Metric.

    Evaluates global empirical distributional divergence of candidate confounder Z between exposure cohorts.
    Dichotomizes exposure X at median into treated (D = 1) and control (D = 0) cohorts.
    Discretizes confounder Z into empirical quantile bins and calculates:
      L1 = 0.5 * sum_{k=1}^K |n_{1k}/n_1 - n_{0k}/n_0| in [0, 1].
    Threshold: L1 > 0.25 certifies severe empirical multivariate imbalance and lack of causal common support.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, False
    try:
        med_x = float(np.median(x))
        treated = (x > med_x)
        z1 = z[treated]
        z0 = z[~treated]
        n1 = len(z1)
        n0 = len(z0)
        if n1 < 6 or n0 < 6:
            return None, False

        n_bins = int(np.clip(np.floor(np.cbrt(n) * 2), 4, 10))
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(z, quantiles)
        bin_edges[0] -= 1e-6
        bin_edges[-1] += 1e-6

        hist1, _ = np.histogram(z1, bins=bin_edges)
        hist0, _ = np.histogram(z0, bins=bin_edges)

        freq1 = hist1 / float(n1)
        freq0 = hist0 / float(n0)

        l1_stat = float(0.5 * np.sum(np.abs(freq1 - freq0)))
        is_imbal = bool(l1_stat > 0.25)
        return l1_stat, is_imbal
    except Exception:
        return None, False


def rubin_parallelism_interaction_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Donald B. Rubin's (1979) Regression Surface Non-Parallelism Test.

    Evaluates whether the slope of outcome Y on confounder Z differs across exposure cohorts (D = I(X > median(X))):
      Restricted Model (parallel surfaces):   Y = b_0 + b_1*D + b_2*Z + e
      Unrestricted Model (interaction):      Y = b_0 + b_1*D + b_2*Z + gamma*(D*Z) + eta
      F_Rubin = ((SSR_R - SSR_UR) / 1) / (SSR_UR / (N - 4)) ~ F(1, N - 4).
    Rejection (F >= 3.841, p < 0.05) certifies non-parallel response surfaces, proving that linear adjustment
    is misspecified due to treatment effect heterogeneity.
    """
    n = len(x)
    if n < 20 or np.std(x) < 1e-9 or np.std(z) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        d = (x > med_x).astype(float)
        if np.std(d) < 1e-9:
            return None, None, False

        # Restricted design matrix [1, D, Z]
        X_res = np.column_stack([np.ones(n), d, z])
        b_res, _, _, _ = np.linalg.lstsq(X_res, y, rcond=None)
        ssr_res = float(np.sum((y - X_res @ b_res) ** 2))

        # Unrestricted design matrix [1, D, Z, D * Z]
        X_unres = np.column_stack([np.ones(n), d, z, d * z])
        b_unres, _, _, _ = np.linalg.lstsq(X_unres, y, rcond=None)
        ssr_unres = float(np.sum((y - X_unres @ b_unres) ** 2))

        df_num = 1
        df_den = max(1, n - 4)

        if ssr_unres < 1e-12:
            return 0.0, 1.0, False

        f_stat = float(((ssr_res - ssr_unres) / df_num) / (ssr_unres / df_den))
        f_stat = max(0.0, f_stat)
        p_val = float(stats.f.sf(f_stat, df_num, df_den))
        is_nonparallel = bool(f_stat >= 3.841 and p_val < 0.05)
        return f_stat, p_val, is_nonparallel
    except Exception:
        return None, None, False


def hosking_l_moments_normality_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates J. R. M. Hosking's (1990) L-Moments Ratio Residual Normality Test.

    Evaluates standardized OLS residuals using unbiased sample L-moments (linear combinations of order statistics):
      t_3 = l_3 / l_2 (sample L-skewness),  t_4 = l_4 / l_2 (sample L-kurtosis).
    For a Gaussian distribution, tau_3 = 0, tau_4 = 0.122602.
    Asymptotic variances: Var(t_3) = 0.1866 / n,  Var(t_4) = 0.0883 / n.
      L_stat = (t_3^2 / Var(t_3)) + ((t_4 - 0.122602)^2 / Var(t_4)) ~ Chi-squared(2).
    Rejection (L_stat >= 5.991, p < 0.05) certifies residual non-Gaussianity robust against extreme outlier distortion.
    """
    n = len(x)
    p = 2
    if n <= p + 4 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        s = float(np.std(res))
        if s < 1e-12:
            return 0.0, 1.0, False

        z_sorted = np.sort(res / s)

        i_vec = np.arange(1, n + 1, dtype=float)
        c1 = i_vec - 1.0
        c1_rev = n - i_vec

        comb_n_2 = n * (n - 1.0) / 2.0
        comb_n_3 = n * (n - 1.0) * (n - 2.0) / 6.0
        comb_n_4 = n * (n - 1.0) * (n - 2.0) * (n - 3.0) / 24.0

        if comb_n_4 < 1.0:
            return None, None, False

        w2 = (c1 - c1_rev)
        l2 = float(np.sum(w2 * z_sorted) / (2.0 * comb_n_2))
        if abs(l2) < 1e-12:
            return 0.0, 1.0, False

        w3 = (c1 * (c1 - 1.0) / 2.0 - 2.0 * c1 * c1_rev + c1_rev * (c1_rev - 1.0) / 2.0)
        l3 = float(np.sum(w3 * z_sorted) / (3.0 * comb_n_3))

        w4 = (c1 * (c1 - 1.0) * (c1 - 2.0) / 6.0
              - 3.0 * (c1 * (c1 - 1.0) / 2.0) * c1_rev
              + 3.0 * c1 * (c1_rev * (c1_rev - 1.0) / 2.0)
              - c1_rev * (c1_rev - 1.0) * (c1_rev - 2.0) / 6.0)
        l4 = float(np.sum(w4 * z_sorted) / (4.0 * comb_n_4))

        t3 = l3 / l2
        t4 = l4 / l2

        var_t3 = 0.1866 / n
        var_t4 = 0.0883 / n

        l_stat = float((t3 ** 2) / var_t3 + ((t4 - 0.122602) ** 2) / var_t4)
        p_val = float(stats.chi2.sf(l_stat, df=2))
        is_non_normal = bool(l_stat >= 5.991 and p_val < 0.05)
        return l_stat, p_val, is_non_normal
    except Exception:
        return None, None, False


def atkinson_influence_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], int, bool]:
    """Calculates Anthony C. Atkinson's (1981, 1985) Modified Leverage-Weighted Influence Metric.

    Evaluates observation influence by amplifying studentized deleted residuals with design leverage:
      A_i = |t_i| * sqrt([(n - p) / p] * [h_ii / (1 - h_ii)]),
    where t_i = e_i * sqrt((n - p - 1) / (SSE * (1 - h_ii) - e_i^2)).
    Cutoff threshold: A_thresh = 2.0 * sqrt(p / (n - p)).
    Flags influential observations that distort OLS parameters without triggering standard Cook's distance alerts.
    """
    n = len(x)
    p = 2
    if n <= p + 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, 0, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        H = X_mat @ np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T
        h_diag = np.clip(np.diag(H), 1e-6, 1.0 - 1e-6)

        sse = float(np.sum(res ** 2))
        if sse < 1e-12:
            return 0.0, 0, False

        # Externally studentized (deleted) residuals
        denom = sse * (1.0 - h_diag) - (res ** 2)
        denom = np.maximum(1e-12, denom)
        t_deleted = res * np.sqrt((n - p - 1.0) / denom)

        leverage_factor = np.sqrt(((n - p) / float(p)) * (h_diag / (1.0 - h_diag)))
        atkinson_scores = np.abs(t_deleted) * leverage_factor

        threshold = 2.0 * np.sqrt(float(p) / max(1.0, float(n - p)))
        max_atkinson = float(np.max(atkinson_scores))
        outlier_count = int(np.sum(atkinson_scores > threshold))
        is_influential = bool(outlier_count > 0 and max_atkinson > threshold)
        return max_atkinson, outlier_count, is_influential
    except Exception:
        return None, 0, False


def pustejovsky_funnel_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Pustejovsky & Rodgers' (2019) Variance-Stabilized Funnel Asymmetry Test.

    Evaluates reporting and publication bias across stratified 2x2 contingency tables
    by regressing variance-stabilized log odds ratios against effective sample precision:
      y_k = ln(OR_k) * sqrt(n_tilde_k) = beta_0 + beta_1 * sqrt(n_tilde_k) + epsilon_k,
    where n_tilde_k = (n_1k * n_2k) / (n_1k + n_2k) is the effective stratum sample size.
    Testing H_0: beta_0 = 0 (symmetry and absence of small-study sample-size distortion).
    t_PR = beta_0 / SE(beta_0) ~ t(K - 2).
    Rejection (p < 0.05 or |t_PR| >= 1.96) flags significant sample-size distortion across strata.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        y_list = []
        prec_list = []
        for tab in tables_2x2:
            if isinstance(tab, (tuple, list)):
                if len(tab) == 4:
                    a, b, c, d = float(tab[0]), float(tab[1]), float(tab[2]), float(tab[3])
                else:
                    tab = np.array(tab, dtype=float)
                    a, b = float(tab[0, 0]), float(tab[0, 1])
                    c, d = float(tab[1, 0]), float(tab[1, 1])
            elif isinstance(tab, np.ndarray) and tab.shape == (2, 2):
                a, b = float(tab[0, 0]), float(tab[0, 1])
                c, d = float(tab[1, 0]), float(tab[1, 1])
            else:
                arr = np.array(tab, dtype=float).ravel()
                a, b, c, d = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

            # Haldane-Anscombe continuity correction
            a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            theta = np.log((a_c * d_c) / (b_c * c_c))
            n1 = a + b
            n2 = c + d
            n_tot = n1 + n2
            if n_tot < 1.0 or n1 < 0.5 or n2 < 0.5:
                continue
            n_tilde = (n1 * n2) / n_tot
            prec = np.sqrt(n_tilde)
            y_k = theta * prec
            y_list.append(y_k)
            prec_list.append(prec)

        k = len(y_list)
        if k < 3:
            return None, None, False

        y_arr = np.array(y_list, dtype=float)
        p_arr = np.array(prec_list, dtype=float)

        if np.std(p_arr) < 1e-9:
            return 0.0, 1.0, False

        X_mat = np.column_stack([np.ones(k), p_arr])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y_arr, rcond=None)
        res = y_arr - X_mat @ beta

        df_e = max(1, k - 2)
        s_sq = float(np.sum(res ** 2) / df_e)
        cov_mat = s_sq * np.linalg.pinv(X_mat.T @ X_mat)
        se_intercept = np.sqrt(max(1e-12, cov_mat[0, 0]))

        t_pr = float(beta[0] / se_intercept)
        p_val = float(2.0 * stats.t.sf(abs(t_pr), df=df_e))
        is_asym = bool(p_val < 0.05)
        return t_pr, p_val, is_asym
    except Exception:
        return None, None, False


def moses_scale_dispersion_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Lincoln E. Moses' (1963) Non-Parametric Sub-Sample Rank Dispersion Test.

    Evaluates scale/spread differences between conditioning groups without assuming equal medians or Gaussianity.
    Divides observations into disjoint sub-samples of size k=3, computes within-subsample sum of squared deviations,
    and applies a Mann-Whitney U test to the sub-sample dispersion values.
    Under H_0: sigma_1 = sigma_2, the distribution of sub-sample dispersions is identical.
    Rejection (p < 0.05) certifies non-parametric scale spread heterogeneity.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 6 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        s1 = valid_samples[0]
        s2 = valid_samples[1]

        k_sub = 3
        m1 = len(s1) // k_sub
        m2 = len(s2) // k_sub
        if m1 < 2 or m2 < 2:
            k_sub = 2
            m1 = len(s1) // k_sub
            m2 = len(s2) // k_sub
            if m1 < 2 or m2 < 2:
                return None, None, False

        # Compute within-subsample sum of squared deviations
        c1 = []
        for i in range(m1):
            sub = s1[i * k_sub : (i + 1) * k_sub]
            c1.append(float(np.sum((sub - np.mean(sub)) ** 2)))
        c2 = []
        for j in range(m2):
            sub = s2[j * k_sub : (j + 1) * k_sub]
            c2.append(float(np.sum((sub - np.mean(sub)) ** 2)))

        c1 = np.array(c1, dtype=float)
        c2 = np.array(c2, dtype=float)

        res = stats.mannwhitneyu(c1, c2, alternative='two-sided')
        u_stat = float(res.statistic)
        p_val = float(res.pvalue)
        is_hetero = bool(p_val < 0.05)
        return u_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def eqq_covariate_balance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Ho, Imai, King, & Stuart's (2007) Empirical Q-Q Covariate Balance Metric.

    Dichotomizes exposure X at median into treated (D = 1) and control (D = 0) cohorts.
    Evaluates empirical quantile-quantile differences across 99 percentiles, standardized
    by the pooled standard deviation of candidate confounder Z:
      eQQ_mean = (1 / 99) * sum_{k=1}^{99} |q_{1, k} - q_{0, k}| / sigma_pooled,
      eQQ_max  = max_{k} |q_{1, k} - q_{0, k}| / sigma_pooled.
    Threshold: eQQ_mean > 0.15 certifies severe distributional imbalance across the common support.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        treated = (x > med_x)
        z1 = z[treated]
        z0 = z[~treated]
        n1 = len(z1)
        n0 = len(z0)
        if n1 < 6 or n0 < 6:
            return None, None, False

        s1_var, s0_var = float(np.var(z1, ddof=1)), float(np.var(z0, ddof=1))
        s_pooled = np.sqrt(((n1 - 1) * s1_var + (n0 - 1) * s0_var) / max(1, n1 + n0 - 2))
        if s_pooled < 1e-9:
            return 0.0, 0.0, False

        percentiles = np.linspace(1, 99, 99)
        q1 = np.percentile(z1, percentiles)
        q0 = np.percentile(z0, percentiles)

        abs_diffs = np.abs(q1 - q0) / s_pooled
        eqq_mean = float(np.mean(abs_diffs))
        eqq_max = float(np.max(abs_diffs))
        is_imbal = bool(eqq_mean > 0.15 or eqq_max > 0.35)
        return eqq_mean, eqq_max, is_imbal
    except Exception:
        return None, None, False


def lin_interaction_heterogeneity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Winston Lin's (2013) Agnostic Treatment-Covariate Centered Interaction Test.

    Evaluates effect heterogeneity across confounder gradients using centered exposure and confounder:
      Y = beta_0 + beta_1 * (D - D_bar) + beta_2 * (Z - Z_bar) + beta_3 * (D - D_bar)*(Z - Z_bar) + epsilon.
    Under H_0: beta_3 = 0 (constant treatment effect, invariant across confounder levels).
    t_Lin = beta_3 / SE(beta_3) ~ t(n - 4).
    Rejection (p < 0.05 or |t_Lin| >= 1.96) proves treatment effect heterogeneity along the confounder gradient,
    demonstrating that standard constant-effect adjustment is misspecified.
    """
    n = len(x)
    if n < 20 or np.std(x) < 1e-9 or np.std(z) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        d = (x > med_x).astype(float)
        if np.std(d) < 1e-9:
            return None, None, False

        d_cent = d - np.mean(d)
        z_cent = z - np.mean(z)
        dz_inter = d_cent * z_cent

        X_mat = np.column_stack([np.ones(n), d_cent, z_cent, dz_inter])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        df_e = max(1, n - 4)
        s_sq = float(np.sum(res ** 2) / df_e)
        cov_mat = s_sq * np.linalg.pinv(X_mat.T @ X_mat)
        se_int = np.sqrt(max(1e-12, cov_mat[3, 3]))

        t_lin = float(beta[3] / se_int)
        p_val = float(2.0 * stats.t.sf(abs(t_lin), df=df_e))
        is_hetero = bool(p_val < 0.05)
        return t_lin, p_val, is_hetero
    except Exception:
        return None, None, False


def shapiro_wilk_normality_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Shapiro & Wilk's (1965) Omnibus Residual Normality Test (W).

    Evaluates regression residuals against the Gaussian null using the ratio of the best linear
    unbiased estimator of variance (based on order statistics) to the sample variance:
      W = (sum a_i e_{(i)})^2 / sum (e_i - e_bar)^2.
    Rejection (p < 0.05) provides optimal power against diverse non-Gaussian alternatives
    (asymmetry, leptokurtosis, platykurtosis, multimodality).
    """
    n = len(x)
    p = 2
    if n <= p + 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        if np.std(res) < 1e-12:
            return 1.0, 1.0, False

        # In case n > 5000, take representative sample for Shapiro-Wilk calculation if needed
        res_calc = res if n <= 5000 else np.random.RandomState(42).choice(res, 5000, replace=False)
        w_stat, p_val = stats.shapiro(res_calc)
        w_stat = float(w_stat)
        p_val = float(p_val)
        is_non_norm = bool(p_val < 0.05)
        return w_stat, p_val, is_non_norm
    except Exception:
        return None, None, False


def hadi_influence_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], int, bool]:
    """Calculates Ali S. Hadi's (1992) Measure of Overall Potential Influence in Regression (H_i^2).

    Decomposes observation influence into residual potential and leverage potential:
      H_i^2 = [(p + 1) / (1 - h_ii)] * [e_i^2 / ((1 - h_ii) * SSE)] + [h_ii / (1 - h_ii)].
    Cutoff threshold: H_crit = mean(H^2) + 2 * std(H^2) or 3 * (p / n).
    Flags observations that exert disproportionate potential influence on both the fitted hyperplane
    and the covariance structure.
    """
    n = len(x)
    p = 2
    if n <= p + 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, 0, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        H = X_mat @ np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T
        h_diag = np.clip(np.diag(H), 1e-6, 1.0 - 1e-6)

        sse = float(np.sum(res ** 2))
        if sse < 1e-12:
            return 0.0, 0, False

        # Hadi's influence measure
        term1 = ((p + 1.0) / (1.0 - h_diag)) * ((res ** 2) / ((1.0 - h_diag) * sse))
        term2 = h_diag / (1.0 - h_diag)
        hadi_scores = term1 + term2

        mean_h = np.mean(hadi_scores)
        std_h = np.std(hadi_scores)
        threshold = float(mean_h + 2.0 * std_h)
        if threshold < (3.0 * p / n):
            threshold = float(3.0 * p / n)

        max_hadi = float(np.max(hadi_scores))
        outlier_count = int(np.sum(hadi_scores > threshold))
        is_influential = bool(outlier_count > 0 and max_hadi > threshold)
        return max_hadi, outlier_count, is_influential
    except Exception:
        return None, 0, False


class DevilsAdvocateAgent:
    """Adversarial stress tester searching for Simpson's Paradox, confounders, and instability."""

    def __init__(
        self,
        n_bootstrap: int = 150,
        min_subgroup_size: int = 8,
        random_seed: int = 42,
        confounder_attenuation_threshold: float = 60.0,
    ):
        self.n_bootstrap = n_bootstrap
        self.min_subgroup_size = min_subgroup_size
        self.random_seed = random_seed
        self.confounder_attenuation_threshold = confounder_attenuation_threshold
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

                # Safely convert to float, handling categorical strings (e.g. 'Male'/'Female') via category codes
                if pd.api.types.is_numeric_dtype(sub_clean[source]):
                    s_vals = pd.to_numeric(sub_clean[source], errors='coerce')
                else:
                    s_vals = pd.Series(pd.Categorical(sub_clean[source]).codes, index=sub_clean.index, dtype=float)

                if pd.api.types.is_numeric_dtype(sub_clean[target]):
                    t_vals = pd.to_numeric(sub_clean[target], errors='coerce')
                else:
                    t_vals = pd.Series(pd.Categorical(sub_clean[target]).codes, index=sub_clean.index, dtype=float)

                valid_pair = ~(s_vals.isna() | t_vals.isna())
                if valid_pair.sum() < self.min_subgroup_size:
                    continue

                r_sub, p_sub = stats.pearsonr(s_vals[valid_pair], t_vals[valid_pair])
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

        # Advanced Metric 1: Cochran-Mantel-Haenszel (CMH) & Tarone's Homogeneity Test for Binary Targets
        cmh_or: Optional[float] = None
        cmh_p: Optional[float] = None
        cmh_chi2: Optional[float] = None
        or_reversal = False
        tarone_stat: Optional[float] = None
        tarone_p: Optional[float] = None
        simpson_e_val: Optional[float] = None
        simpson_e_desc: Optional[str] = None
        woolf_chi2: Optional[float] = None
        woolf_p: Optional[float] = None
        is_woolf_hetero: bool = False
        ft_chi2: Optional[float] = None
        ft_p: Optional[float] = None
        is_ft_hetero: bool = False
        peto_or: Optional[float] = None
        peto_z: Optional[float] = None
        peto_p: Optional[float] = None
        peto_q: Optional[float] = None
        peto_qp: Optional[float] = None
        is_peto_hetero: bool = False
        fligner_stat: Optional[float] = None
        fligner_p: Optional[float] = None
        is_fligner_hetero: bool = False
        harbord_t: Optional[float] = None
        harbord_p: Optional[float] = None
        is_harbord_asym: bool = False
        bf_stat: Optional[float] = None
        bf_p: Optional[float] = None
        is_bf_hetero: bool = False
        peters_t: Optional[float] = None
        peters_p: Optional[float] = None
        is_peters_asym: bool = False
        kish_deff: Optional[float] = None
        strat_var_ratio: Optional[float] = None
        rucker_t: Optional[float] = None
        rucker_p: Optional[float] = None
        is_rucker_asym: bool = False
        levene_stat: Optional[float] = None
        levene_p: Optional[float] = None
        is_levene_hetero: bool = False
        macaskill_t: Optional[float] = None
        macaskill_p: Optional[float] = None
        is_macaskill_asym: bool = False
        bartlett_stat: Optional[float] = None
        bartlett_p: Optional[float] = None
        is_bartlett_hetero: bool = False
        deeks_t: Optional[float] = None
        deeks_p: Optional[float] = None
        is_deeks_asym: bool = False
        mood_z: Optional[float] = None
        mood_p: Optional[float] = None
        is_mood_hetero: bool = False
        thompson_sharp_t: Optional[float] = None
        thompson_sharp_p: Optional[float] = None
        is_thompson_asym: bool = False
        ansari_w: Optional[float] = None
        ansari_p: Optional[float] = None
        is_ansari_hetero: bool = False
        moreno_t: Optional[float] = None
        moreno_p: Optional[float] = None
        is_moreno_asym: bool = False
        siegel_tukey_w: Optional[float] = None
        siegel_tukey_p: Optional[float] = None
        is_siegel_tukey_hetero: bool = False
        sterne_t: Optional[float] = None
        sterne_p: Optional[float] = None
        is_sterne_asym: bool = False
        klotz_w: Optional[float] = None
        klotz_p: Optional[float] = None
        is_klotz_hetero: bool = False
        lfk_val: Optional[float] = None
        is_lfk_asym: bool = False
        conover_stat: Optional[float] = None
        conover_p: Optional[float] = None
        is_conover_hetero: bool = False

        is_binary_target = df[target].dropna().nunique() == 2
        if is_binary_target and len(breakdowns) >= 2 and claim.claim_type in ("proportion_difference", "group_difference"):
            best_g = claim.metadata.get("best_group") or claim.metadata.get("max_group")
            worst_g = claim.metadata.get("worst_group") or claim.metadata.get("min_group")
            pos_val = claim.metadata.get("positive_value")
            if pos_val is None:
                pos_val = 1 if 1 in df[target].values else sorted(df[target].dropna().unique())[-1]

            num_sum = 0.0
            denom_sum = 0.0
            sum_a = 0.0
            sum_ea = 0.0
            sum_vara = 0.0
            tables_2x2 = []

            for grp in sorted(groups, key=str):
                sub = df[df[conf_col] == grp]
                sub_clean = sub[[source, target]].dropna()
                if len(sub_clean) < self.min_subgroup_size:
                    continue
                a = float(len(sub_clean[(sub_clean[source] == best_g) & (sub_clean[target] == pos_val)]))
                b = float(len(sub_clean[(sub_clean[source] == best_g) & (sub_clean[target] != pos_val)]))
                c = float(len(sub_clean[(sub_clean[source] == worst_g) & (sub_clean[target] == pos_val)]))
                d = float(len(sub_clean[(sub_clean[source] == worst_g) & (sub_clean[target] != pos_val)]))
                n_k = a + b + c + d
                if n_k > 1 and (a + b) > 0 and (c + d) > 0 and (a + c) > 0 and (b + d) > 0:
                    tables_2x2.append((a, b, c, d))
                    num_sum += (a * d) / n_k
                    denom_sum += (b * c) / n_k
                    e_a = ((a + b) * (a + c)) / n_k
                    var_a = ((a + b) * (c + d) * (a + c) * (b + d)) / (n_k**2 * (n_k - 1))
                    sum_a += a
                    sum_ea += e_a
                    sum_vara += var_a

            if denom_sum > 0:
                cmh_or = float(num_sum / denom_sum)
                if sum_vara > 1e-9:
                    cmh_chi2 = float(((abs(sum_a - sum_ea) - 0.5) ** 2) / sum_vara)
                    cmh_p = float(stats.chi2.sf(cmh_chi2, df=1))
                if cmh_or < 1.0:
                    or_reversal = True
                
                # Breslow-Day & Tarone's Homogeneity Test
                bd_stat, bd_p, tarone_stat, tarone_p = breslow_day_homogeneity_test(tables_2x2, cmh_or)
                # Woolf's Stratified Heterogeneity Test with Haldane-Anscombe correction
                woolf_chi2, woolf_p, is_woolf_hetero = woolf_stratified_heterogeneity_test(tables_2x2)
                # Freeman-Tukey (1950) Variance-Stabilizing Double-Arcsine Homogeneity Test
                ft_chi2, ft_p, is_ft_hetero = freeman_tukey_double_arcsine_test(tables_2x2)
                # Peto et al. (1977) Stratified Odds Ratio & Heterogeneity Test
                peto_or, peto_z, peto_p, peto_q, peto_qp, is_peto_hetero = peto_stratified_odds_ratio_test(tables_2x2)
                # Harbord, Egger, & Sterne (2006) Modified Score Test for Funnel Plot Asymmetry
                harbord_t, harbord_p, is_harbord_asym = harbord_modified_score_test(tables_2x2)
                # Peters et al. (2006) Regression Test for Funnel Plot Asymmetry
                peters_t, peters_p, is_peters_asym = peters_regression_asymmetry_test(tables_2x2)
                # Rücker, Schwarzer, & Carpenter (2008) Arcsine Test for Funnel Plot Asymmetry
                rucker_t, rucker_p, is_rucker_asym = rucker_arcsine_asymmetry_test(tables_2x2)
                # Macaskill, Walter, & Irwig (2001) Sample-Size Weighted Regression Test for Asymmetry
                macaskill_t, macaskill_p, is_macaskill_asym = macaskill_regression_asymmetry_test(tables_2x2)
                # Jon Deeks, Peter Macaskill, & Les Irwig (2005) Diagnostic Odds Ratio (DOR) Asymmetry Regression Test
                deeks_t, deeks_p, is_deeks_asym = deeks_funnel_asymmetry_test(tables_2x2)
                # S. G. Thompson & S. J. Sharp (1999) Random-Effects Weighted Funnel Asymmetry Test
                thompson_sharp_t, thompson_sharp_p, is_thompson_asym = thompson_sharp_funnel_asymmetry_test(tables_2x2)
                # Moreno et al. (2009) Variance-Weighted Regression Test for Asymmetry
                moreno_t, moreno_p, is_moreno_asym = moreno_funnel_asymmetry_test(tables_2x2)
                # Jonathan Sterne & Matthias Egger (2000) Inverse-Variance Weighted Funnel Asymmetry Regression Test
                sterne_t, sterne_p, is_sterne_asym = sterne_funnel_asymmetry_test(tables_2x2)
                # Luis Furuya-Kanamori & Suhail A. R. Doi (2018) LFK Index of Funnel Plot Asymmetry
                lfk_val, is_lfk_asym = lfk_index_asymmetry_test(tables_2x2)
                # Stanley & Doucouliagos (2012, 2014) PET-PEESE Meta-Regression Falsification Test
                pet_t, pet_p, peese_t, peese_p, is_pet_asym = pet_peese_meta_regression_test(tables_2x2)
                # Tang & Liu (2000) Linear Regression Funnel Asymmetry Test
                tang_t, tang_p, is_tang_asym = tang_funnel_asymmetry_test(tables_2x2)
                # Pustejovsky & Rodgers (2019) Variance-Stabilized Funnel Asymmetry Test
                pust_t, pust_p, is_pust_asym = pustejovsky_funnel_asymmetry_test(tables_2x2)
                # E-value for unmeasured confounding
                simpson_e_val, simpson_e_desc = calculate_e_value(cmh_or)
        else:
            bd_stat, bd_p = None, None
            pet_t, pet_p, peese_t, peese_p, is_pet_asym = None, None, None, None, False
            tang_t, tang_p, is_tang_asym = None, None, False
            pust_t, pust_p, is_pust_asym = None, None, False

        # Advanced Metric 2: Cohen's d Effect Size for Group Differences
        cohens_d: Optional[float] = None
        if claim.claim_type == "group_difference":
            best_g = claim.metadata.get("best_group") or claim.metadata.get("max_group")
            worst_g = claim.metadata.get("worst_group") or claim.metadata.get("min_group")
            sub_clean = df[[source, target]].dropna()
            s1 = sub_clean[sub_clean[source] == best_g][target].astype(float)
            s2 = sub_clean[sub_clean[source] == worst_g][target].astype(float)
            if len(s1) > 1 and len(s2) > 1:
                s1_var, s2_var = s1.var(ddof=1), s2.var(ddof=1)
                s_pooled = np.sqrt(((len(s1) - 1) * s1_var + (len(s2) - 1) * s2_var) / max(1, len(s1) + len(s2) - 2))
                if s_pooled > 1e-9:
                    cohens_d = float((s1.mean() - s2.mean()) / s_pooled)

        # Advanced Metric 3: Higgins & Thompson I^2 Inconsistency & Cochran's Q for Subgroup Heterogeneity
        cochrans_q, i_sq = higgins_i_squared_heterogeneity(
            subgroup_stats, subgroup_weights, is_correlation=(claim.claim_type == "correlation")
        )

        # Advanced Metric 4: Begg & Mazumdar Funnel Plot Asymmetry Test across Strata
        begg_tau, begg_p = begg_funnel_asymmetry_test(subgroup_stats, subgroup_weights)

        # Advanced Metric 5: DerSimonian & Laird Between-Stratum Variance tau^2
        tau_sq = dersimonian_laird_tau_squared(
            subgroup_stats, subgroup_weights, cochrans_q, is_correlation=(claim.claim_type == "correlation")
        )

        # Advanced Metric 6: Egger's Linear Regression Funnel Asymmetry Test
        egger_t, egger_p = eggers_funnel_regression_test(
            subgroup_stats, subgroup_weights, is_correlation=(claim.claim_type == "correlation")
        )

        # Advanced Metric 7: Duval & Tweedie (2000) Trim and Fill Algorithm
        tf_k0, tf_stat = duval_tweedie_trim_and_fill(
            subgroup_stats, subgroup_weights, is_correlation=(claim.claim_type == "correlation")
        )

        # Advanced Metric 8: Hausman (1978) Stratification Endogeneity Specification Test
        haus_stat, haus_p, is_haus_incon = hausman_specification_test(
            df, source, target, conf_col
        )

        # Advanced Metric 9: Vuong (1989) Likelihood Ratio Non-Nested Model Selection Test
        vuong_z_val, vuong_p_val, is_strat_fav = vuong_model_selection_test(
            df, source, target, conf_col
        )

        # Advanced Metric 10: Higgins H^2 Relative Heterogeneity & Galbraith Radial Outlier Strata
        higgins_h2, galb_cnt, galb_outliers = calculate_higgins_h_and_galbraith_outliers(
            subgroup_stats, subgroup_weights, [b.group_name for b in breakdowns], cochrans_q=cochrans_q
        )

        # Advanced Metric 11: Kruskal & Wallis (1952) Non-Parametric Rank-Sum Heterogeneity Test
        kw_h, kw_p, is_kw_hetero = None, None, False
        try:
            target_series = pd.to_numeric(df[target], errors='coerce') if pd.api.types.is_numeric_dtype(df[target]) else pd.Series(pd.Categorical(df[target]).codes, index=df.index, dtype=float)
            kw_samples = []
            for grp in sorted(groups, key=str):
                grp_mask = (df[conf_col] == grp)
                s = target_series[grp_mask].dropna().to_numpy(dtype=float)
                if len(s) >= 3 and np.std(s) > 1e-9:
                    kw_samples.append(s)
            kw_h, kw_p, is_kw_hetero = kruskal_wallis_strata_test(kw_samples)
            fligner_stat, fligner_p, is_fligner_hetero = fligner_killeen_homogeneity_test(kw_samples)
            bf_stat, bf_p, is_bf_hetero = brown_forsythe_homogeneity_test(kw_samples)
            levene_stat, levene_p, is_levene_hetero = levene_homogeneity_test(kw_samples)
            # Maurice Bartlett (1937) Likelihood Ratio Scale Homogeneity Test
            bartlett_stat, bartlett_p, is_bartlett_hetero = bartlett_homogeneity_test(kw_samples)
            # Alexander M. Mood (1954) Non-Parametric Median-Rank Dispersion Scale Test
            mood_z, mood_p, is_mood_hetero = mood_dispersion_scale_test(kw_samples)
            # C. E. Ansari & R. A. Bradley (1960) Non-Parametric Rank-Sum Scale Dispersion Test
            ansari_w, ansari_p, is_ansari_hetero = ansari_bradley_scale_test(kw_samples)
            # Sidney Siegel & John W. Tukey (1960) Non-Parametric Rank Extremes Scale Test
            siegel_tukey_w, siegel_tukey_p, is_siegel_tukey_hetero = siegel_tukey_scale_test(kw_samples)
            # Jerome Klotz (1962) Non-Parametric Normal Scores Rank Scale Dispersion Test
            klotz_w, klotz_p, is_klotz_hetero = klotz_normal_scores_scale_test(kw_samples)
            # W. J. Conover & Ronald L. Iman (1981) Squared Ranks Scale Dispersion Test
            conover_stat, conover_p, is_conover_hetero = conover_squared_ranks_scale_test(kw_samples)
            # B. V. Sukhatme (1957, 1958) Non-Parametric Rank-Order Scale Dispersion Test
            sukhatme_z, sukhatme_p, is_sukhatme_hetero = sukhatme_scale_dispersion_test(kw_samples)
            # Capon (1961) Expected Normal Order Statistics Scale Dispersion Test
            capon_z, capon_p, is_capon_hetero = capon_scale_dispersion_test(kw_samples)
            # Moses (1963) Non-Parametric Sub-Sample Rank Dispersion Test
            moses_u, moses_p, is_moses_hetero = moses_scale_dispersion_test(kw_samples)
            # Leslie Kish (1965) Stratification Design Effect & Variance Efficiency
            subgroup_sizes = [len(df[df[conf_col] == grp]) for grp in sorted(groups, key=str)]
            subgroup_vars = [float(np.var(s, ddof=1)) if len(s) > 1 else 0.0 for s in kw_samples]
            crude_var = float(np.var(target_series, ddof=1)) if len(target_series) > 1 else 1.0
            kish_deff, strat_var_ratio = calculate_kish_design_effect(subgroup_sizes, subgroup_vars, crude_var)
        except Exception:
            pass

        cmh_note = f" [CMH Common OR: {cmh_or:.3f}, p={cmh_p:.3e}]" if cmh_or is not None and cmh_p is not None else ""
        if bd_p is not None:
            cmh_note += f" [Breslow-Day: p={bd_p:.3e}]"
        if tarone_p is not None:
            cmh_note += f" [Tarone Homogeneity: p={tarone_p:.3e}]"
        if simpson_e_val is not None:
            cmh_note += f" [E-value: {simpson_e_val:.2f}]"
        if i_sq is not None:
            cmh_note += f" [I² Heterogeneity: {i_sq:.1f}%]"
        if higgins_h2 is not None and higgins_h2 > 1.0:
            cmh_note += f" [Higgins H²: {higgins_h2:.2f}]"
        if galb_cnt > 0:
            cmh_note += f" [Galbraith Outliers: {galb_cnt} ({', '.join(galb_outliers)})]"
        if begg_p is not None:
            cmh_note += f" [Begg Funnel: τ={begg_tau:.2f}, p={begg_p:.3e}]"
        if tau_sq is not None:
            cmh_note += f" [DerSimonian-Laird τ²: {tau_sq:.4f}]"
        if egger_p is not None:
            cmh_note += f" [Egger Test: t={egger_t:.2f}, p={egger_p:.3e}]"
        if harbord_p is not None:
            cmh_note += f" [Harbord Score: t={harbord_t:.2f}, p={harbord_p:.3e}]"
        if peters_p is not None:
            cmh_note += f" [Peters Test: t={peters_t:.2f}, p={peters_p:.3e}]"
        if rucker_p is not None:
            cmh_note += f" [Rücker Arcsine: t={rucker_t:.2f}, p={rucker_p:.3e}]"
        if macaskill_p is not None:
            cmh_note += f" [Macaskill Funnel: t={macaskill_t:.2f}, p={macaskill_p:.3e}]"
        if deeks_p is not None:
            cmh_note += f" [Deeks DOR Funnel: t={deeks_t:.2f}, p={deeks_p:.3e}]"
        if thompson_sharp_p is not None:
            cmh_note += f" [Thompson-Sharp Funnel: t={thompson_sharp_t:.2f}, p={thompson_sharp_p:.3e}]"
        if moreno_p is not None:
            cmh_note += f" [Moreno Funnel: t={moreno_t:.2f}, p={moreno_p:.3e}]"
        if sterne_p is not None:
            cmh_note += f" [Sterne Funnel: t={sterne_t:.2f}, p={sterne_p:.3e}]"
        if levene_p is not None:
            cmh_note += f" [Levene Scale: W={levene_stat:.2f}, p={levene_p:.3e}]"
        if bartlett_p is not None:
            cmh_note += f" [Bartlett Scale: B={bartlett_stat:.2f}, p={bartlett_p:.3e}]"
        if mood_p is not None:
            cmh_note += f" [Mood Scale: Z={mood_z:.2f}, p={mood_p:.3e}]"
        if ansari_p is not None:
            cmh_note += f" [Ansari-Bradley Scale: W={ansari_w:.2f}, p={ansari_p:.3e}]"
        if siegel_tukey_p is not None:
            cmh_note += f" [Siegel-Tukey Scale: W={siegel_tukey_w:.2f}, p={siegel_tukey_p:.3e}]"
        if klotz_p is not None:
            cmh_note += f" [Klotz Scale: Z={klotz_w:.2f}, p={klotz_p:.3e}]"
        if lfk_val is not None:
            cmh_note += f" [LFK Index: {lfk_val:.2f}{' (Asym)' if is_lfk_asym else ''}]"
        if conover_p is not None:
            cmh_note += f" [Conover Scale: Z={conover_stat:.2f}, p={conover_p:.3e}]"
        if pet_p is not None:
            cmh_note += f" [PET Asymmetry: t={pet_t:.2f}, p={pet_p:.3e}]"
        if peese_p is not None:
            cmh_note += f" [PEESE Limit: t={peese_t:.2f}, p={peese_p:.3e}]"
        if sukhatme_p is not None:
            cmh_note += f" [Sukhatme Scale: Z={sukhatme_z:.2f}, p={sukhatme_p:.3e}]"
        if tang_p is not None:
            cmh_note += f" [Tang Funnel: t={tang_t:.2f}, p={tang_p:.3e}]"
        if pust_p is not None:
            cmh_note += f" [Pustejovsky Funnel: t={pust_t:.2f}, p={pust_p:.3e}]"
        if capon_p is not None:
            cmh_note += f" [Capon Scale: Z={capon_z:.2f}, p={capon_p:.3e}]"
        if moses_p is not None:
            cmh_note += f" [Moses Scale: U={moses_u:.1f}, p={moses_p:.3e}]"
        if kish_deff is not None:
            cmh_note += f" [Kish Deff: {kish_deff:.2f}, Var Ratio={strat_var_ratio:.2f}]"
        if tf_k0 > 0:
            cmh_note += f" [Trim & Fill: k0={tf_k0}, adj={tf_stat:.4f}]"
        if haus_p is not None:
            cmh_note += f" [Hausman Test: H={haus_stat:.2f}, p={haus_p:.3e}]"
        if vuong_p_val is not None:
            cmh_note += f" [Vuong Test: z={vuong_z_val:.2f}, p={vuong_p_val:.3e}]"
        if woolf_p is not None:
            cmh_note += f" [Woolf Test: p={woolf_p:.3e}]"
        if ft_p is not None:
            cmh_note += f" [Freeman-Tukey: p={ft_p:.3e}]"
        if kw_p is not None:
            cmh_note += f" [Kruskal-Wallis: H={kw_h:.2f}, p={kw_p:.3e}]"

        verdict = (
            f"SIMPSON'S PARADOX CONFIRMED: Stratifying by '{conf_col}' reverses the relationship in "
            f"{reversal_count}/{valid_groups} ({reversal_pct:.0%}) subgroups. "
            f"Global statistic was {claim.statistic_value:.4f}, but weighted within-group statistic is {weighted_stat:.4f}.{cmh_note}"
            if paradox_detected
            else f"No paradox under '{conf_col}': {valid_groups - reversal_count}/{valid_groups} subgroups align with global trend.{cmh_note}"
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
            cmh_odds_ratio=cmh_or,
            cmh_p_value=cmh_p,
            cmh_chi2=cmh_chi2,
            cohens_d=cohens_d,
            is_odds_ratio_reversal=or_reversal,
            tarone_homogeneity_p=tarone_p,
            tarone_chi2=tarone_stat,
            e_value=simpson_e_val,
            e_value_interpretation=simpson_e_desc,
            i_squared=i_sq,
            cochrans_q=cochrans_q,
            begg_funnel_tau=begg_tau,
            begg_funnel_p=begg_p,
            tau_squared=tau_sq,
            egger_test_t=egger_t,
            egger_test_p=egger_p,
            trim_and_fill_k0=tf_k0,
            trim_and_fill_adjusted_stat=tf_stat,
            hausman_stat=haus_stat,
            hausman_p=haus_p,
            is_hausman_inconsistent=is_haus_incon,
            breslow_day_chi2=bd_stat,
            breslow_day_p=bd_p,
            vuong_z=vuong_z_val,
            vuong_p=vuong_p_val,
            is_stratification_favored=is_strat_fav,
            higgins_h_squared=higgins_h2,
            galbraith_outlier_count=galb_cnt,
            galbraith_outlier_subgroups=galb_outliers,
            woolf_chi2=woolf_chi2,
            woolf_p=woolf_p,
            is_woolf_heterogeneous=is_woolf_hetero,
            freeman_tukey_chi2=ft_chi2,
            freeman_tukey_p=ft_p,
            is_freeman_tukey_heterogeneous=is_ft_hetero,
            kruskal_wallis_h=kw_h,
            kruskal_wallis_p=kw_p,
            is_kruskal_heterogeneous=is_kw_hetero,
            peto_odds_ratio=peto_or,
            peto_z_stat=peto_z,
            peto_p_value=peto_p,
            peto_heterogeneity_q=peto_q,
            peto_heterogeneity_p=peto_qp,
            is_peto_heterogeneous=is_peto_hetero,
            fligner_killeen_stat=fligner_stat,
            fligner_killeen_p=fligner_p,
            is_fligner_heterogeneous=is_fligner_hetero,
            harbord_t_stat=harbord_t,
            harbord_p_value=harbord_p,
            is_harbord_asymmetric=is_harbord_asym,
            brown_forsythe_stat=bf_stat,
            brown_forsythe_p=bf_p,
            is_brown_forsythe_heterogeneous=is_bf_hetero,
            peters_test_t=peters_t,
            peters_test_p=peters_p,
            is_peters_asymmetric=is_peters_asym,
            kish_design_effect=kish_deff,
            stratified_variance_ratio=strat_var_ratio,
            rucker_test_t=rucker_t,
            rucker_test_p=rucker_p,
            is_rucker_asymmetric=is_rucker_asym,
            levene_stat=levene_stat,
            levene_p=levene_p,
            is_levene_heterogeneous=is_levene_hetero,
            macaskill_test_t=macaskill_t,
            macaskill_test_p=macaskill_p,
            is_macaskill_asymmetric=is_macaskill_asym,
            bartlett_stat=bartlett_stat,
            bartlett_p=bartlett_p,
            is_bartlett_heterogeneous=is_bartlett_hetero,
            deeks_test_t=deeks_t,
            deeks_test_p=deeks_p,
            is_deeks_asymmetric=is_deeks_asym,
            mood_stat_z=mood_z,
            mood_p_value=mood_p,
            is_mood_heterogeneous=is_mood_hetero,
            thompson_sharp_t=thompson_sharp_t,
            thompson_sharp_p=thompson_sharp_p,
            is_thompson_asymmetric=is_thompson_asym,
            ansari_bradley_stat=ansari_w,
            ansari_bradley_p=ansari_p,
            is_ansari_heterogeneous=is_ansari_hetero,
            moreno_test_t=moreno_t,
            moreno_test_p=moreno_p,
            is_moreno_asymmetric=is_moreno_asym,
            siegel_tukey_stat=siegel_tukey_w,
            siegel_tukey_p=siegel_tukey_p,
            is_siegel_tukey_heterogeneous=is_siegel_tukey_hetero,
            sterne_test_t=sterne_t,
            sterne_test_p=sterne_p,
            is_sterne_asymmetric=is_sterne_asym,
            klotz_scale_stat=klotz_w,
            klotz_scale_p=klotz_p,
            is_klotz_heterogeneous=is_klotz_hetero,
            lfk_index=lfk_val,
            is_lfk_asymmetric=is_lfk_asym,
            conover_stat_z=conover_stat,
            conover_p_value=conover_p,
            is_conover_heterogeneous=is_conover_hetero,
            pet_test_t=pet_t,
            pet_test_p=pet_p,
            peese_test_t=peese_t,
            peese_test_p=peese_p,
            is_pet_asymmetric=is_pet_asym,
            sukhatme_stat_z=sukhatme_z,
            sukhatme_p_value=sukhatme_p,
            is_sukhatme_heterogeneous=is_sukhatme_hetero,
            tang_test_t=tang_t,
            tang_test_p=tang_p,
            is_tang_asymmetric=is_tang_asym,
            capon_stat_z=capon_z,
            capon_p_value=capon_p,
            is_capon_heterogeneous=is_capon_hetero,
            pustejovsky_test_t=pust_t,
            pustejovsky_test_p=pust_p,
            is_pustejovsky_asymmetric=is_pust_asym,
            moses_stat_u=moses_u,
            moses_stat_p=moses_p,
            is_moses_heterogeneous=is_moses_hetero,
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
        top_vif = None

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
                    top_vif = float(1.0 / max(1e-4, 1.0 - r_xz**2))
            except Exception:
                continue

        raw_r2 = float(r_xy ** 2)
        partial_r2 = float(best_partial ** 2)
        variance_loss = float(max(0.0, (raw_r2 - partial_r2) / max(1e-9, raw_r2) * 100.0))

        # Causal Sensitivity Metric: VanderWeele-Ding E-value for unmeasured confounding
        d_approx = 2.0 * abs(r_xy) / max(1e-6, np.sqrt(max(1e-9, 1.0 - r_xy**2)))
        rr_approx = float(np.exp(0.91 * d_approx))
        e_val, e_desc = calculate_e_value(rr_approx)

        # Advanced Causal Sensitivity Metric: Cinelli & Hazlett (2020) Robustness Value (RV_q=1 and RV_alpha)
        n_obs = len(df[[source, target]].dropna())
        rv_point, rv_alpha = calculate_cinelli_hazlett_rv(best_partial, n=n_obs)

        # Advanced Mediation Diagnostic: Baron-Kenny & Sobel Mediation Test for Top Confounder
        sobel_z, sobel_p, prop_med = None, None, None
        if top_conf is not None:
            sub_sob = df[[source, target, top_conf]].dropna()
            if len(sub_sob) >= 10:
                try:
                    r_xz_s, _ = stats.pearsonr(sub_sob[source], sub_sob[top_conf])
                    r_yz_s, _ = stats.pearsonr(sub_sob[target], sub_sob[top_conf])
                    sobel_z, sobel_p, prop_med = sobel_mediation_test(r_xy, r_xz_s, r_yz_s, n=len(sub_sob))
                except Exception:
                    pass

        # Advanced Bayesian Model Comparison: Kass & Raftery (1995) Bayes Factor BF_10 & Delta BIC
        bf10, delta_bic, bf_interp = None, None, None
        if top_conf is not None:
            sub_bf = df[[source, target, top_conf]].dropna()
            if len(sub_bf) >= 10:
                try:
                    r_xz_b, _ = stats.pearsonr(sub_bf[source], sub_bf[top_conf])
                    r_yz_b, _ = stats.pearsonr(sub_bf[target], sub_bf[top_conf])
                    bf10, delta_bic, bf_interp = calculate_bayes_factor_bic(r_xy, r_xz_b, r_yz_b, n=len(sub_bf))
                except Exception:
                    pass

        # Advanced Specification Diagnostic: Ramsey RESET Test & Oster (2019) Selection Bound
        reset_f, reset_p, is_reset_bad = None, None, False
        oster_delta_val = None
        f2_val = None
        itcv_val, impact_val, is_itcv_bad = None, None, False
        delta_aic_val, er_aic_val, w_aic_val = None, None, None
        if top_conf is not None:
            sub_res = df[[source, target, top_conf]].dropna()
            if len(sub_res) >= 20:
                try:
                    x_arr = sub_res[source].to_numpy(dtype=float)
                    z_arr = sub_res[top_conf].to_numpy(dtype=float)
                    y_arr = sub_res[target].to_numpy(dtype=float)
                    reset_f, reset_p, is_reset_bad = ramsey_reset_test(x_arr, z_arr, y_arr)
                    r_xz_o, _ = stats.pearsonr(x_arr, z_arr)
                    r_yz_o, _ = stats.pearsonr(y_arr, z_arr)
                    oster_delta_val = calculate_oster_delta(r_xy, r_xz_o, r_yz_o, n=len(sub_res))
                    denom_f2 = max(1e-9, 1.0 - r_xz_o ** 2)
                    r2_full_f2 = min(0.9999, max(0.0, (r_xy ** 2 + r_yz_o ** 2 - 2.0 * r_xy * r_yz_o * r_xz_o) / denom_f2))
                    f2_val = calculate_cohens_f2(r2_full_f2, r_yz_o ** 2)
                    itcv_val, impact_val, is_itcv_bad = calculate_frank_itcv(r_xy, r_xz_o, r_yz_o, n=len(sub_res))
                    delta_aic_val, er_aic_val, w_aic_val = calculate_akaike_weights(r_xy, r_xz_o, r_yz_o, n=len(sub_res))
                except Exception:
                    pass

        # Advanced Causal Sensitivity Metric: Cinelli-Hazlett Critical Confounder Boundary R2_crit
        rcrit_val = calculate_cinelli_hazlett_rcrit(best_partial)

        # Advanced Causal Sensitivity Metric: Rosenbaum (1987, 2002) Sensitivity Bounds (Gamma)
        gamma_crit, p_upper, is_rosen_sens = calculate_rosenbaum_bounds(r_xy, n=n_obs)

        # Advanced Econometric Metric: Manski (1990, 2003) Non-Parametric Partial Identification Bounds
        sub_pair = df[[source, target]].dropna()
        if pd.api.types.is_numeric_dtype(sub_pair[source]):
            x_arr_m = pd.to_numeric(sub_pair[source], errors='coerce').to_numpy(dtype=float)
        else:
            x_arr_m = pd.Series(pd.Categorical(sub_pair[source]).codes, index=sub_pair.index, dtype=float).to_numpy()
        if pd.api.types.is_numeric_dtype(sub_pair[target]):
            y_arr_m = pd.to_numeric(sub_pair[target], errors='coerce').to_numpy(dtype=float)
        else:
            y_arr_m = pd.Series(pd.Categorical(sub_pair[target]).codes, index=sub_pair.index, dtype=float).to_numpy()
        val_m = ~(np.isnan(x_arr_m) | np.isnan(y_arr_m))
        manski_l, manski_h, manski_zero = calculate_manski_bounds(x_arr_m[val_m], y_arr_m[val_m])
        im_ci_l, im_ci_h, im_cn, im_excl_zero = calculate_imbens_manski_ci(x_arr_m[val_m], y_arr_m[val_m])

        # Advanced Econometric Metric: Altonji, Elder & Taber (2005) Selection Ratio on Unobservables
        aet_val, is_aet_vuln = None, False
        mallows_cp_val, mallows_disc_val, is_mallows_bad = None, None, False
        j_stat, j_p, is_j_bad = None, None, False
        dwh_f, dwh_p, is_dwh = None, None, False
        sy_f, is_weak_conf = None, False
        bkw_cond, bkw_max_prop, is_bkw = None, None, False
        sargan_j, sargan_p, is_sargan_rej = None, None, False
        crump_ratio, is_crump_def = None, False
        hansen_f, hansen_p, has_hansen = None, None, False
        hl_chi2, hl_p, is_hl_miscal = None, None, False
        hac_se, hac_ratio, is_hac_inf = None, None, False
        aipw_tau, aipw_disc, is_aipw_conf = None, None, False
        leamer_b_min, leamer_b_max, is_leamer_frag = None, None, False
        dml_tau, dml_disc, is_dml_conf = None, None, False
        cinelli_k, is_cinelli_frag = None, False
        rubin_b, rubin_r, is_rubin_imbal = None, None, False
        bm_w, bm_p, has_bm_effect = None, None, False
        smd_val, mahal_val, is_smd_imbal = None, None, False
        imbens_diff_val, is_imbens_excess = None, False
        austin_vr, is_austin_imbal = None, False
        stuart_disc, is_stuart_disc = None, False
        hb_chi2, hb_p, is_hb_imbal = None, None, False
        caliper_rate, caliper_rad, is_caliper_disc = None, None, False
        gps_r, gps_t, gps_p, is_gps_imbal = None, None, None, False
        austin_ks_d, austin_ks_p, is_austin_ks_imbal = None, None, False
        l1_val, is_l1_imbal = None, False
        rubin_f, rubin_p, is_rubin_nonpar = None, None, False
        eqq_mean, eqq_max, is_eqq_imbal = None, None, False
        lin_t, lin_p, is_lin_hetero = None, None, False
        if top_conf is not None:
            sub_aet = df[[source, target, top_conf]].dropna()
            if len(sub_aet) >= 10:
                try:
                    r_xz_a, _ = stats.pearsonr(sub_aet[source], sub_aet[top_conf])
                    r_yz_a, _ = stats.pearsonr(sub_aet[target], sub_aet[top_conf])
                    aet_val, is_aet_vuln = calculate_altonji_elder_taber_ratio(r_xy, r_xz_a, r_yz_a)
                except Exception:
                    pass
            if len(sub_aet) >= 15:
                try:
                    x_mal = sub_aet[source].to_numpy(dtype=float)
                    z_mal = sub_aet[top_conf].to_numpy(dtype=float)
                    y_mal = sub_aet[target].to_numpy(dtype=float)
                    mallows_cp_val, mallows_disc_val, is_mallows_bad = calculate_mallows_cp(x_mal, z_mal, y_mal)
                    j_stat, j_p, is_j_bad = davidson_mackinnon_j_test(x_mal, z_mal, y_mal)
                    dwh_f, dwh_p, is_dwh = durbin_wu_hausman_test(x_mal, z_mal, y_mal)
                    sy_f, is_weak_conf = stock_yogo_first_stage_test(x_mal, z_mal)
                    bkw_cond, bkw_max_prop, is_bkw = belsley_kuh_welsch_collinearity_test(x_mal, z_mal)
                    sargan_j, sargan_p, is_sargan_rej = sargan_overidentifying_test(x_mal, z_mal, y_mal)
                    crump_ratio, is_crump_def = crump_propensity_overlap_test(x_mal, z_mal)
                    hansen_f, hansen_p, has_hansen = hansen_threshold_regime_test(x_mal, z_mal, y_mal)
                    hl_chi2, hl_p, is_hl_miscal = hosmer_lemeshow_test(x_mal, z_mal)
                    hac_se, hac_ratio, is_hac_inf = newey_west_hac_test(x_mal, z_mal, y_mal)
                    # Robins-Rotnitzky-Zhao (1994) & Imbens (2004) AIPW Doubly Robust Treatment Effect
                    aipw_tau, aipw_disc, is_aipw_conf = aipw_doubly_robust_test(x_mal, z_mal, y_mal)
                    # Chernozhukov et al. (2018) Double Machine Learning (DML) Neyman-Orthogonalized Treatment Effect
                    dml_tau, dml_disc, is_dml_conf = double_machine_learning_test(x_mal, z_mal, y_mal)
                    # Carlos Cinelli & Chad Hazlett (2020) Extreme Confounder Relative Strength Multiplier
                    cinelli_k, is_cinelli_frag = cinelli_extreme_confounder_test(x_mal, z_mal, y_mal, rv_point)
                    # Donald B. Rubin (2001) Propensity Score Covariate Balance Diagnostics
                    rubin_b, rubin_r, is_rubin_imbal = rubin_covariate_balance_test(x_mal, z_mal, y_mal)
                    # Edgar Brunner & Ullrich Munzel (2000) Non-Parametric Relative Treatment Effect
                    bm_w, bm_p, has_bm_effect = brunner_munzel_relative_effect_test(x_mal, y_mal)
                    # Rosenbaum & Rubin (1985) Standardized Mean Difference & Mahalanobis Distance
                    smd_val, mahal_val, is_smd_imbal = rosenbaum_rubin_smd_balance_test(x_mal, z_mal)
                    # Guido Imbens & Donald Rubin (2015) Normalized Difference Metric
                    imbens_diff_val, is_imbens_excess = imbens_rubin_normalized_difference_test(x_mal, z_mal)
                    # Peter C. Austin (2009) Covariate Variance Ratio Balance Test
                    austin_vr, is_austin_imbal = austin_covariate_variance_ratio_test(x_mal, z_mal)
                    # Elizabeth A. Stuart (2010) Prognostic Score Discrepancy Test
                    stuart_disc, is_stuart_disc = stuart_prognostic_score_discrepancy_test(x_mal, z_mal, y_mal)
                    # Ben B. Hansen & Jake Bowers (2008) Omnibus Covariate Balance Test
                    hb_chi2, hb_p, is_hb_imbal = hansen_bowers_balance_test(x_mal, z_mal)
                    # Paul R. Rosenbaum & Donald B. Rubin (1985) Caliper Discordance Metric
                    caliper_rate, caliper_rad, is_caliper_disc = rosenbaum_rubin_caliper_discordance_test(x_mal, z_mal)
                    # Kosuke Imai & David A. van Dyk (2004) Generalized Propensity Score Balance Test
                    gps_r, gps_t, gps_p, is_gps_imbal = imai_van_dyk_gps_balance_test(x_mal, z_mal)
                    # Peter C. Austin (2008, 2011) Kolmogorov-Smirnov Distributional Covariate Balance Test
                    austin_ks_d, austin_ks_p, is_austin_ks_imbal = austin_ks_covariate_balance_test(x_mal, z_mal)
                    # Stefano M. Iacus, Gary King, & Giuseppe Porro (2011, 2012) L1 Multivariate Imbalance Metric
                    l1_val, is_l1_imbal = l1_multivariate_imbalance_test(x_mal, z_mal)
                    # Donald B. Rubin (1979) Regression Surface Non-Parallelism Test
                    rubin_f, rubin_p, is_rubin_nonpar = rubin_parallelism_interaction_test(x_mal, z_mal, y_mal)
                    # Ho, Imai, King, & Stuart (2007) Empirical Q-Q Covariate Balance Metric
                    eqq_mean, eqq_max, is_eqq_imbal = eqq_covariate_balance_test(x_mal, z_mal)
                    # Winston Lin (2013) Agnostic Treatment-Covariate Centered Interaction Test
                    lin_t, lin_p, is_lin_hetero = lin_interaction_heterogeneity_test(x_mal, z_mal, y_mal)
                except Exception:
                    pass

        # Edward Leamer (1983) Extreme Bounds Analysis (EBA) across candidate confounders
        try:
            leamer_b_min, leamer_b_max, is_leamer_frag = leamer_extreme_bounds_test(
                df, source, target, candidates, max_candidates=4
            )
        except Exception:
            pass

        is_conf = max_att >= self.confounder_attenuation_threshold and top_conf is not None
        is_collinear = top_vif is not None and top_vif >= 5.0
        vif_str = f", VIF={top_vif:.2f}" if top_vif is not None else ""
        rv_str = f", RV={rv_point:.2f} (R²crit={rcrit_val:.2f})"
        rosen_str = f", Rosenbaum Γcrit={gamma_crit:.2f}" if gamma_crit is not None else ""
        manski_str = f", Manski=[{manski_l:.3f}, {manski_h:.3f}]" if manski_l is not None else ""
        im_str = f", Imbens-Manski=[{im_ci_l:.3f}, {im_ci_h:.3f}]" if im_ci_l is not None else ""
        aet_str = f", AET={aet_val:.2f}" if aet_val is not None else ""
        mal_str = f", Mallows Cp={mallows_cp_val:.2f} (ΔCp={mallows_disc_val:.2f})" if mallows_cp_val is not None else ""
        j_str = f", J-test={j_stat:.2f} (p={j_p:.3e})" if j_p is not None else ""
        crump_str = f", Crump overlap={crump_ratio:.1%}" if crump_ratio is not None else ""
        hansen_str = f", Hansen threshold F={hansen_f:.2f} (p={hansen_p:.3e})" if hansen_p is not None else ""
        hl_str = f", Hosmer-Lemeshow χ²={hl_chi2:.2f} (p={hl_p:.3e})" if hl_p is not None else ""
        hac_str = f", HAC SE ratio={hac_ratio:.2f}" if hac_ratio is not None else ""
        aipw_str = f", AIPW τ={aipw_tau:.3f} (Δ={aipw_disc:.1%})" if aipw_disc is not None else ""
        dml_str = f", DML τ={dml_tau:.3f} (Δ={dml_disc:.1%})" if dml_disc is not None else ""
        cinelli_str = f", Cinelli k={cinelli_k:.2f}x" if cinelli_k is not None else ""
        rubin_str = f", Rubin B={rubin_b:.1f}% (R={rubin_r:.2f})" if rubin_b is not None else ""
        bm_str = f", Brunner-Munzel W={bm_w:.2f} (p={bm_p:.3e})" if bm_w is not None else ""
        leamer_str = f", Leamer EBA=[{leamer_b_min:.3f}, {leamer_b_max:.3f}]" if leamer_b_min is not None else ""
        sob_str = f", Sobel p={sobel_p:.3e}" if sobel_p is not None else ""
        bf_str = f", BF10={bf10:.2f}" if bf10 is not None else ""
        reset_str = f", RESET p={reset_p:.3e}" if reset_p is not None else ""
        oster_str = f", Oster δ*={oster_delta_val:.2f}" if oster_delta_val is not None else ""
        f2_str = f", Cohen's f²={f2_val:.3f}" if f2_val is not None else ""
        itcv_str = f", ITCV={itcv_val:.3f}" if itcv_val is not None else ""
        aic_str = f", w_AIC={w_aic_val:.3f}" if w_aic_val is not None else ""
        desc = (
            f"Correlation between '{source}' and '{target}' collapses from r = {r_xy:.4f} (R²={raw_r2:.3f}) to partial r = {best_partial:.4f} (partial R²={partial_r2:.3f}) "
            f"({max_att:.1f}% attenuation, {variance_loss:.1f}% variance loss{vif_str}{rv_str}{rosen_str}{manski_str}{im_str}{aet_str}{mal_str}{j_str}{crump_str}{hansen_str}{hl_str}{hac_str}{aipw_str}{dml_str}{cinelli_str}{rubin_str}{bm_str}{leamer_str}{sob_str}{bf_str}{reset_str}{oster_str}{f2_str}{itcv_str}{aic_str}) when controlling for '{top_conf}'. [E-value: {e_val:.2f}]"
            if is_conf
            else f"Relationship is robust to linear confounding across tested covariates (max attenuation: {max_att:.1f}%{vif_str}{rv_str}{rosen_str}{manski_str}{im_str}{aet_str}{mal_str}{j_str}{crump_str}{hansen_str}{hl_str}{hac_str}{aipw_str}{dml_str}{cinelli_str}{rubin_str}{bm_str}{leamer_str}{bf_str}{reset_str}{oster_str}{f2_str}{itcv_str}{aic_str}). [E-value: {e_val:.2f}]"
        )

        return ConfounderAudit(
            is_confounded=is_conf,
            top_confounder=top_conf,
            raw_correlation=r_xy,
            partial_correlation=float(best_partial),
            attenuation_percentage=float(max_att),
            description=desc,
            raw_r_squared=raw_r2,
            partial_r_squared=partial_r2,
            variance_explained_loss_pct=variance_loss,
            e_value=e_val,
            e_value_interpretation=e_desc,
            top_confounder_vif=top_vif,
            is_collinear=is_collinear,
            robustness_value=rv_point,
            robustness_value_alpha=rv_alpha,
            sobel_z=sobel_z,
            sobel_p=sobel_p,
            proportion_mediated=prop_med,
            bayes_factor_bf10=bf10,
            bic_delta=delta_bic,
            bayes_factor_interpretation=bf_interp,
            ramsey_reset_f=reset_f,
            ramsey_reset_p=reset_p,
            is_reset_misspecified=is_reset_bad,
            oster_delta=oster_delta_val,
            cohens_f2=f2_val,
            cinelli_hazlett_rcrit=rcrit_val,
            frank_itcv=itcv_val,
            frank_observed_impact=impact_val,
            is_itcv_exceeded=is_itcv_bad,
            akaike_delta_aic=delta_aic_val,
            akaike_evidence_ratio=er_aic_val,
            akaike_model_weight=w_aic_val,
            rosenbaum_gamma_crit=gamma_crit,
            rosenbaum_p_upper=p_upper,
            is_rosenbaum_sensitive=is_rosen_sens,
            manski_bound_low=manski_l,
            manski_bound_high=manski_h,
            manski_contains_zero=manski_zero,
            aet_selection_ratio=aet_val,
            is_aet_vulnerable=is_aet_vuln,
            imbens_manski_ci_low=im_ci_l,
            imbens_manski_ci_high=im_ci_h,
            imbens_manski_cn=im_cn,
            imbens_manski_excludes_zero=im_excl_zero,
            mallows_cp=mallows_cp_val,
            cp_bias_discrepancy=mallows_disc_val,
            is_mallows_biased=is_mallows_bad,
            davidson_mackinnon_j_stat=j_stat,
            davidson_mackinnon_p=j_p,
            is_j_test_rejected=is_j_bad,
            durbin_wu_hausman_f=dwh_f,
            durbin_wu_hausman_p=dwh_p,
            is_dwh_endogenous=is_dwh,
            stock_yogo_f_stat=sy_f,
            is_weak_confounder=is_weak_conf,
            bkw_condition_index=bkw_cond,
            bkw_max_variance_prop=bkw_max_prop,
            is_bkw_collinear=is_bkw,
            sargan_j_stat=sargan_j,
            sargan_p_value=sargan_p,
            is_sargan_rejected=is_sargan_rej,
            crump_overlap_ratio=crump_ratio,
            is_crump_overlap_deficient=is_crump_def,
            hansen_threshold_f=hansen_f,
            hansen_threshold_p=hansen_p,
            has_hansen_threshold_regime=has_hansen,
            hosmer_lemeshow_chi2=hl_chi2,
            hosmer_lemeshow_p=hl_p,
            is_propensity_miscalibrated=is_hl_miscal,
            newey_west_hac_se=hac_se,
            hac_se_ratio=hac_ratio,
            is_hac_inflated=is_hac_inf,
            aipw_treatment_effect=aipw_tau,
            aipw_discrepancy=aipw_disc,
            is_aipw_confounded=is_aipw_conf,
            leamer_bound_min=leamer_b_min,
            leamer_bound_max=leamer_b_max,
            is_leamer_fragile=is_leamer_frag,
            dml_treatment_effect=dml_tau,
            dml_discrepancy=dml_disc,
            is_dml_confounded=is_dml_conf,
            cinelli_k_multiplier=cinelli_k,
            is_cinelli_fragile=is_cinelli_frag,
            rubin_b_score=rubin_b,
            rubin_r_ratio=rubin_r,
            is_rubin_imbalanced=is_rubin_imbal,
            brunner_munzel_w=bm_w,
            brunner_munzel_p=bm_p,
            has_brunner_munzel_effect=has_bm_effect,
            smd_covariate_balance=smd_val,
            mahalanobis_distance=mahal_val,
            is_smd_imbalanced=is_smd_imbal,
            imbens_normalized_diff=imbens_diff_val,
            is_imbens_diff_excessive=is_imbens_excess,
            austin_variance_ratio=austin_vr,
            is_austin_imbalanced=is_austin_imbal,
            stuart_prognostic_discrepancy=stuart_disc,
            is_prognostic_discrepant=is_stuart_disc,
            hansen_bowers_chi2=hb_chi2,
            hansen_bowers_p=hb_p,
            is_hansen_bowers_imbalanced=is_hb_imbal,
            caliper_discordance_rate=caliper_rate,
            caliper_threshold=caliper_rad,
            is_caliper_discordant=is_caliper_disc,
            gps_balance_r=gps_r,
            gps_balance_t=gps_t,
            gps_balance_p=gps_p,
            is_gps_imbalanced=is_gps_imbal,
            austin_ks_stat=austin_ks_d,
            austin_ks_p_value=austin_ks_p,
            is_austin_ks_imbalanced=is_austin_ks_imbal,
            l1_multivariate_imbalance=l1_val,
            is_l1_imbalanced=is_l1_imbal,
            rubin_parallelism_f=rubin_f,
            rubin_parallelism_p=rubin_p,
            is_rubin_nonparallel=is_rubin_nonpar,
            eqq_mean_diff=eqq_mean,
            eqq_max_diff=eqq_max,
            is_eqq_imbalanced=is_eqq_imbal,
            lin_interaction_t=lin_t,
            lin_interaction_p=lin_p,
            is_lin_heterogeneous=is_lin_hetero,
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

        t_start = time.perf_counter()
        
        # Safely extract numeric arrays for bootstrap
        if pd.api.types.is_numeric_dtype(sub[source]):
            x_series = pd.to_numeric(sub[source], errors='coerce')
        else:
            x_series = pd.Series(pd.Categorical(sub[source]).codes, index=sub.index, dtype=float)

        if pd.api.types.is_numeric_dtype(sub[target]):
            y_series = pd.to_numeric(sub[target], errors='coerce')
        else:
            y_series = pd.Series(pd.Categorical(sub[target]).codes, index=sub.index, dtype=float)

        valid_m = ~(x_series.isna() | y_series.isna())
        x_vals = x_series[valid_m].to_numpy(dtype=float)
        y_vals = y_series[valid_m].to_numpy(dtype=float)
        n = len(x_vals)
        if n < 5:
            return StabilityAudit(
                is_stable=False,
                bootstrap_mean=0.0,
                ci_95_low=0.0,
                ci_95_high=0.0,
                zero_crossing=True,
                trimmed_statistic=0.0,
                description="Insufficient valid numeric records for bootstrap estimation.",
            )

        # Vectorized non-parametric bootstrap resampling for 10x-50x speedup
        boot_idx = self.rng.integers(0, n, size=(self.n_bootstrap, n))
        bx = x_vals[boot_idx]
        by = y_vals[boot_idx]

        bx_c = bx - bx.mean(axis=1, keepdims=True)
        by_c = by - by.mean(axis=1, keepdims=True)

        cov = (bx_c * by_c).sum(axis=1)
        var_x = (bx_c ** 2).sum(axis=1)
        var_y = (by_c ** 2).sum(axis=1)

        denom = np.sqrt(np.maximum(1e-12, var_x * var_y))
        boot_r = cov / denom
        valid_r = boot_r[np.isfinite(boot_r)]

        if len(valid_r) == 0:
            valid_r = np.array([claim.statistic_value])

        ci_low, ci_high = np.percentile(valid_r, [2.5, 97.5])
        zero_crossing = bool(ci_low < 0 < ci_high)

        res = y_vals - (np.polyval(np.polyfit(x_vals, y_vals, 1), x_vals) if np.std(x_vals) > 0 else 0)
        non_outliers = np.abs(res) <= np.percentile(np.abs(res), 97.5)
        trim_r = float(stats.pearsonr(x_vals[non_outliers], y_vals[non_outliers])[0]) if np.sum(non_outliers) > 10 else claim.statistic_value

        is_stable = (not zero_crossing) and (np.sign(trim_r) == np.sign(claim.statistic_value))
        exec_ms = float((time.perf_counter() - t_start) * 1000.0)
        std_err = float(np.std(valid_r))

        # Monotonic rank correlation and discrepancy metric
        try:
            spearman_rho, _ = stats.spearmanr(x_vals, y_vals)
            spearman_rho = float(spearman_rho)
            rank_disc = float(abs(claim.statistic_value - spearman_rho))
        except Exception:
            spearman_rho = None
            rank_disc = None

        # Outlier leverage score (diagonal of hat matrix H = X(X^T X)^(-1)X^T)
        x_diff = x_vals - np.mean(x_vals)
        x_ss = float(np.sum(x_diff ** 2))
        if x_ss > 1e-9:
            h_ii = (1.0 / n) + (x_diff ** 2) / x_ss
            max_lev = float(np.max(h_ii))
            high_lev_count = int(np.sum(h_ii > (4.0 / n)))
        else:
            h_ii = np.full(n, 1.0 / n)
            max_lev = 0.0
            high_lev_count = 0

        # Bayesian sign invariance empirical probability
        if claim.statistic_value > 0:
            sign_inv_prob = float(np.mean(valid_r > 0))
        elif claim.statistic_value < 0:
            sign_inv_prob = float(np.mean(valid_r < 0))
        else:
            sign_inv_prob = 1.0

        # Cook's Distance Outlier Influence Diagnostic
        # D_i = (e_i^2 / (p * MSE)) * (h_ii / (1 - h_ii)^2)
        p_params = 2
        fitted_y = np.polyval(np.polyfit(x_vals, y_vals, 1), x_vals) if np.std(x_vals) > 0 else np.full_like(y_vals, np.mean(y_vals))
        residuals = y_vals - fitted_y
        sse = float(np.sum(residuals ** 2))
        mse = sse / max(1, n - p_params)

        if x_ss > 1e-9 and mse > 1e-12:
            denom_lev = np.maximum(1e-6, (1.0 - h_ii) ** 2)
            cooks_d = (residuals ** 2) / (p_params * mse) * (h_ii / denom_lev)
            max_cooks = float(np.max(cooks_d))
            cooks_influential = cooks_d > (4.0 / n)
            cooks_count = int(np.sum(cooks_influential))

            non_cooks = ~cooks_influential
            if np.sum(non_cooks) > 10 and np.std(x_vals[non_cooks]) > 1e-9 and np.std(y_vals[non_cooks]) > 1e-9:
                cooks_trim_r = float(stats.pearsonr(x_vals[non_cooks], y_vals[non_cooks])[0])
                cooks_shift = float(abs(claim.statistic_value - cooks_trim_r))
            else:
                cooks_trim_r = float(trim_r)
                cooks_shift = 0.0
        else:
            max_cooks = 0.0
            cooks_count = 0
            cooks_trim_r = float(trim_r)
            cooks_shift = 0.0

        # Advanced Diagnostic: DFFITS Regression Influence Diagnostic
        # DFFITS_i = t_i * sqrt(h_ii / (1 - h_ii)), cutoff = 2 * sqrt(p / n)
        if x_ss > 1e-9 and mse > 1e-12:
            denom_h = np.maximum(1e-6, 1.0 - h_ii)
            mse_i = np.maximum(1e-12, ((n - p_params) * mse - (residuals ** 2) / denom_h) / max(1, n - p_params - 1))
            t_stud = residuals / np.sqrt(mse_i * denom_h)
            dffits = t_stud * np.sqrt(h_ii / denom_h)
            max_dffits_val = float(np.max(np.abs(dffits)))
            dffits_threshold = 2.0 * np.sqrt(p_params / n)
            dffits_count = int(np.sum(np.abs(dffits) > dffits_threshold))

            # Advanced Diagnostic: Belsley-Kuh-Welsch DFBETAS Slope Influence Diagnostic
            denom_dfb = np.sqrt(np.maximum(1e-12, (mse_i / x_ss) * denom_h))
            dfbetas = (residuals * x_diff / x_ss) / denom_dfb
            max_dfbetas_val = float(np.max(np.abs(dfbetas)))
            dfbetas_threshold = 2.0 / np.sqrt(n)
            dfbetas_count = int(np.sum(np.abs(dfbetas) > dfbetas_threshold))
        else:
            max_dffits_val = 0.0
            dffits_count = 0
            max_dfbetas_val = 0.0
            dfbetas_count = 0

        # Advanced Falsification Battery: Non-parametric Exact Permutation Shuffle Test
        n_perms = 150
        perm_idx = np.array([self.rng.permutation(n) for _ in range(n_perms)])
        py_perm = y_vals[perm_idx]
        py_perm_c = py_perm - py_perm.mean(axis=1, keepdims=True)
        cov_perm = (x_diff * py_perm_c).sum(axis=1)
        denom_perm = np.sqrt(np.maximum(1e-12, x_ss * (py_perm_c ** 2).sum(axis=1)))
        r_perms = cov_perm / denom_perm
        perm_p = float(np.mean(np.abs(r_perms) >= abs(claim.statistic_value)))

        # Advanced Diagnostic: Breusch-Pagan Lagrange Multiplier Test for Heteroscedasticity
        bp_lm, bp_p, is_hetero = breusch_pagan_test(x_vals, residuals)

        # Advanced Diagnostic: White (1980) General Heteroscedasticity Lagrange Multiplier Test
        white_lm, white_p, is_white_hetero = white_heteroscedasticity_test(x_vals, residuals)

        # Advanced Diagnostic: Jarque-Bera Goodness-of-Fit Normality Test on Residuals
        jb_stat, jb_p, is_non_norm = jarque_bera_normality_test(residuals)

        # Advanced Diagnostic: Quantile Discrepancy Index (QDI) across 25th, 50th, 75th percentiles
        qdi_val, qdi_flag = calculate_quantile_discrepancy(x_vals, y_vals)

        # Advanced Diagnostic: Durbin-Watson Autocorrelation Test on Residuals
        dw_stat, has_serial = durbin_watson_test(residuals)

        # Advanced Diagnostic: MacKinnon-White (1985) HC3 Heteroscedasticity-Consistent Robust Test
        hc3_se, hc3_p, hc3_sig_lost = mackinnon_white_hc3_test(x_vals, y_vals, residuals, h_ii)

        # Advanced Diagnostic: Theil-Sen (1950) Robust Non-Parametric Median Slope
        ts_slope, ts_disc = calculate_theil_sen_discrepancy(x_vals, y_vals)

        # Advanced Diagnostic: Efron & Tibshirani (1993) BCa Bootstrap 95% Confidence Interval
        bca_low, bca_high, bca_zero = calculate_bca_bootstrap_ci(valid_r, claim.statistic_value, x_vals, y_vals)

        # Advanced Diagnostic: Gregory Chow (1960) Structural Break & Parameter Stability F-Test
        chow_f, chow_p, has_break = chow_structural_break_test(x_vals, y_vals)

        # Advanced Diagnostic: Andrew Gelman & John Carlin (2014) Type S (Sign) & Type M (Magnitude) Design Analysis
        type_s, type_m, retro_pow, is_type_m_vuln = calculate_gelman_carlin_design_analysis(
            estimate=claim.statistic_value,
            se=std_err,
        )

        # Advanced Diagnostic: Donald Andrews (1993) Supremum Wald Test for Unknown Structural Break
        andrews_w, andrews_frac, andrews_p, has_andrews_break = andrews_sup_wald_test(x_vals, y_vals)

        # Advanced Diagnostic: Brown, Durbin, & Evans (1975) Recursive Residual CUSUM Test
        cusum_s, cusum_p, has_cusum_drift = brown_durbin_evans_cusum_test(x_vals, y_vals)

        # Advanced Diagnostic: Koenker (1981) Studentized Breusch-Pagan Test
        koenker_lm, koenker_p, is_koenker_hetero = koenker_studentized_breusch_pagan_test(x_vals, y_vals)

        # Advanced Diagnostic: Harvey & Collier (1977) Recursive Residual Linearity Test
        hc_t, hc_p, has_hc_nonlin = harvey_collier_linearity_test(x_vals, y_vals)

        # Advanced Diagnostic: Huber (1964) Robust M-Estimator Regression Slope
        huber_slope, huber_disc, is_huber_distorted = huber_m_estimator_test(x_vals, y_vals)

        # Advanced Diagnostic: Brown, Durbin, & Evans (1975) / Ploberger & Krämer (1989) CUSUM of Squares Test
        cusumsq_val, cusumsq_crit, has_cusumsq = brown_durbin_evans_cusumsq_test(x_vals, y_vals)

        # Advanced Diagnostic: Rousseeuw (1984) Least Trimmed Squares (LTS) High-Breakdown Slope
        lts_slope, lts_disc, is_lts_distorted = least_trimmed_squares_test(x_vals, y_vals)

        # Advanced Diagnostic: Andrew Siegel (1982) Repeated Medians High-Breakdown Slope
        siegel_slope, siegel_disc, is_siegel_distorted = siegel_repeated_medians_test(x_vals, y_vals)

        # Advanced Diagnostic: Breusch & Godfrey (1978) Higher-Order Serial Correlation LM Test
        bg_lm, bg_p, has_bg_autocorr = breusch_godfrey_serial_correlation_test(x_vals, y_vals)

        # Advanced Diagnostic: Abraham Wald & Jacob Wolfowitz (1940) Runs Test for Residual Independence
        runs_z, runs_p, has_runs_drift = wald_wolfowitz_runs_test(x_vals, y_vals)

        # Advanced Diagnostic: Maurice Kendall (1938) Tau-b Rank Concordance Discrepancy
        kendall_tau, kendall_disc, is_kendall_distorted = kendall_tau_b_test(x_vals, y_vals)

        # Advanced Diagnostic: Stephen M. Goldfeld & Richard E. Quandt (1965) Split-Regime Heteroskedasticity Test
        gq_f, gq_p, has_gq_hetero = goldfeld_quandt_heteroskedasticity_test(x_vals, y_vals)

        # Advanced Diagnostic: S. S. Shapiro & R. S. Francia (1972) Residual Normality Test
        sf_w, sf_p, is_sf_non_normal = shapiro_francia_normality_test(x_vals, y_vals)

        # Advanced Diagnostic: Ralph D'Agostino & E. S. Pearson (1973) Omnibus Normality Test
        dp_k2, dp_p, is_dp_non_normal = dagostino_pearson_normality_test(x_vals, y_vals)

        # Advanced Diagnostic: Herbert Glejser (1969) Regression Heteroskedasticity Test
        glejser_t, glejser_p, has_glejser_hetero = glejser_heteroskedasticity_test(x_vals, y_vals)

        # Advanced Diagnostic: Anderson & Darling (1954) Empirical Distribution Function Residual Normality Test
        ad_stat, ad_crit, is_ad_non_normal = anderson_darling_normality_test(x_vals, y_vals)

        # Advanced Diagnostic: Andrew C. Harvey (1976) Multiplicative Error Variance Heteroskedasticity LM Test
        harvey_lm, harvey_p, has_harvey_hetero = harvey_multiplicative_heteroskedasticity_test(x_vals, y_vals)

        # Advanced Diagnostic: Hubert W. Lilliefors (1967) Composite Normality Goodness-of-Fit Test
        lill_stat, lill_p, is_lill_non_normal = lilliefors_normality_test(x_vals, y_vals)

        # Advanced Diagnostic: Richard E. Park (1966) Log-Linear Error Variance Heteroskedasticity Test
        park_t, park_p, has_park_hetero = park_loglinear_heteroskedasticity_test(x_vals, y_vals)

        # Advanced Diagnostic: Harald Cramér (1928) & Richard von Mises (1928) EDF Residual Normality Test
        cvm_w2, cvm_p, is_cvm_non_normal = cramer_von_mises_normality_test(x_vals, y_vals)

        # Advanced Diagnostic: R. Dennis Cook & Sanford Weisberg (1983) Fitted-Value Heteroskedasticity Score Test
        cw_score, cw_p, has_cw_hetero = cook_weisberg_heteroskedasticity_test(x_vals, y_vals)

        # Advanced Diagnostic: Nicolaas H. Kuiper (1960) Invariant EDF Residual Normality Test
        kuiper_v, kuiper_v_star, kuiper_p, is_kuiper_non_normal = kuiper_normality_test(x_vals, y_vals)

        # Advanced Diagnostic: Belsley, Kuh, & Welsch (1980) COVRATIO Covariance Determinant Influence Diagnostic
        covratio_max_dev, covratio_cnt, is_covratio_infl = covratio_influence_test(x_vals, y_vals)

        # Advanced Diagnostic: J. R. M. Hosking (1990) L-Moments Ratio Residual Normality Test
        hosking_l, hosking_p, is_hosking_non_normal = hosking_l_moments_normality_test(x_vals, y_vals)

        # Advanced Diagnostic: Anthony C. Atkinson (1981, 1985) Modified Leverage-Weighted Influence Metric
        atkinson_max, atkinson_cnt, is_atkinson_infl = atkinson_influence_test(x_vals, y_vals)

        # Advanced Diagnostic: Samuel S. Shapiro & Martin B. Wilk (1965) Omnibus Residual Normality Test
        sw_w, sw_p, is_sw_non_normal = shapiro_wilk_normality_test(x_vals, y_vals)

        # Advanced Diagnostic: Ali S. Hadi (1992) Measure of Overall Potential Influence in Regression
        hadi_max, hadi_cnt, is_hadi_infl = hadi_influence_test(x_vals, y_vals)

        is_stable = (not zero_crossing) and (not bca_zero) and (np.sign(trim_r) == np.sign(claim.statistic_value)) and (sign_inv_prob >= 0.90) and (perm_p < 0.05)

        disc_str = f", Δr_rank={rank_disc:.3f}" if rank_disc is not None else ""
        cooks_str = f", Cook's D_max={max_cooks:.3f} ({cooks_count} influential)" if max_cooks > 0 else ""
        dffits_str = f", DFFITS_max={max_dffits_val:.3f}" if max_dffits_val > 0 else ""
        dfb_str = f", DFBETAS_max={max_dfbetas_val:.3f}" if max_dfbetas_val > 0 else ""
        bp_str = f", BP_p={bp_p:.3e}" if bp_p is not None else ""
        white_str = f", White_p={white_p:.3e}" if white_p is not None else ""
        chow_str = f", Chow_p={chow_p:.3e}" if chow_p is not None else ""
        andrews_str = f", Andrews_SupW={andrews_w:.2f} (p={andrews_p:.3e})" if andrews_w is not None else ""
        cusum_str = f", CUSUM_S={cusum_s:.3f} (p={cusum_p:.3e})" if cusum_s is not None else ""
        cusumsq_str = f", CUSUMSQ={cusumsq_val:.3f} (crit={cusumsq_crit:.3f})" if cusumsq_val is not None else ""
        koenker_str = f", Koenker_p={koenker_p:.3e}" if koenker_p is not None else ""
        jb_str = f", JB_p={jb_p:.3e}" if jb_p is not None else ""
        qdi_str = f", QDI={qdi_val:.2f}" if qdi_val is not None else ""
        dw_str = f", DW={dw_stat:.2f}" if dw_stat is not None else ""
        hc3_str = f", HC3_p={hc3_p:.3e}" if hc3_p is not None else ""
        ts_str = f", Δ_TS={ts_disc:.1%}" if ts_disc is not None else ""
        lts_str = f", Δ_LTS={lts_disc:.1%}" if lts_disc is not None else ""
        siegel_str = f", Δ_Siegel={siegel_disc:.1%}" if siegel_disc is not None else ""
        bg_str = f", BG_p={bg_p:.3e}" if bg_p is not None else ""
        runs_str = f", Runs_Z={runs_z:.2f} (p={runs_p:.3e})" if runs_z is not None else ""
        kendall_str = f", Δ_Kendall={kendall_disc:.1%}" if kendall_disc is not None else ""
        gq_str = f", GQ_p={gq_p:.3e}" if gq_p is not None else ""
        glejser_str = f", Glejser_p={glejser_p:.3e}" if glejser_p is not None else ""
        sf_str = f", SF_W={sf_w:.3f} (p={sf_p:.3e})" if sf_w is not None else ""
        dp_str = f", DP_K2={dp_k2:.2f} (p={dp_p:.3e})" if dp_k2 is not None else ""
        bca_str = f", BCa=[{bca_low:.4f}, {bca_high:.4f}]"
        type_m_str = f", Type M={type_m:.2f} (Type S={type_s:.1%})" if type_m is not None else ""
        desc = f"Vectorized Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]{bca_str} (SE={std_err:.4f}, P(sign)={sign_inv_prob:.1%}, p_perm={perm_p:.3e}, {exec_ms:.1f}ms{disc_str}{cooks_str}{dffits_str}{dfb_str}{bp_str}{koenker_str}{white_str}{gq_str}{glejser_str}{chow_str}{andrews_str}{cusum_str}{cusumsq_str}{jb_str}{sf_str}{dp_str}{qdi_str}{dw_str}{bg_str}{runs_str}{hc3_str}{ts_str}{lts_str}{siegel_str}{kendall_str}{type_m_str}). Trimmed r = {trim_r:.4f}."

        return StabilityAudit(
            is_stable=is_stable,
            bootstrap_mean=float(np.mean(valid_r)),
            ci_95_low=float(ci_low),
            ci_95_high=float(ci_high),
            zero_crossing=zero_crossing,
            trimmed_statistic=float(trim_r),
            description=desc,
            bootstrap_std_err=std_err,
            execution_time_ms=exec_ms,
            spearman_rho=spearman_rho,
            rank_discrepancy=rank_disc,
            max_leverage=max_lev,
            high_leverage_points=high_lev_count,
            sign_invariance_prob=sign_inv_prob,
            max_cooks_distance=max_cooks,
            cooks_outlier_count=cooks_count,
            cooks_trimmed_statistic=cooks_trim_r,
            cooks_statistic_shift=cooks_shift,
            max_dffits=max_dffits_val,
            dffits_outlier_count=dffits_count,
            permutation_p_value=perm_p,
            breusch_pagan_lm=bp_lm,
            breusch_pagan_p=bp_p,
            is_heteroscedastic=is_hetero,
            quantile_discrepancy_index=qdi_val,
            qdi_flag=qdi_flag,
            max_dfbetas=max_dfbetas_val,
            dfbetas_outlier_count=dfbetas_count,
            jarque_bera_stat=jb_stat,
            jarque_bera_p=jb_p,
            is_non_normal=is_non_norm,
            durbin_watson_stat=dw_stat,
            has_serial_correlation=has_serial,
            hc3_robust_se=hc3_se,
            hc3_robust_p=hc3_p,
            hc3_significance_lost=hc3_sig_lost,
            theil_sen_slope=ts_slope,
            theil_sen_discrepancy=ts_disc,
            bca_ci_low=bca_low,
            bca_ci_high=bca_high,
            bca_zero_crossing=bca_zero,
            white_lm_stat=white_lm,
            white_lm_p=white_p,
            is_white_heteroscedastic=is_white_hetero,
            chow_f_stat=chow_f,
            chow_p_value=chow_p,
            has_structural_break=has_break,
            type_s_error_prob=type_s,
            type_m_exaggeration_ratio=type_m,
            retrodesign_power=retro_pow,
            is_type_m_vulnerable=is_type_m_vuln,
            andrews_sup_w=andrews_w,
            andrews_break_fraction=andrews_frac,
            andrews_p_value=andrews_p,
            has_unknown_structural_break=has_andrews_break,
            cusum_stat=cusum_s,
            cusum_p_approx=cusum_p,
            has_cusum_instability=has_cusum_drift,
            koenker_lm_stat=koenker_lm,
            koenker_p_value=koenker_p,
            is_koenker_heteroscedastic=is_koenker_hetero,
            harvey_collier_t=hc_t,
            harvey_collier_p=hc_p,
            has_harvey_collier_nonlinearity=has_hc_nonlin,
            huber_m_slope=huber_slope,
            huber_slope_discrepancy=huber_disc,
            is_huber_outlier_distorted=is_huber_distorted,
            cusumsq_stat=cusumsq_val,
            cusumsq_critical_val=cusumsq_crit,
            has_cusumsq_instability=has_cusumsq,
            lts_slope=lts_slope,
            lts_slope_discrepancy=lts_disc,
            is_lts_outlier_distorted=is_lts_distorted,
            siegel_slope=siegel_slope,
            siegel_slope_discrepancy=siegel_disc,
            is_siegel_outlier_distorted=is_siegel_distorted,
            breusch_godfrey_lm=bg_lm,
            breusch_godfrey_p=bg_p,
            has_higher_order_autocorrelation=has_bg_autocorr,
            runs_test_z=runs_z,
            runs_test_p=runs_p,
            has_runs_nonrandomness=has_runs_drift,
            kendall_tau_b=kendall_tau,
            kendall_discrepancy=kendall_disc,
            is_kendall_distorted=is_kendall_distorted,
            goldfeld_quandt_f=gq_f,
            goldfeld_quandt_p=gq_p,
            has_goldfeld_quandt_heteroskedasticity=has_gq_hetero,
            shapiro_francia_w=sf_w,
            shapiro_francia_p=sf_p,
            is_shapiro_francia_non_normal=is_sf_non_normal,
            dagostino_pearson_k2=dp_k2,
            dagostino_pearson_p=dp_p,
            is_dagostino_non_normal=is_dp_non_normal,
            glejser_t_stat=glejser_t,
            glejser_p_value=glejser_p,
            has_glejser_heteroskedasticity=has_glejser_hetero,
            anderson_darling_stat=ad_stat,
            anderson_darling_crit=ad_crit,
            is_anderson_non_normal=is_ad_non_normal,
            harvey_lm_stat=harvey_lm,
            harvey_lm_p=harvey_p,
            has_harvey_heteroskedasticity=has_harvey_hetero,
            lilliefors_stat=lill_stat,
            lilliefors_p_value=lill_p,
            is_lilliefors_non_normal=is_lill_non_normal,
            park_t_stat=park_t,
            park_p_value=park_p,
            has_park_heteroskedasticity=has_park_hetero,
            cramer_von_mises_w2=cvm_w2,
            cramer_von_mises_p=cvm_p,
            is_cvm_non_normal=is_cvm_non_normal,
            cook_weisberg_score=cw_score,
            cook_weisberg_p=cw_p,
            has_cook_weisberg_heteroskedasticity=has_cw_hetero,
            kuiper_v_stat=kuiper_v,
            kuiper_v_star=kuiper_v_star,
            kuiper_p_value=kuiper_p,
            is_kuiper_non_normal=is_kuiper_non_normal,
            max_covratio_dev=covratio_max_dev,
            covratio_outlier_count=covratio_cnt,
            is_covratio_influential=is_covratio_infl,
            hosking_l_stat=hosking_l,
            hosking_l_p=hosking_p,
            is_hosking_non_normal=is_hosking_non_normal,
            max_atkinson_distance=atkinson_max,
            atkinson_outlier_count=atkinson_cnt,
            is_atkinson_influential=is_atkinson_infl,
            shapiro_wilk_w=sw_w,
            shapiro_wilk_p=sw_p,
            is_shapiro_wilk_non_normal=is_sw_non_normal,
            max_hadi_influence=hadi_max,
            hadi_outlier_count=hadi_cnt,
            is_hadi_influential=is_hadi_infl,
        )

    def _audit_leakage(self, claim: Claim, df: pd.DataFrame) -> bool:
        if claim.claim_type == "correlation" and abs(claim.statistic_value) >= 0.98:
            return True
        if claim.claim_type in ("group_difference", "proportion_difference"):
            diff = claim.metadata.get("diff", 0.0)
            if diff >= 0.99 or np.isinf(claim.statistic_value):
                return True
        return False
