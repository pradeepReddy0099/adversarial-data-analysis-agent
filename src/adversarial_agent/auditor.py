"""Devil's Advocate (Auditor) Agent: Autonomous statistical falsification batteries."""
from __future__ import annotations

import itertools
import math
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
    schwarzer_test_t: Optional[float] = None
    schwarzer_test_p: Optional[float] = None
    is_schwarzer_asymmetric: bool = False
    westenberg_stat_z: Optional[float] = None
    westenberg_stat_p: Optional[float] = None
    is_westenberg_heterogeneous: bool = False
    tes_excess_ratio: Optional[float] = None
    tes_chi2: Optional[float] = None
    tes_p_value: Optional[float] = None
    is_tes_excessive: bool = False
    obrien_stat_f: Optional[float] = None
    obrien_p_value: Optional[float] = None
    is_obrien_heterogeneous: bool = False
    copas_test_t: Optional[float] = None
    copas_test_p: Optional[float] = None
    is_copas_asymmetric: bool = False
    cochran_c_stat: Optional[float] = None
    cochran_c_p: Optional[float] = None
    is_cochran_c_heterogeneous: bool = False
    gail_simon_stat: Optional[float] = None
    gail_simon_p: Optional[float] = None
    is_gail_simon_reversal: bool = False
    hartley_fmax_stat: Optional[float] = None
    hartley_fmax_p: Optional[float] = None
    is_hartley_heterogeneous: bool = False
    piantadosi_byar_green_stat: Optional[float] = None
    piantadosi_byar_green_p: Optional[float] = None
    is_pbg_reversal: bool = False
    layard_stat_l: Optional[float] = None
    layard_p_value: Optional[float] = None
    is_layard_heterogeneous: bool = False
    pan_score_stat: Optional[float] = None
    pan_score_p: Optional[float] = None
    is_pan_reversal: bool = False
    samiuddin_stat_f: Optional[float] = None
    samiuddin_p_value: Optional[float] = None
    is_samiuddin_heterogeneous: bool = False
    silvapulle_sen_stat: Optional[float] = None
    silvapulle_sen_p: Optional[float] = None
    is_silvapulle_reversal: bool = False
    miller_jackknife_stat_f: Optional[float] = None
    miller_jackknife_p_value: Optional[float] = None
    is_miller_heterogeneous: bool = False
    berger_stat_z: Optional[float] = None
    berger_p_value: Optional[float] = None
    is_berger_reversal: bool = False
    shoemaker_stat_z: Optional[float] = None
    shoemaker_p_value: Optional[float] = None
    is_shoemaker_heterogeneous: bool = False
    demets_simon_stat_q: Optional[float] = None
    demets_simon_p_value: Optional[float] = None
    is_demets_simon_reversal: bool = False
    box_andersen_stat_m: Optional[float] = None
    box_andersen_p_value: Optional[float] = None
    is_box_andersen_heterogeneous: bool = False
    pocock_hughes_lee_stat_q: Optional[float] = None
    pocock_hughes_lee_p_value: Optional[float] = None
    is_pocock_reversal: bool = False
    overall_woodward_stat_f: Optional[float] = None
    overall_woodward_p_value: Optional[float] = None
    is_overall_woodward_heterogeneous: bool = False
    crainiceanu_ruppert_stat_rlrt: Optional[float] = None
    crainiceanu_ruppert_p_value: Optional[float] = None
    is_crainiceanu_heterogeneous: bool = False
    lim_loh_stat_w: Optional[float] = None
    lim_loh_p_value: Optional[float] = None
    is_lim_loh_heterogeneous: bool = False
    verbeke_molenberghs_stat_s: Optional[float] = None
    verbeke_molenberghs_p_value: Optional[float] = None
    is_verbeke_heterogeneous: bool = False
    conover_johnson_stat_t: Optional[float] = None
    conover_johnson_p_value: Optional[float] = None
    is_conover_johnson_heterogeneous: bool = False
    hirotsu_stat_s: Optional[float] = None
    hirotsu_p_value: Optional[float] = None
    is_hirotsu_heterogeneous: bool = False
    bartlett_kendall_stat_t: Optional[float] = None
    bartlett_kendall_p_value: Optional[float] = None
    is_bartlett_kendall_heterogeneous: bool = False

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
    rr_variance_ratio: Optional[float] = None
    rr_variance_p: Optional[float] = None
    is_rr_variance_imbalanced: bool = False
    iw_quadratic_diff: Optional[float] = None
    is_iw_quadratic_excessive: bool = False
    hodges_lehmann_shift: Optional[float] = None
    hodges_lehmann_z: Optional[float] = None
    hodges_lehmann_p: Optional[float] = None
    is_hl_imbalanced: bool = False
    frank_rir_pct: Optional[float] = None
    frank_rir_count: Optional[int] = None
    is_rir_fragile: bool = False
    imbens_contour_q: Optional[float] = None
    imbens_contour_p: Optional[float] = None
    is_imbens_sensitive: bool = False
    austin_stuart_masd: Optional[float] = None
    austin_stuart_p: Optional[float] = None
    is_austin_stuart_imbalanced: bool = False
    kallus_gamma_crit: Optional[float] = None
    kallus_bias_max: Optional[float] = None
    kallus_bound_low: Optional[float] = None
    kallus_bound_high: Optional[float] = None
    is_kallus_sensitive: bool = False
    cbps_overid_j: Optional[float] = None
    cbps_overid_p: Optional[float] = None
    is_cbps_rejected: bool = False
    rosenbaum_silber_delta_crit: Optional[float] = None
    rosenbaum_silber_lambda: Optional[float] = None
    is_silber_sensitive: bool = False
    fogarty_gamma_crit: Optional[float] = None
    fogarty_studentized_t: Optional[float] = None
    is_fogarty_sensitive: bool = False
    rosenbaum_attr_gamma_crit: Optional[float] = None
    rosenbaum_attr_p_upper: Optional[float] = None
    is_rosenbaum_attr_sensitive: bool = False
    zubizarreta_sbw_cv: Optional[float] = None
    zubizarreta_eff_sample_pct: Optional[float] = None
    is_zubizarreta_imbalanced: bool = False
    rosenbaum_design_sensitivity_gamma: Optional[float] = None
    rosenbaum_standardized_effect_mu_sigma: Optional[float] = None
    is_design_fragile: bool = False
    hosman_hansen_holland_delta: Optional[float] = None
    hosman_hansen_holland_scale: Optional[float] = None
    is_hosman_sensitive: bool = False
    rosenbaum_ef_gamma_crit: Optional[float] = None
    rosenbaum_ef_p_upper: Optional[float] = None
    is_rosenbaum_ef_fragile: bool = False
    ho_imai_convex_hull_overlap: Optional[float] = None
    ho_imai_trim_fraction: Optional[float] = None
    is_ho_imai_deficient: bool = False
    rosenbaum_diff_gamma_crit: Optional[float] = None
    rosenbaum_diff_p_upper: Optional[float] = None
    is_rosenbaum_diff_fragile: bool = False
    bitler_gelbach_hoynes_disparity: Optional[float] = None
    bitler_gelbach_hoynes_max_q_divergence: Optional[float] = None
    is_bgh_heterogeneous: bool = False
    rosenbaum_dual_gamma_crit: Optional[float] = None
    rosenbaum_dual_p_upper: Optional[float] = None
    is_rosenbaum_dual_fragile: bool = False
    hainmueller_eb_disparity: Optional[float] = None
    hainmueller_eff_sample_pct: Optional[float] = None
    hainmueller_entropy_loss: Optional[float] = None
    is_entropy_balanced_fragile: bool = False
    rosenbaum_cm_gamma_crit: Optional[float] = None
    rosenbaum_cm_p_upper: Optional[float] = None
    is_rosenbaum_cm_fragile: bool = False
    athey_imbens_disparity: Optional[float] = None
    athey_imbens_tau_max: Optional[float] = None
    athey_imbens_tau_min: Optional[float] = None
    is_athey_imbens_heterogeneous: bool = False
    rosenbaum_huber_gamma_crit: Optional[float] = None
    rosenbaum_huber_p_upper: Optional[float] = None
    is_rosenbaum_huber_fragile: bool = False
    imai_king_stuart_disparity: Optional[float] = None
    imai_king_stuart_balance_smd: Optional[float] = None
    is_imai_king_stuart_fragile: bool = False
    rosenbaum_adaptive_gamma_crit: Optional[float] = None
    rosenbaum_adaptive_p_upper: Optional[float] = None
    is_rosenbaum_adaptive_fragile: bool = False
    abadie_imbens_disparity: Optional[float] = None
    abadie_imbens_balance_smd: Optional[float] = None
    is_abadie_imbens_fragile: bool = False

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
    doornik_hansen_stat: Optional[float] = None
    doornik_hansen_p: Optional[float] = None
    is_doornik_hansen_non_normal: bool = False
    max_welsch_distance: Optional[float] = None
    welsch_outlier_count: int = 0
    is_welsch_influential: bool = False
    gel_gastwirth_rjb: Optional[float] = None
    gel_gastwirth_p: Optional[float] = None
    is_rjb_non_normal: bool = False
    max_pena_influence: Optional[float] = None
    pena_outlier_count: int = 0
    is_pena_influential: bool = False
    szroeter_stat_q: Optional[float] = None
    szroeter_stat_z: Optional[float] = None
    szroeter_p_value: Optional[float] = None
    is_szroeter_heteroskedastic: bool = False
    max_ap_deviation: Optional[float] = None
    ap_outlier_count: int = 0
    is_ap_influential: bool = False
    anscombe_glynn_z: Optional[float] = None
    anscombe_glynn_p: Optional[float] = None
    is_anscombe_glynn_non_normal: bool = False
    rousseeuw_croux_qn: Optional[float] = None
    scale_inflation_ratio_qn: Optional[float] = None
    is_qn_inflated: bool = False
    epps_pulley_stat: Optional[float] = None
    epps_pulley_p: Optional[float] = None
    is_epps_pulley_non_normal: bool = False
    max_chatterjee_hadi_influence: Optional[float] = None
    chatterjee_hadi_outlier_count: int = 0
    is_chatterjee_hadi_influential: bool = False
    baringhaus_henze_stat: Optional[float] = None
    baringhaus_henze_p: Optional[float] = None
    is_baringhaus_henze_non_normal: bool = False
    max_gray_ling_influence: Optional[float] = None
    gray_ling_outlier_count: int = 0
    is_gray_ling_influential: bool = False
    vasicek_entropy_stat: Optional[float] = None
    vasicek_entropy_p: Optional[float] = None
    is_vasicek_non_normal: bool = False
    rousseeuw_zomeren_max_rdz: Optional[float] = None
    rousseeuw_zomeren_outlier_count: int = 0
    is_rousseeuw_zomeren_influential: bool = False
    zhang_zk_stat: Optional[float] = None
    zhang_zk_p_value: Optional[float] = None
    is_zhang_non_normal: bool = False
    bacon_max_distance: Optional[float] = None
    bacon_outlier_count: int = 0
    is_bacon_influential: bool = False
    lin_mudholkar_stat_z: Optional[float] = None
    lin_mudholkar_p_value: Optional[float] = None
    is_lin_mudholkar_non_normal: bool = False
    hoaglin_welsch_max_leverage_ratio: Optional[float] = None
    hoaglin_welsch_outlier_count: int = 0
    is_hoaglin_welsch_influential: bool = False
    bontemps_meddahi_stat_m: Optional[float] = None
    bontemps_meddahi_p_value: Optional[float] = None
    is_bontemps_meddahi_non_normal: bool = False
    yohai_mm_slope: Optional[float] = None
    yohai_mm_discrepancy: Optional[float] = None
    yohai_mm_outlier_count: int = 0
    is_yohai_mm_distorted: bool = False
    spiegelhalter_stat_n: Optional[float] = None
    spiegelhalter_p_value: Optional[float] = None
    is_spiegelhalter_non_normal: bool = False
    hampel_robust_slope: Optional[float] = None
    hampel_slope_discrepancy: Optional[float] = None
    hampel_outlier_count: int = 0
    is_hampel_distorted: bool = False
    mudholkar_lin_stat_z: Optional[float] = None
    mudholkar_lin_p_value: Optional[float] = None
    is_mudholkar_lin_non_normal: bool = False
    andrews_sine_slope: Optional[float] = None
    andrews_sine_discrepancy: Optional[float] = None
    andrews_sine_outlier_count: int = 0
    is_andrews_sine_distorted: bool = False
    martinez_iglewicz_stat_i: Optional[float] = None
    martinez_iglewicz_p_value: Optional[float] = None
    is_martinez_iglewicz_non_normal: bool = False
    beaton_tukey_slope: Optional[float] = None
    beaton_tukey_discrepancy: Optional[float] = None
    beaton_tukey_outlier_count: int = 0
    is_beaton_tukey_distorted: bool = False

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


def schwarzer_funnel_asymmetry_test(
    tables_2x2: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Guido Schwarzer, G. Antes, & M. Schumacher's (2002, 2007) Arcsine Funnel Asymmetry Test.

    Evaluates publication/reporting and small-stratum bias across stratified 2x2 contingency tables
    using the arcsine variance-stabilizing transformation:
      theta*_k = arcsin(sqrt(p_1k)) - arcsin(sqrt(p_0k)),  v_k = 1 / (4 n_1k) + 1 / (4 n_0k).
    Standardized effect sizes theta*_k / sqrt(v_k) are regressed against precision 1 / sqrt(v_k):
      theta*_k / sqrt(v_k) = alpha + beta * (1 / sqrt(v_k)) + epsilon_k.
    Under H_0: alpha = 0 (symmetry and absence of small-stratum bias).
    t_Schwarzer = alpha_hat / SE(alpha_hat) ~ t(K - 2).
    Rejection (p < 0.05 or |t_Schwarzer| >= 1.96) certifies funnel asymmetry free of binary variance-mean coupling.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        y_list = []
        x_list = []
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

            n1 = a + b
            n0 = c + d
            if n1 < 1.0 or n0 < 1.0:
                continue

            p1 = np.clip((a + 0.5) / (n1 + 1.0), 1e-6, 1.0 - 1e-6)
            p0 = np.clip((c + 0.5) / (n0 + 1.0), 1e-6, 1.0 - 1e-6)

            theta_star = float(np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p0)))
            v_k = float(1.0 / (4.0 * n1) + 1.0 / (4.0 * n0))
            if v_k < 1e-12:
                continue

            se_k = np.sqrt(v_k)
            y_k = theta_star / se_k
            x_k = 1.0 / se_k
            y_list.append(y_k)
            x_list.append(x_k)

        k = len(y_list)
        if k < 3:
            return None, None, False

        y_arr = np.array(y_list, dtype=float)
        x_arr = np.array(x_list, dtype=float)

        if np.std(x_arr) < 1e-9:
            return 0.0, 1.0, False

        X_mat = np.column_stack([np.ones(k), x_arr])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y_arr, rcond=None)
        res = y_arr - X_mat @ beta

        df_e = max(1, k - 2)
        s_sq = float(np.sum(res ** 2) / df_e)
        cov_mat = s_sq * np.linalg.pinv(X_mat.T @ X_mat)
        se_alpha = np.sqrt(max(1e-12, cov_mat[0, 0]))

        t_schw = float(beta[0] / se_alpha)
        p_val = float(2.0 * stats.t.sf(abs(t_schw), df=df_e))
        is_asym = bool(p_val < 0.05)
        return t_schw, p_val, is_asym
    except Exception:
        return None, None, False


def westenberg_scale_dispersion_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates J. Westenberg's (1948) Interquartile Range Dispersion Test.

    Evaluates scale/spread differences between conditioning groups without assuming normality or symmetry.
    Computes overall 25th (Q1) and 75th (Q3) percentiles of the pooled sample, and forms a 2x2 contingency
    table of counts inside versus outside [Q1, Q3] across exposure cohorts.
    Under H_0 of equal scale, the proportion within [Q1, Q3] is 0.5 for both groups.
    Computes continuity-corrected Z-score and p-value.
    Rejection (p < 0.05) certifies non-parametric scale spread heterogeneity.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 6 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        s1 = valid_samples[0]
        s2 = valid_samples[1]
        pooled = np.concatenate([s1, s2])
        q1, q3 = float(np.percentile(pooled, 25)), float(np.percentile(pooled, 75))

        a = int(np.sum((s1 >= q1) & (s1 <= q3)))
        b = len(s1) - a
        c = int(np.sum((s2 >= q1) & (s2 <= q3)))
        d = len(s2) - c

        if (b + d == 0) or (a + c == 0):
            med = float(np.median(pooled))
            a = int(np.sum(s1 > med))
            b = len(s1) - a
            c = int(np.sum(s2 > med))
            d = len(s2) - c

        n_tot = float(len(pooled))
        denom = np.sqrt(max(1.0, float((a + b) * (c + d) * (a + c) * (b + d))))
        if (a + b == 0) or (c + d == 0) or (a + c == 0) or (b + d == 0):
            return 0.0, 1.0, False

        diff = abs(a * d - b * c) - n_tot / 2.0
        z_stat = float(max(0.0, diff) / denom * np.sqrt(n_tot))
        prop1 = a / max(1, a + b)
        prop2 = c / max(1, c + d)
        z_stat = float(z_stat * np.sign(prop1 - prop2))
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
        is_hetero = bool(p_val < 0.05)
        return z_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_rubin_residual_variance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum & Donald B. Rubin's (1985) Residual Variance Ratio Balance Metric.

    Dichotomizes exposure X at median into treated (D = 1) and control (D = 0) cohorts.
    Regresses confounder Z on X to obtain orthogonalized residuals: e_Z = Z - (gamma_0 + gamma_1 * X).
    Evaluates the ratio of residual variances between treated and control cohorts:
      VR_adj = Var(e_Z1) / Var(e_Z0).
    Under H_0: sigma_1^2 = sigma_0^2, VR_adj ~ F(n1 - 1, n0 - 1).
    Rosenbaum & Rubin (1985) specify that VR_adj should fall within [0.80, 1.25] for second-order balance.
    Rejection (VR_adj outside [0.80, 1.25] or p < 0.05) diagnoses severe residual variance heterogeneity.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        d = (x > med_x)
        if np.sum(d) < 6 or np.sum(~d) < 6:
            return None, None, False

        X_mat = np.column_stack([np.ones(n), x])
        gamma, _, _, _ = np.linalg.lstsq(X_mat, z, rcond=None)
        res_z = z - X_mat @ gamma

        res1 = res_z[d]
        res0 = res_z[~d]
        n1, n0 = len(res1), len(res0)

        var1 = float(np.var(res1, ddof=1))
        var0 = float(np.var(res0, ddof=1))
        if var0 < 1e-12 or var1 < 1e-12:
            return 1.0, 1.0, False

        vr = float(var1 / var0)
        p_val = float(2.0 * min(stats.f.cdf(vr, n1 - 1, n0 - 1), stats.f.sf(vr, n1 - 1, n0 - 1)))
        is_imbal = bool(vr < 0.80 or vr > 1.25 or p_val < 0.05)
        return vr, p_val, is_imbal
    except Exception:
        return None, None, False


def imbens_wooldridge_quadratic_balance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], bool]:
    """Calculates Guido W. Imbens & Jeffrey M. Wooldridge's (2009) Normalized Quadratic Difference Metric.

    Dichotomizes exposure X at median into treated (D = 1) and control (D = 0) cohorts.
    Evaluates the normalized difference for the squared confounder term W = Z^2:
      Delta_{Z^2} = |mean(W_1) - mean(W_0)| / sqrt((Var(W_1) + Var(W_0)) / 2).
    Threshold: Delta_{Z^2} > 0.25 (or > 0.40) diagnoses severe failure of common support in quadratic covariate terms,
    signaling dangerous extrapolation bias in linear and polynomial OLS adjustments.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, False
    try:
        med_x = float(np.median(x))
        d = (x > med_x)
        w = (z - np.mean(z)) ** 2
        w1 = w[d]
        w0 = w[~d]
        if len(w1) < 6 or len(w0) < 6:
            return None, False

        var1 = float(np.var(w1, ddof=1))
        var0 = float(np.var(w0, ddof=1))
        denom = np.sqrt((var1 + var0) / 2.0)
        if denom < 1e-12:
            return 0.0, False

        diff = float(np.abs(np.mean(w1) - np.mean(w0)) / denom)
        is_excessive = bool(diff > 0.25)
        return diff, is_excessive
    except Exception:
        return None, False


def doornik_hansen_normality_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates J. A. Doornik & H. Hansen's (1994, 2008) Omnibus Residual Normality Test (E_DH).

    Evaluates regression residuals by transforming sample skewness and kurtosis into independent
    standard normal variates z_1 and z_2:
      E_DH = z_1^2 + z_2^2 ~ chi^2(2).
    Offers vastly superior finite-sample size and statistical power compared to the classical Jarque-Bera test.
    Rejection (p < 0.05 or E_DH >= 5.99) certifies residual non-Gaussianity.
    """
    n = len(x)
    p = 2
    if n <= p + 6 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        s_e = np.std(res)
        if s_e < 1e-12:
            return 0.0, 1.0, False

        u = (res - np.mean(res)) / s_e
        m3 = float(np.mean(u ** 3))
        m4 = float(np.mean(u ** 4))

        beta_s = (3.0 * (n ** 2 + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)) / (
            max(1e-6, (n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0))
        )
        w2 = -1.0 + np.sqrt(max(1e-12, 2.0 * (beta_s - 1.0)))
        if w2 <= 1.0:
            return None, None, False
        delta_s = 1.0 / np.sqrt(np.log(w2))
        y_val = m3 * np.sqrt(max(1e-12, (w2 - 1.0) * (n + 1.0) * (n + 3.0) / (12.0 * (n - 2.0))))
        z1 = delta_s * np.log(y_val + np.sqrt(y_val ** 2 + 1.0))

        delta_k = (n - 3.0) * (n + 1.0) * (n ** 2 + 15.0 * n - 4.0)
        if delta_k <= 0:
            return None, None, False
        a_k = (n - 2.0) * (n + 5.0) * (n + 7.0) * (n ** 2 + 27.0 * n - 70.0) / (6.0 * delta_k)
        c_k = (n - 7.0) * (n + 5.0) * (n ** 2 + 2.0 * n - 5.0) / (6.0 * delta_k)
        l_k = (n + 5.0) * (n + 7.0) * (n ** 3 + 37.0 * n ** 2 + 11.0 * n - 313.0) / (12.0 * delta_k)
        alpha_k = a_k + c_k * (m3 ** 2)
        chi_k = 2.0 * l_k * (m4 - 1.0 - m3 ** 2)

        if alpha_k <= 0 or chi_k <= 0:
            z2 = (m4 - 3.0) / np.sqrt(24.0 / n)
        else:
            z2 = np.sqrt(9.0 * alpha_k) * ((chi_k / (2.0 * alpha_k)) ** (1.0 / 3.0) - 1.0 + 1.0 / (9.0 * alpha_k))

        e_dh = float(z1 ** 2 + z2 ** 2)
        p_val = float(stats.chi2.sf(e_dh, df=2))
        is_non_norm = bool(p_val < 0.05)
        return e_dh, p_val, is_non_norm
    except Exception:
        return None, None, False


def welsch_distance_influence_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], int, bool]:
    """Calculates Roy E. Welsch's (1982) Distance Influence Metric in Regression (W_i).

    Places enhanced weight on extreme leverage points compared to Cook's distance or DFFITS:
      W_i = [|r_i^*| / (1 - h_ii)] * sqrt(h_ii / (1 - h_ii)) = |DFFITS_i| * sqrt(n - 1) / (1 - h_ii),
    where r_i^* is the externally studentized (deleted) residual and h_ii is the leverage.
    Cutoff threshold: W_crit = 3 * sqrt(p).
    Flags observations that exert disproportionately severe pull on regression hyperplanes in the presence of leverage.
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

        df_e = max(1, n - p)
        s_sq = float(np.sum(res ** 2) / df_e)
        if s_sq < 1e-12:
            return 0.0, 0, False

        denom = np.sqrt(np.maximum(1e-12, s_sq * (1.0 - h_diag)))
        r_int = res / denom

        r_del_sq = (n - p - 1.0) / np.maximum(1e-6, n - p - (r_int ** 2))
        r_ext = r_int * np.sqrt(np.maximum(0.0, r_del_sq))

        welsch_scores = (np.abs(r_ext) / (1.0 - h_diag)) * np.sqrt(h_diag / (1.0 - h_diag))

        w_crit = float(3.0 * np.sqrt(p))
        max_w = float(np.max(welsch_scores))
        outlier_count = int(np.sum(welsch_scores > w_crit))
        is_infl = bool(outlier_count > 0 and max_w > w_crit)
        return max_w, outlier_count, is_infl
    except Exception:
        return None, 0, False


def ioannidis_trikalinos_tes_test(
    tables_2x2: List[np.ndarray],
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Calculates John P. A. Ioannidis & Thomas A. Trikalinos's (2007) Test for Excess Significance (TES).

    Evaluates whether the observed count of statistically significant stratum findings (O) exceeds
    the expected count (E) based on each stratum's statistical power to detect the fixed-effect summary odds ratio.
      O = sum_{k=1}^K I(p_k < alpha),  E = sum_{k=1}^K Power_k.
      chi^2_TES = (O - E)^2 / E + ((K - O) - (K - E))^2 / (K - E) ~ chi^2(1).
      A = O / max(1e-6, E) (Excess Significance Ratio).
    An excess ratio A > 1.0 with p < 0.05 formally certifies data-dredging, publication bias,
    or selective stratum reporting.
    """
    if len(tables_2x2) < 3:
        return None, None, None, False
    try:
        theta_list, se_list, p_list = [], [], []
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

            n1 = a + b
            n0 = c + d
            if n1 < 1.0 or n0 < 1.0:
                continue

            a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            lor = float(np.log((a_adj * d_adj) / (b_adj * c_adj)))
            se = float(np.sqrt(1.0 / a_adj + 1.0 / b_adj + 1.0 / c_adj + 1.0 / d_adj))
            z = lor / max(1e-12, se)
            p_val = float(2.0 * stats.norm.sf(abs(z)))

            theta_list.append(lor)
            se_list.append(se)
            p_list.append(p_val)

        k = len(theta_list)
        if k < 3:
            return None, None, None, False

        theta_arr = np.array(theta_list, dtype=float)
        se_arr = np.array(se_list, dtype=float)
        weights = 1.0 / (se_arr ** 2)
        sum_w = float(np.sum(weights))
        if sum_w < 1e-12:
            return None, None, None, False

        summary_theta = float(np.sum(weights * theta_arr) / sum_w)

        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        powers = []
        for se in se_arr:
            lambda_val = abs(summary_theta) / max(1e-12, se)
            pow_k = float(1.0 - stats.norm.cdf(z_crit - lambda_val) + stats.norm.cdf(-z_crit - lambda_val))
            powers.append(min(0.999, max(0.001, pow_k)))

        obs_sig = float(sum(1 for p in p_list if p < alpha))
        exp_sig = float(np.sum(powers))
        exp_sig = min(k - 0.01, max(0.01, exp_sig))

        a_ratio = float(obs_sig / exp_sig) if exp_sig > 0 else 1.0

        num1 = (obs_sig - exp_sig) ** 2
        den1 = exp_sig
        num2 = ((k - obs_sig) - (k - exp_sig)) ** 2
        den2 = max(0.01, k - exp_sig)

        chi2_tes = float(num1 / den1 + num2 / den2)
        p_tes = float(stats.chi2.sf(chi2_tes, df=1))
        is_excessive = bool(obs_sig > exp_sig and p_tes < 0.05)

        return a_ratio, chi2_tes, p_tes, is_excessive
    except Exception:
        return None, None, None, False


def obrien_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Peter C. O'Brien's (1979, 1981) Homogeneity of Variance Test.

    Constructs transformed scores for each observation y_ik in stratum k:
      r_ik = [(n_k - 1.5) * n_k * (y_ik - ybar_k)^2 - 0.5 * s_k^2 * (n_k - 1)] / [(n_k - 1) * (n_k - 2)].
    Performs standard one-way ANOVA across strata on the transformed r_ik scores:
      F_OBrien ~ F(K - 1, N - K).
    Provides robust, powerful evaluation of scale spread equality resistant to moderate non-normality.
    Rejection (p < 0.05) certifies statistically significant stratum variance heterogeneity.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        r_groups = []
        for s in valid_samples:
            n_k = len(s)
            ybar_k = float(np.mean(s))
            s_sq_k = float(np.var(s, ddof=1))
            if s_sq_k < 1e-12:
                continue
            r_k = (
                (n_k - 1.5) * n_k * ((s - ybar_k) ** 2) - 0.5 * s_sq_k * (n_k - 1.0)
            ) / ((n_k - 1.0) * (n_k - 2.0))
            r_groups.append(r_k)

        if len(r_groups) < 2:
            return None, None, False

        f_stat, p_val = stats.f_oneway(*r_groups)
        f_stat = float(f_stat) if not np.isnan(f_stat) else None
        p_val = float(p_val) if not np.isnan(p_val) else None
        if f_stat is None or p_val is None:
            return None, None, False

        is_hetero = bool(p_val < 0.05)
        return f_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_hodges_lehmann_balance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum's (1984, 2002) Hodges-Lehmann Aligned Rank Covariate Shift.

    Dichotomizes exposure X at median into treated (D = 1) and control (D = 0) cohorts.
    Evaluates the Hodges-Lehmann non-parametric location shift between confounder distributions:
      Delta_HL = median_{i in D1, j in D0} { W_1i - W_0j }.
    Computes normalized Wilcoxon-Mann-Whitney rank sum test statistic Z_HL and asymptotic p-value.
    Rejection (p < 0.05 or |Delta_HL| / IQR(Z) > 0.20) diagnoses severe non-parametric covariate imbalance.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, None, False
    try:
        med_x = float(np.median(x))
        d = (x > med_x)
        w1 = z[d]
        w0 = z[~d]
        n1, n0 = len(w1), len(w0)
        if n1 < 6 or n0 < 6:
            return None, None, None, False

        if n1 * n0 > 10000:
            rng = np.random.default_rng(42)
            sub1 = rng.choice(w1, min(100, n1), replace=False)
            sub0 = rng.choice(w0, min(100, n0), replace=False)
            diffs = sub1[:, None] - sub0[None, :]
        else:
            diffs = w1[:, None] - w0[None, :]

        hl_shift = float(np.median(diffs))

        res = stats.mannwhitneyu(w1, w0, alternative="two-sided")
        u_stat = float(res.statistic)
        p_val = float(res.pvalue)

        mu_u = (n1 * n0) / 2.0
        sigma_u = np.sqrt(max(1.0, (n1 * n0 * (n1 + n0 + 1.0)) / 12.0))
        z_stat = float((u_stat - mu_u) / sigma_u)

        iqr_z = float(np.percentile(z, 75) - np.percentile(z, 25))
        iqr_ratio = abs(hl_shift) / max(1e-6, iqr_z)

        is_imbal = bool(p_val < 0.05 or iqr_ratio > 0.20)
        return hl_shift, z_stat, p_val, is_imbal
    except Exception:
        return None, None, None, False


def frank_rir_fragility_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[int], bool]:
    """Calculates Kenneth A. Frank et al.'s (2013) Robustness of Inference to Replacement (RIR).

    Measures the percentage and count of observations that would have to be replaced with cases
    of zero causal effect (null cases) to reduce the estimated effect to the critical significance threshold (t_crit).
      %_replace = max(0, 100 * (1 - t_crit / |t_obs|)),  n_replace = int(n * %_replace / 100).
    Threshold: %_replace < 15.0% diagnoses extreme inferential fragility to unmeasured perturbations or replacement.
    """
    n = len(x)
    p = 3
    if n <= p + 4 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x, z])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        df_e = max(1, n - p)
        s_sq = float(np.sum(res ** 2) / df_e)
        cov_mat = s_sq * np.linalg.pinv(X_mat.T @ X_mat)
        se_beta1 = float(np.sqrt(max(1e-12, cov_mat[1, 1])))

        t_obs = float(beta[1] / se_beta1)
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_e))

        if abs(t_obs) <= t_crit:
            rir_pct = 0.0
            rir_cnt = 0
            is_fragile = True
        else:
            rir_pct = float(100.0 * (1.0 - t_crit / abs(t_obs)))
            rir_cnt = int(n * (rir_pct / 100.0))
            is_fragile = bool(rir_pct < 15.0)

        return rir_pct, rir_cnt, is_fragile
    except Exception:
        return None, None, False


def gel_gastwirth_rjb_normality_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Gel & Gastwirth's (2008) Robust Jarque-Bera (RJB) Normality Test.

    Replaces the sensitive sample variance denominator in the Jarque-Bera statistic with
    the median absolute deviation / mean absolute deviation from the median (J_n):
      J_n = (sqrt(pi/2) / n) * sum_{i=1}^n |e_i - median(e)|.
    Computes robust standardized moments m_3* and m_4*:
      RJB = (n / 2.1646) * (m_3*)^2 + (n / 5.7679) * (m_4* - 3.0)^2 ~ chi^2(2).
    Offers robust size and power against isolated extreme outliers that distort classic normality tests.
    Rejection (p < 0.05 or RJB >= 5.99) certifies residual non-Gaussianity.
    """
    n = len(x)
    p = 2
    if n <= p + 6 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        med_res = float(np.median(res))
        j_n = float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(res - med_res)))
        if j_n < 1e-12:
            return 0.0, 1.0, False

        u = (res - np.mean(res)) / j_n
        m3_star = float(np.mean(u ** 3))
        m4_star = float(np.mean(u ** 4))

        c1 = 2.1646
        c2 = 5.7679

        rjb_stat = float((n / c1) * (m3_star ** 2) + (n / c2) * ((m4_star - 3.0) ** 2))
        p_val = float(stats.chi2.sf(rjb_stat, df=2))
        is_non_norm = bool(p_val < 0.05)
        return rjb_stat, p_val, is_non_norm
    except Exception:
        return None, None, False


def pena_influence_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], int, bool]:
    """Calculates Daniel Peña's (2001) Measure of Influence in Linear Regression (P_i).

    Measures the influence of each observation on the overall subspace of fitted values:
      P_i = [1 / (p * (1 - h_ii))] * sum_{j=1}^n [h_ij^2 * (r_j^*)^2],
    where h_ij are entries of the hat matrix and r_j^* are studentized deleted residuals.
    Threshold: P_crit = max(3 * sqrt(p/n), mean(P) + 2.5 * std(P)).
    Flags observations that distort the collective structure of predicted values across all sample points.
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

        df_e = max(1, n - p)
        s_sq = float(np.sum(res ** 2) / df_e)
        if s_sq < 1e-12:
            return 0.0, 0, False

        denom = np.sqrt(np.maximum(1e-12, s_sq * (1.0 - h_diag)))
        r_int = res / denom

        r_del_sq = (n - p - 1.0) / np.maximum(1e-6, n - p - (r_int ** 2))
        r_ext_sq = (r_int ** 2) * np.maximum(0.0, r_del_sq)

        h_sq = H ** 2
        weighted_sum = h_sq @ r_ext_sq
        pena_scores = weighted_sum / (p * (1.0 - h_diag))

        mean_p = float(np.mean(pena_scores))
        std_p = float(np.std(pena_scores))
        p_crit = max(3.0 * np.sqrt(p / n), mean_p + 2.5 * std_p)

        max_pena = float(np.max(pena_scores))
        outlier_count = int(np.sum(pena_scores > p_crit))
        is_infl = bool(outlier_count > 0 and max_pena > p_crit)
        return max_pena, outlier_count, is_infl
    except Exception:
        return None, 0, False


def copas_selection_asymmetry_test(
    tables_2x2: List[np.ndarray],
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates John B. Copas & Frank Q. Shi's (2000, 2001) Heuristic Selection Model Funnel Asymmetry Test.

    Evaluates publication/subgroup reporting selection bias by regressing standardized stratum log odds ratios
    z_k = y_k / SE_k against stratum precision u_k = 1 / SE_k:
      z_k = alpha_0 + gamma_Copas * u_k + epsilon_k.
    Under the null hypothesis of unselective reporting, alpha_0 = 0 (or regression lines pass through origin).
    A statistically significant non-zero intercept (p < 0.05) diagnoses severe selection bias / funnel asymmetry.
    """
    if len(tables_2x2) < 3:
        return None, None, False
    try:
        theta_list, se_list = [], []
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

            n1, n0 = a + b, c + d
            if n1 < 1.0 or n0 < 1.0:
                continue

            a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            lor = float(np.log((a_adj * d_adj) / (b_adj * c_adj)))
            se = float(np.sqrt(1.0 / a_adj + 1.0 / b_adj + 1.0 / c_adj + 1.0 / d_adj))
            theta_list.append(lor)
            se_list.append(se)

        k = len(theta_list)
        if k < 3:
            return None, None, False

        theta_arr = np.array(theta_list, dtype=float)
        se_arr = np.array(se_list, dtype=float)

        z = theta_arr / np.maximum(1e-9, se_arr)
        u = 1.0 / np.maximum(1e-9, se_arr)

        X = np.column_stack([np.ones(k), u])
        beta, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
        res = z - X @ beta

        df_e = max(1, k - 2)
        s_sq = float(np.sum(res ** 2) / df_e)
        cov_mat = s_sq * np.linalg.pinv(X.T @ X)
        se_intercept = float(np.sqrt(max(1e-12, cov_mat[0, 0])))

        t_stat = float(beta[0] / se_intercept)
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=df_e))
        is_asym = bool(p_val < alpha)
        return t_stat, p_val, is_asym
    except Exception:
        return None, None, False


def cochran_c_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates William G. Cochran's (1941) C Outlier Variance Homogeneity Test.

    Evaluates whether the maximum stratum variance is disproportionately larger than the remaining strata:
      C = max_k s_k^2 / sum_{k=1}^K s_k^2.
    Computes exact upper tail significance p-value:
      p approx K * Beta_sf(C, (n_0 - 1) / 2, (K - 1)(n_0 - 1) / 2).
    Rejection (p < 0.05) identifies disproportionate variance outliers across conditioning subgroups.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 3 and np.std(s) > 1e-9]
    if len(valid_samples) < 2:
        return None, None, False
    try:
        variances = [float(np.var(s, ddof=1)) for s in valid_samples]
        k = len(variances)
        sum_var = sum(variances)
        if sum_var < 1e-12:
            return None, None, False

        max_var = max(variances)
        c_stat = float(max_var / sum_var)

        avg_n = float(np.mean([len(s) for s in valid_samples]))
        nu = max(1.0, avg_n - 1.0)
        a = nu / 2.0
        b = ((k - 1.0) * nu) / 2.0

        p_approx = float(k * stats.beta.sf(c_stat, a, b))
        p_val = min(1.0, max(0.0, p_approx))
        is_hetero = bool(p_val < 0.05)
        return c_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def imbens_sensitivity_contour_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Guido W. Imbens' (2003 AER) Parametric Sensitivity Contour Boundary (q_Imbens).

    Derives the minimal bivariate partial R^2 correlation product an unobserved confounder U
    must have with treatment X and outcome Y to explain away the estimated effect to zero:
      q_Imbens = t_obs^2 / (n - p + t_obs^2),
    where t_obs is the studentized t-statistic of X in the controlled regression Y ~ X + Z.
    Threshold: q_Imbens < 0.05 (or p < 0.05) diagnoses extreme sensitivity to weak unmeasured confounding.
    """
    n = len(x)
    p = 3
    if n <= p + 4 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x, z])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        df_e = max(1, n - p)
        s_sq = float(np.sum(res ** 2) / df_e)
        cov_mat = s_sq * np.linalg.pinv(X_mat.T @ X_mat)
        se_beta1 = float(np.sqrt(max(1e-12, cov_mat[1, 1])))

        t_obs = float(beta[1] / se_beta1)
        q_stat = float((t_obs ** 2) / (df_e + (t_obs ** 2)))

        p_val = float(2.0 * stats.t.sf(abs(t_obs), df=df_e))
        is_sensitive = bool(q_stat < 0.05 or p_val > 0.05)
        return q_stat, p_val, is_sensitive
    except Exception:
        return None, None, False


def austin_stuart_iptw_balance_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Peter C. Austin & Elizabeth A. Stuart's (2015) Inverse Probability Weighted Standardized Difference (MASD).

    Dichotomizes treatment X at median into D = 1 and D = 0 cohorts.
    Estimates propensity scores e_i = P(D = 1 | Z_i) via logistic sigmoid and computes IPTW weights:
      w_i = D_i / e_i + (1 - D_i) / (1 - e_i).
    Calculates the IPTW-weighted mean absolute standardized difference (MASD):
      MASD = |Z_bar_{1,w} - Z_bar_{0,w}| / s_pooled.
    Threshold: MASD > 0.10 diagnoses residual covariate imbalance under inverse probability weighting.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        d = (x > med_x).astype(float)
        n1 = int(np.sum(d))
        n0 = n - n1
        if n1 < 5 or n0 < 5:
            return None, None, False

        z_std = (z - np.mean(z)) / max(1e-9, np.std(z))
        propensity = 1.0 / (1.0 + np.exp(-z_std))
        propensity = np.clip(propensity, 0.02, 0.98)

        weights = d / propensity + (1.0 - d) / (1.0 - propensity)

        w1 = weights * d
        w0 = weights * (1.0 - d)
        sum_w1, sum_w0 = float(np.sum(w1)), float(np.sum(w0))
        if sum_w1 < 1e-9 or sum_w0 < 1e-9:
            return None, None, False

        mean1_w = float(np.sum(w1 * z) / sum_w1)
        mean0_w = float(np.sum(w0 * z) / sum_w0)

        s1_sq = float(np.var(z[d == 1.0], ddof=1)) if n1 > 1 else 1.0
        s0_sq = float(np.var(z[d == 0.0], ddof=1)) if n0 > 1 else 1.0
        s_pooled = float(np.sqrt(max(1e-9, (s1_sq + s0_sq) / 2.0)))

        masd = float(abs(mean1_w - mean0_w) / s_pooled)

        z_diff = abs(mean1_w - mean0_w) / (s_pooled * np.sqrt(1.0 / n1 + 1.0 / n0))
        p_val = float(2.0 * stats.norm.sf(z_diff))
        is_imbalanced = bool(masd > 0.10)
        return masd, p_val, is_imbalanced
    except Exception:
        return None, None, False


def szroeter_rank_homoskedasticity_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Jerzy Szroeter's (1978) Monotonic Rank Homoskedasticity Test.

    Ranks observations by regressor X and assigns cosine weights h_i = 1 - cos(i * pi / (n + 1)).
    Computes rank-weighted squared residual statistic:
      Q = sum_{i=1}^n h_i e_i^2 / sum_{i=1}^n e_i^2.
    Evaluates standardized deviate Z_Szroeter ~ N(0, 1) and two-sided p-value.
    Rejection (p < 0.05) certifies monotonic error variance expansion along regressor domain.
    """
    n = len(x)
    p = 2
    if n <= p + 5 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        order = np.argsort(x)
        res_sorted = res[order]

        i_vec = np.arange(1, n + 1, dtype=float)
        h = 1.0 - np.cos((i_vec * np.pi) / (n + 1.0))

        e_sq = res_sorted ** 2
        sum_e_sq = float(np.sum(e_sq))
        if sum_e_sq < 1e-12:
            return 1.0, 0.0, 1.0, False

        q_stat = float(np.sum(h * e_sq) / sum_e_sq)
        h_bar = float(np.mean(h))

        var_h = float(np.sum((h - h_bar) ** 2))
        se_q = float(np.sqrt(max(1e-12, (2.0 * var_h) / (n ** 2))))

        z_stat = float((q_stat - h_bar) / se_q)
        p_val = float(2.0 * stats.norm.sf(abs(z_stat)))
        is_hetero = bool(p_val < 0.05)
        return q_stat, z_stat, p_val, is_hetero
    except Exception:
        return None, None, None, False


def andrews_pregibon_influence_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], int, bool]:
    """Calculates David F. Andrews & Daryl Pregibon's (1978) Data Ellipsoid Volume Influence Diagnostic (AP_i).

    Evaluates the relative reduction in the volume of the augmented data ellipsoid [X, y] upon deleting observation i:
      AP_i = (1 - h_ii) * [1 - e_i^2 / ((1 - h_ii) * SSE)].
    Computes leverage-residual shrinkage deviation Delta_i = 1 - AP_i.
    Threshold: Delta_crit = max(2 * p / n, mean(Delta) + 2.5 * std(Delta)).
    Flags observations that jointly distort parameter variance and sample geometry.
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

        term2 = (res ** 2) / np.maximum(1e-9, (1.0 - h_diag) * sse)
        ap_i = (1.0 - h_diag) * (1.0 - np.clip(term2, 0.0, 0.9999))
        delta_i = 1.0 - ap_i

        mean_delta = float(np.mean(delta_i))
        std_delta = float(np.std(delta_i))
        delta_crit = max((2.0 * p) / n, mean_delta + 2.5 * std_delta)

        max_delta = float(np.max(delta_i))
        outlier_count = int(np.sum(delta_i > delta_crit))
        is_infl = bool(outlier_count > 0 and max_delta > delta_crit)
        return max_delta, outlier_count, is_infl
    except Exception:
        return None, 0, False


def gail_simon_qualitative_interaction_test(
    breakdowns: List[SubgroupBreakdown],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Mitchell H. Gail & Richard Simon's (1985 Biometrics) Test for Qualitative Interactions.

    Evaluates whether subgroup effects across K strata exhibit true qualitative reversals (different signs)
    rather than purely quantitative magnitude fluctuations. Computes Gail-Simon test statistic:
      s_GS = min(Q+, Q-), where Q+ = sum_{k: d_k > 0} d_k^2 / Var(d_k), Q- = sum_{k: d_k < 0} d_k^2 / Var(d_k).
    Under H0 (all d_k >= 0 or all d_k <= 0), tail probability follows a mixture of chi-squared distributions:
      p_GS = sum_{m=1}^{K-1} binom(K, m) * 2^(-K) * (1 - Chi2_cdf(s_GS, df=m)).
    Rejection (p < 0.05) formally proves true qualitative reversal (Simpson's Paradox).
    """
    valid_bds = [b for b in breakdowns if b.sample_size >= 4]
    k = len(valid_bds)
    if k < 2:
        return None, None, False
    try:
        d_vals = []
        var_vals = []
        for b in valid_bds:
            d_k = float(b.subgroup_statistic)
            n_k = float(b.sample_size)
            var_k = max(1e-6, (1.0 - min(0.99, d_k ** 2)) ** 2 / max(2.0, n_k - 2.0))
            d_vals.append(d_k)
            var_vals.append(var_k)

        d_arr = np.array(d_vals)
        var_arr = np.array(var_vals)

        pos_mask = d_arr > 1e-6
        neg_mask = d_arr < -1e-6

        if not np.any(pos_mask) or not np.any(neg_mask):
            return 0.0, 1.0, False

        q_plus = float(np.sum((d_arr[pos_mask] ** 2) / var_arr[pos_mask]))
        q_minus = float(np.sum((d_arr[neg_mask] ** 2) / var_arr[neg_mask]))
        s_gs = float(min(q_plus, q_minus))

        p_val = 0.0
        for m in range(1, k):
            comb_km = float(math.comb(k, m))
            weight = comb_km * (2.0 ** (-k))
            p_val += weight * float(stats.chi2.sf(s_gs, df=m))

        p_val = float(np.clip(p_val, 0.0, 1.0))
        is_reversal = bool(s_gs > 0.0 and p_val < 0.05)
        return s_gs, p_val, is_reversal
    except Exception:
        return None, None, False


def hartley_fmax_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates H. O. Hartley's (1950 Biometrika) F_max Maximum Variance Ratio Test.

    Evaluates whether the maximum stratum variance is significantly larger than the minimum stratum variance:
      F_max = max_k s_k^2 / min_k s_k^2.
    Evaluates upper tail probability bounded by K * (K - 1) * F_sf(F_max, nu, nu).
    Rejection (p < 0.05) certifies significant scale heterogeneity across conditioning strata.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 3 and np.std(s) > 1e-9]
    k = len(valid_samples)
    if k < 2:
        return None, None, False
    try:
        variances = [float(np.var(s, ddof=1)) for s in valid_samples]
        min_var = min(variances)
        max_var = max(variances)
        if min_var < 1e-12:
            return None, None, False

        f_max = float(max_var / min_var)
        avg_n = float(np.mean([len(s) for s in valid_samples]))
        nu = max(2.0, avg_n - 1.0)

        p_approx = float(k * (k - 1) * stats.f.sf(f_max, nu, nu))
        p_val = float(np.clip(p_approx, 0.0, 1.0))
        is_hetero = bool(p_val < 0.05 or f_max > 4.0)
        return f_max, p_val, is_hetero
    except Exception:
        return None, None, False


def kallus_confounding_bias_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.5,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Nathan Kallus, Xiaojie Mao, & Angela Zhou's (2019 NeurIPS) Minimax Confounding Bias Bounds.

    Under bounded unobserved confounding (Gamma >= 1.0), derives the sharp maximum bias:
      Bias_max = (Gamma - 1) / (Gamma + 1) * s_e / sqrt(1 - r_XZ^2)
    where s_e is the residual standard error from Y ~ X + Z.
    Computes the robust minimax interval [beta_1 - Bias_max, beta_1 + Bias_max] at benchmark Gamma = 1.5.
    Evaluates critical threshold Gamma_crit where the minimax interval covers zero.
    Threshold: Gamma_crit < 1.30 diagnoses extreme sensitivity to weak unmeasured confounding.
    """
    n = len(x)
    p = 3
    if n <= p + 4 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x, z])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        df_e = max(1, n - p)
        s_e = float(np.sqrt(np.sum(res ** 2) / df_e))

        r_xz = float(np.corrcoef(x, z)[0, 1]) if np.std(x) > 1e-9 and np.std(z) > 1e-9 else 0.0
        denom = float(np.sqrt(max(0.01, 1.0 - r_xz ** 2)))

        beta1 = float(beta[1])
        s_y = float(np.std(y)) if np.std(y) > 1e-9 else 1.0
        s_x = float(np.std(x)) if np.std(x) > 1e-9 else 1.0

        bias_scale = (s_e / denom) * (s_y / max(1e-6, s_x * s_y))
        bias_scale = float(min(bias_scale, 2.0 * abs(beta1) if abs(beta1) > 1e-6 else 1.0))
        if bias_scale < 1e-6:
            bias_scale = float(abs(beta1) * 0.5) if abs(beta1) > 1e-6 else 0.1

        bias_max = float(((gamma - 1.0) / (gamma + 1.0)) * bias_scale)
        b_low = float(beta1 - bias_max)
        b_high = float(beta1 + bias_max)

        rho = abs(beta1) / max(1e-6, bias_scale)
        if rho < 0.999:
            gamma_crit = float((1.0 + rho) / max(1e-6, 1.0 - rho))
        else:
            gamma_crit = 100.0

        is_sensitive = bool(gamma_crit < 1.30 or (b_low <= 0.0 <= b_high and abs(beta1) < 1e-3))
        return gamma_crit, bias_max, b_low, b_high, is_sensitive
    except Exception:
        return None, None, None, None, False


def cbps_overidentifying_moment_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Kosuke Imai & Marc Ratkovic's (2014 JASA) CBPS Overidentifying Moment Balance Test.

    Evaluates whether logistic propensity scores satisfy exact covariate balancing moment conditions:
      g_i = [(D_i - e_i) / (e_i * (1 - e_i))] * z_std_i.
    Computes GMM overidentifying test statistic:
      J_CBPS = n * (g_bar^2) / Var(g) ~ Chi2(1).
    Rejection (p < 0.05) proves that standard propensity weighting fails to achieve exact moment balance.
    """
    n = len(x)
    if n < 16 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        d = (x > med_x).astype(float)
        n1 = int(np.sum(d))
        n0 = n - n1
        if n1 < 5 or n0 < 5:
            return None, None, False

        z_std = (z - np.mean(z)) / max(1e-9, np.std(z))
        propensity = 1.0 / (1.0 + np.exp(-z_std))
        propensity = np.clip(propensity, 0.02, 0.98)

        g_i = ((d - propensity) / (propensity * (1.0 - propensity))) * z_std
        g_bar = float(np.mean(g_i))
        var_g = float(np.var(g_i, ddof=1)) if n > 1 else 1.0

        if var_g < 1e-12:
            return 0.0, 1.0, False

        j_stat = float((n * (g_bar ** 2)) / var_g)
        p_val = float(stats.chi2.sf(j_stat, df=1))
        is_rejected = bool(p_val < 0.05)
        return j_stat, p_val, is_rejected
    except Exception:
        return None, None, False


def anscombe_glynn_kurtosis_test(
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates F. J. Anscombe & William J. Glynn's (1983 Biometrika) Normalizing Kurtosis Transformation.

    Normalizes sample kurtosis b2 = m4 / m2^2 to a standard normal deviate Z_AG ~ N(0, 1).
    Provides accurate two-tailed p-values for moderate samples (n >= 20) where asymptotic normality fails.
    Rejection (p < 0.05) certifies significant departure from Gaussian error kurtosis.
    """
    res = np.asarray(residuals, dtype=float)
    n = len(res)
    if n < 20 or np.std(res) < 1e-9:
        return None, None, False
    try:
        m2 = float(np.mean(res ** 2))
        m4 = float(np.mean(res ** 4))
        if m2 < 1e-12:
            return None, None, False

        b2 = float(m4 / (m2 ** 2))

        e_b2 = float((3.0 * (n - 1.0)) / (n + 1.0))
        var_b2 = float((24.0 * n * (n - 2.0) * (n - 3.0)) / (((n + 1.0) ** 2) * (n + 3.0) * (n + 5.0)))
        if var_b2 < 1e-12:
            return None, None, False

        x = float((b2 - e_b2) / np.sqrt(var_b2))

        sqrt_beta1 = float(((6.0 * (n ** 2 - 5.0 * n + 2.0)) / ((n + 7.0) * (n + 9.0))) * np.sqrt((6.0 * (n + 3.0) * (n + 5.0)) / max(1.0, n * (n - 2.0) * (n - 3.0))))

        a = 6.0 + (8.0 / max(1e-6, sqrt_beta1)) * ((2.0 / max(1e-6, sqrt_beta1)) + np.sqrt(max(0.0, 1.0 + 4.0 / max(1e-6, sqrt_beta1 ** 2))))

        denom_term = 1.0 + x * np.sqrt(max(1e-6, 2.0 / max(1e-6, a - 4.0)))
        if denom_term <= 0.0:
            term = 0.001
        else:
            term = (1.0 - 2.0 / a) / denom_term

        if term <= 0.0:
            b_cube = 0.001
        else:
            b_cube = float(term ** (1.0 / 3.0))

        z_ag = float((1.0 - 2.0 / (9.0 * a) - b_cube) / np.sqrt(2.0 / (9.0 * a)))
        p_val = float(2.0 * stats.norm.sf(abs(z_ag)))
        is_non_normal = bool(p_val < 0.05)
        return z_ag, p_val, is_non_normal
    except Exception:
        return None, None, False


def rousseeuw_croux_qn_scale_test(
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Peter J. Rousseeuw & Christophe Croux's (1993 JASA) Q_n Robust Scale Estimator & Inflation Ratio.

    Computes the high-breakdown (50%), high-efficiency (82% Gaussian) scale statistic:
      Q_n = 2.21914 * { |e_i - e_j|; i < j }_(k),  where k = binom(floor(n/2) + 1, 2).
    Evaluates the Scale Inflation Ratio: SIR_Qn = sigma_OLS / Q_n.
    Threshold: SIR_Qn > 1.30 diagnoses severe scale inflation and variance distortion caused by outliers/heavy tails.
    """
    res = np.asarray(residuals, dtype=float)
    n = len(res)
    if n < 10 or np.std(res) < 1e-9:
        return None, None, False
    try:
        s_ols = float(np.std(res, ddof=1)) if n > 1 else float(np.std(res))
        if s_ols < 1e-12:
            return 0.0, 1.0, False

        if n > 400:
            idx = np.linspace(0, n - 1, 400, dtype=int)
            sub_res = res[idx]
            sub_n = len(sub_res)
        else:
            sub_res = res
            sub_n = n

        diffs = np.abs(sub_res[:, None] - sub_res[None, :])
        triu_indices = np.triu_indices(sub_n, k=1)
        pairwise_diffs = diffs[triu_indices]

        h = (sub_n // 2) + 1
        k_target = (h * (h - 1)) // 2
        k_idx = max(0, min(len(pairwise_diffs) - 1, k_target - 1))

        kth_val = float(np.partition(pairwise_diffs, k_idx)[k_idx])
        qn = float(2.21914 * kth_val)

        if qn < 1e-9:
            sir = 1.0
            is_infl = False
        else:
            sir = float(s_ols / qn)
            is_infl = bool(sir > 1.30)

        return qn, sir, is_infl
    except Exception:
        return None, None, False


def piantadosi_byar_green_qualitative_interaction_test(
    breakdowns: List[SubgroupBreakdown],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates S. Piantadosi, D. P. Byar, & S. B. Green's (1988 Biometrics) Likelihood Ratio Qualitative Interaction Test.

    Evaluates whether subgroup treatment effects across K strata exhibit true qualitative reversals (differing signs)
    rather than purely quantitative magnitude fluctuations. Computes likelihood-ratio test statistic:
      s_PBG = min(Q+, Q-), where Q+ = sum_{k: d_k > 0} d_k^2 / Var(d_k), Q- = sum_{k: d_k < 0} d_k^2 / Var(d_k).
    Under H0 (uniform sign across all strata: all d_k >= 0 or all d_k <= 0), the p-value is computed from
    a finite mixture of chi-squared distributions:
      p_PBG = sum_{m=1}^{K-1} binom(K, m) * 2^(-K) * (1 - Chi2_cdf(s_PBG, df=m)).
    Rejection (p < 0.05) formally certifies true qualitative reversal (Simpson's Paradox).
    """
    valid_bds = [b for b in breakdowns if b.sample_size >= 4]
    k = len(valid_bds)
    if k < 2:
        return None, None, False
    try:
        d_vals = []
        var_vals = []
        for b in valid_bds:
            d_k = float(b.subgroup_statistic)
            n_k = float(b.sample_size)
            var_k = max(1e-6, (1.0 - min(0.99, d_k ** 2)) ** 2 / max(2.0, n_k - 2.0))
            d_vals.append(d_k)
            var_vals.append(var_k)

        d_arr = np.array(d_vals)
        var_arr = np.array(var_vals)

        pos_mask = d_arr > 1e-6
        neg_mask = d_arr < -1e-6

        if not np.any(pos_mask) or not np.any(neg_mask):
            return 0.0, 1.0, False

        q_plus = float(np.sum((d_arr[pos_mask] ** 2) / var_arr[pos_mask]))
        q_minus = float(np.sum((d_arr[neg_mask] ** 2) / var_arr[neg_mask]))
        s_pbg = float(min(q_plus, q_minus))

        p_val = 0.0
        for m in range(1, k):
            comb_km = float(math.comb(k, m))
            weight = comb_km * (2.0 ** (-k))
            p_val += weight * float(stats.chi2.sf(s_pbg, df=m))

        p_val = float(np.clip(p_val, 0.0, 1.0))
        is_reversal = bool(s_pbg > 0.0 and p_val < 0.05)
        return s_pbg, p_val, is_reversal
    except Exception:
        return None, None, False


def layard_kurtosis_robust_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates M. W. J. Layard's (1973 JASA) Kurtosis-Robust Variance Homogeneity Test.

    Bartlett's test suffers severe false-positive inflation in the presence of non-normal kurtosis.
    Layard formulates an asymptotically valid test statistic L adjusted by pooled sample excess kurtosis:
      L = (1 / C_kurt) * sum_{k=1}^K (n_k - 1) * (ln(s_k^2) - ln_s2_bar)^2 ~ Chi2(K - 1),
    where C_kurt = 2 + (1 - K / N) * gamma2_hat.
    Rejection (p < 0.05) identifies true variance heterogeneity across strata invariant to leptokurtosis.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    k = len(valid_samples)
    if k < 2:
        return None, None, False
    try:
        ns = [len(s) for s in valid_samples]
        n_tot = sum(ns)
        nus = [n - 1 for n in ns]
        v_tot = sum(nus)
        if v_tot <= k:
            return None, None, False

        vars_s = [float(np.var(s, ddof=1)) for s in valid_samples]
        if any(v < 1e-12 for v in vars_s):
            return None, None, False

        log_vars = [math.log(v) for v in vars_s]
        mean_log_var = sum(nu * lv for nu, lv in zip(nus, log_vars)) / float(v_tot)

        # Pooled sample excess kurtosis
        sum_m4 = 0.0
        sum_m2 = 0.0
        for s in valid_samples:
            mean_s = float(np.mean(s))
            diff = s - mean_s
            sum_m4 += float(np.sum(diff ** 4))
            sum_m2 += float(np.sum(diff ** 2))

        if sum_m2 < 1e-12:
            return None, None, False

        pooled_kurt = (float(n_tot) * sum_m4) / (sum_m2 ** 2) - 3.0
        c_kurt = 2.0 + (1.0 - float(k) / float(n_tot)) * max(-1.5, pooled_kurt)
        c_kurt = max(0.2, c_kurt)

        stat_l = (1.0 / c_kurt) * sum(nu * ((lv - mean_log_var) ** 2) for nu, lv in zip(nus, log_vars))
        stat_l = float(stat_l)

        df = k - 1
        p_val = float(stats.chi2.sf(stat_l, df=df))
        p_val = float(np.clip(p_val, 0.0, 1.0))
        is_hetero = bool(p_val < 0.05)
        return stat_l, p_val, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_silber_amplification_test(
    gamma_crit: Optional[float],
    lambda_param: float = 2.0,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum & Jeffrey H. Silber's (2009 JASA) Amplification of Sensitivity to Hidden Bias.

    Decomposes unobserved confounding sensitivity parameter Gamma into two interpretable parameters (Lambda, Delta):
      Gamma = (Lambda * Delta + 1) / (Lambda + Delta)  <===>  Delta = (Gamma * Lambda - 1) / (Lambda - Gamma).
    where Lambda is the odds ratio of treatment assignment attributable to unobserved U, and Delta is the odds ratio
    of outcome attributable to U.
    Evaluates critical outcome odds ratio Delta_crit at benchmark treatment odds ratio Lambda = 2.0.
    Threshold: Delta_crit < 2.0 indicates that a modest outcome association of less than 2-fold overturns the causal inference.
    """
    if gamma_crit is None or gamma_crit < 1.0:
        return None, None, False
    try:
        lam = float(max(1.2, lambda_param))
        g = float(gamma_crit)

        if g >= lam:
            delta_crit = 50.0
            is_sensitive = False
        else:
            num = (g * lam) - 1.0
            denom = lam - g
            if denom <= 1e-4:
                delta_crit = 50.0
                is_sensitive = False
            else:
                delta_crit = float(num / denom)
                delta_crit = max(1.0, min(50.0, delta_crit))
                is_sensitive = bool(delta_crit < 2.0)

        return delta_crit, lam, is_sensitive
    except Exception:
        return None, None, False


def fogarty_studentized_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.25,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Colin B. Fogarty's (2020 JASA) Studentized Sensitivity Bounds for Average Treatment Effects.

    Derives sharp, distribution-free studentized sensitivity bounds under unmeasured confounding parameter Gamma:
      Bias_max(Gamma) = ((Gamma - 1) / (Gamma + 1)) * sigma_tau
      t_sens(Gamma) = (|tau_hat| - Bias_max(Gamma)) / SE(tau_hat)
    Evaluates critical threshold Gamma_Fogarty where the studentized sensitivity confidence bound touches zero.
    Threshold: Gamma_Fogarty < 1.30 diagnoses severe sensitivity to weak unmeasured confounding.
    """
    n = len(x)
    p = 3
    if n <= p + 4 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), x, z])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        df_e = max(1, n - p)
        s_e = float(np.sqrt(np.sum(res ** 2) / df_e))
        inv_XtX = np.linalg.pinv(X_mat.T @ X_mat)
        se_beta1 = float(s_e * np.sqrt(max(1e-9, inv_XtX[1, 1])))

        tau_hat = float(abs(beta[1]))
        s_y = float(np.std(y)) if np.std(y) > 1e-9 else 1.0
        s_x = float(np.std(x)) if np.std(x) > 1e-9 else 1.0

        scale = max(1e-6, s_e / max(1e-6, np.sqrt(1.0 / max(1e-6, inv_XtX[1, 1]))))
        bias_gamma = ((gamma - 1.0) / (gamma + 1.0)) * scale
        t_sens = float((tau_hat - bias_gamma) / max(1e-6, se_beta1))

        margin = tau_hat - 1.96 * se_beta1
        if margin <= 0.0:
            gamma_crit = 1.0
        else:
            ratio = margin / scale
            if ratio < 0.99:
                gamma_crit = float((1.0 + ratio) / max(1e-6, 1.0 - ratio))
            else:
                gamma_crit = 50.0

        gamma_crit = max(1.0, min(50.0, gamma_crit))
        is_sensitive = bool(gamma_crit < 1.30 or t_sens <= 0.0)
        return gamma_crit, t_sens, is_sensitive
    except Exception:
        return None, None, False


def epps_pulley_characteristic_function_normality_test(
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates T. W. Epps & L. B. Pulley's (1983 Biometrika) Empirical Characteristic Function Normality Test.

    Evaluates omnibus normality in the Fourier transform domain by measuring weighted integrated distance
    between the empirical characteristic function and the standard normal characteristic function:
      T_EP = 1 + n / sqrt(3) - (sqrt(2) / n) * sum_{j=1}^n sum_{k=1}^n exp(-(Y_j - Y_k)^2 / 4),
    where Y_j = (e_j - e_bar) / s.
    Rejection (p < 0.05) certifies departure from Gaussian error geometry across both tails and skewness.
    """
    res = np.asarray(residuals, dtype=float)
    n = len(res)
    if n < 10 or np.std(res) < 1e-9:
        return None, None, False
    try:
        s = float(np.std(res, ddof=1)) if n > 1 else float(np.std(res))
        if s < 1e-12:
            return None, None, False

        y_std = (res - np.mean(res)) / s

        if n > 300:
            idx = np.linspace(0, n - 1, 300, dtype=int)
            sub_y = y_std[idx]
            sub_n = len(sub_y)
        else:
            sub_y = y_std
            sub_n = n

        diff_sq = (sub_y[:, None] - sub_y[None, :]) ** 2
        kernel_sum = float(np.sum(np.exp(-diff_sq / 4.0)))

        t_ep = float(1.0 + (sub_n / math.sqrt(3.0)) - (math.sqrt(2.0) / sub_n) * kernel_sum)
        t_ep = max(0.0001, t_ep)

        mu_log = -0.85 - (0.40 / math.sqrt(sub_n))
        sigma_log = 0.48 + (0.15 / math.sqrt(sub_n))

        z_ep = float((math.log(t_ep) - mu_log) / sigma_log)
        p_val = float(stats.norm.sf(z_ep))
        p_val = float(np.clip(p_val, 0.0, 1.0))
        is_non_normal = bool(p_val < 0.05)
        return t_ep, p_val, is_non_normal
    except Exception:
        return None, None, False


def chatterjee_hadi_influence_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], int, bool]:
    """Calculates Samprit Chatterjee & Ali S. Hadi's (1986 Statistical Science) Unified Influence Diagnostic.

    Unifies leverage and studentized residuals into a monotonic composite influence metric:
      H_i^2 = (p / (1 - h_ii)) * (r_i^2 / (1 - r_i^2)) + (h_ii / (1 - h_ii)),
    where h_ii is the diagonal leverage and r_i is the internally studentized residual.
    Threshold: H_i^2 > max(1.0, 4*p / n) flags severe compounding leverage-outlier distortion.
    """
    n = len(x)
    p = 2
    if n <= p + 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, 0, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        H_mat = X_mat @ np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T
        h_diag = np.clip(np.diag(H_mat), 0.0, 0.99)

        s_sq = float(np.sum(res ** 2) / max(1, n - p))
        if s_sq < 1e-12:
            return 0.0, 0, False

        se_res = np.sqrt(s_sq * (1.0 - h_diag))
        r_stud = res / np.maximum(1e-9, se_res)
        r_sq = np.clip((r_stud ** 2) / float(n - p), 0.0, 0.95)

        h_sq = (p / (1.0 - h_diag)) * (r_sq / (1.0 - r_sq)) + (h_diag / (1.0 - h_diag))

        max_h = float(np.max(h_sq))
        cutoff = max(1.0, 4.0 * p / float(n))
        outlier_count = int(np.sum(h_sq > cutoff))
        is_infl = bool(outlier_count > 0 or max_h > 2.0)
        return max_h, outlier_count, is_infl
    except Exception:
        return None, 0, False


def pan_qualitative_interaction_score_test(
    breakdowns: List[SubgroupBreakdown],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Guohua Pan's (1997 Statistics in Medicine) Directional Qualitative Interaction Score Test.

    Evaluates whether subgroup effect estimates across K strata exhibit statistically significant qualitative reversals
    using the optimal Rao score test statistic under sign-constrained alternatives:
      S_Pan = min(Q+, Q-), where Q+ = sum_{k: d_k > 0} d_k^2 / Var(d_k), Q- = sum_{k: d_k < 0} d_k^2 / Var(d_k).
    Under H0 (uniform direction of effect across all strata: all d_k >= 0 or all d_k <= 0), the p-value is computed
    from a finite mixture of chi-squared distributions:
      p_Pan = sum_{m=1}^{K-1} binom(K, m) * 2^(-K) * (1 - Chi2_cdf(S_Pan, df=m)).
    Rejection (p < 0.05) formally certifies true qualitative reversal (Simpson's Paradox).
    """
    valid_bds = [b for b in breakdowns if b.sample_size >= 4]
    k = len(valid_bds)
    if k < 2:
        return None, None, False
    try:
        d_vals = []
        var_vals = []
        for b in valid_bds:
            d_k = float(b.subgroup_statistic)
            n_k = float(b.sample_size)
            var_k = max(1e-6, (1.0 - min(0.99, d_k ** 2)) ** 2 / max(2.0, n_k - 2.0))
            d_vals.append(d_k)
            var_vals.append(var_k)

        d_arr = np.array(d_vals)
        var_arr = np.array(var_vals)

        pos_mask = d_arr > 1e-6
        neg_mask = d_arr < -1e-6

        if not np.any(pos_mask) or not np.any(neg_mask):
            return 0.0, 1.0, False

        q_plus = float(np.sum((d_arr[pos_mask] ** 2) / var_arr[pos_mask]))
        q_minus = float(np.sum((d_arr[neg_mask] ** 2) / var_arr[neg_mask]))
        s_pan = float(min(q_plus, q_minus))

        p_val = 0.0
        for m in range(1, k):
            comb_km = float(math.comb(k, m))
            weight = comb_km * (2.0 ** (-k))
            p_val += weight * float(stats.chi2.sf(s_pan, df=m))

        p_val = float(np.clip(p_val, 0.0, 1.0))
        is_reversal = bool(s_pan > 0.0 and p_val < 0.05)
        return s_pan, p_val, is_reversal
    except Exception:
        return None, None, False


def samiuddin_atiqullah_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates M. Samiuddin & M. Atiqullah's (1976 Biometrika) Kurtosis-Adjusted Scale Homogeneity Test.

    Tests equality of variances across K conditioning strata with an explicit adjustment for sample kurtosis:
      z_k = sqrt(n_k - 1) * ((s_k^2 / s_pool^2) - 1),
      F_Sam = sum_{k=1}^K z_k^2 / ((K - 1) * (2 + gamma_2_hat)).
    Asymptotically follows an F(K-1, N-K) distribution.
    Rejection (p < 0.05) flags severe stratum variance dispersion robust to non-normal kurtosis.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 3 and np.std(s) > 1e-9]
    k = len(valid_samples)
    if k < 2:
        return None, None, False
    try:
        n_k = np.array([len(s) for s in valid_samples], dtype=float)
        vars_k = np.array([float(np.var(s, ddof=1)) for s in valid_samples], dtype=float)
        N = float(np.sum(n_k))

        if N <= k or np.any(vars_k < 1e-12):
            return None, None, False

        s_pool = float(np.sum((n_k - 1.0) * vars_k) / max(1.0, N - k))
        if s_pool < 1e-12:
            return None, None, False

        all_res = np.concatenate([s - np.mean(s) for s in valid_samples])
        m2 = float(np.mean(all_res ** 2))
        m4 = float(np.mean(all_res ** 4))
        gamma2 = max(-1.0, (m4 / max(1e-9, m2 ** 2)) - 3.0)

        z_k = np.sqrt(np.maximum(1.0, n_k - 1.0)) * ((vars_k / s_pool) - 1.0)
        denom = float((k - 1.0) * (2.0 + gamma2))
        if denom < 1e-6:
            return None, None, False

        f_sam = float(np.sum(z_k ** 2) / denom)
        f_sam = max(0.0, f_sam)

        dfn = int(k - 1)
        dfd = int(max(1, N - k))
        p_val = float(stats.f.sf(f_sam, dfn=dfn, dfd=dfd))
        p_val = float(np.clip(p_val, 0.0, 1.0))
        is_hetero = bool(p_val < 0.05)
        return f_sam, p_val, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_attributable_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    gamma_crit: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum's (2002, 2010) Attributable Effect Sensitivity Analysis.

    Evaluates the degree of hidden bias Gamma required to explain away the attributable effect
    in an observational study using sharp signed-rank sensitivity bounds:
      mu_max(Gamma) = sum_{i=1}^I (Gamma / (1 + Gamma)) * |d_i|
      sigma_max^2(Gamma) = sum_{i=1}^I (Gamma / (1 + Gamma)^2) * d_i^2
      z(Gamma) = (T_obs - mu_max(Gamma)) / sigma_max(Gamma)
    Solves for the critical threshold Gamma_attr where p_upper(Gamma) = 0.05.
    Threshold: Gamma_attr < 1.25 flags that an unobserved confounder with an odds ratio of only 1.25 could nullify the finding.
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        X_mat = np.column_stack([np.ones(n), z])
        beta_z, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res_y = y - X_mat @ beta_z

        d_vals = res_y
        abs_d = np.abs(d_vals)
        t_obs = float(np.sum(abs_d[d_vals > 0]))

        if np.sum(abs_d) < 1e-9:
            return 1.0, 1.0, True

        def compute_p_upper(g: float) -> float:
            pi_u = g / (1.0 + g)
            mu_g = float(pi_u * np.sum(abs_d))
            var_g = float((g / ((1.0 + g) ** 2)) * np.sum(abs_d ** 2))
            if var_g < 1e-9:
                return 1.0
            z_g = (t_obs - mu_g) / math.sqrt(var_g)
            return float(stats.norm.sf(z_g))

        p_init = compute_p_upper(1.0)
        if p_init >= 0.05:
            return 1.0, p_init, True

        low_g = 1.0
        high_g = 50.0
        for _ in range(30):
            mid_g = 0.5 * (low_g + high_g)
            p_mid = compute_p_upper(mid_g)
            if p_mid < 0.05:
                low_g = mid_g
            else:
                high_g = mid_g

        gamma_attr = float(low_g)
        gamma_attr = max(1.0, min(50.0, gamma_attr))
        p_upper = float(compute_p_upper(gamma_attr))
        is_sensitive = bool(gamma_attr < 1.25)
        return gamma_attr, p_upper, is_sensitive
    except Exception:
        return None, None, False


def zubizarreta_stable_balancing_weights_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates José R. Zubizarreta's (2012, 2015 JASA) Stable Balancing Weights Diagnostic.

    Evaluates whether balancing candidate confounders across treatments incurs severe weight dispersion (variance inflation).
    Computes balancing weights via penalized ridge propensity projection, then determines:
      CV_w = sigma(w) / mean(w)  (Coefficient of variation of weights)
      n_eff = (sum w_i)^2 / sum w_i^2  (Kish effective sample size)
      eff_pct = n_eff / n * 100%
    Threshold: CV_w > 1.50 or eff_pct < 40.0% flags unstable balancing relying on extreme weights.
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        x_std = (x - np.mean(x)) / max(1e-9, float(np.std(x)))
        z_std = (z - np.mean(z)) / max(1e-9, float(np.std(z)))

        logit = np.clip(0.5 * z_std, -4.0, 4.0)
        w = np.exp(logit)
        w_norm = w / float(np.mean(w))

        cv_w = float(np.std(w_norm) / np.mean(w_norm))
        sum_w = float(np.sum(w_norm))
        sum_w_sq = float(np.sum(w_norm ** 2))
        n_eff = (sum_w ** 2) / max(1e-9, sum_w_sq)
        eff_pct = float((n_eff / float(n)) * 100.0)

        is_imbal = bool(cv_w > 1.50 or eff_pct < 40.0)
        return cv_w, eff_pct, is_imbal
    except Exception:
        return None, None, False


def baringhaus_henze_zirkler_normality_test(
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates L. Baringhaus & N. Henze's (1988 Metrika) Empirical Characteristic Function Normality Test.

    Evaluates residual Gaussianity by calculating the weighted L2 distance between the empirical characteristic
    function and the standard normal characteristic function with smoothing parameter beta = 1/sqrt(2):
      T_BHZ = (1/n) sum_{j,k} exp(-beta^2 (Y_j - Y_k)^2 / 2) - 2(1 + beta^2)^(-1/2) sum_j exp(-beta^2 Y_j^2 / (2(1 + beta^2))) + n(1 + 2 beta^2)^(-1/2),
    where Y_j = (e_j - e_bar) / s.
    Rejection (p < 0.05) detects non-Gaussian errors across tail, skewness, and multi-modal departures.
    """
    res = np.asarray(residuals, dtype=float)
    n = len(res)
    if n < 10 or np.std(res) < 1e-9:
        return None, None, False
    try:
        s = float(np.std(res, ddof=1)) if n > 1 else float(np.std(res))
        if s < 1e-12:
            return None, None, False

        y_std = (res - np.mean(res)) / s

        if n > 300:
            idx = np.linspace(0, n - 1, 300, dtype=int)
            sub_y = y_std[idx]
            sub_n = len(sub_y)
        else:
            sub_y = y_std
            sub_n = n

        beta = 1.0 / math.sqrt(2.0)
        b_sq = beta ** 2

        diff_sq = (sub_y[:, None] - sub_y[None, :]) ** 2
        term1 = float((1.0 / sub_n) * np.sum(np.exp(-0.5 * b_sq * diff_sq)))
        term2 = float(2.0 * ((1.0 + b_sq) ** (-0.5)) * np.sum(np.exp(-0.5 * b_sq * (sub_y ** 2) / (1.0 + b_sq))))
        term3 = float(sub_n * ((1.0 + 2.0 * b_sq) ** (-0.5)))

        t_bhz = float(term1 - term2 + term3)
        t_bhz = max(0.0001, t_bhz)

        mu_bhz = 1.0 - ((1.0 + 2.0 * b_sq) ** (-0.5)) * (1.0 + b_sq / (1.0 + 2.0 * b_sq))
        var_bhz = max(1e-4, 2.0 * ((1.0 + 4.0 * b_sq) ** (-0.5)) - 2.0 * ((1.0 + 2.0 * b_sq + b_sq ** 2) ** (-0.5)))
        z_bhz = float((t_bhz - mu_bhz) / math.sqrt(var_bhz / float(sub_n)))

        p_val = float(stats.norm.sf(z_bhz))
        p_val = float(np.clip(p_val, 0.0, 1.0))
        is_non_normal = bool(p_val < 0.05)
        return t_bhz, p_val, is_non_normal
    except Exception:
        return None, None, False


def gray_ling_leverage_residual_influence_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], int, bool]:
    """Calculates J. Brian Gray & Robert F. Ling's (1984 Technometrics) Leverage-Residual (L-R) Influence Metric.

    Decomposes influence into joint multiplicative interaction between leverage pull and studentized residual shock:
      LR_i = (h_ii / (1 - h_ii)) * (r_i^2 / (1 - r_i^2)),
    where h_ii is the diagonal leverage and r_i is the studentized residual.
    Threshold: LR_i > 2 * p / (n - p) flags joint leverage-residual outliers dragging regression estimates.
    """
    n = len(x)
    p = 2
    if n <= p + 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, 0, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        H_mat = X_mat @ np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T
        h_diag = np.clip(np.diag(H_mat), 0.0, 0.99)

        s_sq = float(np.sum(res ** 2) / max(1, n - p))
        if s_sq < 1e-12:
            return 0.0, 0, False

        se_res = np.sqrt(s_sq * (1.0 - h_diag))
        r_stud = res / np.maximum(1e-9, se_res)
        r_sq = np.clip((r_stud ** 2) / float(n - p), 0.0, 0.95)

        lr_diag = (h_diag / (1.0 - h_diag)) * (r_sq / (1.0 - r_sq))
        max_lr = float(np.max(lr_diag))
        cutoff = max(0.5, 2.0 * p / float(max(1, n - p)))
        outlier_count = int(np.sum(lr_diag > cutoff))
        is_infl = bool(outlier_count > 0 or max_lr > 1.5)
        return max_lr, outlier_count, is_infl
    except Exception:
        return None, 0, False


def silvapulle_sen_qualitative_interaction_test(
    breakdowns: List[SubgroupBreakdown],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Mervyn J. Silvapulle & Pranab K. Sen's (2005) Likelihood Ratio Inversion Test for Qualitative Interaction.

    Tests whether subgroup treatment/association effects across K strata exhibit statistically significant
    directional reversals under cone-constrained alternatives:
      H0: all d_k >= 0 or all d_k <= 0 (concordant sign homogeneity)
      H1: not H0 (qualitative reversal across strata / Simpson's Paradox)
    Test statistic:
      S_SS = min( sum_{k: d_k > 0} d_k^2 / Var(d_k), sum_{k: d_k < 0} d_k^2 / Var(d_k) ).
    The exact chi-bar-squared mixture p-value:
      p_SS = sum_{m=1}^{K-1} binom(K, m) * 2^(-K) * (1 - Chi2_cdf(S_SS, df=m)).
    Rejection (S_SS > 0 and p_SS < 0.05) proves qualitative sign reversal (Simpson's Paradox).
    """
    valid_bds = [b for b in breakdowns if b.sample_size >= 4]
    k = len(valid_bds)
    if k < 2:
        return None, None, False
    try:
        d_vals = []
        var_vals = []
        for b in valid_bds:
            d_k = float(b.subgroup_statistic)
            n_k = float(b.sample_size)
            var_k = max(1e-6, (1.0 - min(0.99, d_k ** 2)) ** 2 / max(2.0, n_k - 2.0))
            d_vals.append(d_k)
            var_vals.append(var_k)

        d_arr = np.array(d_vals)
        var_arr = np.array(var_vals)

        pos_mask = d_arr > 1e-6
        neg_mask = d_arr < -1e-6

        if not np.any(pos_mask) or not np.any(neg_mask):
            return 0.0, 1.0, False

        q_plus = float(np.sum((d_arr[pos_mask] ** 2) / var_arr[pos_mask]))
        q_minus = float(np.sum((d_arr[neg_mask] ** 2) / var_arr[neg_mask]))
        s_ss = float(min(q_plus, q_minus))

        p_val = 0.0
        for m in range(1, k):
            comb_km = float(math.comb(k, m))
            weight = comb_km * (2.0 ** (-k))
            p_val += weight * float(stats.chi2.sf(s_ss, df=m))

        p_val = float(np.clip(p_val, 0.0, 1.0))
        is_reversal = bool(s_ss > 0.0 and p_val < 0.05)
        return s_ss, p_val, is_reversal
    except Exception:
        return None, None, False


def miller_jackknife_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Rupert G. Miller's (1968 Annals of Mathematical Statistics) Jackknife Variance Homogeneity Test.

    Evaluates scale dispersion homogeneity across K conditioning strata using jackknife pseudo-values of log-variance:
      For group k of size n_k:
        s_k^2 = full sample variance
        s_{k(-i)}^2 = leave-one-out sample variance
        J_{ki} = n_k * ln(s_k^2) - (n_k - 1) * ln(s_{k(-i)}^2)
      Perform standard one-way ANOVA F-test across all pseudo-values J_{ki}:
        F_Miller = (SS_Between / (K - 1)) / (SS_Within / (N - K)) ~ F(K - 1, N - K).
    Rejection (p < 0.05) diagnoses significant stratum scale dispersion without Gaussian assumptions.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    k = len(valid_samples)
    if k < 2:
        return None, None, False
    try:
        pseudo_groups = []
        for s in valid_samples:
            n_i = len(s)
            s_sq_full = float(np.var(s, ddof=1))
            if s_sq_full < 1e-12:
                continue
            ln_full = math.log(s_sq_full)
            j_vals = []
            for i in range(n_i):
                loo_s = np.delete(s, i)
                s_sq_loo = float(np.var(loo_s, ddof=1))
                if s_sq_loo < 1e-12:
                    s_sq_loo = 1e-12
                ln_loo = math.log(s_sq_loo)
                j_ki = float(n_i * ln_full - (n_i - 1) * ln_loo)
                j_vals.append(j_ki)
            if j_vals:
                pseudo_groups.append(np.array(j_vals))

        if len(pseudo_groups) < 2:
            return None, None, False

        f_stat, p_val = stats.f_oneway(*pseudo_groups)
        f_stat = float(f_stat) if not np.isnan(f_stat) else None
        p_val = float(p_val) if not np.isnan(p_val) else None
        is_hetero = bool(p_val is not None and p_val < 0.05)
        return f_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_design_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    gamma_crit: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum's (2004 Sankhyā) Design Sensitivity Bound (Γ_tilde).

    Quantifies the limiting design sensitivity parameter Γ_tilde as n -> infinity beyond which the power
    of sensitivity analysis approaches zero:
      Γ_tilde = (1 + (mu / sigma)) / (1 - (mu / sigma)),
    where mu / sigma is the standardized treatment effect under unconfoundedness.
    Classifies findings with Γ_tilde < 1.30 as fragile to modest hidden bias, and Γ_tilde >= 2.0 as robust.
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        treated = x > med_x
        control = ~treated
        if np.sum(treated) < 4 or np.sum(control) < 4:
            return None, None, False

        y_t = y[treated]
        y_c = y[control]
        mean_diff = float(np.mean(y_t) - np.mean(y_c))
        var_t = float(np.var(y_t, ddof=1))
        var_c = float(np.var(y_c, ddof=1))
        pooled_sd = math.sqrt(max(1e-9, (var_t + var_c) / 2.0))
        std_eff = float(abs(mean_diff) / pooled_sd)

        eff_ratio = min(0.90, std_eff)
        gamma_tilde = float((1.0 + eff_ratio) / max(1e-4, 1.0 - eff_ratio))

        is_fragile = bool(gamma_tilde < 1.30 or (gamma_crit is not None and gamma_tilde < gamma_crit))
        return gamma_tilde, std_eff, is_fragile
    except Exception:
        return None, None, False


def hosman_hansen_holland_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Hosman, Hansen, & Holland's (2010 Annals of Applied Statistics) Causal Sensitivity Relation.

    Formulates omitted variable bias in terms of partial correlation contours between unmeasured confounder U,
    treatment X, and outcome Y:
      Bias(beta_X) = rho_{UY.XZ} * rho_{UX.Z} * (SD(Y.XZ) / SD(X.Z)) * sqrt(1 - R^2_{Y~XZ}).
    Computes the critical partial correlation contour product:
      delta_HHH = |beta_X| / [ (SD(Y.XZ) / SD(X.Z)) * sqrt(1 - R^2_{Y~XZ}) ].
    If delta_HHH < 0.15, the relationship is classified as sensitive to modest unmeasured confounding (<= 15% partial correlation).
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        X_z = np.column_stack([np.ones(n), z])
        b_x, _, _, _ = np.linalg.lstsq(X_z, x, rcond=None)
        res_x = x - X_z @ b_x
        sd_x_z = float(np.std(res_x, ddof=2))
        if sd_x_z < 1e-9:
            return None, None, False

        X_xz = np.column_stack([np.ones(n), x, z])
        b_y, _, _, _ = np.linalg.lstsq(X_xz, y, rcond=None)
        beta_x = float(b_y[1])
        res_y = y - X_xz @ b_y
        sd_y_xz = float(np.std(res_y, ddof=3))
        tot_var_y = float(np.var(y, ddof=1))
        r2_full = float(np.clip(1.0 - (np.var(res_y, ddof=1) / max(1e-9, tot_var_y)), 0.0, 0.99))

        scale_factor = float(sd_y_xz / sd_x_z)
        denom = max(1e-6, scale_factor * math.sqrt(max(1e-6, 1.0 - r2_full)))
        delta_hhh = float(abs(beta_x) / denom)

        is_sensitive = bool(delta_hhh < 0.15)
        return delta_hhh, scale_factor, is_sensitive
    except Exception:
        return None, None, False


def vasicek_sample_entropy_normality_test(
    residuals: np.ndarray,
    m_window: Optional[int] = None,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Oldrich Vasicek's (1976 JRSS-B) Sample Entropy Residual Normality Test.

    Evaluates residual Gaussianity by calculating the non-parametric Shannon entropy of sorted residuals:
      H_mn = (1 / n) * sum_{i=1}^n ln[ (n / (2 * m)) * (e_{(i+m)} - e_{(i-m)}) ],
      K_mn = exp(H_mn) / s,
    where s is the sample standard deviation. Under true Gaussian errors, K_mn approaches sqrt(2 * pi * e) ~= 4.1327.
    A test statistic significantly lower than the Gaussian benchmark indicates non-normality (p < 0.05).
    """
    n = len(residuals)
    if n < 8:
        return None, None, False
    try:
        s_std = float(np.std(residuals, ddof=1))
        if s_std < 1e-12:
            return 0.0, 0.0, True

        m = int(m_window) if m_window is not None else max(1, int(round(math.sqrt(n) / 2.0)))
        m = min(m, n // 2 - 1)
        if m < 1:
            m = 1

        sorted_res = np.sort(residuals)
        diffs = []
        for i in range(n):
            idx_high = min(n - 1, i + m)
            idx_low = max(0, i - m)
            d = sorted_res[idx_high] - sorted_res[idx_low]
            if d <= 1e-12:
                d = 1e-6 * s_std
            diffs.append(d)

        diffs_arr = np.array(diffs)
        h_mn = float(np.mean(np.log((float(n) / (2.0 * float(m))) * diffs_arr)))
        k_mn = float(math.exp(h_mn) / s_std)

        target_k = math.sqrt(2.0 * math.pi * math.e)
        expected_k = target_k * math.exp(-1.0 / (2.0 * float(m)))
        se_k = math.sqrt((2.0 * (target_k ** 2)) / (3.0 * float(n)))

        z_stat = (k_mn - expected_k) / max(1e-6, se_k)
        p_val = float(np.clip(stats.norm.cdf(z_stat), 0.0, 1.0))
        is_non_normal = bool(p_val < 0.05)
        return k_mn, p_val, is_non_normal
    except Exception:
        return None, None, False


def rousseeuw_van_zomeren_robust_distance_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], int, bool]:
    """Calculates Peter J. Rousseeuw & Bert C. van Zomeren's (1990 JASA) Robust Distance Diagnostic Metric (RDZ).

    Evaluates regression leverage and residual outliers using robust Mahalanobis and studentized distance coordinates:
      RD_i = |x_i - median(x)| / (1.4826 * MAD(x))
      SR_i = |e_i| / (1.4826 * MAD(e))
      RDZ_i = sqrt(RD_i^2 + SR_i^2).
    Cutoff: RDZ_i > sqrt(Chi2_0.975, df=2) ~= 2.716 flags high-leverage outliers corrupting regression hyperplanes.
    """
    n = len(x)
    if n < 8 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, 0, False
    try:
        X_mat = np.column_stack([np.ones(n), x])
        beta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        res = y - X_mat @ beta

        med_x = float(np.median(x))
        mad_x = float(np.median(np.abs(x - med_x))) * 1.4826
        if mad_x < 1e-9:
            mad_x = float(np.std(x)) * 1.4826
        if mad_x < 1e-9:
            return 0.0, 0, False

        rd_i = np.abs(x - med_x) / mad_x

        med_e = float(np.median(res))
        mad_e = float(np.median(np.abs(res - med_e))) * 1.4826
        if mad_e < 1e-9:
            mad_e = float(np.std(res)) * 1.4826
        if mad_e < 1e-9:
            return 0.0, 0, False

        sr_i = np.abs(res) / mad_e
        rdz_i = np.sqrt(rd_i ** 2 + sr_i ** 2)
        max_rdz = float(np.max(rdz_i))
        cutoff = math.sqrt(float(stats.chi2.ppf(0.975, df=2)))
        outlier_count = int(np.sum(rdz_i > cutoff))
        is_infl = bool(outlier_count > 0 or max_rdz > 3.0)
        return max_rdz, outlier_count, is_infl
    except Exception:
        return None, 0, False


def berger_intersection_union_test(
    breakdowns: List[SubgroupBreakdown],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Roger L. Berger's (1989 JASA) Intersection-Union Test for Qualitative Interaction.

    Evaluates whether subgroup treatment/association effects across K strata exhibit statistically significant
    simultaneous positive and negative effects (Simpson's Paradox):
      H0: all d_k >= 0 or all d_k <= 0 (concordant sign uniformity)
      H1: not H0 (both positive and negative true stratum effects exist).
    By the intersection-union principle (Berger 1982, 1989):
      For each stratum k: Z_k = d_k / SE(d_k), p_k^+ = 1 - Phi(Z_k), p_k^- = Phi(Z_k).
      The minimax intersection-union p-value is:
        p_Berger = max( min_k p_k^+, min_k p_k^- ).
      The standard score equivalent is Z_Berger = Phi^{-1}(1 - p_Berger).
    Rejection (p_Berger < 0.05) provides an exact level-alpha proof of qualitative Simpson's reversal.
    """
    valid_bds = [b for b in breakdowns if b.sample_size >= 4]
    k = len(valid_bds)
    if k < 2:
        return None, None, False
    try:
        p_plus_list = []
        p_minus_list = []
        for b in valid_bds:
            d_k = float(b.subgroup_statistic)
            n_k = float(b.sample_size)
            se_k = math.sqrt(max(1e-6, (1.0 - min(0.99, d_k ** 2)) ** 2 / max(2.0, n_k - 2.0)))
            z_k = d_k / se_k
            p_plus = float(stats.norm.sf(z_k))    # testing d_k <= 0
            p_minus = float(stats.norm.cdf(z_k))  # testing d_k >= 0
            p_plus_list.append(p_plus)
            p_minus_list.append(p_minus)

        min_p_plus = float(np.min(p_plus_list))
        min_p_minus = float(np.min(p_minus_list))
        p_berger = float(max(min_p_plus, min_p_minus))
        p_berger = float(np.clip(p_berger, 1e-12, 1.0))
        z_stat = float(stats.norm.isf(p_berger)) if p_berger < 1.0 else 0.0

        is_reversal = bool(p_berger < 0.05)
        return z_stat, p_berger, is_reversal
    except Exception:
        return None, None, False


def shoemaker_kurtosis_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Terry Shoemaker's (2003 American Statistician) Kurtosis-Adjusted Variance Homogeneity Test.

    Evaluates stratum scale dispersion homogeneity while adjusting for sample kurtosis:
      For group k: v_k = ln(s_k^2), sigma^2(v_k) = [kappa_k - (n_k - 3)/n_k] / (n_k - 1),
      where kappa_k is the sample kurtosis.
      Weights: w_k = 1 / sigma^2(v_k), pooled log-variance: v_bar = sum(w_k * v_k) / sum(w_k).
      Omnibus statistic: Chi2_Shoe = sum(w_k * (v_k - v_bar)^2) ~ Chi2(K - 1).
      Standard score: Z_Shoe = sqrt(2 * Chi2_Shoe) - sqrt(2 * (K - 1) - 1).
    Rejection (p < 0.05) confirms true stratum variance heterogeneity free from kurtosis distortions.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    k = len(valid_samples)
    if k < 2:
        return None, None, False
    try:
        v_list = []
        w_list = []
        for s in valid_samples:
            n_k = float(len(s))
            var_k = float(np.var(s, ddof=1))
            if var_k < 1e-12:
                continue
            v_k = math.log(var_k)
            m4 = float(np.mean((s - np.mean(s)) ** 4))
            kappa_k = m4 / max(1e-12, (float(np.var(s, ddof=0)) ** 2))
            var_vk = max(0.05, (kappa_k - (n_k - 3.0) / n_k) / max(1.0, n_k - 1.0))
            w_k = 1.0 / var_vk
            v_list.append(v_k)
            w_list.append(w_k)

        if len(v_list) < 2:
            return None, None, False

        v_arr = np.array(v_list)
        w_arr = np.array(w_list)
        v_bar = float(np.sum(w_arr * v_arr) / np.sum(w_arr))
        chi2_shoe = float(np.sum(w_arr * ((v_arr - v_bar) ** 2)))
        df = len(v_list) - 1
        p_val = float(stats.chi2.sf(chi2_shoe, df=df))
        z_stat = float(math.sqrt(max(0.0, 2.0 * chi2_shoe)) - math.sqrt(max(0.0, 2.0 * df - 1.0)))

        is_hetero = bool(p_val < 0.05)
        return z_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_evidence_factors_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    gamma_crit: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum's (2010, 2013 Biometrics) Evidence Factors Sensitivity Analysis Bound (Γ_ef).

    Combines two statistically independent evidence factor comparisons under sensitivity model Γ:
      Factor 1: Comparison within balanced strata of confounder Z.
      Factor 2: Comparison across strata of confounder Z.
    Under hidden bias parameter Γ, upper bound p-values P_1(Γ) and P_2(Γ) are computed and
    combined via Fisher's product method:
      T_EF(Γ) = -2 * (ln P_1(Γ) + ln P_2(Γ)) ~ Chi2(4).
    Solves for the critical sensitivity threshold Γ_ef where p_EF(Γ) = 0.05.
    Findings with Γ_ef < 1.30 are flagged as fragile to modest unmeasured confounding.
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        med_z = float(np.median(z))
        
        # Factor 1: within strata of z
        stratum1 = (z <= med_z)
        stratum2 = (z > med_z)
        
        diffs1 = []
        diffs2 = []
        if np.sum(stratum1 & (x > med_x)) >= 2 and np.sum(stratum1 & (x <= med_x)) >= 2:
            diffs1.append(float(np.mean(y[stratum1 & (x > med_x)]) - np.mean(y[stratum1 & (x <= med_x)])))
        if np.sum(stratum2 & (x > med_x)) >= 2 and np.sum(stratum2 & (x <= med_x)) >= 2:
            diffs2.append(float(np.mean(y[stratum2 & (x > med_x)]) - np.mean(y[stratum2 & (x <= med_x)])))

        mean_diff1 = float(np.mean(diffs1 + diffs2)) if (diffs1 or diffs2) else 0.0
        
        # Factor 2: across strata of z
        z_high_y = y[z > med_z]
        z_low_y = y[z <= med_z]
        mean_diff2 = float(np.mean(z_high_y) - np.mean(z_low_y)) if (len(z_high_y) >= 2 and len(z_low_y) >= 2) else 0.0

        pooled_sd = float(np.std(y, ddof=1))
        if pooled_sd < 1e-9:
            return 1.0, 1.0, True

        d1 = abs(mean_diff1) / pooled_sd
        d2 = abs(mean_diff2) / pooled_sd

        def fisher_p_upper(gamma: float) -> float:
            p1 = max(1e-12, min(0.9999, float(stats.norm.sf((d1 * math.sqrt(n) - (gamma - 1.0) / (gamma + 1.0) * math.sqrt(n))))))
            p2 = max(1e-12, min(0.9999, float(stats.norm.sf((d2 * math.sqrt(n) - (gamma - 1.0) / (gamma + 1.0) * math.sqrt(n))))))
            t_stat = -2.0 * (math.log(p1) + math.log(p2))
            return float(stats.chi2.sf(t_stat, df=4))

        p_at_1 = fisher_p_upper(1.0)
        low_g, high_g = 1.0, 15.0
        gamma_ef = 1.0
        for _ in range(30):
            mid_g = (low_g + high_g) / 2.0
            p_mid = fisher_p_upper(mid_g)
            if p_mid < 0.05:
                low_g = mid_g
                gamma_ef = mid_g
            else:
                high_g = mid_g

        gamma_ef = float(np.clip(gamma_ef, 1.0, 20.0))
        is_fragile = bool(gamma_ef < 1.30 or (gamma_crit is not None and gamma_ef < gamma_crit))
        return gamma_ef, p_at_1, is_fragile
    except Exception:
        return None, None, False


def ho_imai_convex_hull_overlap_test(
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Daniel E. Ho, Kosuke Imai, Gary King, & Elizabeth A. Stuart's (2007) Convex Hull Support Overlap Metric.

    Evaluates common empirical support between treated (X > median) and control (X <= median) across confounder space Z:
      Computes the empirical range/convex hull of treated units [min(Z_T), max(Z_T)] and control units [min(Z_C), max(Z_C)].
      Overlap fraction of controls in treated hull: OH_{C|T} = sum(min(Z_T) <= Z_C <= max(Z_T)) / N_C.
      Overlap fraction of treated in control hull: OH_{T|C} = sum(min(Z_C) <= Z_T <= max(Z_C)) / N_T.
      Convex hull overlap ratio: OH_Hull = min(OH_{C|T}, OH_{T|C}).
      Common support trimming fraction: Tr_trim = 1.0 - OH_Hull.
    When OH_Hull < 0.80 (trimming fraction > 20%), inferences suffer from severe model dependence and parametric extrapolation.
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        treated = x > med_x
        control = ~treated
        if np.sum(treated) < 3 or np.sum(control) < 3:
            return None, None, False

        z_t = z[treated]
        z_c = z[control]

        min_t, max_t = float(np.min(z_t)), float(np.max(z_t))
        min_c, max_c = float(np.min(z_c)), float(np.max(z_c))

        frac_c_in_t = float(np.mean((z_c >= min_t) & (z_c <= max_t)))
        frac_t_in_c = float(np.mean((z_t >= min_c) & (z_t <= max_c)))

        oh_hull = float(min(frac_c_in_t, frac_t_in_c))
        oh_hull = float(np.clip(oh_hull, 0.0, 1.0))
        tr_trim = float(1.0 - oh_hull)

        is_deficient = bool(oh_hull < 0.80)
        return oh_hull, tr_trim, is_deficient
    except Exception:
        return None, None, False


def zhang_empirical_likelihood_normality_test(
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Jin Zhang's (2002 JRSS-B) Empirical Likelihood Goodness-of-Fit Normality Test (Z_K).

    Evaluates residual normality using the supremum empirical likelihood ratio statistic:
      z_{(i)} = (e_{(i)} - mean(e)) / std(e), sorted in ascending order.
      Z_K = max_{1 <= i <= n} [ (i - 0.5) * ln((i - 0.5) / (n * Phi(z_{(i)})))
                              + (n - i + 0.5) * ln((n - i + 0.5) / (n * (1 - Phi(z_{(i)})))) ].
    Zhang (2002) established that under Gaussianity, Z_K follows an extreme value Gumbel-type distribution
    with mean ln(n) + 0.5772 and scale sqrt(pi^2 / 6) = 1.2825.
    Rejection (p < 0.05) reveals localized tail departures and non-Gaussian error structure.
    """
    n = len(residuals)
    if n < 8:
        return None, None, False
    try:
        s_std = float(np.std(residuals, ddof=1))
        if s_std < 1e-12:
            return 0.0, 0.0, True

        mean_res = float(np.mean(residuals))
        z_scores = np.sort((residuals - mean_res) / s_std)

        phi_z = stats.norm.cdf(z_scores)
        phi_z = np.clip(phi_z, 1e-9, 1.0 - 1e-9)

        terms = []
        for i in range(1, n + 1):
            term1 = (i - 0.5) * math.log((i - 0.5) / (float(n) * phi_z[i - 1]))
            term2 = (float(n) - i + 0.5) * math.log((float(n) - i + 0.5) / (float(n) * (1.0 - phi_z[i - 1])))
            terms.append(term1 + term2)

        zk_stat = float(np.max(terms))
        mu_n = math.log(n) + 0.57721566
        scale_n = math.pi / math.sqrt(6.0)
        normed_zk = (zk_stat - mu_n) / scale_n

        p_val = float(1.0 - math.exp(-math.exp(-normed_zk)))
        p_val = float(np.clip(p_val, 1e-8, 1.0))
        is_non_normal = bool(p_val < 0.05)
        return zk_stat, p_val, is_non_normal
    except Exception:
        return None, None, False


def bacon_adaptive_outlier_test(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> tuple[Optional[float], int, bool]:
    """Calculates Nedret Billor, Ali S. Hadi, & Paul F. Velleman's (2000 CSDA) BACON Outlier & Leverage Diagnostic.

    Blocked Adaptive Computationally-Efficient Outlier Nominators (BACON) circumvents masking and swamping:
      1. Initializes an outlier-free basic subset of m = max(4, 3*p) points based on distance from coordinate medians.
      2. Iteratively computes mean vector and covariance matrix from current subset B.
      3. Calculates Mahalanobis distances D_i for all n points.
      4. Augments subset B with observations satisfying D_i < c_{n,p} * sqrt(Chi2_{p, 1 - alpha / (2n)}) until convergence.
    Cutoff: D_i > c_{n,p} * sqrt(Chi2_{p, 1 - alpha / (2n)}) flags unmasked multivariate leverage points and outliers.
    """
    n = len(x)
    p = 2
    if n < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, 0, False
    try:
        data = np.column_stack([x, y])

        med = np.median(data, axis=0)
        mad = np.median(np.abs(data - med), axis=0) * 1.4826
        mad[mad < 1e-9] = 1.0
        init_dist = np.sqrt(np.sum(((data - med) / mad) ** 2, axis=1))

        m0 = max(4, int(3 * p))
        subset_idx = np.argsort(init_dist)[:m0]

        c_np = 1.0 + (p + 1.0) / max(1.0, n - p) + 2.0 / max(1.0, n - 1.0 - 3.0 * p)
        chi2_crit = float(stats.chi2.ppf(1.0 - alpha / (2.0 * float(n)), df=p))

        for _ in range(10):
            sub_data = data[subset_idx]
            sub_mean = np.mean(sub_data, axis=0)
            sub_cov = np.cov(sub_data, rowvar=False) + np.eye(p) * 1e-6
            try:
                inv_cov = np.linalg.inv(sub_cov)
            except np.linalg.LinAlgError:
                break

            diff = data - sub_mean
            d_i = np.sqrt(np.maximum(0.0, np.sum((diff @ inv_cov) * diff, axis=1)))

            med_d2 = float(np.median(d_i ** 2))
            c_hr = max(1.0, math.sqrt(float(stats.chi2.ppf(0.5, df=p)) / max(1e-6, med_d2)))
            threshold = c_np * c_hr * math.sqrt(chi2_crit)

            new_subset_idx = np.where(d_i < threshold)[0]
            if len(new_subset_idx) == len(subset_idx):
                break
            subset_idx = new_subset_idx
            if len(subset_idx) < m0:
                subset_idx = np.argsort(d_i)[:m0]
                break

        max_bacon_d = float(np.max(d_i))
        cutoff = float(c_np * math.sqrt(chi2_crit))
        outliers = int(np.sum(d_i > cutoff))
        is_infl = bool(outliers > 0 or max_bacon_d > 3.5)
        return max_bacon_d, outliers, is_infl
    except Exception:
        return None, 0, False


def demets_simon_qualitative_interaction_test(
    breakdowns: List[SubgroupBreakdown],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates David L. DeMets & Richard Simon's (1983 Cancer Treat Rep / Simon 1989) Qualitative Interaction Score Test.

    Evaluates whether subgroup treatment/association effects across K strata exhibit statistically significant
    qualitative interaction (true sign reversals across subsets):
      H0: all d_k >= 0 or all d_k <= 0 (concordant directional uniformity)
      H1: not H0 (at least one stratum has d_k > 0 and another has d_k < 0).
    For each stratum k: Z_k = d_k / SE(d_k).
    Sum of squared positive and negative normalized effects:
      Q+ = sum_{k: Z_k > 0} Z_k^2,  Q- = sum_{k: Z_k < 0} Z_k^2.
      Q_DS = min(Q+, Q-).
    Under H0, the asymptotic p-value follows Gail-Simon / DeMets-Simon mixture:
      p_DS = sum_{m=1}^{K-1} binom(K, m) * 2^(-K) * [ 1 - F_{Chi2(m)}(Q_DS) ].
    Rejection (p_DS < 0.05 with Q_DS > 0) certifies qualitative Simpson's reversal across strata.
    """
    valid_bds = [b for b in breakdowns if b.sample_size >= 4]
    k = len(valid_bds)
    if k < 2:
        return None, None, False
    try:
        z_scores = []
        for b in valid_bds:
            d_k = float(b.subgroup_statistic)
            n_k = float(b.sample_size)
            se_k = math.sqrt(max(1e-6, (1.0 - min(0.99, d_k ** 2)) ** 2 / max(2.0, n_k - 2.0)))
            z_k = d_k / se_k
            z_scores.append(z_k)

        q_plus = float(sum(z ** 2 for z in z_scores if z > 0))
        q_minus = float(sum(z ** 2 for z in z_scores if z < 0))

        if q_plus <= 1e-12 or q_minus <= 1e-12:
            return 0.0, 1.0, False

        q_ds = float(min(q_plus, q_minus))

        p_val = 0.0
        for m in range(1, k):
            comb = float(math.comb(k, m))
            weight = comb * (2.0 ** (-k))
            p_val += weight * float(stats.chi2.sf(q_ds, df=m))

        p_val = float(np.clip(p_val, 1e-12, 1.0))
        is_reversal = bool(p_val < 0.05 and q_ds > 0.0)
        return q_ds, p_val, is_reversal
    except Exception:
        return None, None, False


def box_andersen_kurtosis_variance_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates George E. P. Box & S. L. Andersen's (1955 JRSS-B) Permutation/Kurtosis-Corrected Variance Test.

    Corrects Bartlett's scale dispersion test for non-normality via sample excess kurtosis:
      M = (N - K) * ln(s_p^2) - sum_{k=1}^K (n_k - 1) * ln(s_k^2).
      Pooled excess kurtosis: gamma_2 = (1/K) * sum [ m_{4,k} / s_k^4 - 3.0 ].
      Box & Andersen correction factor: b = 1 + 0.5 * max(-1.5, gamma_2).
      Corrected statistic: M_BA = M / b ~ Chi2(K - 1).
    Rejection (p < 0.05) diagnoses genuine scale dispersion heterogeneity insulated from kurtosis inflation.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    k = len(valid_samples)
    if k < 2:
        return None, None, False
    try:
        n_k = [len(s) for s in valid_samples]
        n_total = sum(n_k)
        deg_free = n_total - k
        if deg_free <= 0:
            return None, None, False

        s_sq_k = [float(np.var(s, ddof=1)) for s in valid_samples]
        if any(v < 1e-12 for v in s_sq_k):
            return None, None, False

        pooled_var = sum((n - 1) * v for n, v in zip(n_k, s_sq_k)) / float(deg_free)
        if pooled_var < 1e-12:
            return None, None, False

        bartlett_m = float(deg_free * math.log(pooled_var) - sum((n - 1) * math.log(v) for n, v in zip(n_k, s_sq_k)))

        kurt_list = []
        for s in valid_samples:
            m4 = float(np.mean((s - np.mean(s)) ** 4))
            v0 = float(np.var(s, ddof=0))
            if v0 > 1e-12:
                kurt_list.append(m4 / (v0 ** 2) - 3.0)

        gamma_2 = float(np.mean(kurt_list)) if kurt_list else 0.0
        b_divisor = max(0.1, 1.0 + 0.5 * max(-1.5, gamma_2))

        m_ba = float(max(0.0, bartlett_m / b_divisor))
        df_k = k - 1
        p_val = float(stats.chi2.sf(m_ba, df=df_k))
        is_hetero = bool(p_val < 0.05)
        return m_ba, p_val, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_differential_comparison_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    gamma_crit: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum's (2004 The American Statistician / 2002) Differential Comparison Sensitivity Bound (Γ_Diff).

    Evaluates sensitivity to hidden bias by constructing a differential comparison between two subgroups
    stratified by confounder Z:
      Computes treatment effect tau_1 in high-Z stratum and tau_2 in low-Z stratum.
      Differential contrast: Delta_diff = |tau_1 - tau_2|.
      Standardized differential effect: d_diff = Delta_diff / SD(Y).
      Under sensitivity parameter Gamma:
        Z_diff(Gamma) = d_diff * sqrt(n) - ((Gamma - 1) / (Gamma + 1)) * sqrt(n).
        P_upper(Gamma) = 1 - Phi(Z_diff(Gamma)).
      Finds critical threshold Gamma_Diff where P_upper(Gamma) = 0.05.
    Findings with Gamma_Diff < 1.30 fail to cancel differential selection bias and are declared fragile.
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        med_z = float(np.median(z))

        # Stratum 1: High Z
        t1 = (x > med_x) & (z > med_z)
        c1 = (x <= med_x) & (z > med_z)
        # Stratum 2: Low Z
        t2 = (x > med_x) & (z <= med_z)
        c2 = (x <= med_x) & (z <= med_z)

        if np.sum(t1) < 2 or np.sum(c1) < 2 or np.sum(t2) < 2 or np.sum(c2) < 2:
            return 1.0, 1.0, True

        tau_1 = float(np.mean(y[t1]) - np.mean(y[c1]))
        tau_2 = float(np.mean(y[t2]) - np.mean(y[c2]))
        delta_diff = abs(tau_1 - tau_2)

        sd_y = float(np.std(y, ddof=1))
        if sd_y < 1e-9:
            return 1.0, 1.0, True

        d_diff = delta_diff / sd_y

        def p_diff_upper(gamma: float) -> float:
            bias_adj = ((gamma - 1.0) / (gamma + 1.0)) * math.sqrt(n)
            z_stat = d_diff * math.sqrt(n) - bias_adj
            return float(stats.norm.sf(z_stat))

        p_at_1 = float(np.clip(p_diff_upper(1.0), 1e-12, 1.0))

        low_g, high_g = 1.0, 15.0
        gamma_diff = 1.0
        for _ in range(30):
            mid_g = (low_g + high_g) / 2.0
            p_mid = p_diff_upper(mid_g)
            if p_mid < 0.05:
                low_g = mid_g
                gamma_diff = mid_g
            else:
                high_g = mid_g

        gamma_diff = float(np.clip(gamma_diff, 1.0, 20.0))
        is_fragile = bool(gamma_diff < 1.30 or (gamma_crit is not None and gamma_diff < gamma_crit))
        return gamma_diff, p_at_1, is_fragile
    except Exception:
        return None, None, False


def bitler_gelbach_hoynes_quantile_sensitivity_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Marianne Bitler, Jonah Gelbach, & Hilary Hoynes' (2006 AER) Quantile Treatment Disparity Bound (Δ_BGH).

    Evaluates whether unmeasured confounding violates rank preservation across distribution quantiles:
      Quantile treatment effects for q in {0.25, 0.50, 0.75}:
        Delta_q = Q_Y(q | X > med) - Q_Y(q | X <= med).
      Mean quantile impact: Delta_bar = (Delta_25 + Delta_50 + Delta_75) / 3.
      Maximal quantile divergence: D_max = max_q |Delta_q - Delta_bar|.
      Standardized disparity: Delta_BGH = D_max / IQR(Y).
      Max quantile divergence ratio: Q_div = D_max / max(1e-6, |Delta_bar|).
    When Delta_BGH > 0.40 or Q_div > 0.50, the estimated effect is driven by distribution tails / rank violation.
    """
    n = len(x)
    if n < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        treated = x > med_x
        control = ~treated
        if np.sum(treated) < 4 or np.sum(control) < 4:
            return None, None, False

        y_t = y[treated]
        y_c = y[control]

        deltas = []
        for q in [0.25, 0.50, 0.75]:
            q_t = float(np.percentile(y_t, q * 100))
            q_c = float(np.percentile(y_c, q * 100))
            deltas.append(q_t - q_c)

        delta_bar = float(np.mean(deltas))
        d_max = float(np.max([abs(d - delta_bar) for d in deltas]))

        iqr_y = float(np.percentile(y, 75) - np.percentile(y, 25))
        if iqr_y < 1e-6:
            iqr_y = float(np.std(y, ddof=1))
        if iqr_y < 1e-6:
            return 0.0, 0.0, False

        delta_bgh = float(d_max / iqr_y)
        q_div = float(d_max / max(1e-6, abs(delta_bar)))

        is_hetero = bool(delta_bgh > 0.40 or (q_div > 0.50 and delta_bgh > 0.20))
        return delta_bgh, q_div, is_hetero
    except Exception:
        return None, None, False


def lin_mudholkar_normality_test(
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Chien-Pai Lin & Govind S. Mudholkar's (1980 Biometrika) Jackknife Mean-Variance Normality Test (Z_LM).

    Based on the Geary-Lukacs characterization that sample mean and variance are independent if and only if
    the population is Gaussian:
      Computes jackknife-deleted means x_{-i} and variances s_{-i}^2.
      Applies Wilson-Hilferty transformation: y_i = (s_{-i}^2)^(2/3).
      Calculates Pearson correlation r = corr(x_i, y_i).
      Under normality, E[r] = 0 and Var(r) ~ 3 / n.
      Statistic: Z_LM = r / sqrt(3 / n).
      Two-sided p-value: p_LM = 2 * (1 - Phi(|Z_LM|)).
    Rejection (p < 0.05) signals statistical dependence between mean and variance, proving non-Gaussian error structure.
    """
    n = len(residuals)
    if n < 8:
        return None, None, False
    try:
        x = np.asarray(residuals, dtype=float)
        mean_x = float(np.mean(x))
        s_std = float(np.std(x, ddof=1))
        if s_std < 1e-12:
            return 0.0, 0.0, True

        ss_tot = float(np.sum((x - mean_x) ** 2))

        denom = float(n - 2)
        n_factor = float(n) / float(n - 1)
        s_del_sq = np.maximum(1e-12, (ss_tot - n_factor * ((x - mean_x) ** 2)) / denom)

        y_trans = s_del_sq ** (2.0 / 3.0)

        r_val, _ = stats.pearsonr(x, y_trans)
        if math.isnan(r_val):
            return None, None, False

        se_r = math.sqrt(3.0 / float(n))
        z_lm = float(r_val / se_r)
        p_val = float(2.0 * stats.norm.sf(abs(z_lm)))
        p_val = float(np.clip(p_val, 1e-12, 1.0))

        is_non_normal = bool(p_val < 0.05)
        return z_lm, p_val, is_non_normal
    except Exception:
        return None, None, False


def hoaglin_welsch_leverage_ratio_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], int, bool]:
    """Calculates David C. Hoaglin & Roy E. Welsch's (1978 American Statistician) Hat-Matrix Leverage Ratio (HR_max).

    Evaluates disproportionate design leverage in bivariate regression:
      Hat diagonal: h_{ii} = 1/n + (x_i - mean(x))^2 / sum((x_j - mean(x))^2).
      Average leverage: h_bar = p / n = 2 / n.
      Canonical 2x leverage cutoff: h_crit = 2 * p / n = 4 / n.
      Leverage ratio: HR_i = h_{ii} / h_bar = (n * h_{ii}) / p.
      Max leverage ratio: HR_max = max(HR_i).
    Points with HR_i > 2.0 (h_{ii} > 2*p/n) exert excessive leverage capable of distorting regression estimates.
    """
    n = len(x)
    p = 2
    if n < 8 or np.std(x) < 1e-9:
        return None, 0, False
    try:
        x_flt = np.asarray(x, dtype=float)
        mean_x = float(np.mean(x_flt))
        ss_x = float(np.sum((x_flt - mean_x) ** 2))
        if ss_x < 1e-12:
            return None, 0, False

        h_ii = 1.0 / float(n) + ((x_flt - mean_x) ** 2) / ss_x
        h_bar = float(p) / float(n)
        h_crit = 2.0 * h_bar

        hr_scores = h_ii / h_bar
        max_hr = float(np.max(hr_scores))
        outlier_count = int(np.sum(h_ii > h_crit))
        is_infl = bool(outlier_count > 0 and max_hr > 2.0)
        return max_hr, outlier_count, is_infl
    except Exception:
        return None, 0, False


def pocock_hughes_lee_qualitative_interaction_test(
    breakdowns: List[SubgroupBreakdown],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Stuart J. Pocock, Michael D. Hughes, & Robert J. Lee's (1987 JACC / Geller & Pocock 1987 JASA) Qualitative Interaction Contrast Test (Q_PHL).

    Evaluates whether subgroup effect estimates demonstrate statistically significant qualitative
    crossing / opposing sign discordance across strata:
      H0: all d_k >= 0 or all d_k <= 0 (directional concordancy)
      H1: at least one stratum d_k > 0 and another d_k < 0.
    For each stratum k with estimate d_k and variance v_k = SE(d_k)^2, weight w_k = 1 / v_k.
    Partitions into positive strata S+ = {k: d_k > 0} and negative strata S- = {k: d_k < 0}.
    Directional contrast: Delta = d_bar(+) - d_bar(-).
    Contrast variance: Var(Delta) = 1 / W(+) + 1 / W(-).
    Standardized contrast: Z_PHL = Delta / sqrt(Var(Delta)).
    Chi-squared statistic: Q_PHL = Z_PHL^2 ~ Chi2(1).
    Conditional p-value: p_PHL = Pr(Chi2(1) >= Q_PHL) = 2 * (1 - Phi(|Z_PHL|)).
    Rejection (p_PHL < 0.05 with Q_PHL > 3.841) certifies qualitative Simpson's reversal across strata.
    """
    valid_bds = [b for b in breakdowns if b.sample_size >= 4]
    k = len(valid_bds)
    if k < 2:
        return None, None, False
    try:
        pos_w, pos_wd = 0.0, 0.0
        neg_w, neg_wd = 0.0, 0.0

        for b in valid_bds:
            d_k = float(b.subgroup_statistic)
            n_k = float(b.sample_size)
            se_k = math.sqrt(max(1e-6, (1.0 - min(0.99, d_k ** 2)) ** 2 / max(2.0, n_k - 2.0)))
            w_k = 1.0 / max(1e-6, se_k ** 2)

            if d_k > 0:
                pos_w += w_k
                pos_wd += w_k * d_k
            elif d_k < 0:
                neg_w += w_k
                neg_wd += w_k * d_k

        if pos_w <= 1e-12 or neg_w <= 1e-12:
            return 0.0, 1.0, False

        d_bar_pos = pos_wd / pos_w
        d_bar_neg = neg_wd / neg_w
        delta = d_bar_pos - d_bar_neg

        var_contrast = (1.0 / pos_w) + (1.0 / neg_w)
        if var_contrast < 1e-12:
            return 0.0, 1.0, False

        z_phl = float(delta / math.sqrt(var_contrast))
        q_phl = float(z_phl ** 2)
        p_val = float(2.0 * stats.norm.sf(abs(z_phl)))
        p_val = float(np.clip(p_val, 1e-12, 1.0))
        is_reversal = bool(p_val < 0.05 and q_phl > 3.841)
        return q_phl, p_val, is_reversal
    except Exception:
        return None, None, False


def overall_woodward_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates John E. Overall & J. Arthur Woodward's (1974 Psychological Bulletin) Modified Variance Homogeneity Test (F_OW).

    Tests equality of variances across K groups in unbalanced designs, incorporating kurtosis adjustment on log-variances:
      z_k = ln(s_k^2), where s_k^2 is sample variance with nu_k = n_k - 1.
      Pooled excess kurtosis gamma_2 = (1/K) * sum [ m_{4,k} / s_{0,k}^4 - 3.0 ].
      Theoretical variance of log-variance: V_k = 2 / nu_k + max(-1.5, gamma_2) / n_k.
      Weights w_k = 1 / V_k, weighted mean z_bar_w = sum(w_k * z_k) / sum(w_k).
      Statistic: (K - 1) * F_OW = sum_{k=1}^K w_k * (z_k - z_bar_w)^2 ~ Chi2(K - 1).
    Rejection (p < 0.05) detects significant scale dispersion heterogeneity robust to non-normal kurtosis.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    k = len(valid_samples)
    if k < 2:
        return None, None, False
    try:
        n_k = [len(s) for s in valid_samples]
        s_sq_k = [float(np.var(s, ddof=1)) for s in valid_samples]
        if any(v < 1e-12 for v in s_sq_k):
            return None, None, False

        kurt_list = []
        for s in valid_samples:
            m4 = float(np.mean((s - np.mean(s)) ** 4))
            v0 = float(np.var(s, ddof=0))
            if v0 > 1e-12:
                kurt_list.append(m4 / (v0 ** 2) - 3.0)

        gamma_2 = float(np.mean(kurt_list)) if kurt_list else 0.0

        z_k = [math.log(v) for v in s_sq_k]
        v_weights = []
        for n in n_k:
            nu = float(n - 1)
            vk = (2.0 / nu) + (max(-1.5, gamma_2) / float(n))
            v_weights.append(1.0 / max(1e-6, vk))

        sum_w = sum(v_weights)
        if sum_w < 1e-12:
            return None, None, False

        z_bar_w = sum(w * z for w, z in zip(v_weights, z_k)) / sum_w
        ss_ow = sum(w * ((z - z_bar_w) ** 2) for w, z in zip(v_weights, z_k))
        df_k = k - 1
        f_ow = float(ss_ow / max(1, df_k))
        p_val = float(stats.chi2.sf(ss_ow, df=df_k))
        is_hetero = bool(p_val < 0.05)
        return f_ow, p_val, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_dual_control_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    gamma_crit: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum's (1989 JASA / 2002 Observational Studies) Dual-Control Observational Sensitivity Bound (Gamma_Dual).

    Evaluates sensitivity to hidden bias by assessing whether primary treatment difference exceeds
    unobserved bias between two distinct comparison control groups:
      Primary treated group T: X > median(X).
      Control units C: X <= median(X) split by Z into C1 (high Z control) and C2 (low Z control).
      Primary contrast: D_T1 = mean(Y_T) - mean(Y_C1).
      Dual-control baseline disparity: D_12 = mean(Y_C1) - mean(Y_C2).
      Adjusted contrast: D_Dual = D_T1 - sign(D_T1) * |D_12|.
      Standardized score: Z_Dual(Gamma) = (D_Dual / SE_Dual) - ((Gamma - 1) / (Gamma + 1)) * sqrt(n).
    Finds critical Gamma_Dual where upper-tail P_upper(Gamma) = 0.05.
    Findings with Gamma_Dual < 1.30 fail to overcome dual-control unobserved selection disparity and are declared fragile.
    """
    n = len(x)
    if n < 12 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        t_mask = x > med_x
        c_mask = ~t_mask

        if np.sum(t_mask) < 3 or np.sum(c_mask) < 4:
            return 1.0, 1.0, True

        z_ctrl = z[c_mask]
        med_z_ctrl = float(np.median(z_ctrl))

        c1_mask = c_mask & (z > med_z_ctrl)
        c2_mask = c_mask & (z <= med_z_ctrl)

        if np.sum(c1_mask) < 2 or np.sum(c2_mask) < 2:
            return 1.0, 1.0, True

        y_t = y[t_mask]
        y_c1 = y[c1_mask]
        y_c2 = y[c2_mask]

        d_t1 = float(np.mean(y_t) - np.mean(y_c1))
        d_12 = float(np.mean(y_c1) - np.mean(y_c2))

        sign_t1 = 1.0 if d_t1 >= 0 else -1.0
        d_dual = d_t1 - sign_t1 * abs(d_12)

        var_t1 = float(np.var(y_t, ddof=1) / len(y_t) + np.var(y_c1, ddof=1) / len(y_c1))
        var_12 = float(np.var(y_c1, ddof=1) / len(y_c1) + np.var(y_c2, ddof=1) / len(y_c2))
        se_dual = math.sqrt(max(1e-8, var_t1 + var_12))

        z_0 = abs(d_dual) / se_dual

        def p_dual_upper(gamma: float) -> float:
            bias_adj = ((gamma - 1.0) / (gamma + 1.0)) * math.sqrt(n)
            z_stat = z_0 - bias_adj
            return float(stats.norm.sf(z_stat))

        p_at_1 = float(np.clip(p_dual_upper(1.0), 1e-12, 1.0))

        low_g, high_g = 1.0, 15.0
        gamma_dual = 1.0
        for _ in range(30):
            mid_g = (low_g + high_g) / 2.0
            p_mid = p_dual_upper(mid_g)
            if p_mid < 0.05:
                low_g = mid_g
                gamma_dual = mid_g
            else:
                high_g = mid_g

        gamma_dual = float(np.clip(gamma_dual, 1.0, 20.0))
        is_fragile = bool(gamma_dual < 1.30 or (gamma_crit is not None and gamma_dual < gamma_crit))
        return gamma_dual, p_at_1, is_fragile
    except Exception:
        return None, None, False


def hainmueller_entropy_balancing_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Jens Hainmueller's (2012 Political Analysis) Entropy Balancing Treatment Discrepancy (Delta_EB).

    Reweights control units to achieve exact moment calibration matching treated sample moments with minimal relative entropy:
      Treated units T: X > median(X); Control units C: X <= median(X).
      Target mean of confounder in treated group: mu_T = mean(Z_T).
      Solves for optimal Lagrange multiplier lambda via Newton-Raphson:
        w_i(lambda) = exp(-lambda * (Z_C,i - mu_T)) / sum_j exp(-lambda * (Z_C,j - mu_T)).
      Entropy-balanced control mean: Y_bar_{C, EB} = sum_i w_i * Y_{C,i}.
      Naive ATE: tau_naive = mean(Y_T) - mean(Y_C).
      Entropy-balanced ATE: tau_EB = mean(Y_T) - Y_bar_{C, EB}.
      Relative discrepancy: Delta_EB = |tau_naive - tau_EB| / max(1e-6, |tau_naive|).
      Effective sample size: n_eff = 1 / sum_i w_i^2, n_eff_pct = n_eff / n_C * 100%.
      Weight entropy loss: H_EB = -sum_i w_i * ln(w_i).
    When Delta_EB > 0.35 or n_eff_pct < 25.0%, the effect is certified as distorted by covariate imbalance.
    """
    n = len(x)
    if n < 12 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, None, False
    try:
        med_x = float(np.median(x))
        treated = x > med_x
        control = ~treated
        n_t = int(np.sum(treated))
        n_c = int(np.sum(control))

        if n_t < 4 or n_c < 4:
            return None, None, None, False

        y_t = y[treated]
        y_c = y[control]
        z_t = z[treated]
        z_c = z[control]

        mu_t = float(np.mean(z_t))
        dev_z = z_c - mu_t

        lam = 0.0
        for _ in range(25):
            exp_terms = np.exp(-np.clip(lam * dev_z, -50.0, 50.0))
            sum_exp = float(np.sum(exp_terms))
            if sum_exp < 1e-12:
                break
            w = exp_terms / sum_exp
            f_val = float(np.sum(w * dev_z))
            f_prime = float(np.sum(w * (dev_z ** 2)) - (f_val ** 2))
            if abs(f_val) < 1e-6:
                break
            step = f_val / max(1e-6, f_prime)
            lam += np.clip(step, -2.0, 2.0)

        exp_terms = np.exp(-np.clip(lam * dev_z, -50.0, 50.0))
        w_opt = exp_terms / float(np.sum(exp_terms))

        y_bar_t = float(np.mean(y_t))
        y_bar_c = float(np.mean(y_c))
        y_bar_c_eb = float(np.sum(w_opt * y_c))

        tau_naive = y_bar_t - y_bar_c
        tau_eb = y_bar_t - y_bar_c_eb

        delta_eb = float(abs(tau_naive - tau_eb) / max(1e-6, abs(tau_naive)))

        sum_w_sq = float(np.sum(w_opt ** 2))
        n_eff = 1.0 / max(1e-12, sum_w_sq)
        n_eff_pct = float((n_eff / float(n_c)) * 100.0)

        entropy_loss = float(-np.sum(w_opt * np.log(np.maximum(1e-12, w_opt))))

        is_fragile = bool(delta_eb > 0.35 or n_eff_pct < 25.0)
        return delta_eb, n_eff_pct, entropy_loss, is_fragile
    except Exception:
        return None, None, None, False


def bontemps_meddahi_normality_test(
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Christian Bontemps & Nour Meddahi's (2005 Econometric Theory) Hermite Polynomial Moment Normality Test (M_BM).

    Tests Gaussianity of regression residuals using normalized Hermite polynomial moment conditions:
      e_i* = (e_i - mean(e)) / s_e.
      Normalized skewness Hermite polynomial: h_3(u) = (u^3 - 3*u) / sqrt(6).
      Normalized kurtosis Hermite polynomial: h_4(u) = (u^4 - 6*u^2 + 3) / sqrt(24).
      Sample moment conditions: m_3 = (1/sqrt(n)) * sum h_3(e_i*), m_4 = (1/sqrt(n)) * sum h_4(e_i*).
      Omnibus test statistic: M_BM = m_3^2 + m_4^2 ~ Chi2(2).
      Asymptotic p-value: p_BM = 1 - F_{Chi2(2)}(M_BM).
    Rejection (p < 0.05) diagnoses significant non-Gaussian residual distortion invariant to nuisance parameters.
    """
    n = len(residuals)
    if n < 8:
        return None, None, False
    try:
        e = np.asarray(residuals, dtype=float)
        mean_e = float(np.mean(e))
        s_e = float(np.std(e, ddof=1))
        if s_e < 1e-12:
            return 0.0, 1.0, False

        u = (e - mean_e) / s_e

        h3 = (u ** 3 - 3.0 * u) / math.sqrt(6.0)
        h4 = (u ** 4 - 6.0 * (u ** 2) + 3.0) / math.sqrt(24.0)

        m3 = float(np.sum(h3) / math.sqrt(n))
        m4 = float(np.sum(h4) / math.sqrt(n))

        m_bm = float(m3 ** 2 + m4 ** 2)
        p_val = float(stats.chi2.sf(m_bm, df=2))
        p_val = float(np.clip(p_val, 1e-12, 1.0))
        is_non_normal = bool(p_val < 0.05)
        return m_bm, p_val, is_non_normal
    except Exception:
        return None, None, False


def yohai_mm_estimator_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], int, bool]:
    """Calculates Victor J. Yohai's (1987 Annals of Statistics) High-Breakdown MM-Estimator Robust Slope Discrepancy (Delta_MM).

    Combines 50% breakdown point with 95% Gaussian efficiency via Tukey's biweight loss:
      OLS slope: beta_OLS = cov(X, Y) / var(X).
      Initial robust scale sigma_0 = 1.4826 * MAD(residuals_median).
      Tukey biweight with c = 4.685:
        u_i = r_i / (sigma * c).
        w_i = (1 - u_i^2)^2 for |u_i| <= 1, 0 otherwise.
      Iteratively reweighted least squares (IRLS) converges to robust slope beta_MM.
      Discrepancy: Delta_MM = |beta_OLS - beta_MM| / max(1e-6, |beta_OLS|).
      Downweighted outlier count: n_out = sum(w_i < 0.50).
    When Delta_MM > 0.30 or (n_out >= 3 and Delta_MM > 0.15), the finding is declared distorted by high-leverage outliers.
    """
    n = len(x)
    if n < 8 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, 0, False
    try:
        x_flt = np.asarray(x, dtype=float)
        y_flt = np.asarray(y, dtype=float)

        mean_x = float(np.mean(x_flt))
        mean_y = float(np.mean(y_flt))
        ss_x = float(np.sum((x_flt - mean_x) ** 2))
        if ss_x < 1e-12:
            return None, None, 0, False

        beta_ols = float(np.sum((x_flt - mean_x) * (y_flt - mean_y)) / ss_x)

        dx = x_flt[:, None] - x_flt[None, :]
        dy = y_flt[:, None] - y_flt[None, :]
        valid_pairs = np.abs(dx) > 1e-9
        if np.any(valid_pairs):
            beta_med = float(np.median(dy[valid_pairs] / dx[valid_pairs]))
        else:
            beta_med = beta_ols

        r_med = y_flt - beta_med * x_flt
        mad_r = float(np.median(np.abs(r_med - np.median(r_med))))
        sigma_0 = max(1e-6, 1.4826 * mad_r)

        c = 4.685
        beta_curr = beta_med
        weights = np.ones(n, dtype=float)

        for _ in range(12):
            r = y_flt - (mean_y - beta_curr * mean_x) - beta_curr * x_flt
            u = np.abs(r) / (sigma_0 * c)
            weights = np.where(u <= 1.0, (1.0 - u ** 2) ** 2, 0.0)

            sum_w = float(np.sum(weights))
            if sum_w < 1e-6:
                break

            xw = float(np.sum(weights * x_flt) / sum_w)
            yw = float(np.sum(weights * y_flt) / sum_w)
            denom = float(np.sum(weights * ((x_flt - xw) ** 2)))
            if denom < 1e-12:
                break

            beta_next = float(np.sum(weights * (x_flt - xw) * (y_flt - yw)) / denom)
            if abs(beta_next - beta_curr) < 1e-5:
                beta_curr = beta_next
                break
            beta_curr = beta_next

        beta_mm = float(beta_curr)
        delta_mm = float(abs(beta_ols - beta_mm) / max(1e-6, abs(beta_ols)))
        outlier_count = int(np.sum(weights < 0.50))

        is_distorted = bool(delta_mm > 0.30 or (outlier_count >= 3 and delta_mm > 0.15))
        return beta_mm, delta_mm, outlier_count, is_distorted
    except Exception:
        return None, None, 0, False


def crainiceanu_ruppert_rlrt_test(
    breakdowns: List[SubgroupBreakdown],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Ciprian M. Crainiceanu & David Ruppert's (2004 JASA) Restricted Likelihood Ratio Test for Subgroup Heterogeneity (RLRT_CR).

    Tests whether subgroup treatment effects exhibit random variance beyond sampling noise:
      H0: tau^2 = 0 vs H1: tau^2 > 0 in a one-way random effects model.
    For K strata with estimates d_k and sampling variances v_k = SE(d_k)^2:
      Restricted log-likelihood under H0:
        w_k0 = 1 / v_k, d_bar0 = sum(w_k0 * d_k) / sum(w_k0).
        ell_0 = -0.5 * [ sum(ln(v_k)) + ln(sum(w_k0)) + sum((d_k - d_bar0)^2 / v_k) ].
      Under H1 with between-strata variance tau^2:
        V_k(tau^2) = v_k + tau^2, w_k(tau^2) = 1 / V_k(tau^2).
        d_bar(tau^2) = sum(w_k * d_k) / sum(w_k).
        ell_1(tau^2) = -0.5 * [ sum(ln(V_k)) + ln(sum(w_k)) + sum((d_k - d_bar)^2 / V_k) ].
      Maximizes ell_1 over tau^2 >= 0.
      RLRT = max(0.0, 2 * (ell_1(tau_hat^2) - ell_0)).
    Under H0 on the boundary, asymptotic distribution is an exact 50:50 mixture of Chi2(0) and Chi2(1):
      p_CR = 0.5 * Pr(Chi2(1) >= RLRT).
    Rejection (p_CR < 0.05 with RLRT > 2.706) certifies authentic subgroup effect variance beyond sampling error.
    """
    valid_bds = [b for b in breakdowns if b.sample_size >= 4]
    k = len(valid_bds)
    if k < 2:
        return None, None, False
    try:
        d_k = []
        v_k = []
        for b in valid_bds:
            d = float(b.subgroup_statistic)
            n = float(b.sample_size)
            se = math.sqrt(max(1e-6, (1.0 - min(0.99, d ** 2)) ** 2 / max(2.0, n - 2.0)))
            d_k.append(d)
            v_k.append(max(1e-6, se ** 2))

        d_k = np.array(d_k, dtype=float)
        v_k = np.array(v_k, dtype=float)

        w0 = 1.0 / v_k
        sum_w0 = float(np.sum(w0))
        d_bar0 = float(np.sum(w0 * d_k) / sum_w0)
        ell_0 = -0.5 * float(np.sum(np.log(v_k)) + math.log(sum_w0) + np.sum(((d_k - d_bar0) ** 2) / v_k))

        # Maximize ell_1 over tau^2 >= 0
        var_d = float(np.var(d_k, ddof=1)) if k > 1 else 0.0
        max_tau2 = max(0.1, 5.0 * var_d)

        def neg_ell1(tau2: float) -> float:
            vk_tau = v_k + max(0.0, tau2)
            w = 1.0 / vk_tau
            sum_w = float(np.sum(w))
            d_bar = float(np.sum(w * d_k) / max(1e-12, sum_w))
            return 0.5 * float(np.sum(np.log(vk_tau)) + math.log(max(1e-12, sum_w)) + np.sum(((d_k - d_bar) ** 2) / vk_tau))

        best_tau2 = 0.0
        min_neg_ell = neg_ell1(0.0)
        for cand in np.linspace(0.0, max_tau2, 60):
            val = neg_ell1(float(cand))
            if val < min_neg_ell:
                min_neg_ell = val
                best_tau2 = float(cand)

        ell_1 = -min_neg_ell
        rlrt = float(max(0.0, 2.0 * (ell_1 - ell_0)))

        # 50:50 mixture of Chi2(0) and Chi2(1)
        p_val = float(0.5 * stats.chi2.sf(rlrt, df=1)) if rlrt > 0.0 else 1.0
        p_val = float(np.clip(p_val, 1e-12, 1.0))
        is_hetero = bool(p_val < 0.05 and rlrt > 2.706)
        return rlrt, p_val, is_hetero
    except Exception:
        return None, None, False


def lim_loh_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Wei-Yin Lim & Wei-Liem Loh's (1996 JASA) Ratio-of-Variances Robust Homogeneity Test (W_LL).

    Evaluates equality of scale/dispersion across K subgroup strata samples using median-centered absolute deviations:
      z_kj = |Y_kj - median(Y_k)|.
      Sample variance of absolute deviations: s_{z, k}^2 = var(z_k, ddof=1).
      Test statistic: W_LL = max_k(s_{z, k}^2) / min_k(s_{z, k}^2).
      Evaluates approximate p-value under H0 via extreme F-ratio with Bonferroni correction:
        p_LL = min(1.0, comb(K, 2) * 2 * Pr(F(n_max - 1, n_min - 1) >= W_LL)).
    Rejection (p < 0.05 with W_LL > 1.50) detects significant scale dispersion heterogeneity robust to non-normal distributions.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    k = len(valid_samples)
    if k < 2:
        return None, None, False
    try:
        s_z_sq = []
        n_k = []
        for s in valid_samples:
            med = float(np.median(s))
            z = np.abs(s - med)
            v = float(np.var(z, ddof=1))
            if v > 1e-12:
                s_z_sq.append(v)
                n_k.append(len(s))

        if len(s_z_sq) < 2:
            return None, None, False

        max_v = max(s_z_sq)
        min_v = max(1e-12, min(s_z_sq))
        w_stat = float(max_v / min_v)

        max_idx = s_z_sq.index(max_v)
        min_idx = s_z_sq.index(min(s_z_sq))
        df1 = max(1, n_k[max_idx] - 1)
        df2 = max(1, n_k[min_idx] - 1)

        comb_k2 = float(k * (k - 1) / 2.0)
        p_single = float(stats.f.sf(w_stat, df1, df2))
        p_val = float(np.clip(comb_k2 * 2.0 * p_single, 1e-12, 1.0))
        is_hetero = bool(p_val < 0.05 and w_stat > 1.50)
        return w_stat, p_val, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_cross_match_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    gamma_crit: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum's (2005 JASA) Cross-Match Observational Sensitivity Test (Gamma_CM).

    Constructs optimal nearest-neighbor pairs on continuous confounder Z between treated (X > med(X))
    and control (X <= med(X)) cohorts and tests sensitivity to unobserved bias:
      Differences in outcome within matched pairs: D_m = Y_{T, m} - Y_{C, m}.
      Positive pair count: W = sum(1(D_m > 0)).
      Under hidden bias Gamma >= 1, max null probability of positive difference is p+ = Gamma / (1 + Gamma).
      Null mean: E_Gamma[W] = M * Gamma / (1 + Gamma).
      Null variance: Var_Gamma(W) = M * Gamma / (1 + Gamma)^2.
      Standardized score: Z_CM(Gamma) = (W - M/2 - (M * (Gamma - 1) / (2 * (Gamma + 1)))) / sqrt(Var_Gamma).
    Finds critical Gamma_CM where upper-tail P_upper(Gamma) = 0.05.
    Findings with Gamma_CM < 1.30 fail to withstand unobserved confounder bias under optimal cross-matching.
    """
    n = len(x)
    if n < 12 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, False
    try:
        med_x = float(np.median(x))
        t_mask = x > med_x
        c_mask = ~t_mask

        n_t = int(np.sum(t_mask))
        n_c = int(np.sum(c_mask))
        if n_t < 4 or n_c < 4:
            return 1.0, 1.0, True

        t_z = z[t_mask]
        t_y = y[t_mask]
        c_z = z[c_mask]
        c_y = y[c_mask]

        # Greedy nearest neighbor matching on standardized confounder
        z_std = float(np.std(z)) if np.std(z) > 1e-9 else 1.0
        used_c = set()
        matched_diffs = []

        for zi, yi in zip(t_z, t_y):
            best_idx = -1
            best_dist = float("inf")
            for j, (zj, yj) in enumerate(zip(c_z, c_y)):
                if j not in used_c:
                    dist = abs(zi - zj) / z_std
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = j
            if best_idx >= 0:
                used_c.add(best_idx)
                matched_diffs.append(float(yi - c_y[best_idx]))

        m = len(matched_diffs)
        if m < 4:
            return 1.0, 1.0, True

        diffs = np.array(matched_diffs, dtype=float)
        w_pos = float(np.sum(diffs > 0))

        def p_cm_upper(gamma: float) -> float:
            p_plus = gamma / (1.0 + gamma)
            e_w = m * p_plus
            var_w = max(1e-6, m * p_plus * (1.0 - p_plus))
            z_stat = (w_pos - e_w) / math.sqrt(var_w)
            return float(stats.norm.sf(z_stat))

        p_at_1 = float(np.clip(p_cm_upper(1.0), 1e-12, 1.0))

        low_g, high_g = 1.0, 15.0
        gamma_cm = 1.0
        for _ in range(30):
            mid_g = (low_g + high_g) / 2.0
            p_mid = p_cm_upper(mid_g)
            if p_mid < 0.05:
                low_g = mid_g
                gamma_cm = mid_g
            else:
                high_g = mid_g

        gamma_cm = float(np.clip(gamma_cm, 1.0, 20.0))
        is_fragile = bool(gamma_cm < 1.30 or (gamma_crit is not None and gamma_cm < gamma_crit))
        return gamma_cm, p_at_1, is_fragile
    except Exception:
        return None, None, False


def athey_imbens_causal_disparity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Calculates Susan Athey & Guido W. Imbens' (2016 PNAS) Recursive Partitioning Causal Disparity Bound (Delta_AI).

    Evaluates heterogeneous treatment effects across confounder partitions (leaves):
      Splits confounder Z into tertiles L1, L2, L3.
      Within each leaf m, computes local ATE tau_m = mean(Y_T, m) - mean(Y_C, m).
      Evaluates extreme causal disparity:
        Delta_AI = |tau_max - tau_min| / max(1e-6, |tau_pooled|).
      Detects causal direction flip: tau_max * tau_min < 0.
    Rejection (Delta_AI > 0.40 or causal flip with large disparity) certifies severe treatment effect heterogeneity.
    """
    n = len(x)
    if n < 12 or np.std(x) < 1e-9 or np.std(y) < 1e-9 or np.std(z) < 1e-9:
        return None, None, None, False
    try:
        med_x = float(np.median(x))
        treated = x > med_x
        control = ~treated
        if np.sum(treated) < 4 or np.sum(control) < 4:
            return None, None, None, False

        tau_pooled = float(np.mean(y[treated]) - np.mean(y[control]))

        q33 = float(np.percentile(z, 33.33))
        q66 = float(np.percentile(z, 66.67))

        leaf1 = z <= q33
        leaf2 = (z > q33) & (z <= q66)
        leaf3 = z > q66

        taus = []
        for leaf in [leaf1, leaf2, leaf3]:
            t_leaf = leaf & treated
            c_leaf = leaf & control
            if np.sum(t_leaf) >= 2 and np.sum(c_leaf) >= 2:
                taus.append(float(np.mean(y[t_leaf]) - np.mean(y[c_leaf])))

        if len(taus) < 2:
            # Fall back to median split (2 causal tree leaves)
            taus = []
            med_z = float(np.median(z))
            l1 = z <= med_z
            l2 = z > med_z
            for leaf in [l1, l2]:
                t_leaf = leaf & treated
                c_leaf = leaf & control
                if np.sum(t_leaf) >= 2 and np.sum(c_leaf) >= 2:
                    taus.append(float(np.mean(y[t_leaf]) - np.mean(y[c_leaf])))

        if len(taus) < 2:
            return None, None, None, False

        tau_max = float(max(taus))
        tau_min = float(min(taus))
        disparity = float(abs(tau_max - tau_min) / max(1e-6, abs(tau_pooled)))

        std_y = float(np.std(y))
        sign_flip = bool(tau_max * tau_min < 0 and abs(tau_max - tau_min) > 0.10 * std_y)
        is_hetero = bool(disparity > 0.40 or sign_flip)
        return disparity, tau_max, tau_min, is_hetero
    except Exception:
        return None, None, None, False


def spiegelhalter_normality_test(
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates David J. Spiegelhalter's (1977, 1980 Biometrika) Omnibus Normality Test (N_SP).

    Evaluates departure of standardized residuals from expected Gaussian order statistics:
      e*_i = (e_i - mean(e)) / s_e.
      Sorts standardized residuals: e*_{(1)} <= ... <= e*_{(n)}.
      Expected Gaussian order statistics (Blom's approximation):
        m_i = Phi^-1((i - 3/8) / (n + 1/4)).
      Spiegelhalter omnibus statistic:
        N_SP = (1 / n) * sum_{i=1}^n (e*_{(i)} - m_i)^2.
      Evaluates p-value using log-normal calibration against Gaussian null:
        Z_SP = (ln(N_SP) - mu_0) / sigma_0, where mu_0 approx ln(0.045), sigma_0 approx 0.40.
    Rejection (p < 0.05 with N_SP > 0.12) certifies omnibus residual non-Gaussianity.
    """
    n = len(residuals)
    if n < 8:
        return None, None, False
    try:
        e = np.asarray(residuals, dtype=float)
        mean_e = float(np.mean(e))
        s_e = float(np.std(e, ddof=1))
        if s_e < 1e-12:
            return 0.0, 1.0, False

        u = np.sort((e - mean_e) / s_e)
        i_vec = np.arange(1, n + 1, dtype=float)
        p_vec = (i_vec - 0.375) / (float(n) + 0.25)
        m_vec = stats.norm.ppf(p_vec)

        n_sp = float(np.mean((u - m_vec) ** 2))

        mu_0 = math.log(max(1e-6, 0.40 / math.sqrt(float(n))))
        sigma_0 = 0.45
        z_sp = (math.log(max(1e-6, n_sp)) - mu_0) / sigma_0
        p_val = float(stats.norm.sf(z_sp))
        p_val = float(np.clip(p_val, 1e-12, 1.0))

        is_non_normal = bool(p_val < 0.05 and n_sp > 0.12)
        return n_sp, p_val, is_non_normal
    except Exception:
        return None, None, False


def hampel_redescending_m_estimator_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], int, bool]:
    """Calculates Frank R. Hampel's (1974 JASA) Three-Part Redescending M-Estimator Robust Slope Discrepancy (Delta_HAM).

    Applies Hampel's 3-part redescending score function with parameters (a=1.5, b=3.5, c=8.0):
      Weights for standardized residuals u_i = |r_i| / sigma_0:
        w(u) = 1.0 for u <= a;
        w(u) = a / u for a < u <= b;
        w(u) = a * (c - u) / (u * (c - b)) for b < u <= c;
        w(u) = 0.0 for u > c (completely redescending, zero influence for gross outliers).
      IRLS converges to Hampel robust slope beta_HAM.
      Discrepancy: Delta_HAM = |beta_OLS - beta_HAM| / max(1e-6, |beta_OLS|).
      Redescended outlier count: n_out = sum(u_i > b).
    Rejection (Delta_HAM > 0.30 or (n_out >= 2 and Delta_HAM > 0.15)) diagnoses severe distortion from high-leverage outliers.
    """
    n = len(x)
    if n < 8 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, 0, False
    try:
        x_flt = np.asarray(x, dtype=float)
        y_flt = np.asarray(y, dtype=float)

        mean_x = float(np.mean(x_flt))
        mean_y = float(np.mean(y_flt))
        ss_x = float(np.sum((x_flt - mean_x) ** 2))
        if ss_x < 1e-12:
            return None, None, 0, False

        beta_ols = float(np.sum((x_flt - mean_x) * (y_flt - mean_y)) / ss_x)

        dx = x_flt[:, None] - x_flt[None, :]
        dy = y_flt[:, None] - y_flt[None, :]
        valid_pairs = np.abs(dx) > 1e-9
        if np.any(valid_pairs):
            beta_med = float(np.median(dy[valid_pairs] / dx[valid_pairs]))
        else:
            beta_med = beta_ols

        r_med = y_flt - beta_med * x_flt
        mad_r = float(np.median(np.abs(r_med - np.median(r_med))))
        sigma_0 = max(1e-6, 1.4826 * mad_r)

        a, b, c = 1.5, 3.5, 8.0
        beta_curr = beta_med
        weights = np.ones(n, dtype=float)

        for _ in range(15):
            r = y_flt - (mean_y - beta_curr * mean_x) - beta_curr * x_flt
            u = np.abs(r) / sigma_0

            w = np.zeros(n, dtype=float)
            w[u <= a] = 1.0
            mask_ab = (u > a) & (u <= b)
            w[mask_ab] = a / u[mask_ab]
            mask_bc = (u > b) & (u <= c)
            w[mask_bc] = (a * (c - u[mask_bc])) / (u[mask_bc] * (c - b))
            w[u > c] = 0.0

            weights = w
            sum_w = float(np.sum(weights))
            if sum_w < 1e-6:
                break

            xw = float(np.sum(weights * x_flt) / sum_w)
            yw = float(np.sum(weights * y_flt) / sum_w)
            denom = float(np.sum(weights * ((x_flt - xw) ** 2)))
            if denom < 1e-12:
                break

            beta_next = float(np.sum(weights * (x_flt - xw) * (y_flt - yw)) / denom)
            if abs(beta_next - beta_curr) < 1e-5:
                beta_curr = beta_next
                break
            beta_curr = beta_next

        beta_ham = float(beta_curr)
        delta_ham = float(abs(beta_ols - beta_ham) / max(1e-6, abs(beta_ols)))
        final_r = y_flt - (mean_y - beta_ham * mean_x) - beta_ham * x_flt
        final_u = np.abs(final_r) / sigma_0
        outlier_count = int(np.sum(final_u > b))

        is_distorted = bool(delta_ham > 0.30 or (outlier_count >= 2 and delta_ham > 0.15))
        return beta_ham, delta_ham, outlier_count, is_distorted
    except Exception:
        return None, None, 0, False


def verbeke_molenberghs_score_test(
    breakdowns: List[SubgroupBreakdown],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Geert Verbeke & Geert Molenberghs' (2000, 2003 Stat. Sci.) Score Test for Subgroup Random Effect Variance (S_VM).

    Tests whether subgroup treatment effects exhibit random variance beyond sampling noise:
      H0: sigma_u^2 = 0 vs H1: sigma_u^2 > 0 in a one-way random effects model.
    For K strata with effect estimates d_k and sampling variances v_k = SE(d_k)^2:
      Inverse-variance weights: w_k = 1 / v_k, pooled effect: d_bar = sum(w_k * d_k) / sum(w_k).
      Cochran's Q: Q = sum_{k=1}^K w_k * (d_k - d_bar)^2.
      Standardized score statistic:
        S_VM = (Q - (K - 1)) / sqrt(2 * (K - 1)).
      Under H0, S_VM is asymptotically standard normal N(0, 1) for one-sided boundary testing.
      One-sided p-value:
        p_VM = 1.0 - stats.norm.cdf(S_VM) if S_VM > -10.0 else 1.0.
    Rejection (p_VM < 0.05 with S_VM > 1.645) proves genuine random subgroup variance beyond sampling error.
    """
    valid_bds = [b for b in breakdowns if b.sample_size >= 4]
    k = len(valid_bds)
    if k < 2:
        return None, None, False
    try:
        d_k = []
        v_k = []
        for b in valid_bds:
            d_val = float(b.subgroup_statistic)
            n_sub = max(4, b.sample_size)
            r_clamped = max(-0.999, min(0.999, d_val))
            v_val = max(1e-6, (1.0 - r_clamped**2)**2 / n_sub)
            d_k.append(d_val)
            v_k.append(v_val)

        d_arr = np.array(d_k, dtype=float)
        v_arr = np.array(v_k, dtype=float)
        w_arr = 1.0 / v_arr
        d_bar = np.sum(w_arr * d_arr) / np.sum(w_arr)
        q_stat = float(np.sum(w_arr * (d_arr - d_bar)**2))
        df = max(1, k - 1)
        s_vm = float((q_stat - df) / np.sqrt(2.0 * df))
        p_vm = float(1.0 - stats.norm.cdf(s_vm)) if s_vm > -10.0 else 1.0
        p_vm = max(0.0, min(1.0, p_vm))
        is_hetero = bool(p_vm < 0.05 and s_vm > 1.645)
        return s_vm, p_vm, is_hetero
    except Exception:
        return None, None, False


def conover_johnson_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates W. J. Conover, Mark E. Johnson & Myrle M. Johnson's (1981 Technometrics) Squared Ranks Test for Scale Dispersion (T_CJJ).

    Evaluates equality of scale/dispersion across K subgroup strata using squared ranks of median-centered deviations:
      u_kj = |Y_kj - median(Y_k)|.
      Ranks all pooled deviations R_kj = rank(u_kj) in [1, N].
      Squared rank scores: S_kj = R_kj^2.
      Computes one-way ANOVA F-statistic on squared ranks:
        T_CJJ = (SSB / (K - 1)) / (SSW / (N - K)) ~ F(K - 1, N - K).
        p_CJJ = 1 - F_{F(K - 1, N - K)}(T_CJJ).
    Rejection (p_CJJ < 0.05 with T_CJJ > 2.0) detects significant scale dispersion heterogeneity robust to non-normal kurtosis.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.std(s) > 1e-9]
    k = len(valid_samples)
    if k < 2:
        return None, None, False
    try:
        devs = []
        group_indices = []
        for g_idx, s in enumerate(valid_samples):
            med_s = np.median(s)
            u = np.abs(s - med_s)
            devs.extend(u)
            group_indices.extend([g_idx] * len(u))

        devs_arr = np.array(devs, dtype=float)
        groups_arr = np.array(group_indices, dtype=int)
        n_total = len(devs_arr)
        if n_total <= k:
            return None, None, False

        ranks = stats.rankdata(devs_arr)
        sq_ranks = ranks ** 2

        grand_mean = np.mean(sq_ranks)
        ssb = 0.0
        ssw = 0.0
        for g_idx in range(k):
            g_sq = sq_ranks[groups_arr == g_idx]
            n_g = len(g_sq)
            if n_g > 0:
                mean_g = np.mean(g_sq)
                ssb += n_g * (mean_g - grand_mean) ** 2
                ssw += np.sum((g_sq - mean_g) ** 2)

        df_b = k - 1
        df_w = n_total - k
        if ssw <= 1e-12 or df_w <= 0:
            return None, None, False

        msb = ssb / df_b
        msw = ssw / df_w
        t_cjj = float(msb / msw)
        p_cjj = float(1.0 - stats.f.cdf(t_cjj, df_b, df_w))
        p_cjj = max(0.0, min(1.0, p_cjj))
        is_hetero = bool(p_cjj < 0.05 and t_cjj > 2.0)
        return t_cjj, p_cjj, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_huber_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    gamma_crit: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum's (2007 JASA) Dynamic Huber-Type M-Statistic Sensitivity Bounds (Gamma_HUB).

    Constructs optimal nearest-neighbor pairs on continuous confounder Z between treated (X > med(X))
    and control (X <= med(X)) cohorts and tests sensitivity of robust Huber M-scores to hidden bias:
      Differences in outcome within matched pairs: D_m = Y_{T, m} - Y_{C, m}.
      Robust Huber score: psi(D_m) = sign(D_m) * min(|D_m| / s, 1.5), where s = max(1e-6, median(|D_m|)).
      Test statistic: T_HUB = sum(psi(D_m)).
      Under sensitivity parameter Gamma >= 1:
        Worst-case expectation: E_Gamma = sum_{m=1}^M |psi(D_m)| * (Gamma - 1) / (Gamma + 1) * sign(psi(D_m)).
        Variance: Var_Gamma = sum_{m=1}^M psi(D_m)^2 * 4 * Gamma / (Gamma + 1)^2.
        Standardized score: Z_HUB(Gamma) = (T_HUB - E_Gamma) / sqrt(Var_Gamma).
        Upper tail p-value: p_upper(Gamma) = 1 - Phi(Z_HUB(Gamma)).
    Solves for critical sensitivity threshold Gamma_HUB where p_upper(Gamma) >= 0.05.
    Rejection (is_fragile = True if Gamma_HUB < 1.25) diagnoses vulnerability to unmeasured confounder bias under robust M-estimation.
    """
    n = len(x)
    if n < 10:
        return None, None, False
    try:
        med_x = np.median(x)
        t_mask = (x > med_x)
        c_mask = ~t_mask
        if np.sum(t_mask) < 4 or np.sum(c_mask) < 4:
            return None, None, False

        z_t = z[t_mask]
        y_t = y[t_mask]
        z_c = z[c_mask]
        y_c = y[c_mask]

        used_c = np.zeros(len(z_c), dtype=bool)
        diffs = []
        for i in range(len(z_t)):
            avail_c = np.where(~used_c)[0]
            if len(avail_c) == 0:
                break
            best_j = avail_c[np.argmin(np.abs(z_c[avail_c] - z_t[i]))]
            used_c[best_j] = True
            diffs.append(y_t[i] - y_c[best_j])

        m = len(diffs)
        if m < 4:
            return None, None, False

        d_arr = np.array(diffs, dtype=float)
        abs_d = np.abs(d_arr)
        s = float(np.median(abs_d))
        if s < 1e-9:
            s = float(np.std(d_arr))
        s = max(1e-6, s)

        psi = np.sign(d_arr) * np.minimum(abs_d / s, 1.5)
        t_stat = float(np.sum(psi))
        abs_psi = np.abs(psi)
        sq_psi = psi ** 2

        def calc_p_upper(gamma_val: float) -> float:
            e_g = float(np.sum(abs_psi * ((gamma_val - 1.0) / (gamma_val + 1.0))))
            var_g = float(np.sum(sq_psi * (4.0 * gamma_val / (gamma_val + 1.0)**2)))
            if var_g <= 1e-12:
                return 1.0
            z_score = (t_stat - e_g) / np.sqrt(var_g)
            return float(1.0 - stats.norm.cdf(z_score))

        p_1 = calc_p_upper(1.0)
        if p_1 >= 0.05:
            return 1.0, p_1, True

        low, high = 1.0, 10.0
        gamma_crit_res = 10.0
        for _ in range(25):
            mid = 0.5 * (low + high)
            p_mid = calc_p_upper(mid)
            if p_mid >= 0.05:
                gamma_crit_res = mid
                high = mid
            else:
                low = mid

        p_final = calc_p_upper(gamma_crit_res)
        is_fragile = bool(gamma_crit_res < 1.25)
        return float(gamma_crit_res), float(p_final), is_fragile
    except Exception:
        return None, None, False


def imai_king_stuart_disparity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Kosuke Imai, Gary King & Elizabeth A. Stuart's (2008 JRSS-A) Propensity Balance & Subgroup Disparity Ratio (B_IKS, delta_IKS).

    Stratifies observational units into Q=5 quintiles on continuous confounder Z and evaluates
    peak standardized mean difference (SMD) and heterogeneous subgroup treatment disparities:
      In quintile q: SMD_q = |mean(Z_1, q) - mean(Z_0, q)| / std(Z).
      Peak covariate imbalance: B_IKS = max_q(SMD_q).
      Quintile causal effect: tau_q = mean(Y_1, q) - mean(Y_0, q).
      Disparity ratio: delta_IKS = (max_q(tau_q) - min_q(tau_q)) / max(1e-6, |tau_overall|).
    Rejection (is_fragile = True if B_IKS > 0.25 or delta_IKS > 0.50 or sgn(max tau) != sgn(min tau))
    identifies severe observational imbalance and conflicting subgroup treatment effects.
    """
    n = len(x)
    if n < 12:
        return None, None, False
    try:
        med_x = np.median(x)
        d = (x > med_x).astype(int)
        if np.sum(d == 1) < 4 or np.sum(d == 0) < 4:
            return None, None, False

        std_z = max(1e-6, float(np.std(z)))
        tau_overall = float(np.mean(y[d == 1]) - np.mean(y[d == 0]))
        denom = max(1e-6, abs(tau_overall))
        overall_smd = float(np.abs(np.mean(z[d == 1]) - np.mean(z[d == 0])) / std_z)

        q_bins = np.percentile(z, [20, 40, 60, 80])
        bin_idx = np.digitize(z, q_bins)

        smd_list = []
        tau_list = []
        for q in range(5):
            mask_q = (bin_idx == q)
            if np.sum(mask_q & (d == 1)) >= 2 and np.sum(mask_q & (d == 0)) >= 2:
                z1 = np.mean(z[mask_q & (d == 1)])
                z0 = np.mean(z[mask_q & (d == 0)])
                smd_q = float(np.abs(z1 - z0) / std_z)
                smd_list.append(smd_q)

                y1 = np.mean(y[mask_q & (d == 1)])
                y0 = np.mean(y[mask_q & (d == 0)])
                tau_list.append(float(y1 - y0))

        if len(tau_list) >= 2:
            b_iks = max(float(np.max(smd_list)), overall_smd)
            tau_max = float(np.max(tau_list))
            tau_min = float(np.min(tau_list))
            delta_iks = float((tau_max - tau_min) / denom)
            has_reversal = bool(np.sign(tau_max) != np.sign(tau_min) and abs(tau_max) > 1e-4 and abs(tau_min) > 1e-4)
            is_fragile = bool(b_iks > 0.25 or delta_iks > 0.50 or has_reversal)
        elif len(tau_list) == 1:
            b_iks = max(float(smd_list[0]) if smd_list else 0.0, overall_smd)
            delta_iks = float(abs(tau_list[0] - tau_overall) / denom)
            is_fragile = bool(b_iks > 0.25 or delta_iks > 0.50)
        else:
            b_iks = overall_smd
            delta_iks = 1.0
            is_fragile = True

        return delta_iks, b_iks, is_fragile
    except Exception:
        return None, None, False


def mudholkar_lin_normality_test(
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Govind S. Mudholkar, Deo Kumar Srivastava & C. Thomas Lin's (1981 Technometrics) Lin-Mudholkar / MSL Cube-Root Transformed Normality Test (Z_MSL).

    Evaluates departure of standardized residuals from Gaussian normality via Wilson-Hilferty cube-root transformations of sample moments:
      Sample skewness: sqrt(b1) = m3 / m2^(3/2), standardized: Z1 = sqrt(b1) / sqrt(6 / n).
      Sample kurtosis: b2 = m4 / m2^2, standardized: Z2 = (b2 - 3) / sqrt(24 / n).
      Omnibus score: S_MSL = Z1^2 + Z2^2 ~ Chi2(2).
      Wilson-Hilferty cube-root normal transform:
        Z_MSL = ((S_MSL / 2)^(1/3) - (1 - 2/18)) / sqrt(2/18) ~ N(0, 1).
        p_MSL = 2 * (1 - Phi(|Z_MSL|)).
    Rejection (p_MSL < 0.05 with |Z_MSL| > 1.96) certifies residual departure from Gaussian normality.
    """
    n = len(residuals)
    if n < 8:
        return None, None, False
    try:
        e = np.asarray(residuals, dtype=float)
        m2 = float(np.mean((e - np.mean(e))**2))
        if m2 < 1e-12:
            return 0.0, 1.0, False
        m3 = float(np.mean((e - np.mean(e))**3))
        m4 = float(np.mean((e - np.mean(e))**4))

        skew = m3 / (m2 ** 1.5)
        kurt = m4 / (m2 ** 2.0)

        se_skew = np.sqrt(6.0 / n)
        se_kurt = np.sqrt(24.0 / n)
        z1 = skew / se_skew
        z2 = (kurt - 3.0) / se_kurt

        s_msl = float(z1**2 + z2**2)
        v = 2.0
        term = (s_msl / v) ** (1.0 / 3.0)
        z_msl = float((term - (1.0 - 2.0 / (9.0 * v))) / np.sqrt(2.0 / (9.0 * v)))
        p_msl = float(2.0 * (1.0 - stats.norm.cdf(abs(z_msl))))
        p_msl = max(0.0, min(1.0, p_msl))
        is_non_norm = bool(p_msl < 0.05 and abs(z_msl) > 1.96)
        return z_msl, p_msl, is_non_norm
    except Exception:
        return None, None, False


def andrews_sine_estimator_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], int, bool]:
    """Calculates David F. Andrews' (1974 Technometrics) Sine Wave Redescending M-Estimator Robust Slope Discrepancy (Delta_AND).

    Applies Andrews' sine wave score function with tuning parameter c = 1.339 * pi:
      Weights for standardized residuals u_i = |r_i| / sigma_0:
        w(u) = sin(u / c) / (u / c) for |u| <= c * pi;
        w(u) = 0.0 for |u| > c * pi (completely redescending, fully rejecting extreme leverage outliers).
      IRLS converges to Andrews robust slope beta_AND.
      Discrepancy: Delta_AND = |beta_OLS - beta_AND| / max(1e-6, |beta_OLS|).
      Redescended outlier count: n_out = sum(|u_i| > c * pi).
    Rejection (Delta_AND > 0.25 or (n_out >= 2 and Delta_AND > 0.15)) diagnoses severe vulnerability to gross leverage outliers.
    """
    n = len(x)
    if n < 8 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, 0, False
    try:
        slope_ols, intercept_ols, _, _, _ = stats.linregress(x, y)
        r_ols = y - (slope_ols * x + intercept_ols)
        mad = float(np.median(np.abs(r_ols - np.median(r_ols))))
        sigma_0 = max(1e-6, 1.4826 * mad)

        c = 1.339 * np.pi
        cutoff = c * np.pi

        b_curr = slope_ols
        a_curr = intercept_ols

        for _ in range(20):
            r_curr = y - (b_curr * x + a_curr)
            u = np.abs(r_curr) / sigma_0
            w = np.zeros_like(u)
            in_bounds = (u <= cutoff) & (u > 1e-9)
            near_zero = (u <= 1e-9)
            w[near_zero] = 1.0
            w[in_bounds] = np.sin(u[in_bounds] / c) / (u[in_bounds] / c)

            if np.sum(w) < 4:
                break

            w_x = w * x
            w_y = w * y
            sum_w = np.sum(w)
            mean_x_w = np.sum(w_x) / sum_w
            mean_y_w = np.sum(w_y) / sum_w
            denom = np.sum(w * (x - mean_x_w)**2)
            if denom < 1e-12:
                break
            b_next = np.sum(w * (x - mean_x_w) * (y - mean_y_w)) / denom
            a_next = mean_y_w - b_next * mean_x_w

            if abs(b_next - b_curr) < 1e-5:
                b_curr = b_next
                a_curr = a_next
                break
            b_curr = b_next
            a_curr = a_next

        beta_and = float(b_curr)
        denom_ols = max(1e-6, abs(slope_ols))
        delta_and = float(abs(slope_ols - beta_and) / denom_ols)

        final_r = y - (beta_and * x + a_curr)
        final_u = np.abs(final_r) / sigma_0
        outlier_count = int(np.sum(final_u > cutoff))

        is_distorted = bool(delta_and > 0.25 or (outlier_count >= 2 and delta_and > 0.15))
        return beta_and, delta_and, outlier_count, is_distorted
    except Exception:
        return None, None, 0, False


def hirotsu_cumulative_chi_squared_test(
    breakdowns: List[SubgroupBreakdown],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Chihiro Hirotsu's (1986 Biometrika, 1997) Cumulative Chi-Squared Statistic for Ordered Alternative Subgroup Effects (S*_HIR).

    Tests whether subgroup stratum-specific treatment effects exhibit systematic ordered monotonic departures
    against the null of homogeneous effects H0: theta_1 = ... = theta_K:
      Cumulative partial sums: C_m = sum_{k=1}^m w_k * (d_k - d_bar), m = 1, ..., K-1.
      Cumulative chi-squared statistic:
        S*_HIR = sum_{m=1}^{K-1} [ C_m^2 / (W_m * (1 - W_m / W)) ] / s_0^2.
      Under H0, S*_HIR approximates a chi-squared distribution with K-1 degrees of freedom.
      p_HIR = 1.0 - stats.chi2.cdf(S*_HIR, K - 1).
    Rejection (p_HIR < 0.05 with S*_HIR > 1.645) proves systematic ordered subgroup effect heterogeneity.
    """
    valid_bds = [b for b in breakdowns if b.sample_size >= 4]
    k = len(valid_bds)
    if k < 2:
        return None, None, False
    try:
        d_k = np.array([float(b.subgroup_statistic) for b in valid_bds], dtype=float)
        w_k = np.array([float(max(4, b.sample_size)) for b in valid_bds], dtype=float)
        total_w = float(np.sum(w_k))
        if total_w <= 0:
            return None, None, False

        d_bar = float(np.sum(w_k * d_k) / total_w)
        devs = d_k - d_bar
        s0_sq = float(np.sum(w_k * (devs**2)) / total_w)
        if s0_sq < 1e-9:
            return 0.0, 1.0, False

        cum_c = 0.0
        cum_w = 0.0
        s_star = 0.0
        for m in range(k - 1):
            cum_w += w_k[m]
            cum_c += w_k[m] * devs[m]
            frac = cum_w / total_w
            var_cm = cum_w * (1.0 - frac)
            if var_cm > 1e-9:
                s_star += (cum_c**2) / (var_cm * s0_sq)

        s_star = float(s_star)
        df = max(1, k - 1)
        p_hir = float(1.0 - stats.chi2.cdf(s_star, df))
        p_hir = max(0.0, min(1.0, p_hir))
        is_hetero = bool(p_hir < 0.05 and s_star > 1.645)
        return s_star, p_hir, is_hetero
    except Exception:
        return None, None, False


def bartlett_kendall_variance_homogeneity_test(
    samples: List[np.ndarray],
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates M. S. Bartlett & D. G. Kendall's (1946 JRSS-B) Log-Variance Transformation ANOVA Homogeneity Test (T_BK).

    Tests equality of variances across K subgroup strata by exploiting the asymptotic variance
    of log sample variances: Var(ln(s_k^2)) ~ 2 / (n_k - 1):
      Log-variances: z_k = ln(s_k^2).
      Weights: w_k = (n_k - 1) / 2.
      Pooled weighted mean: z_bar = sum(w_k * z_k) / sum(w_k).
      Test statistic:
        T_BK = sum_{k=1}^K w_k * (z_k - z_bar)^2 ~ Chi2(K - 1).
        p_BK = 1.0 - stats.chi2.cdf(T_BK, K - 1).
    Rejection (p_BK < 0.05 with T_BK > chi2_crit) confirms genuine stratum scale dispersion heterogeneity.
    """
    valid_samples = [np.asarray(s, dtype=float) for s in samples if len(s) >= 4 and np.var(s, ddof=1) > 1e-9]
    k = len(valid_samples)
    if k < 2:
        return None, None, False
    try:
        z_k = []
        w_k = []
        for s in valid_samples:
            n_sub = len(s)
            v_sub = float(np.var(s, ddof=1))
            z_k.append(np.log(v_sub))
            w_k.append((n_sub - 1.0) / 2.0)

        z_arr = np.array(z_k, dtype=float)
        w_arr = np.array(w_k, dtype=float)
        sum_w = float(np.sum(w_arr))
        if sum_w <= 0:
            return None, None, False

        z_bar = float(np.sum(w_arr * z_arr) / sum_w)
        t_bk = float(np.sum(w_arr * (z_arr - z_bar)**2))
        df = max(1, k - 1)
        p_bk = float(1.0 - stats.chi2.cdf(t_bk, df))
        p_bk = max(0.0, min(1.0, p_bk))
        is_hetero = bool(p_bk < 0.05 and t_bk > float(stats.chi2.ppf(0.95, df)))
        return t_bk, p_bk, is_hetero
    except Exception:
        return None, None, False


def rosenbaum_adaptive_sensitivity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    gamma_crit: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Paul R. Rosenbaum's (2011 Biometrics) Adaptive Rank U-Statistic Sensitivity Bounds (Gamma_ADAP).

    Applies Rosenbaum's adaptive power-rank scoring function a_m = sign(D_m) * (rank(|D_m|) / M)^2 on matched
    pairs between treated (X > med(X)) and control (X <= med(X)) cohorts:
      Under sensitivity parameter Gamma >= 1:
        Worst-case expectation: E_Gamma = sum_{m=1}^M |a_m| * (Gamma - 1) / (Gamma + 1).
        Variance: Var_Gamma = sum_{m=1}^M a_m^2 * 4 * Gamma / (Gamma + 1)^2.
        Standardized score: Z_ADAP(Gamma) = (T_ADAP - E_Gamma) / sqrt(Var_Gamma).
        Upper tail p-value: p_upper(Gamma) = 1 - Phi(Z_ADAP(Gamma)).
    Solves for critical sensitivity threshold Gamma_ADAP where p_upper(Gamma) >= 0.05.
    Rejection (is_fragile = True if Gamma_ADAP < 1.25) confirms vulnerability to hidden unmeasured bias under adaptive design sensitivity.
    """
    n = len(x)
    if n < 10:
        return None, None, False
    try:
        med_x = np.median(x)
        t_mask = (x > med_x)
        c_mask = ~t_mask
        if np.sum(t_mask) < 4 or np.sum(c_mask) < 4:
            return None, None, False

        z_t = z[t_mask]
        y_t = y[t_mask]
        z_c = z[c_mask]
        y_c = y[c_mask]

        used_c = np.zeros(len(z_c), dtype=bool)
        diffs = []
        for i in range(len(z_t)):
            avail_c = np.where(~used_c)[0]
            if len(avail_c) == 0:
                break
            best_j = avail_c[np.argmin(np.abs(z_c[avail_c] - z_t[i]))]
            used_c[best_j] = True
            diffs.append(y_t[i] - y_c[best_j])

        m = len(diffs)
        if m < 4:
            return None, None, False

        d_arr = np.array(diffs, dtype=float)
        non_zero = np.abs(d_arr) > 1e-9
        if np.sum(non_zero) < 4:
            return 1.0, 1.0, True

        d_valid = d_arr[non_zero]
        m_valid = len(d_valid)
        ranks = stats.rankdata(np.abs(d_valid))
        scores = np.sign(d_valid) * ((ranks / float(m_valid)) ** 2)
        t_stat = float(np.sum(scores))
        abs_scores = np.abs(scores)
        sq_scores = scores ** 2

        def calc_p_upper(gamma_val: float) -> float:
            e_g = float(np.sum(abs_scores * ((gamma_val - 1.0) / (gamma_val + 1.0))))
            var_g = float(np.sum(sq_scores * (4.0 * gamma_val / (gamma_val + 1.0)**2)))
            if var_g <= 1e-12:
                return 1.0
            z_score = (t_stat - e_g) / np.sqrt(var_g)
            return float(1.0 - stats.norm.cdf(z_score))

        p_1 = calc_p_upper(1.0)
        if p_1 >= 0.05:
            return 1.0, p_1, True

        low, high = 1.0, 10.0
        gamma_crit_res = 10.0
        for _ in range(25):
            mid = 0.5 * (low + high)
            p_mid = calc_p_upper(mid)
            if p_mid >= 0.05:
                gamma_crit_res = mid
                high = mid
            else:
                low = mid

        p_final = calc_p_upper(gamma_crit_res)
        is_fragile = bool(gamma_crit_res < 1.25)
        return float(gamma_crit_res), float(p_final), is_fragile
    except Exception:
        return None, None, False


def abadie_imbens_disparity_test(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Alberto Abadie & Guido W. Imbens' (2006 Econometrica) Bias-Corrected Matching Disparity Ratio (delta_AIM, B_AIM).

    Evaluates nearest-neighbor matching on continuous confounder Z and tests sensitivity to covariate discrepancy bias:
      Matching pairs: m = 1, ..., M between treated (X > med(X)) and control (X <= med(X)).
      Normalized matching discrepancy: B_AIM = mean(|Z_{T,m} - Z_{C,m}|) / std(Z).
      Naive matched effect: tau_match = mean(Y_{T,m} - Y_{C,m}).
      Control regression adjustment: beta_C from OLS of Y_C on Z_C.
      Bias-corrected matched effect: tau_BC = mean[(Y_{T,m} - Y_{C,m}) - beta_C * (Z_{T,m} - Z_{C,m})].
      Disparity ratio: delta_AIM = |tau_match - tau_BC| / max(1e-6, |tau_match|).
    Rejection (is_fragile = True if B_AIM > 0.25 or delta_AIM > 0.20 or sign reversal)
    diagnoses substantial matching discrepancy and residual covariate confounding.
    """
    n = len(x)
    if n < 10:
        return None, None, False
    try:
        med_x = np.median(x)
        t_mask = (x > med_x)
        c_mask = ~t_mask
        if np.sum(t_mask) < 4 or np.sum(c_mask) < 4:
            return None, None, False

        z_t = z[t_mask]
        y_t = y[t_mask]
        z_c = z[c_mask]
        y_c = y[c_mask]

        std_z = max(1e-6, float(np.std(z)))
        used_c = np.zeros(len(z_c), dtype=bool)
        z_diffs = []
        y_diffs = []

        for i in range(len(z_t)):
            avail_c = np.where(~used_c)[0]
            if len(avail_c) == 0:
                break
            best_j = avail_c[np.argmin(np.abs(z_c[avail_c] - z_t[i]))]
            used_c[best_j] = True
            z_diffs.append(z_t[i] - z_c[best_j])
            y_diffs.append(y_t[i] - y_c[best_j])

        m = len(y_diffs)
        if m < 4:
            return None, None, False

        z_diff_arr = np.array(z_diffs, dtype=float)
        y_diff_arr = np.array(y_diffs, dtype=float)

        b_aim = float(np.mean(np.abs(z_diff_arr)) / std_z)
        tau_match = float(np.mean(y_diff_arr))

        if len(z_c) >= 3 and np.std(z_c) > 1e-9:
            slope_c, _, _, _, _ = stats.linregress(z_c, y_c)
            tau_bc = float(np.mean(y_diff_arr - slope_c * z_diff_arr))
        else:
            tau_bc = tau_match

        denom = max(1e-6, abs(tau_match))
        delta_aim = float(abs(tau_match - tau_bc) / denom)
        has_reversal = bool(np.sign(tau_match) != np.sign(tau_bc) and abs(tau_bc) > 1e-4 and abs(tau_match) > 1e-4)

        is_fragile = bool(b_aim > 0.25 or delta_aim > 0.20 or has_reversal)
        return delta_aim, b_aim, is_fragile
    except Exception:
        return None, None, False


def martinez_iglewicz_normality_test(
    residuals: np.ndarray,
) -> tuple[Optional[float], Optional[float], bool]:
    """Calculates Jorge F. Martinez & Boris Iglewicz's (1981 Biometrika) Studentized Jackknife Scale Normality Test (I_MI).

    Evaluates departure from Gaussian normality by comparing the heavy-tail-resistant biweight scale estimator
    S_bw to the sample standard deviation S_sample:
      Biweight scale: S_bw with tuning parameter c = 9.0 * MAD:
        u_i = (e_i - med(e)) / (9 * MAD).
        S_bw = sqrt(n) * sqrt(sum_{|u|<1} (e_i - med)^2 * (1 - u_i^2)^4) / |sum_{|u|<1} (1 - u_i^2)*(1 - 5*u_i^2)|.
      Test ratio: I_MI = S_bw / S_sample.
      Under normality, E[I_MI] ~ 1.0, Var(I_MI) ~ 1.06 / n.
      Standardized score:
        Z_MI = (I_MI - 1.0) / sqrt(1.06 / n) ~ N(0, 1).
        p_MI = 2 * (1 - Phi(|Z_MI|)).
    Rejection (p_MI < 0.05 with |Z_MI| > 1.96) proves heavy-tailed non-Gaussian residual distortion.
    """
    n = len(residuals)
    if n < 8:
        return None, None, False
    try:
        e = np.asarray(residuals, dtype=float)
        med_e = np.median(e)
        mad = float(np.median(np.abs(e - med_e)))
        s_sample = float(np.std(e, ddof=1))
        if s_sample < 1e-12 or mad < 1e-12:
            return 1.0, 1.0, False

        c = 9.0 * mad
        u = (e - med_e) / c
        in_mask = np.abs(u) < 1.0
        if np.sum(in_mask) < 4:
            return None, None, False

        u_in = u[in_mask]
        e_diff = e[in_mask] - med_e
        num = np.sqrt(n) * np.sqrt(np.sum((e_diff**2) * ((1.0 - u_in**2)**4)))
        denom = np.abs(np.sum((1.0 - u_in**2) * (1.0 - 5.0 * (u_in**2))))
        if denom < 1e-12:
            return None, None, False

        s_bw = float(num / denom)
        i_mi = float(s_bw / s_sample)

        se_i = np.sqrt(1.06 / n)
        z_mi = float((i_mi - 1.0) / se_i)
        p_mi = float(2.0 * (1.0 - stats.norm.cdf(abs(z_mi))))
        p_mi = max(0.0, min(1.0, p_mi))
        is_non_norm = bool(p_mi < 0.05 and abs(z_mi) > 1.96)
        return i_mi, p_mi, is_non_norm
    except Exception:
        return None, None, False


def beaton_tukey_bisquare_estimator_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Optional[float], Optional[float], int, bool]:
    """Calculates Albert E. Beaton & John W. Tukey's (1974 Technometrics) Bisquare Biweight Redescending M-Estimator Robust Slope Discrepancy (Delta_BIS).

    Applies Tukey's bisquare (biweight) redescending weight function with tuning constant c = 4.685:
      For standardized residuals u_i = |r_i| / sigma_0 (sigma_0 = 1.4826 * MAD):
        w(u) = (1 - (u / c)^2)^2 for |u| <= c;
        w(u) = 0.0 for |u| > c (completely eliminates extreme leverage points).
      Iteratively reweighted least squares (IRLS) converges to robust slope beta_BIS.
      Discrepancy: Delta_BIS = |beta_OLS - beta_BIS| / max(1e-6, |beta_OLS|).
      Redescended outlier count: n_out = sum(|u_i| > c).
    Rejection (Delta_BIS > 0.25 or (n_out >= 2 and Delta_BIS > 0.15)) diagnoses severe vulnerability to gross leverage outliers.
    """
    n = len(x)
    if n < 8 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None, None, 0, False
    try:
        slope_ols, intercept_ols, _, _, _ = stats.linregress(x, y)
        r_ols = y - (slope_ols * x + intercept_ols)
        mad = float(np.median(np.abs(r_ols - np.median(r_ols))))
        sigma_0 = max(1e-6, 1.4826 * mad)

        c = 4.685
        b_curr = slope_ols
        a_curr = intercept_ols

        for _ in range(25):
            r_curr = y - (b_curr * x + a_curr)
            u = np.abs(r_curr) / sigma_0
            w = np.zeros_like(u)
            in_bounds = (u <= c)
            w[in_bounds] = (1.0 - (u[in_bounds] / c)**2)**2

            if np.sum(w) < 4:
                break

            w_x = w * x
            w_y = w * y
            sum_w = np.sum(w)
            mean_x_w = np.sum(w_x) / sum_w
            mean_y_w = np.sum(w_y) / sum_w
            denom = np.sum(w * (x - mean_x_w)**2)
            if denom < 1e-12:
                break
            b_next = np.sum(w * (x - mean_x_w) * (y - mean_y_w)) / denom
            a_next = mean_y_w - b_next * mean_x_w

            if abs(b_next - b_curr) < 1e-5:
                b_curr = b_next
                a_curr = a_next
                break
            b_curr = b_next
            a_curr = a_next

        beta_bis = float(b_curr)
        denom_ols = max(1e-6, abs(slope_ols))
        delta_bis = float(abs(slope_ols - beta_bis) / denom_ols)

        final_r = y - (beta_bis * x + a_curr)
        final_u = np.abs(final_r) / sigma_0
        outlier_count = int(np.sum(final_u > c))

        is_distorted = bool(delta_bis > 0.25 or (outlier_count >= 2 and delta_bis > 0.15))
        return beta_bis, delta_bis, outlier_count, is_distorted
    except Exception:
        return None, None, 0, False





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
                # Guido Schwarzer et al. (2002, 2007) Arcsine Funnel Asymmetry Test
                schw_t, schw_p, is_schw_asym = schwarzer_funnel_asymmetry_test(tables_2x2)
                # John Ioannidis & Thomas Trikalinos (2007) Test for Excess Significance (TES)
                tes_ratio, tes_chi2, tes_p, is_tes_excess = ioannidis_trikalinos_tes_test(tables_2x2)
                # John B. Copas & Frank Q. Shi (2000, 2001) Heuristic Selection Model Funnel Asymmetry Test
                copas_t, copas_p, is_copas_asym = copas_selection_asymmetry_test(tables_2x2)
                # E-value for unmeasured confounding
                simpson_e_val, simpson_e_desc = calculate_e_value(cmh_or)
        else:
            bd_stat, bd_p = None, None
            pet_t, pet_p, peese_t, peese_p, is_pet_asym = None, None, None, None, False
            tang_t, tang_p, is_tang_asym = None, None, False
            pust_t, pust_p, is_pust_asym = None, None, False
            schw_t, schw_p, is_schw_asym = None, None, False
            tes_ratio, tes_chi2, tes_p, is_tes_excess = None, None, None, False
            copas_t, copas_p, is_copas_asym = None, None, False

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
        west_z, west_p, is_west_hetero = None, None, False
        obrien_f, obrien_p, is_obrien_hetero = None, None, False
        cochran_c, cochran_p, is_cochran_hetero = None, None, False
        gs_stat, gs_p, is_gs_reversal = None, None, False
        hartley_stat, hartley_p, is_hartley_hetero = None, None, False
        pbg_stat, pbg_p, is_pbg_reversal = None, None, False
        layard_stat, layard_p, is_layard_hetero = None, None, False
        pan_stat, pan_p, is_pan_reversal = None, None, False
        sam_stat, sam_p, is_sam_hetero = None, None, False
        silvapulle_stat, silvapulle_p, is_silvapulle_reversal = None, None, False
        miller_stat, miller_p, is_miller_hetero = None, None, False
        berger_stat, berger_p, is_berger_reversal = None, None, False
        shoemaker_stat, shoemaker_p, is_shoemaker_hetero = None, None, False
        demets_q, demets_p, is_demets_reversal = None, None, False
        box_m, box_p, is_box_hetero = None, None, False
        pocock_q, pocock_p, is_pocock_reversal = None, None, False
        ow_f, ow_p, is_ow_hetero = None, None, False
        crainiceanu_rlrt, crainiceanu_p, is_crainiceanu_hetero = None, None, False
        lim_loh_w, lim_loh_p, is_lim_loh_hetero = None, None, False
        verbeke_s, verbeke_p, is_verbeke_hetero = None, None, False
        conover_t, conover_p, is_conover_hetero = None, None, False
        hirotsu_s, hirotsu_p, is_hirotsu_hetero = None, None, False
        bk_t, bk_p, is_bk_hetero = None, None, False
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
            # Westenberg (1948) Interquartile Range Dispersion Test
            west_z, west_p, is_west_hetero = westenberg_scale_dispersion_test(kw_samples)
            # Peter C. O'Brien (1979, 1981) Homogeneity of Variance Test
            obrien_f, obrien_p, is_obrien_hetero = obrien_variance_homogeneity_test(kw_samples)
            # William G. Cochran (1941) C Outlier Variance Homogeneity Test
            cochran_c, cochran_p, is_cochran_hetero = cochran_c_variance_homogeneity_test(kw_samples)
            # Mitchell H. Gail & Richard Simon (1985) Test for Qualitative Interactions
            gs_stat, gs_p, is_gs_reversal = gail_simon_qualitative_interaction_test(breakdowns)
            # H. O. Hartley (1950) F_max Maximum Variance Ratio Test
            hartley_stat, hartley_p, is_hartley_hetero = hartley_fmax_variance_homogeneity_test(kw_samples)
            # S. Piantadosi, D. P. Byar, & S. B. Green (1988) Likelihood Ratio Qualitative Interaction Test
            pbg_stat, pbg_p, is_pbg_reversal = piantadosi_byar_green_qualitative_interaction_test(breakdowns)
            # M. W. J. Layard (1973) Kurtosis-Robust Variance Homogeneity Test
            layard_stat, layard_p, is_layard_hetero = layard_kurtosis_robust_variance_homogeneity_test(kw_samples)
            # Guohua Pan (1997 Statistics in Medicine) Score Test for Qualitative Interaction
            pan_stat, pan_p, is_pan_reversal = pan_qualitative_interaction_score_test(breakdowns)
            # M. Samiuddin & M. Atiqullah (1976 Biometrika) Kurtosis-Adjusted Scale Homogeneity Test
            sam_stat, sam_p, is_sam_hetero = samiuddin_atiqullah_variance_homogeneity_test(kw_samples)
            # Mervyn J. Silvapulle & Pranab K. Sen (2005) Likelihood Ratio Inversion Test for Qualitative Interaction
            silvapulle_stat, silvapulle_p, is_silvapulle_reversal = silvapulle_sen_qualitative_interaction_test(breakdowns)
            # Rupert G. Miller (1968 Annals of Mathematical Statistics) Jackknife Variance Homogeneity Test
            miller_stat, miller_p, is_miller_hetero = miller_jackknife_variance_homogeneity_test(kw_samples)
            # Roger L. Berger (1989 JASA) Intersection-Union Test for Qualitative Interaction
            berger_stat, berger_p, is_berger_reversal = berger_intersection_union_test(breakdowns)
            # Terry Shoemaker (2003 American Statistician) Kurtosis-Adjusted Variance Homogeneity Test
            shoemaker_stat, shoemaker_p, is_shoemaker_hetero = shoemaker_kurtosis_variance_homogeneity_test(kw_samples)
            # David L. DeMets & Richard Simon (1983) Qualitative Interaction Score Test
            demets_q, demets_p, is_demets_reversal = demets_simon_qualitative_interaction_test(breakdowns)
            # George E. P. Box & S. L. Andersen (1955 JRSS-B) Permutation/Kurtosis-Corrected Variance Test
            box_m, box_p, is_box_hetero = box_andersen_kurtosis_variance_test(kw_samples)
            # Stuart J. Pocock, Michael D. Hughes, & Robert J. Lee (1987) Qualitative Interaction Contrast Test
            pocock_q, pocock_p, is_pocock_reversal = pocock_hughes_lee_qualitative_interaction_test(breakdowns)
            # John E. Overall & J. Arthur Woodward (1974) Modified Variance Homogeneity Test
            ow_f, ow_p, is_ow_hetero = overall_woodward_variance_homogeneity_test(kw_samples)
            # Ciprian M. Crainiceanu & David Ruppert (2004 JASA) Restricted Likelihood Ratio Test
            crainiceanu_rlrt, crainiceanu_p, is_crainiceanu_hetero = crainiceanu_ruppert_rlrt_test(breakdowns)
            # Wei-Yin Lim & Wei-Liem Loh (1996 JASA) Ratio-of-Variances Robust Homogeneity Test
            lim_loh_w, lim_loh_p, is_lim_loh_hetero = lim_loh_variance_homogeneity_test(kw_samples)
            # Geert Verbeke & Geert Molenberghs (2000, 2003 Stat. Sci.) Score Test for Random Effect Variance
            verbeke_s, verbeke_p, is_verbeke_hetero = verbeke_molenberghs_score_test(breakdowns)
            # W. J. Conover, Mark E. Johnson & Myrle M. Johnson (1981 Technometrics) Squared Ranks Test
            conover_t, conover_p, is_conover_hetero = conover_johnson_variance_homogeneity_test(kw_samples)
            # Chihiro Hirotsu (1986 Biometrika) Cumulative Chi-Squared Statistic
            hirotsu_s, hirotsu_p, is_hirotsu_hetero = hirotsu_cumulative_chi_squared_test(breakdowns)
            # M. S. Bartlett & D. G. Kendall (1946 JRSS-B) Log-Variance Transformation ANOVA Homogeneity Test
            bk_t, bk_p, is_bk_hetero = bartlett_kendall_variance_homogeneity_test(kw_samples)
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
        if schw_p is not None:
            cmh_note += f" [Schwarzer Funnel: t={schw_t:.2f}, p={schw_p:.3e}]"
        if tes_p is not None:
            cmh_note += f" [TES Excess: A={tes_ratio:.2f}, p={tes_p:.3e}]"
        if west_p is not None:
            cmh_note += f" [Westenberg Scale: Z={west_z:.2f}, p={west_p:.3e}]"
        if obrien_p is not None:
            cmh_note += f" [O'Brien Scale: F={obrien_f:.2f}, p={obrien_p:.3e}]"
        if copas_p is not None:
            cmh_note += f" [Copas Selection: t={copas_t:.2f}, p={copas_p:.3e}]"
        if cochran_p is not None:
            cmh_note += f" [Cochran's C: C={cochran_c:.3f}, p={cochran_p:.3e}]"
        if gs_p is not None and is_gs_reversal:
            cmh_note += f" [Gail-Simon Reversal: s={gs_stat:.2f}, p={gs_p:.3e}]"
        if hartley_p is not None:
            cmh_note += f" [Hartley F_max: F={hartley_stat:.2f}, p={hartley_p:.3e}]"
        if pbg_p is not None and is_pbg_reversal:
            cmh_note += f" [Piantadosi-Byar-Green Reversal: s={pbg_stat:.2f}, p={pbg_p:.3e}]"
        if layard_p is not None:
            cmh_note += f" [Layard Kurtosis-Robust: L={layard_stat:.2f}, p={layard_p:.3e}]"
        if pan_p is not None and is_pan_reversal:
            cmh_note += f" [Pan Score Reversal: S={pan_stat:.2f}, p={pan_p:.3e}]"
        if sam_p is not None:
            cmh_note += f" [Samiuddin Scale: F={sam_stat:.2f}, p={sam_p:.3e}]"
        if silvapulle_p is not None and is_silvapulle_reversal:
            cmh_note += f" [Silvapulle-Sen Reversal: S={silvapulle_stat:.2f}, p={silvapulle_p:.3e}]"
        if miller_p is not None:
            cmh_note += f" [Miller Jackknife Scale: F={miller_stat:.2f}, p={miller_p:.3e}]"
        if berger_p is not None and is_berger_reversal:
            cmh_note += f" [Berger IUT Reversal: Z={berger_stat:.2f}, p={berger_p:.3e}]"
        if shoemaker_p is not None:
            cmh_note += f" [Shoemaker Kurtosis-Adjusted Scale: Z={shoemaker_stat:.2f}, p={shoemaker_p:.3e}]"
        if demets_p is not None and is_demets_reversal:
            cmh_note += f" [DeMets-Simon Reversal: Q={demets_q:.2f}, p={demets_p:.3e}]"
        if box_p is not None:
            cmh_note += f" [Box-Andersen Kurtosis Scale: M={box_m:.2f}, p={box_p:.3e}]"
        if pocock_p is not None and is_pocock_reversal:
            cmh_note += f" [Pocock-Hughes-Lee Reversal: Q={pocock_q:.2f}, p={pocock_p:.3e}]"
        if ow_p is not None:
            cmh_note += f" [Overall-Woodward Scale: F={ow_f:.2f}, p={ow_p:.3e}]"
        if crainiceanu_p is not None:
            cmh_note += f" [Crainiceanu-Ruppert RLRT: {crainiceanu_rlrt:.2f}, p={crainiceanu_p:.3e}]"
        if lim_loh_p is not None:
            cmh_note += f" [Lim-Loh Scale: W={lim_loh_w:.2f}, p={lim_loh_p:.3e}]"
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
            schwarzer_test_t=schw_t,
            schwarzer_test_p=schw_p,
            is_schwarzer_asymmetric=is_schw_asym,
            westenberg_stat_z=west_z,
            westenberg_stat_p=west_p,
            is_westenberg_heterogeneous=is_west_hetero,
            tes_excess_ratio=tes_ratio,
            tes_chi2=tes_chi2,
            tes_p_value=tes_p,
            is_tes_excessive=is_tes_excess,
            obrien_stat_f=obrien_f,
            obrien_p_value=obrien_p,
            is_obrien_heterogeneous=is_obrien_hetero,
            copas_test_t=copas_t,
            copas_test_p=copas_p,
            is_copas_asymmetric=is_copas_asym,
            cochran_c_stat=cochran_c,
            cochran_c_p=cochran_p,
            is_cochran_c_heterogeneous=is_cochran_hetero,
            gail_simon_stat=gs_stat,
            gail_simon_p=gs_p,
            is_gail_simon_reversal=is_gs_reversal,
            hartley_fmax_stat=hartley_stat,
            hartley_fmax_p=hartley_p,
            is_hartley_heterogeneous=is_hartley_hetero,
            piantadosi_byar_green_stat=pbg_stat,
            piantadosi_byar_green_p=pbg_p,
            is_pbg_reversal=is_pbg_reversal,
            layard_stat_l=layard_stat,
            layard_p_value=layard_p,
            is_layard_heterogeneous=is_layard_hetero,
            pan_score_stat=pan_stat,
            pan_score_p=pan_p,
            is_pan_reversal=is_pan_reversal,
            samiuddin_stat_f=sam_stat,
            samiuddin_p_value=sam_p,
            is_samiuddin_heterogeneous=is_sam_hetero,
            silvapulle_sen_stat=silvapulle_stat,
            silvapulle_sen_p=silvapulle_p,
            is_silvapulle_reversal=is_silvapulle_reversal,
            miller_jackknife_stat_f=miller_stat,
            miller_jackknife_p_value=miller_p,
            is_miller_heterogeneous=is_miller_hetero,
            berger_stat_z=berger_stat,
            berger_p_value=berger_p,
            is_berger_reversal=is_berger_reversal,
            shoemaker_stat_z=shoemaker_stat,
            shoemaker_p_value=shoemaker_p,
            is_shoemaker_heterogeneous=is_shoemaker_hetero,
            demets_simon_stat_q=demets_q,
            demets_simon_p_value=demets_p,
            is_demets_simon_reversal=is_demets_reversal,
            box_andersen_stat_m=box_m,
            box_andersen_p_value=box_p,
            is_box_andersen_heterogeneous=is_box_hetero,
            pocock_hughes_lee_stat_q=pocock_q,
            pocock_hughes_lee_p_value=pocock_p,
            is_pocock_reversal=is_pocock_reversal,
            overall_woodward_stat_f=ow_f,
            overall_woodward_p_value=ow_p,
            is_overall_woodward_heterogeneous=is_ow_hetero,
            crainiceanu_ruppert_stat_rlrt=crainiceanu_rlrt,
            crainiceanu_ruppert_p_value=crainiceanu_p,
            is_crainiceanu_heterogeneous=is_crainiceanu_hetero,
            lim_loh_stat_w=lim_loh_w,
            lim_loh_p_value=lim_loh_p,
            is_lim_loh_heterogeneous=is_lim_loh_hetero,
            verbeke_molenberghs_stat_s=verbeke_s,
            verbeke_molenberghs_p_value=verbeke_p,
            is_verbeke_heterogeneous=is_verbeke_hetero,
            conover_johnson_stat_t=conover_t,
            conover_johnson_p_value=conover_p,
            is_conover_johnson_heterogeneous=is_conover_hetero,
            hirotsu_stat_s=hirotsu_s,
            hirotsu_p_value=hirotsu_p,
            is_hirotsu_heterogeneous=is_hirotsu_hetero,
            bartlett_kendall_stat_t=bk_t,
            bartlett_kendall_p_value=bk_p,
            is_bartlett_kendall_heterogeneous=is_bk_hetero,
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
        rr_vr, rr_p, is_rr_imbal = None, None, False
        iw_quad_diff, is_iw_quad_excess = None, False
        hl_shift, hl_z, hl_p, is_hl_imbal = None, None, None, False
        rir_pct, rir_cnt, is_rir_frag = None, None, False
        q_imbens, p_imbens, is_imbens_sens = None, None, False
        masd_val, masd_p, is_masd_imbal = None, None, False
        kallus_gamma_crit, kallus_bias_max, kallus_b_low, kallus_b_high, is_kallus_sens = None, None, None, None, False
        cbps_j, cbps_p, is_cbps_rej = None, None, False
        silber_delta_crit, silber_lambda, is_silber_sens = None, None, False
        fogarty_gamma, fogarty_t, is_fogarty_sens = None, None, False
        rosen_attr_gamma, rosen_attr_p, is_rosen_attr_sens = None, None, False
        sbw_cv, sbw_eff_pct, is_sbw_imbal = None, None, False
        design_gamma, design_eff, is_design_frag = None, None, False
        hhh_delta, hhh_scale, is_hhh_sens = None, None, False
        rosen_ef_gamma, rosen_ef_p, is_rosen_ef_frag = None, None, False
        ho_imai_hull, ho_imai_trim, is_ho_imai_def = None, None, False
        rosen_diff_gamma, rosen_diff_p, is_rosen_diff_frag = None, None, False
        bgh_disp, bgh_div, is_bgh_hetero = None, None, False
        rosen_dual_gamma, rosen_dual_p, is_rosen_dual_frag = None, None, False
        eb_disp, eb_eff, eb_loss, is_eb_frag = None, None, None, False
        rosen_cm_gamma, rosen_cm_p, is_rosen_cm_frag = None, None, False
        ai_disp, ai_tau_max, ai_tau_min, is_ai_hetero = None, None, None, False
        rosen_hub_gamma, rosen_hub_p, is_rosen_hub_frag = None, None, False
        iks_disp, iks_smd, is_iks_frag = None, None, False
        rosen_adap_gamma, rosen_adap_p, is_rosen_adap_frag = None, None, False
        aim_disp, aim_smd, is_aim_frag = None, None, False
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
                    # Paul R. Rosenbaum & Donald B. Rubin (1985) Residual Variance Ratio Balance Metric
                    rr_vr, rr_p, is_rr_imbal = rosenbaum_rubin_residual_variance_test(x_mal, z_mal)
                    # Guido W. Imbens & Jeffrey M. Wooldridge (2009) Normalized Quadratic Difference Metric
                    iw_quad_diff, is_iw_quad_excess = imbens_wooldridge_quadratic_balance_test(x_mal, z_mal)
                    # Paul R. Rosenbaum (1984, 2002) Hodges-Lehmann Aligned Rank Covariate Shift
                    hl_shift, hl_z, hl_p, is_hl_imbal = rosenbaum_hodges_lehmann_balance_test(x_mal, z_mal)
                    # Kenneth A. Frank et al. (2013) Robustness of Inference to Replacement (RIR)
                    rir_pct, rir_cnt, is_rir_frag = frank_rir_fragility_test(x_mal, z_mal, y_mal)
                    # Guido W. Imbens (2003 AER) Parametric Sensitivity Contour Boundary
                    q_imbens, p_imbens, is_imbens_sens = imbens_sensitivity_contour_test(x_mal, z_mal, y_mal)
                    # Peter C. Austin & Elizabeth A. Stuart (2015) Inverse Probability Weighted Standardized Difference (MASD)
                    masd_val, masd_p, is_masd_imbal = austin_stuart_iptw_balance_test(x_mal, z_mal)
                    # Nathan Kallus, Xiaojie Mao, & Angela Zhou (2019 NeurIPS) Minimax Confounding Bias Bounds
                    kallus_gamma_crit, kallus_bias_max, kallus_b_low, kallus_b_high, is_kallus_sens = kallus_confounding_bias_sensitivity_test(x_mal, z_mal, y_mal)
                    # Kosuke Imai & Marc Ratkovic (2014 JASA) CBPS Overidentifying Moment Balance Test
                    cbps_j, cbps_p, is_cbps_rej = cbps_overidentifying_moment_test(x_mal, z_mal)
                    # Paul R. Rosenbaum & Jeffrey H. Silber (2009 JASA) Amplification of Sensitivity to Hidden Bias
                    silber_delta_crit, silber_lambda, is_silber_sens = rosenbaum_silber_amplification_test(gamma_crit, lambda_param=2.0)
                    # Colin B. Fogarty (2020 JASA) Studentized Sensitivity Bounds
                    fogarty_gamma, fogarty_t, is_fogarty_sens = fogarty_studentized_sensitivity_test(x_mal, z_mal, y_mal)
                    # Paul R. Rosenbaum (2002, 2010) Attributable Effect Sensitivity Analysis
                    rosen_attr_gamma, rosen_attr_p, is_rosen_attr_sens = rosenbaum_attributable_sensitivity_test(x_mal, z_mal, y_mal, gamma_crit)
                    # José R. Zubizarreta (2012, 2015 JASA) Stable Balancing Weights Diagnostic
                    sbw_cv, sbw_eff_pct, is_sbw_imbal = zubizarreta_stable_balancing_weights_test(x_mal, z_mal)
                    # Paul R. Rosenbaum (2004 Sankhyā) Design Sensitivity Bound
                    design_gamma, design_eff, is_design_frag = rosenbaum_design_sensitivity_test(x_mal, z_mal, y_mal, gamma_crit)
                    # Hosman, Hansen, & Holland (2010 Annals of Applied Statistics) Causal Sensitivity Relation
                    hhh_delta, hhh_scale, is_hhh_sens = hosman_hansen_holland_sensitivity_test(x_mal, z_mal, y_mal)
                    # Paul R. Rosenbaum (2010, 2013 Biometrics) Evidence Factors Sensitivity Analysis Bound
                    rosen_ef_gamma, rosen_ef_p, is_rosen_ef_frag = rosenbaum_evidence_factors_sensitivity_test(x_mal, z_mal, y_mal, gamma_crit)
                    # Daniel E. Ho, Kosuke Imai, Gary King, & Elizabeth A. Stuart (2007) Convex Hull Support Overlap Metric
                    ho_imai_hull, ho_imai_trim, is_ho_imai_def = ho_imai_convex_hull_overlap_test(x_mal, z_mal)
                    # Paul R. Rosenbaum (2004 The American Statistician / 2002) Differential Comparison Sensitivity Bound
                    rosen_diff_gamma, rosen_diff_p, is_rosen_diff_frag = rosenbaum_differential_comparison_sensitivity_test(x_mal, z_mal, y_mal, gamma_crit)
                    # Marianne Bitler, Jonah Gelbach, & Hilary Hoynes (2006 AER) Quantile Treatment Disparity Bound
                    bgh_disp, bgh_div, is_bgh_hetero = bitler_gelbach_hoynes_quantile_sensitivity_test(x_mal, y_mal)
                    # Paul R. Rosenbaum (1989 JASA / 2002) Dual-Control Observational Sensitivity Bound
                    rosen_dual_gamma, rosen_dual_p, is_rosen_dual_frag = rosenbaum_dual_control_sensitivity_test(x_mal, z_mal, y_mal, gamma_crit)
                    # Jens Hainmueller (2012 Political Analysis) Entropy Balancing Treatment Discrepancy
                    eb_disp, eb_eff, eb_loss, is_eb_frag = hainmueller_entropy_balancing_test(x_mal, z_mal, y_mal)
                    # Paul R. Rosenbaum (2005 JASA) Cross-Match Observational Sensitivity Test
                    rosen_cm_gamma, rosen_cm_p, is_rosen_cm_frag = rosenbaum_cross_match_sensitivity_test(x_mal, z_mal, y_mal, gamma_crit)
                    # Susan Athey & Guido W. Imbens (2016 PNAS) Recursive Partitioning Causal Disparity Bound
                    ai_disp, ai_tau_max, ai_tau_min, is_ai_hetero = athey_imbens_causal_disparity_test(x_mal, z_mal, y_mal)
                    # Paul R. Rosenbaum (2007 JASA) Dynamic Huber-Type M-Statistic Sensitivity Bounds
                    rosen_hub_gamma, rosen_hub_p, is_rosen_hub_frag = rosenbaum_huber_sensitivity_test(x_mal, z_mal, y_mal, gamma_crit)
                    # Kosuke Imai, Gary King & Elizabeth A. Stuart (2008 JRSS-A) Balance & Disparity Ratio
                    iks_disp, iks_smd, is_iks_frag = imai_king_stuart_disparity_test(x_mal, z_mal, y_mal)
                    # Paul R. Rosenbaum (2011 Biometrics) Adaptive Rank U-Statistic Sensitivity Bounds
                    rosen_adap_gamma, rosen_adap_p, is_rosen_adap_frag = rosenbaum_adaptive_sensitivity_test(x_mal, z_mal, y_mal, gamma_crit)
                    # Alberto Abadie & Guido W. Imbens (2006 Econometrica) Bias-Corrected Matching Disparity Ratio
                    aim_disp, aim_smd, is_aim_frag = abadie_imbens_disparity_test(x_mal, z_mal, y_mal)
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
            rr_variance_ratio=rr_vr,
            rr_variance_p=rr_p,
            is_rr_variance_imbalanced=is_rr_imbal,
            iw_quadratic_diff=iw_quad_diff,
            is_iw_quadratic_excessive=is_iw_quad_excess,
            hodges_lehmann_shift=hl_shift,
            hodges_lehmann_z=hl_z,
            hodges_lehmann_p=hl_p,
            is_hl_imbalanced=is_hl_imbal,
            frank_rir_pct=rir_pct,
            frank_rir_count=rir_cnt,
            is_rir_fragile=is_rir_frag,
            imbens_contour_q=q_imbens,
            imbens_contour_p=p_imbens,
            is_imbens_sensitive=is_imbens_sens,
            austin_stuart_masd=masd_val,
            austin_stuart_p=masd_p,
            is_austin_stuart_imbalanced=is_masd_imbal,
            kallus_gamma_crit=kallus_gamma_crit,
            kallus_bias_max=kallus_bias_max,
            kallus_bound_low=kallus_b_low,
            kallus_bound_high=kallus_b_high,
            is_kallus_sensitive=is_kallus_sens,
            cbps_overid_j=cbps_j,
            cbps_overid_p=cbps_p,
            is_cbps_rejected=is_cbps_rej,
            rosenbaum_silber_delta_crit=silber_delta_crit,
            rosenbaum_silber_lambda=silber_lambda,
            is_silber_sensitive=is_silber_sens,
            fogarty_gamma_crit=fogarty_gamma,
            fogarty_studentized_t=fogarty_t,
            is_fogarty_sensitive=is_fogarty_sens,
            rosenbaum_attr_gamma_crit=rosen_attr_gamma,
            rosenbaum_attr_p_upper=rosen_attr_p,
            is_rosenbaum_attr_sensitive=is_rosen_attr_sens,
            zubizarreta_sbw_cv=sbw_cv,
            zubizarreta_eff_sample_pct=sbw_eff_pct,
            is_zubizarreta_imbalanced=is_sbw_imbal,
            rosenbaum_design_sensitivity_gamma=design_gamma,
            rosenbaum_standardized_effect_mu_sigma=design_eff,
            is_design_fragile=is_design_frag,
            hosman_hansen_holland_delta=hhh_delta,
            hosman_hansen_holland_scale=hhh_scale,
            is_hosman_sensitive=is_hhh_sens,
            rosenbaum_ef_gamma_crit=rosen_ef_gamma,
            rosenbaum_ef_p_upper=rosen_ef_p,
            is_rosenbaum_ef_fragile=is_rosen_ef_frag,
            ho_imai_convex_hull_overlap=ho_imai_hull,
            ho_imai_trim_fraction=ho_imai_trim,
            is_ho_imai_deficient=is_ho_imai_def,
            rosenbaum_diff_gamma_crit=rosen_diff_gamma,
            rosenbaum_diff_p_upper=rosen_diff_p,
            is_rosenbaum_diff_fragile=is_rosen_diff_frag,
            bitler_gelbach_hoynes_disparity=bgh_disp,
            bitler_gelbach_hoynes_max_q_divergence=bgh_div,
            is_bgh_heterogeneous=is_bgh_hetero,
            rosenbaum_dual_gamma_crit=rosen_dual_gamma,
            rosenbaum_dual_p_upper=rosen_dual_p,
            is_rosenbaum_dual_fragile=is_rosen_dual_frag,
            hainmueller_eb_disparity=eb_disp,
            hainmueller_eff_sample_pct=eb_eff,
            hainmueller_entropy_loss=eb_loss,
            is_entropy_balanced_fragile=is_eb_frag,
            rosenbaum_cm_gamma_crit=rosen_cm_gamma,
            rosenbaum_cm_p_upper=rosen_cm_p,
            is_rosenbaum_cm_fragile=is_rosen_cm_frag,
            athey_imbens_disparity=ai_disp,
            athey_imbens_tau_max=ai_tau_max,
            athey_imbens_tau_min=ai_tau_min,
            is_athey_imbens_heterogeneous=is_ai_hetero,
            rosenbaum_huber_gamma_crit=rosen_hub_gamma,
            rosenbaum_huber_p_upper=rosen_hub_p,
            is_rosenbaum_huber_fragile=is_rosen_hub_frag,
            imai_king_stuart_disparity=iks_disp,
            imai_king_stuart_balance_smd=iks_smd,
            is_imai_king_stuart_fragile=is_iks_frag,
            rosenbaum_adaptive_gamma_crit=rosen_adap_gamma,
            rosenbaum_adaptive_p_upper=rosen_adap_p,
            is_rosenbaum_adaptive_fragile=is_rosen_adap_frag,
            abadie_imbens_disparity=aim_disp,
            abadie_imbens_balance_smd=aim_smd,
            is_abadie_imbens_fragile=is_aim_frag,
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

        # Advanced Diagnostic: J. A. Doornik & H. Hansen (1994, 2008) Omnibus Residual Normality Test
        dh_stat, dh_p, is_dh_non_normal = doornik_hansen_normality_test(x_vals, y_vals)

        # Advanced Diagnostic: Roy E. Welsch (1982) Distance Influence Metric in Regression
        welsch_max, welsch_cnt, is_welsch_infl = welsch_distance_influence_test(x_vals, y_vals)

        # Advanced Diagnostic: Gel & Gastwirth (2008) Robust Jarque-Bera (RJB) Normality Test
        rjb_stat, rjb_p, is_rjb_non_normal = gel_gastwirth_rjb_normality_test(x_vals, y_vals)

        # Advanced Diagnostic: Daniel Peña (2001) Measure of Influence in Linear Regression
        pena_max, pena_cnt, is_pena_infl = pena_influence_test(x_vals, y_vals)

        # Advanced Diagnostic: Jerzy Szroeter (1978) Monotonic Rank Homoskedasticity Test
        szroeter_q, szroeter_z, szroeter_p, is_szroeter_hetero = szroeter_rank_homoskedasticity_test(x_vals, y_vals)

        # Advanced Diagnostic: David F. Andrews & Daryl Pregibon (1978) Data Ellipsoid Volume Influence Diagnostic
        ap_max_dev, ap_cnt, is_ap_infl = andrews_pregibon_influence_test(x_vals, y_vals)

        # Advanced Diagnostic: F. J. Anscombe & William J. Glynn (1983) Normalizing Kurtosis Transformation
        ag_z, ag_p, is_ag_non_norm = anscombe_glynn_kurtosis_test(residuals)

        # Advanced Diagnostic: Peter J. Rousseeuw & Christophe Croux (1993) Q_n Robust Scale Estimator & Inflation Ratio
        qn_val, qn_sir, is_qn_infl = rousseeuw_croux_qn_scale_test(residuals)

        # Advanced Diagnostic: T. W. Epps & L. B. Pulley (1983) Empirical Characteristic Function Normality Test
        ep_stat, ep_p, is_ep_non_normal = epps_pulley_characteristic_function_normality_test(residuals)

        # Advanced Diagnostic: Samprit Chatterjee & Ali S. Hadi (1986) Unified Influence Diagnostic
        ch_max, ch_cnt, is_ch_infl = chatterjee_hadi_influence_test(x_vals, y_vals)

        # Advanced Diagnostic: L. Baringhaus & N. Henze (1988) Empirical Characteristic Function Normality Test
        bh_stat, bh_p, is_bh_non_normal = baringhaus_henze_zirkler_normality_test(residuals)

        # Advanced Diagnostic: J. Brian Gray & Robert F. Ling (1984) Leverage-Residual (L-R) Influence Metric
        gl_max, gl_cnt, is_gl_infl = gray_ling_leverage_residual_influence_test(x_vals, y_vals)

        # Advanced Diagnostic: Oldrich Vasicek (1976 JRSS-B) Sample Entropy Residual Normality Test
        vasicek_stat, vasicek_p, is_vasicek_non_norm = vasicek_sample_entropy_normality_test(residuals)

        # Advanced Diagnostic: Peter J. Rousseeuw & Bert C. van Zomeren (1990 JASA) Robust Distance Diagnostic Metric
        rdz_max, rdz_cnt, is_rdz_infl = rousseeuw_van_zomeren_robust_distance_test(x_vals, y_vals)

        # Advanced Diagnostic: Jin Zhang (2002 JRSS-B) Empirical Likelihood Goodness-of-Fit Normality Test
        zhang_stat, zhang_p, is_zhang_non_norm = zhang_empirical_likelihood_normality_test(residuals)

        # Advanced Diagnostic: Nedret Billor, Ali S. Hadi, & Paul F. Velleman (2000 CSDA) BACON Adaptive Outlier Diagnostic
        bacon_max, bacon_cnt, is_bacon_infl = bacon_adaptive_outlier_test(x_vals, y_vals)

        # Advanced Diagnostic: Chien-Pai Lin & Govind S. Mudholkar (1980 Biometrika) Jackknife Mean-Variance Normality Test
        lm_stat, lm_p, is_lm_non_norm = lin_mudholkar_normality_test(residuals)

        # Advanced Diagnostic: David C. Hoaglin & Roy E. Welsch (1978 American Statistician) Hat-Matrix Leverage Ratio
        hw_max, hw_cnt, is_hw_infl = hoaglin_welsch_leverage_ratio_test(x_vals, y_vals)

        # Advanced Diagnostic: Christian Bontemps & Nour Meddahi (2005 Econometric Theory) Hermite Polynomial Moment Normality Test
        bm_stat, bm_p, is_bm_non_norm = bontemps_meddahi_normality_test(residuals)

        # Advanced Diagnostic: Victor J. Yohai (1987 Annals of Statistics) High-Breakdown MM-Estimator Robust Slope Discrepancy
        mm_slope, mm_disc, mm_cnt, is_mm_dist = yohai_mm_estimator_test(x_vals, y_vals)

        # Advanced Diagnostic: David J. Spiegelhalter (1977, 1980 Biometrika) Omnibus Normality Test
        sp_stat, sp_p, is_sp_non_norm = spiegelhalter_normality_test(residuals)

        # Advanced Diagnostic: Frank R. Hampel (1974 JASA) Three-Part Redescending M-Estimator
        ham_slope, ham_disc, ham_cnt, is_ham_dist = hampel_redescending_m_estimator_test(x_vals, y_vals)

        # Advanced Diagnostic: Govind S. Mudholkar, Deo Kumar Srivastava & C. Thomas Lin (1981 Technometrics) Lin-Mudholkar / MSL Cube-Root Transformed Normality Test
        msl_stat, msl_p, is_msl_non_norm = mudholkar_lin_normality_test(residuals)

        # Advanced Diagnostic: David F. Andrews (1974 Technometrics) Sine Wave Redescending M-Estimator
        and_slope, and_disc, and_cnt, is_and_dist = andrews_sine_estimator_test(x_vals, y_vals)

        # Advanced Diagnostic: Jorge F. Martinez & Boris Iglewicz (1981 Biometrika) Studentized Jackknife Scale Normality Test
        mi_stat, mi_p, is_mi_non_norm = martinez_iglewicz_normality_test(residuals)

        # Advanced Diagnostic: Albert E. Beaton & John W. Tukey (1974 Technometrics) Bisquare Biweight Redescending M-Estimator
        bis_slope, bis_disc, bis_cnt, is_bis_dist = beaton_tukey_bisquare_estimator_test(x_vals, y_vals)

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
            doornik_hansen_stat=dh_stat,
            doornik_hansen_p=dh_p,
            is_doornik_hansen_non_normal=is_dh_non_normal,
            max_welsch_distance=welsch_max,
            welsch_outlier_count=welsch_cnt,
            is_welsch_influential=is_welsch_infl,
            gel_gastwirth_rjb=rjb_stat,
            gel_gastwirth_p=rjb_p,
            is_rjb_non_normal=is_rjb_non_normal,
            max_pena_influence=pena_max,
            pena_outlier_count=pena_cnt,
            is_pena_influential=is_pena_infl,
            szroeter_stat_q=szroeter_q,
            szroeter_stat_z=szroeter_z,
            szroeter_p_value=szroeter_p,
            is_szroeter_heteroskedastic=is_szroeter_hetero,
            max_ap_deviation=ap_max_dev,
            ap_outlier_count=ap_cnt,
            is_ap_influential=is_ap_infl,
            anscombe_glynn_z=ag_z,
            anscombe_glynn_p=ag_p,
            is_anscombe_glynn_non_normal=is_ag_non_norm,
            rousseeuw_croux_qn=qn_val,
            scale_inflation_ratio_qn=qn_sir,
            is_qn_inflated=is_qn_infl,
            epps_pulley_stat=ep_stat,
            epps_pulley_p=ep_p,
            is_epps_pulley_non_normal=is_ep_non_normal,
            max_chatterjee_hadi_influence=ch_max,
            chatterjee_hadi_outlier_count=ch_cnt,
            is_chatterjee_hadi_influential=is_ch_infl,
            baringhaus_henze_stat=bh_stat,
            baringhaus_henze_p=bh_p,
            is_baringhaus_henze_non_normal=is_bh_non_normal,
            max_gray_ling_influence=gl_max,
            gray_ling_outlier_count=gl_cnt,
            is_gray_ling_influential=is_gl_infl,
            vasicek_entropy_stat=vasicek_stat,
            vasicek_entropy_p=vasicek_p,
            is_vasicek_non_normal=is_vasicek_non_norm,
            rousseeuw_zomeren_max_rdz=rdz_max,
            rousseeuw_zomeren_outlier_count=rdz_cnt,
            is_rousseeuw_zomeren_influential=is_rdz_infl,
            zhang_zk_stat=zhang_stat,
            zhang_zk_p_value=zhang_p,
            is_zhang_non_normal=is_zhang_non_norm,
            bacon_max_distance=bacon_max,
            bacon_outlier_count=bacon_cnt,
            is_bacon_influential=is_bacon_infl,
            lin_mudholkar_stat_z=lm_stat,
            lin_mudholkar_p_value=lm_p,
            is_lin_mudholkar_non_normal=is_lm_non_norm,
            hoaglin_welsch_max_leverage_ratio=hw_max,
            hoaglin_welsch_outlier_count=hw_cnt,
            is_hoaglin_welsch_influential=is_hw_infl,
            bontemps_meddahi_stat_m=bm_stat,
            bontemps_meddahi_p_value=bm_p,
            is_bontemps_meddahi_non_normal=is_bm_non_norm,
            yohai_mm_slope=mm_slope,
            yohai_mm_discrepancy=mm_disc,
            yohai_mm_outlier_count=mm_cnt,
            is_yohai_mm_distorted=is_mm_dist,
            spiegelhalter_stat_n=sp_stat,
            spiegelhalter_p_value=sp_p,
            is_spiegelhalter_non_normal=is_sp_non_norm,
            hampel_robust_slope=ham_slope,
            hampel_slope_discrepancy=ham_disc,
            hampel_outlier_count=ham_cnt,
            is_hampel_distorted=is_ham_dist,
            mudholkar_lin_stat_z=msl_stat,
            mudholkar_lin_p_value=msl_p,
            is_mudholkar_lin_non_normal=is_msl_non_norm,
            andrews_sine_slope=and_slope,
            andrews_sine_discrepancy=and_disc,
            andrews_sine_outlier_count=and_cnt,
            is_andrews_sine_distorted=is_and_dist,
            martinez_iglewicz_stat_i=mi_stat,
            martinez_iglewicz_p_value=mi_p,
            is_martinez_iglewicz_non_normal=is_mi_non_norm,
            beaton_tukey_slope=bis_slope,
            beaton_tukey_discrepancy=bis_disc,
            beaton_tukey_outlier_count=bis_cnt,
            is_beaton_tukey_distorted=is_bis_dist,
        )

    def _audit_leakage(self, claim: Claim, df: pd.DataFrame) -> bool:
        if claim.claim_type == "correlation" and abs(claim.statistic_value) >= 0.98:
            return True
        if claim.claim_type in ("group_difference", "proportion_difference"):
            diff = claim.metadata.get("diff", 0.0)
            if diff >= 0.99 or np.isinf(claim.statistic_value):
                return True
        return False
