"""Devil's Advocate (Auditor) Agent: Autonomous statistical falsification batteries."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
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
                # E-value for unmeasured confounding
                simpson_e_val, simpson_e_desc = calculate_e_value(cmh_or)
        else:
            bd_stat, bd_p = None, None

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

        cmh_note = f" [CMH Common OR: {cmh_or:.3f}, p={cmh_p:.3e}]" if cmh_or is not None and cmh_p is not None else ""
        if bd_p is not None:
            cmh_note += f" [Breslow-Day: p={bd_p:.3e}]"
        if tarone_p is not None:
            cmh_note += f" [Tarone Homogeneity: p={tarone_p:.3e}]"
        if simpson_e_val is not None:
            cmh_note += f" [E-value: {simpson_e_val:.2f}]"
        if i_sq is not None:
            cmh_note += f" [I² Heterogeneity: {i_sq:.1f}%]"
        if begg_p is not None:
            cmh_note += f" [Begg Funnel: τ={begg_tau:.2f}, p={begg_p:.3e}]"
        if tau_sq is not None:
            cmh_note += f" [DerSimonian-Laird τ²: {tau_sq:.4f}]"
        if egger_p is not None:
            cmh_note += f" [Egger Test: t={egger_t:.2f}, p={egger_p:.3e}]"
        if tf_k0 > 0:
            cmh_note += f" [Trim & Fill: k0={tf_k0}, adj={tf_stat:.4f}]"
        if haus_p is not None:
            cmh_note += f" [Hausman Test: H={haus_stat:.2f}, p={haus_p:.3e}]"
        if vuong_p_val is not None:
            cmh_note += f" [Vuong Test: z={vuong_z_val:.2f}, p={vuong_p_val:.3e}]"

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

        is_conf = max_att >= self.confounder_attenuation_threshold and top_conf is not None
        is_collinear = top_vif is not None and top_vif >= 5.0
        vif_str = f", VIF={top_vif:.2f}" if top_vif is not None else ""
        rv_str = f", RV={rv_point:.2f} (R²crit={rcrit_val:.2f})"
        sob_str = f", Sobel p={sobel_p:.3e}" if sobel_p is not None else ""
        bf_str = f", BF10={bf10:.2f}" if bf10 is not None else ""
        reset_str = f", RESET p={reset_p:.3e}" if reset_p is not None else ""
        oster_str = f", Oster δ*={oster_delta_val:.2f}" if oster_delta_val is not None else ""
        f2_str = f", Cohen's f²={f2_val:.3f}" if f2_val is not None else ""
        itcv_str = f", ITCV={itcv_val:.3f}" if itcv_val is not None else ""
        aic_str = f", w_AIC={w_aic_val:.3f}" if w_aic_val is not None else ""
        desc = (
            f"Correlation between '{source}' and '{target}' collapses from r = {r_xy:.4f} (R²={raw_r2:.3f}) to partial r = {best_partial:.4f} (partial R²={partial_r2:.3f}) "
            f"({max_att:.1f}% attenuation, {variance_loss:.1f}% variance loss{vif_str}{rv_str}{sob_str}{bf_str}{reset_str}{oster_str}{f2_str}{itcv_str}{aic_str}) when controlling for '{top_conf}'. [E-value: {e_val:.2f}]"
            if is_conf
            else f"Relationship is robust to linear confounding across tested covariates (max attenuation: {max_att:.1f}%{vif_str}{rv_str}{bf_str}{reset_str}{oster_str}{f2_str}{itcv_str}{aic_str}). [E-value: {e_val:.2f}]"
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

        is_stable = (not zero_crossing) and (not bca_zero) and (np.sign(trim_r) == np.sign(claim.statistic_value)) and (sign_inv_prob >= 0.90) and (perm_p < 0.05)

        disc_str = f", Δr_rank={rank_disc:.3f}" if rank_disc is not None else ""
        cooks_str = f", Cook's D_max={max_cooks:.3f} ({cooks_count} influential)" if max_cooks > 0 else ""
        dffits_str = f", DFFITS_max={max_dffits_val:.3f}" if max_dffits_val > 0 else ""
        dfb_str = f", DFBETAS_max={max_dfbetas_val:.3f}" if max_dfbetas_val > 0 else ""
        bp_str = f", BP_p={bp_p:.3e}" if bp_p is not None else ""
        jb_str = f", JB_p={jb_p:.3e}" if jb_p is not None else ""
        qdi_str = f", QDI={qdi_val:.2f}" if qdi_val is not None else ""
        dw_str = f", DW={dw_stat:.2f}" if dw_stat is not None else ""
        hc3_str = f", HC3_p={hc3_p:.3e}" if hc3_p is not None else ""
        ts_str = f", Δ_TS={ts_disc:.1%}" if ts_disc is not None else ""
        bca_str = f", BCa=[{bca_low:.4f}, {bca_high:.4f}]"
        desc = f"Vectorized Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]{bca_str} (SE={std_err:.4f}, P(sign)={sign_inv_prob:.1%}, p_perm={perm_p:.3e}, {exec_ms:.1f}ms{disc_str}{cooks_str}{dffits_str}{dfb_str}{bp_str}{jb_str}{qdi_str}{dw_str}{hc3_str}{ts_str}). Trimmed r = {trim_r:.4f}."

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
        )

    def _audit_leakage(self, claim: Claim, df: pd.DataFrame) -> bool:
        if claim.claim_type == "correlation" and abs(claim.statistic_value) >= 0.98:
            return True
        if claim.claim_type in ("group_difference", "proportion_difference"):
            diff = claim.metadata.get("diff", 0.0)
            if diff >= 0.99 or np.isinf(claim.statistic_value):
                return True
        return False
