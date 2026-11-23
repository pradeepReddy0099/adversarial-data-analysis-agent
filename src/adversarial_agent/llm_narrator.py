"""LLM Narrator & Executive Copilot for Adversarial Data Analysis Agent.

Acts as a neuro-symbolic bridge: takes the deterministic Referee Dossier
and generates human-readable, domain-aware executive briefings and answers
user queries. Operates with offline heuristic synthesis or live LLM APIs.
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional, List


def calculate_risk_score(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes an automated Executive Decision Risk Score (0-100 scale)."""
    if not findings:
        return {"score": 0, "tier": "LOW", "badge": "🟢 LOW RISK", "summary": "No findings to evaluate."}

    points = 0
    paradox_count = 0
    confounder_count = 0
    unstable_count = 0
    leakage_count = 0
    robust_count = 0

    for f in findings:
        verdict = f.get("verdict", "")
        badge = f.get("verdict_badge", "")
        if "SIMPSON" in verdict or "SIMPSON" in badge:
            points += 35
            paradox_count += 1
        elif "LEAKAGE" in verdict or "LEAKAGE" in badge:
            points += 30
            leakage_count += 1
        elif "CONFOUNDED" in verdict or "CONFOUNDED" in badge or "SPURIOUS" in verdict:
            points += 20
            confounder_count += 1
        elif "UNSTABLE" in verdict or "UNSTABLE" in badge:
            points += 15
            unstable_count += 1
        elif "ROBUST" in verdict or "ROBUST" in badge:
            robust_count += 1

    max_possible = max(1, len(findings) * 35)
    normalized_score = min(100, int((points / max_possible) * 100))

    if normalized_score >= 70 or paradox_count >= 2:
        tier = "CRITICAL"
        badge = "🔴 CRITICAL RISK"
        summary = "Severe risk of catastrophic policy/business errors if acting on aggregate associations. Multiple subgroup inversions detected."
    elif normalized_score >= 45 or paradox_count >= 1:
        tier = "HIGH"
        badge = "🟠 HIGH RISK"
        summary = "Substantial confounding and subgroup reversals present. Disaggregated policy intervention required."
    elif normalized_score >= 20:
        tier = "MODERATE"
        badge = "🟡 MODERATE RISK"
        summary = "Noticeable covariate mediation or sampling instability detected. Requires controlled verification."
    else:
        tier = "LOW"
        badge = "🟢 LOW RISK"
        summary = "Identified associations demonstrate high invariance across sub-populations and pass stability stress tests."

    return {
        "score": normalized_score,
        "tier": tier,
        "badge": badge,
        "summary": summary,
        "counts": {
            "total": len(findings),
            "robust": robust_count,
            "simpson": paradox_count,
            "confounded": confounder_count,
            "unstable": unstable_count,
            "leakage": leakage_count,
        }
    }


class LLMNarrator:
    """Narrates and explains adjudicated findings using LLM APIs or offline heuristic synthesis."""

    def __init__(self, provider: str = "auto", api_key: Optional[str] = None):
        """
        Args:
            provider: 'auto', 'gemini', 'openai', 'anthropic', or 'offline'
            api_key: Optional explicit API key. If omitted, checks environment variables.
        """
        self.provider = provider
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.active_provider = self._detect_provider()

    def _detect_provider(self) -> str:
        if self.provider == "offline":
            return "offline"
        if os.environ.get("GEMINI_API_KEY") or (self.provider == "gemini" and self.api_key):
            return "gemini"
        if os.environ.get("OPENAI_API_KEY") or (self.provider == "openai" and self.api_key):
            return "openai"
        return "offline"

    def generate_executive_briefing(self, findings: List[Dict[str, Any]], dataset_name: str = "Dataset") -> str:
        """Generates an executive briefing explaining paradoxes, robust signals, and confounded relationships."""
        if self.active_provider == "gemini":
            try:
                return self._call_gemini(findings, dataset_name)
            except Exception as e:
                return self._offline_briefing(findings, dataset_name, error_note=f"Gemini API fallback: {e}")
        elif self.active_provider == "openai":
            try:
                return self._call_openai(findings, dataset_name)
            except Exception as e:
                return self._offline_briefing(findings, dataset_name, error_note=f"OpenAI API fallback: {e}")
        else:
            return self._offline_briefing(findings, dataset_name)

    def answer_query(self, query: str, findings: List[Dict[str, Any]], dataset_name: str = "Dataset") -> str:
        """Answers an interactive user question regarding the findings dossier."""
        risk = calculate_risk_score(findings)
        q_lower = query.lower()

        # Offline intelligence for common executive queries
        if any(w in q_lower for w in ["risk", "score", "overall", "health", "safe"]):
            return (
                f"**Executive Decision Risk Score: {risk['score']}/100 ({risk['badge']})**.\n"
                f"{risk['summary']} Across {risk['counts']['total']} claims evaluated: "
                f"{risk['counts']['robust']} are robust, {risk['counts']['simpson']} exhibited Simpson's Paradox, "
                f"and {risk['counts']['confounded']} were confounded."
            )

        if any(w in q_lower for w in ["recommend", "action", "guidance", "next step", "leadership"]):
            paradoxes = [f for f in findings if "SIMPSON" in f.get("verdict", "")]
            if paradoxes:
                top_p = paradoxes[0]
                var = top_p.get("claim", {}).get("source_variable")
                conf = top_p.get("audit", {}).get("simpsons_audit", {}).get("confounder_column", "subgroups")
                return (
                    f"**Immediate Action Directive:** Do not deploy uniform policies targeting '{var}'. "
                    f"Our audit proved that '{var}' inverts direction when conditioned on '{conf}'. "
                    f"Interventions must be segmented at the sub-population level. For robust variables, refer to the verified claims."
                )
            return "All verified findings are invariant across tested covariates. Proceed with controlled rollouts while monitoring stability metrics."

        if any(w in q_lower for w in ["e-value", "e value", "e_value", "unmeasured", "sensitivity", "hidden confound"]):
            e_vals = []
            for f in findings:
                simps_e = f.get("audit", {}).get("simpsons_audit", {}).get("e_value")
                conf_e = f.get("audit", {}).get("confounder_audit", {}).get("e_value")
                ev = simps_e or conf_e
                if ev:
                    e_vals.append(f"• **{f.get('claim', {}).get('source_variable')}**: E-value = {ev:.2f}")
            if e_vals:
                return (
                    "**Causal Sensitivity (VanderWeele-Ding E-values):**\n"
                    "The E-value represents the minimum risk ratio association that an unmeasured confounder must have with both the exposure and outcome to fully nullify the observed effect.\n"
                    + "\n".join(e_vals)
                )
            return "E-value sensitivity metrics are computed for continuous correlations and stratified odds ratios."

        if any(w in q_lower for w in ["tarone", "homogeneity", "interaction", "effect modification"]):
            tarone_findings = [
                f for f in findings
                if f.get("audit", {}).get("simpsons_audit", {}).get("tarone_homogeneity_p") is not None
            ]
            if tarone_findings:
                top_t = tarone_findings[0]
                p_t = top_t["audit"]["simpsons_audit"]["tarone_homogeneity_p"]
                chi_t = top_t["audit"]["simpsons_audit"]["tarone_chi2"]
                v_name = top_t.get("claim", {}).get("source_variable")
                status = "effect modification (odds ratios vary significantly across strata)" if p_t < 0.05 else "homogeneous odds ratios across strata"
                return (
                    f"**Tarone's Homogeneity Test for '{v_name}':** χ² = {chi_t:.2f}, p = {p_t:.3e}. "
                    f"This indicates {status}."
                )
            return "Tarone's test evaluates whether the odds ratio is invariant across subgroups in stratified 2x2xK tables."

        if any(w in q_lower for w in ["dffits", "studentized"]):
            dffits_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                dff = stab.get("max_dffits")
                cnt = stab.get("dffits_outlier_count", 0)
                if dff is not None and dff > 0:
                    v_name = f.get("claim", {}).get("source_variable")
                    dffits_items.append(f"• **{v_name}**: Max |DFFITS| = {dff:.3f} ({cnt} points exceeding 2√(p/n))")
            if dffits_items:
                return (
                    "**DFFITS Regression Influence Diagnostics:**\n"
                    "Measures the number of standard deviations that fitted values change when each observation is removed:\n"
                    + "\n".join(dffits_items)
                )
            return "DFFITS diagnostic identifies observations with outsized influence on predicted values."

        if any(w in q_lower for w in ["cook", "cook's", "leverage", "d_max"]):
            cook_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                d_max = stab.get("max_cooks_distance")
                n_infl = stab.get("cooks_outlier_count", 0)
                shift = stab.get("cooks_statistic_shift", 0.0)
                if d_max is not None and d_max > 0:
                    v_name = f.get("claim", {}).get("source_variable")
                    cook_items.append(f"• **{v_name}**: Max Cook's D = {d_max:.3f} ({n_infl} influential points, correlation shift = {shift:.4f})")
            if cook_items:
                return (
                    "**Cook's Distance Influence Diagnostics (Threshold: D_i > 4/n):**\n"
                    "Quantifies how much regression parameters shift when individual observations are removed:\n"
                    + "\n".join(cook_items)
                )
            return "Cook's distance diagnostic measures the influence of individual data points on parameter stability."

        if any(w in q_lower for w in ["sign invariance", "sign test", "p(sign)", "bayesian"]):
            sign_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                p_sign = stab.get("sign_invariance_prob")
                if p_sign is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    sign_items.append(f"• **{v_name}**: P(sign invariance) = {p_sign:.1%}")
            if sign_items:
                return (
                    "**Empirical Bootstrap Sign Invariance Probabilities:**\n"
                    "The posterior empirical probability that the true effect maintains its claimed direction across resampling:\n"
                    + "\n".join(sign_items)
                )
            return "Sign invariance probability measures directional consistency across 150 non-parametric bootstrap replications."

        if any(w in q_lower for w in ["vif", "collinear", "multicollinear"]):
            vif_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                vif = conf.get("top_confounder_vif")
                if vif is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    top_c = conf.get("top_confounder")
                    flag = " ⚠️ High collinearity (VIF >= 5.0)" if conf.get("is_collinear") else " (Stable VIF < 5.0)"
                    vif_items.append(f"• **{v_name}** controlled by **{top_c}**: VIF = {vif:.2f}{flag}")
            if vif_items:
                return (
                    "**Variance Inflation Factor (VIF) Multicollinearity Diagnostics:**\n"
                    "VIF measures variance inflation when controlling for candidate confounders:\n"
                    + "\n".join(vif_items)
                )
            return "VIF diagnostics evaluate multicollinearity between candidate predictors and confounders."

        if any(w in q_lower for w in ["fdr", "false discovery", "hochberg", "benjamini", "multiple testing"]):
            fdr_items = []
            for f in findings:
                q = f.get("claim", {}).get("fdr_q_value")
                if q is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    p_orig = f.get("claim", {}).get("p_value", 0.0)
                    fdr_items.append(f"• **{v_name}**: Raw p = {p_orig:.3e} → Benjamini-Hochberg FDR q = {q:.3e}")
            if fdr_items:
                return (
                    "**Multiple Hypothesis Testing Correction (Benjamini-Hochberg FDR):**\n"
                    "Guarantees that across all evaluated claims, the expected proportion of false discoveries is bounded:\n"
                    + "\n".join(fdr_items)
                )
            return "Benjamini-Hochberg FDR q-values adjust raw p-values across all discovered candidate claims."

        if any(w in q_lower for w in ["robustness value", "cinelli", "hazlett", "rv"]):
            rv_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                rv = conf.get("robustness_value")
                rv_a = conf.get("robustness_value_alpha")
                if rv is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    rv_items.append(f"• **{v_name}**: RV_q=1 = {rv:.2f} (RV_α=0.05 = {rv_a:.2f})")
            if rv_items:
                return (
                    "**Cinelli & Hazlett (2020) Robustness Value (RV):**\n"
                    "Quantifies the minimum strength of association (partial R²) an unmeasured confounder must have with both exposure and outcome to overturn the finding:\n"
                    + "\n".join(rv_items)
                )
            return "Cinelli-Hazlett Robustness Values define the minimum confounder association needed to nullify point estimates or significance."

        if any(w in q_lower for w in ["i-squared", "i^2", "heterogeneity", "cochran"]):
            i2_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                i2 = simp.get("i_squared")
                q_stat = simp.get("cochrans_q")
                if i2 is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    conf = simp.get("confounder_column")
                    i2_items.append(f"• **{v_name}** stratified by **{conf}**: I² = {i2:.1f}% (Cochran's Q = {q_stat:.2f})")
            if i2_items:
                return (
                    "**Higgins & Thompson I² Inconsistency Index & Cochran's Q:**\n"
                    "Measures the percentage of cross-stratum variation due to genuine heterogeneity rather than random sampling error:\n"
                    + "\n".join(i2_items)
                )
            return "Higgins I² index quantifies cross-stratum effect heterogeneity across conditioning variables."

        if any(w in q_lower for w in ["sobel", "mediation", "baron-kenny", "indirect"]):
            sobel_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                z_sob = conf.get("sobel_z")
                p_sob = conf.get("sobel_p")
                p_med = conf.get("proportion_mediated")
                if p_sob is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    sobel_items.append(f"• **{v_name}** → **{c_name}**: Sobel z = {z_sob:.2f}, p = {p_sob:.3e} ({p_med:.1f}% mediated)")
            if sobel_items:
                return (
                    "**Baron-Kenny & Sobel Mediation Test:**\n"
                    "Evaluates whether covariate attenuation is driven by an intermediate causal mechanism (mediation) rather than pure confounding:\n"
                    + "\n".join(sobel_items)
                )
            return "The Sobel test formally evaluates whether a covariate acts as a significant mediating pathway."

        if any(w in q_lower for w in ["permutation", "shuffle"]):
            perm_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                p_perm = stab.get("permutation_p_value")
                if p_perm is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    perm_items.append(f"• **{v_name}**: Permutation p = {p_perm:.3e}")
            if perm_items:
                return (
                    "**Exact Random Permutation Shuffle Falsification Tests:**\n"
                    "Non-parametric null distribution tests assessing whether the observed effect can arise by chance under label shuffling:\n"
                    + "\n".join(perm_items)
                )
            return "Permutation tests simulate the exact empirical null distribution by shuffling exposure values."

        if any(w in q_lower for w in ["bonferroni", "fwer"]):
            bonf_items = []
            for f in findings:
                b_p = f.get("claim", {}).get("bonferroni_p_value")
                if b_p is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    p_orig = f.get("claim", {}).get("p_value", 0.0)
                    bonf_items.append(f"• **{v_name}**: Raw p = {p_orig:.3e} → Bonferroni p_adj = {b_p:.3e}")
            if bonf_items:
                return (
                    "**Bonferroni FWER Multiple Testing Correction:**\n"
                    "Controls Family-Wise Error Rate (FWER) across m hypotheses: p_adj = min(1.0, m * p):\n"
                    + "\n".join(bonf_items)
                )
            return "Bonferroni correction strictly controls family-wise error rate across multiple hypotheses."

        if any(w in q_lower for w in ["bayes factor", "bic", "kass", "raftery"]):
            bf_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                bf = conf.get("bayes_factor_bf10")
                dbic = conf.get("bic_delta")
                interp = conf.get("bayes_factor_interpretation", "")
                if bf is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    bf_items.append(f"• **{v_name}** vs **{c_name}**: BF_10 = {bf:.2f} (ΔBIC = {dbic:+.2f}) — {interp}")
            if bf_items:
                return (
                    "**Kass & Raftery (1995) Bayes Factor & BIC Model Selection:**\n"
                    "Quantifies the posterior odds ratio favoring the confounder-conditioned model M1 over bivariate claim M0:\n"
                    + "\n".join(bf_items)
                )
            return "Bayes Factors (BF_10) provide Bayesian evidence arbitration between bivariate and confounder-adjusted models."

        if any(w in q_lower for w in ["begg", "funnel"]):
            begg_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                tau = simp.get("begg_funnel_tau")
                p_val = simp.get("begg_funnel_p")
                if p_val is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    conf = simp.get("confounder_column")
                    begg_items.append(f"• **{v_name}** across **{conf}**: Begg's τ = {tau:.2f} (p = {p_val:.3e})")
            if begg_items:
                return (
                    "**Begg & Mazumdar Funnel Plot Asymmetry Test across Strata:**\n"
                    "Tests rank correlation between stratum effect sizes and sampling precision to detect small-subgroup distortion:\n"
                    + "\n".join(begg_items)
                )
            return "Begg's funnel test detects whether effect sizes systematically diverge in small strata."

        if any(w in q_lower for w in ["heteroscedastic", "breusch", "pagan"]):
            bp_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                lm = stab.get("breusch_pagan_lm")
                p_val = stab.get("breusch_pagan_p")
                is_h = stab.get("is_heteroscedastic", False)
                if p_val is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Heteroscedastic" if is_h else " (Homoscedastic)"
                    bp_items.append(f"• **{v_name}**: LM = {lm:.2f}, p = {p_val:.3e}{flag}")
            if bp_items:
                return (
                    "**Breusch-Pagan Lagrange Multiplier Heteroscedasticity Test:**\n"
                    "Tests null hypothesis of constant error variance across exposure levels:\n"
                    + "\n".join(bp_items)
                )
            return "Breusch-Pagan diagnostic tests whether residual variance is non-constant across exposure levels."

        if any(w in q_lower for w in ["quantile", "qdi", "distributional"]):
            qdi_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                qdi = stab.get("quantile_discrepancy_index")
                q_flag = stab.get("qdi_flag", False)
                if qdi is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ High Quantile Discrepancy" if q_flag else " (Uniform Distribution)"
                    qdi_items.append(f"• **{v_name}**: QDI = {qdi:.2f}{flag}")
            if qdi_items:
                return (
                    "**Quantile Discrepancy Index (QDI) across 25th, 50th, 75th Percentiles:**\n"
                    "Quantifies divergence between interquartile regression slopes to detect asymmetric conditional responses:\n"
                    + "\n".join(qdi_items)
                )
            return "Quantile Discrepancy Index measures whether conditional slopes vary dramatically across outcome quantiles."

        if any(w in q_lower for w in ["egger", "egger's"]):
            egger_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                t_stat = simp.get("egger_test_t")
                p_val = simp.get("egger_test_p")
                if p_val is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    conf = simp.get("confounder_column")
                    flag = " ⚠️ Asymmetric" if p_val < 0.05 else " (Symmetric)"
                    egger_items.append(f"• **{v_name}** across **{conf}**: Egger's t = {t_stat:.2f} (p = {p_val:.3e}){flag}")
            if egger_items:
                return (
                    "**Egger et al. (1997) Linear Regression Test for Funnel Asymmetry:**\n"
                    "Regresses standardized effect on precision to detect small-subgroup effect distortion or reporting bias:\n"
                    + "\n".join(egger_items)
                )
            return "Egger's test evaluates funnel plot asymmetry and small-subgroup effect distortion."

        if any(w in q_lower for w in ["dersimonian", "laird", "tau-squared", "tau^2", "between-strata"]):
            tau_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                tau_sq = simp.get("tau_squared")
                if tau_sq is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    conf = simp.get("confounder_column")
                    tau_items.append(f"• **{v_name}** stratified by **{conf}**: DerSimonian-Laird τ² = {tau_sq:.4f}")
            if tau_items:
                return (
                    "**DerSimonian & Laird (1986) Between-Stratum Variance (τ²):**\n"
                    "Quantifies the absolute magnitude of true between-subgroup effect variance beyond random sampling noise:\n"
                    + "\n".join(tau_items)
                )
            return "DerSimonian-Laird τ² estimates the between-stratum effect variance in random-effects meta-analysis."

        if any(w in q_lower for w in ["ramsey", "reset", "misspecification"]):
            reset_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                f_stat = conf.get("ramsey_reset_f")
                p_val = conf.get("ramsey_reset_p")
                is_bad = conf.get("is_reset_misspecified", False)
                if p_val is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = " ⚠️ Functional Misspecification" if is_bad else " (Well-Specified)"
                    reset_items.append(f"• **{v_name}** with **{c_name}**: RESET F = {f_stat:.2f} (p = {p_val:.3e}){flag}")
            if reset_items:
                return (
                    "**Ramsey (1969) RESET Test for Functional Form & Non-linear Confounding:**\n"
                    "Tests whether higher powers of fitted values (ŷ², ŷ³) improve fit, detecting omitted polynomial or interaction confounding:\n"
                    + "\n".join(reset_items)
                )
            return "Ramsey's RESET tests for functional form misspecification and omitted non-linear confounding."

        if any(w in q_lower for w in ["oster", "unobservable selection", "delta bound"]):
            oster_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                d_star = conf.get("oster_delta")
                if d_star is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = " (Robust: |δ*| ≥ 1.0)" if abs(d_star) >= 1.0 else " ⚠️ Vulnerable: |δ*| < 1.0"
                    oster_items.append(f"• **{v_name}** vs **{c_name}**: Oster δ* = {d_star:.2f}{flag}")
            if oster_items:
                return (
                    "**Emily Oster (2019) Selection on Unobservables Bound (δ*):**\n"
                    "Measures how much stronger selection on unobserved confounders must be relative to observed covariates to explain away the effect (R_max = min(1.0, 1.3*R²)):\n"
                    + "\n".join(oster_items)
                )
            return "Oster's δ* bounds the degree of unobservable selection needed to nullify the observed effect."

        if any(w in q_lower for w in ["dfbetas", "slope influence"]):
            dfb_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                max_dfb = stab.get("max_dfbetas")
                cnt = stab.get("dfbetas_outlier_count", 0)
                if max_dfb is not None and max_dfb > 0:
                    v_name = f.get("claim", {}).get("source_variable")
                    dfb_items.append(f"• **{v_name}**: Max |DFBETAS| = {max_dfb:.3f} ({cnt} observations exceeding 2/√n)")
            if dfb_items:
                return (
                    "**Belsley, Kuh & Welsch (1980) DFBETAS Slope Influence Diagnostic:**\n"
                    "Measures the standardized shift in the regression slope coefficient caused by deleting each observation:\n"
                    + "\n".join(dfb_items)
                )
            return "DFBETAS identifies individual observations exerting excessive leverage on the estimated slope parameter."

        if any(w in q_lower for w in ["jarque", "bera", "normality"]):
            jb_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                jb_stat = stab.get("jarque_bera_stat")
                p_val = stab.get("jarque_bera_p")
                is_non = stab.get("is_non_normal", False)
                if p_val is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Non-Gaussian" if is_non else " (Normal Residuals)"
                    jb_items.append(f"• **{v_name}**: JB = {jb_stat:.2f} (p = {p_val:.3e}){flag}")
            if jb_items:
                return (
                    "**Jarque-Bera Goodness-of-Fit Residual Normality Test:**\n"
                    "Tests null hypothesis of Gaussian residual distribution based on skewness and kurtosis:\n"
                    + "\n".join(jb_items)
                )
            return "Jarque-Bera diagnostic tests whether regression residuals satisfy Gaussian normality."

        if any(w in q_lower for w in ["trim and fill", "trim-and-fill", "duval", "publication bias", "adjusted effect"]):
            tf_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                k0 = simp.get("trim_and_fill_k0")
                adj = simp.get("trim_and_fill_adjusted_stat")
                if k0 is not None and k0 > 0:
                    v_name = f.get("claim", {}).get("source_variable")
                    tf_items.append(f"• **{v_name}**: {k0} missing strata imputed, adjusted effect = {adj:.4f}")
            if tf_items:
                return (
                    "**Duval & Tweedie (2000) Trim and Fill Asymmetry Algorithm:**\n"
                    "Imputes suppressed or missing strata on the asymmetric side of the funnel plot to produce a bias-corrected estimate:\n"
                    + "\n".join(tf_items)
                )
            return "Duval & Tweedie Trim and Fill algorithm tests for publication-style or stratum selection asymmetry across subgroups."

        if any(w in q_lower for w in ["hausman", "specification test", "endogeneity", "fixed effects"]):
            h_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                h_stat = simp.get("hausman_stat")
                h_p = simp.get("hausman_p")
                is_incon = simp.get("is_hausman_inconsistent", False)
                if h_p is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Inconsistent / Endogenous" if is_incon else " (Consistent)"
                    h_items.append(f"• **{v_name}**: H = {h_stat:.2f} (p = {h_p:.3e}){flag}")
            if h_items:
                return (
                    "**Hausman (1978) Stratification Specification Test:**\n"
                    "Tests whether Within-Stratum Fixed Effects deviate systematically from Pooled OLS (rejection confirms endogeneity):\n"
                    + "\n".join(h_items)
                )
            return "Hausman specification test formally verifies whether subgroup-level confounding causes pooled estimates to be inconsistent."

        if any(w in q_lower for w in ["cohen's f2", "cohen f2", "f-squared", "f2", "rcrit", "critical boundary"]):
            f2_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                f2 = conf.get("cohens_f2")
                rcrit = conf.get("cinelli_hazlett_rcrit")
                if f2 is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    f2_s = f"f² = {f2:.3f}"
                    rcrit_s = f", R²crit = {rcrit:.2f}" if rcrit is not None else ""
                    f2_items.append(f"• **{v_name}**: {f2_s}{rcrit_s}")
            if f2_items:
                return (
                    "**Cohen's f² Effect Size & Cinelli-Hazlett Critical Boundary:**\n"
                    "Quantifies the unique effect size added by exposure beyond confounders and the critical partial R² needed to nullify effect:\n"
                    + "\n".join(f2_items)
                )
            return "Cohen's f² measures conditional effect size, and Cinelli-Hazlett R²crit identifies the minimal confounder nullification boundary."

        if any(w in q_lower for w in ["durbin", "watson", "serial correlation", "autocorrelation"]):
            dw_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                dw = stab.get("durbin_watson_stat")
                has_s = stab.get("has_serial_correlation", False)
                if dw is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Serial Correlation Detected" if has_s else " (Independent Residuals)"
                    dw_items.append(f"• **{v_name}**: DW = {dw:.2f}{flag}")
            if dw_items:
                return (
                    "**Durbin-Watson (1950) Serial Autocorrelation Test:**\n"
                    "Evaluates first-order residual independence (DW ~ 2.0 indicates independence; DW < 1.5 indicates clustered/ordered sampling):\n"
                    + "\n".join(dw_items)
                )
            return "Durbin-Watson test detects non-independent ordering or spatial/serial clustering in regression residuals."

        if any(w in q_lower for w in ["hc3", "heteroscedasticity-robust", "robust standard error", "mackinnon", "sandwich"]):
            hc3_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                hc3_se = stab.get("hc3_robust_se")
                hc3_p = stab.get("hc3_robust_p")
                lost = stab.get("hc3_significance_lost", False)
                if hc3_p is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Significance Lost" if lost else " (Robust to Heteroscedasticity)"
                    hc3_items.append(f"• **{v_name}**: SE_HC3 = {hc3_se:.4f} (p_HC3 = {hc3_p:.3e}){flag}")
            if hc3_items:
                return (
                    "**MacKinnon & White (1985) HC3 Robust Standard Error Test:**\n"
                    "Computes jackknife-weighted heteroscedasticity-consistent standard errors to guard against inflated false positive rates:\n"
                    + "\n".join(hc3_items)
                )
            return "MacKinnon-White HC3 test evaluates whether findings retain statistical significance under robust sandwich covariance estimation."

        context_str = json.dumps([{
            "pair": f"{f.get('claim', {}).get('source_variable')} -> {f.get('claim', {}).get('target_variable')}",
            "verdict": f.get("verdict"),
            "referee_summary": f.get("synthesis_summary") or f.get("referee_explanation"),
            "takeaway": f.get("actionable_takeaway"),
            "cmh_odds_ratio": f.get("audit", {}).get("simpsons_audit", {}).get("cmh_odds_ratio"),
        } for f in findings], indent=2)

        prompt = (
            f"You are the Adversarial Data Analysis AI Copilot. The user is asking about an empirical audit of '{dataset_name}'.\n"
            f"Here is the adjudicated findings dossier:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            "Answer directly, citing exact subgroup reversals, confounders, or CMH odds ratios when relevant. Keep your answer concise (2-4 sentences)."
        )

        if self.active_provider == "gemini":
            try:
                import urllib.request
                key = self.api_key or os.environ.get("GEMINI_API_KEY")
                for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                        req = urllib.request.Request(
                            url,
                            data=json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8"),
                            headers={"Content-Type": "application/json"}
                        )
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            data = json.loads(resp.read().decode("utf-8"))
                            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    except Exception:
                        continue
            except Exception:
                pass

        # Offline targeted search matching
        matched = []
        for f in findings:
            src = f.get('claim', {}).get('source_variable', '').lower()
            tgt = f.get('claim', {}).get('target_variable', '').lower()
            verdict = f.get('verdict', '').lower()
            if src in q_lower or tgt in q_lower or verdict in q_lower:
                matched.append(f)

        if matched:
            top = matched[0]
            summary = top.get('synthesis_summary') or top.get('referee_explanation') or ''
            takeaway = top.get('actionable_takeaway') or ''
            return f"Regarding '{top.get('claim', {}).get('source_variable')}': {summary} {takeaway}"

        return (
            f"Across {len(findings)} evaluated variables in '{dataset_name}', the Referee adjudicated "
            f"{risk['counts']['robust']} robust, {risk['counts']['simpson']} paradoxical, and {risk['counts']['confounded']} confounded relationships. "
            "You can ask about any specific variable name to see its stratified breakdown."
        )

    def _call_gemini(self, findings: List[Dict[str, Any]], dataset_name: str) -> str:
        import urllib.request
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        risk = calculate_risk_score(findings)

        prompt = (
            f"You are a Senior Principal Data Scientist and AI Auditor reviewing adjudicated findings from '{dataset_name}'.\n"
            f"System Executive Risk Score: {risk['score']}/100 ({risk['badge']}).\n\n"
            "Explain in clear, compelling, executive-level language:\n"
            "1. Executive Decision Risk Scorecard and why naive correlations would mislead leadership.\n"
            "2. Detailed breakdown of any detected Simpson's Paradoxes (citing subgroup reversals or Cochran-Mantel-Haenszel odds ratios).\n"
            "3. Confounder collapse and which findings are 100% robust and safe for policy/business decisions.\n"
            "4. Immediate actionable leadership directives.\n\n"
            f"Findings Data:\n{json.dumps(findings, indent=2)}\n\n"
            "Keep the response structured with clear Markdown headers, bullet points, and under 400 words."
        )

        last_err = None
        for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"All Gemini endpoints failed: {last_err}")

    def _call_openai(self, findings: List[Dict[str, Any]], dataset_name: str) -> str:
        import urllib.request
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        risk = calculate_risk_score(findings)
        url = "https://api.openai.com/v1/chat/completions"
        prompt = (
            f"Analyze these audited statistical findings for '{dataset_name}' (Risk Score: {risk['score']}/100 - {risk['badge']}) "
            f"and provide an executive summary highlighting Simpson's Paradoxes, confounder collapse, and robust signals:\n{json.dumps(findings, indent=2)}"
        )
        payload = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        })
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()

    def _offline_briefing(self, findings: List[Dict[str, Any]], dataset_name: str, error_note: Optional[str] = None) -> str:
        """Deterministic neuro-symbolic narrative generator (Zero-API, 100% offline)."""
        risk = calculate_risk_score(findings)
        robust = [f for f in findings if "ROBUST" in f.get("verdict", "") or "ROBUST" in f.get("verdict_badge", "")]
        paradox = [f for f in findings if "SIMPSON" in f.get("verdict", "") or "SIMPSON" in f.get("verdict_badge", "")]
        leakage = [f for f in findings if "LEAKAGE" in f.get("verdict", "") or "LEAKAGE" in f.get("verdict_badge", "")]
        confounded = [f for f in findings if ("CONFOUNDED" in f.get("verdict", "") or "CONFOUNDED" in f.get("verdict_badge", "")) and f not in leakage and f not in paradox]
        unstable = [f for f in findings if "UNSTABLE" in f.get("verdict", "") or "UNSTABLE" in f.get("verdict_badge", "")]

        lines = [
            f"# 📋 Executive Briefing: Adversarial Audit of {dataset_name}",
            "",
            f"### 🛡️ Decision Risk Status: {risk['badge']} ({risk['score']}/100)",
            f"> *{risk['summary']}*",
            "",
            "---",
            "### 📊 Executive Findings Scorecard",
            "| Variable | Claimed Effect | Adjudicated Status | Primary Threat / Invariance | Decision Directive |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]

        for f in findings:
            claim = f.get("claim", {})
            var = claim.get("source_variable", "Unknown")
            stat_name = claim.get("statistic_name", "Stat")
            stat_val = claim.get("statistic_value", 0.0)
            badge = f.get("verdict_badge", "AUDITED")
            verdict = f.get("verdict", "")

            if "SIMPSON" in verdict or "SIMPSON" in badge:
                threat = f"Subgroup Reversal (Stratified: {f.get('audit', {}).get('simpsons_audit', {}).get('confounder_column')})"
                directive = "❌ DO NOT act globally; tailor locally."
            elif "LEAKAGE" in verdict or "LEAKAGE" in badge:
                threat = "Deterministic Target Leakage"
                directive = "🚫 Strip immediately from model pipeline."
            elif "CONFOUNDED" in verdict or "CONFOUNDED" in badge:
                threat = f"Confounded by {f.get('audit', {}).get('confounder_audit', {}).get('top_confounder')}"
                directive = "⚠️ Spurious effect; control for covariates."
            elif "UNSTABLE" in verdict or "UNSTABLE" in badge:
                threat = "Bootstrap CI Crosses Zero"
                directive = "⏳ Collect larger sample before acting."
            else:
                threat = "Verified Invariant Across Subgroups"
                directive = "✅ Safe for business & policy deployment."

            lines.append(f"| **{var}** | {stat_name}={stat_val:.3f} | {badge} | {threat} | {directive} |")

        lines.append("")

        if paradox:
            lines.append("### ⚠️ CRITICAL WARNING: Simpson's Paradox Inversions Detected")
            for p in paradox:
                claim = p.get("claim", {})
                audit = p.get("audit", {}).get("simpsons_audit", {})
                var = claim.get("source_variable")
                conf = audit.get("confounder_column")
                rev_count = audit.get("reversal_subgroups_count", 0)
                tot_count = audit.get("total_subgroups", 0)
                cmh_or = audit.get("cmh_odds_ratio")

                cmh_str = f" Cochran-Mantel-Haenszel common odds ratio = {cmh_or:.3f}." if cmh_or is not None else ""
                lines.append(
                    f"- **{var}:** The apparent aggregate trend completely reverses direction when conditioned on **`{conf}`** "
                    f"({rev_count}/{tot_count} subgroups reverse).{cmh_str} "
                    f"Any global policy assuming the aggregate relationship will produce outcomes exactly opposite to leadership intent."
                )

        if confounded:
            lines.append("\n### 🔍 Confounded & Spurious Associations")
            for c in confounded:
                claim = c.get("claim", {})
                conf_audit = c.get("audit", {}).get("confounder_audit", {})
                var = claim.get("source_variable")
                top_c = conf_audit.get("top_confounder")
                att = conf_audit.get("attenuation_percentage", 0.0)
                var_loss = conf_audit.get("variance_explained_loss_pct", 0.0)
                lines.append(
                    f"- **{var}:** Association collapses by **{att:.1f}%** ({var_loss:.1f}% reduction in explained variance) "
                    f"when controlling for **`{top_c}`**. The naive correlation is mediated by lurking covariates."
                )

        if robust:
            lines.append("\n### ✅ Robust & Verified Invariants")
            for r in robust:
                claim = r.get("claim", {})
                var = claim.get("source_variable")
                stat_val = claim.get("statistic_value", 0.0)
                stab = r.get("audit", {}).get("stability_audit", {})
                p_sign = stab.get("sign_invariance_prob", 1.0)
                lines.append(
                    f"- **{var}:** Passed all four falsification batteries without subgroup reversal or covariate attenuation. "
                    f"Empirical effect ({claim.get('statistic_name')} = {stat_val:.3f}, P(sign invariance) = {p_sign:.1%}) is robust and recommended for deployment."
                )

        if leakage:
            lines.append("\n### 🚫 Target Leakage Detected")
            for lk in leakage:
                var = lk.get("claim", {}).get("source_variable")
                lines.append(f"- **{var}:** Exhibits near-deterministic correlation with target ($r \\ge 0.98$). Feature is a data artifact.")

        lines.append("\n---")
        lines.append("### 🛡️ Dialectical Robustness & Falsification Matrix")
        lines.append("Our neuro-symbolic multi-agent architecture verifies statistical hypotheses across 8 continuous and categorical stress tests:")
        lines.append("1. **Cochran-Mantel-Haenszel, Tarone Homogeneity, Higgins I², DerSimonian-Laird τ², Egger/Begg Funnel Tests, Duval-Tweedie Trim & Fill & Hausman Endogeneity:** Verifies whether stratified odds ratios reverse, tests cross-stratum invariance, quantifies true heterogeneity percentage ($I^2$) and excess variance ($\\tau^2$), tests small-stratum funnel asymmetry, estimates missing strata via Trim & Fill, and verifies endogeneity via Hausman specification test.")
        lines.append("2. **VanderWeele E-Values, Cinelli-Hazlett Robustness Values (RV) & Critical Confounder Boundary (R²crit) & Cohen's f²:** Establishes mathematical bounds, minimal partial $R^2$ strength, critical confounder boundaries, and unique variance effect sizes ($f^2$) needed to nullify empirical associations.")
        lines.append("3. **Ramsey RESET Specification Test & Oster (2019) Selection Bound (δ*):** Evaluates functional form misspecification/omitted non-linearities and calculates proportional selection on unobservables needed to nullify effect estimates.")
        lines.append("4. **Kass-Raftery Bayes Factors & BIC Model Selection:** Computes exact posterior evidence odds ($BF_{10} = \\exp(\\Delta \\text{BIC} / 2)$) arbitrating between bivariate claims and confounder-adjusted multivariate models.")
        lines.append("5. **Baron-Kenny Sobel Mediation Diagnostics:** Disentangles true exogenous confounding from valid intermediate mechanistic transmission ($X \\to Z \\to Y$).")
        lines.append("6. **Non-Parametric Vectorized Bootstrap, Cook's Distance, DFFITS, DFBETAS, Breusch-Pagan, Jarque-Bera, Durbin-Watson & MacKinnon-White HC3:** Evaluates empirical confidence intervals, sign invariance probability ($P \\ge 90\\%$), parameter-specific influence (DFBETAS), overall leverage (Cook's D, DFFITS), error homoscedasticity, residual Gaussian normality, residual autocorrelation (Durbin-Watson), and heteroscedasticity-consistent robust inference ($HC_3$).")
        lines.append("7. **Quantile Discrepancy Index, Exact Permutation Shuffle & FDR/FWER Correction:** Assesses conditional slope invariance across 25th/50th/75th percentiles, validates empirical null distributions, and controls false discovery rates via Benjamini-Hochberg FDR and Bonferroni FWER.")

        lines.append("\n---")
        lines.append("### 🎯 Immediate Leadership Recommendations")
        lines.append("1. **Quarantine Global Policies:** Immediately halt any decision or marketing/admissions rule based on aggregate trends for variables flagged with Simpson's Paradox.")
        lines.append("2. **Implement Sub-Population Segmentation:** Deploy localized interventions tailored to distinct strata identified by our Devil's Advocate Agent.")
        lines.append("3. **Greenlight Robust Variables:** Confidently proceed with feature selection and production models incorporating verified invariants.")

        if error_note:
            lines.append(f"\n*(Running in local deterministic mode. {error_note})*")
        else:
            lines.append("\n*(Generated by ADAA Neuro-Symbolic Narrative Engine — offline deterministic mode)*")

        return "\n".join(lines)
