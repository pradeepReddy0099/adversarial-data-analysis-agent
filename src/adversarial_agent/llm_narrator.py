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

        if any(w in q_lower for w in ["e-value", "e value", "e_value", "vanderweele", "unmeasured"]) or ("sensitivity" in q_lower and "rosenbaum" not in q_lower and "gamma" not in q_lower):
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

        if "tarone" in q_lower or (any(w in q_lower for w in ["homogeneity", "interaction", "effect modification"]) and not any(k in q_lower for k in ["freeman", "woolf", "breslow"])):
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

        if any(w in q_lower for w in ["robustness value", "cinelli", "hazlett"]) or " rv " in f" {q_lower} " or "(rv)" in q_lower or "rv_" in q_lower:
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

        if any(w in q_lower for w in ["i-squared", "i^2", "heterogeneity", "cochran"]) and not any(w in q_lower for w in ["woolf", "haldane", "anscombe"]):
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

        if (any(w in q_lower for w in ["breusch", "pagan"]) or "heteroscedastic" in q_lower) and "white" not in q_lower:
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

        if any(w in q_lower for w in ["vuong", "non-nested", "kullback", "model selection"]):
            vuong_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                z_v = simp.get("vuong_z")
                p_v = simp.get("vuong_p")
                fav = simp.get("is_stratification_favored", False)
                if p_v is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    conf = simp.get("confounder_column")
                    flag = " ⚠️ Stratified Model Significantly Favored (Rejects Pooled)" if fav else " (Models Statistically Equivalent)"
                    vuong_items.append(f"• **{v_name}** stratified by **{conf}**: Vuong z = {z_v:.2f} (p = {p_v:.3e}){flag}")
            if vuong_items:
                return (
                    "**Vuong (1989) Likelihood Ratio Non-Nested Model Selection Test:**\n"
                    "Compares pooled OLS vs subgroup-stratified model in Kullback-Leibler information distance (rejection indicates pooled claim is misspecified):\n"
                    + "\n".join(vuong_items)
                )
            return "Vuong's non-nested likelihood ratio test rigorously determines whether stratified conditioning provides superior probabilistic fit over crude unstratified claims."

        if any(w in q_lower for w in ["breslow-day", "breslow day", "odds ratio homogeneity"]):
            bd_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                bd_chi2 = simp.get("breslow_day_chi2")
                bd_p = simp.get("breslow_day_p")
                if bd_p is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    conf = simp.get("confounder_column")
                    flag = " ⚠️ Heterogeneous Odds Ratios" if bd_p < 0.05 else " (Homogeneous Across Strata)"
                    bd_items.append(f"• **{v_name}** across **{conf}**: Breslow-Day χ² = {bd_chi2:.2f} (p = {bd_p:.3e}){flag}")
            if bd_items:
                return (
                    "**Breslow-Day (1980) Test for Odds Ratio Homogeneity:**\n"
                    "Tests null hypothesis of equal odds ratios across 2x2xK contingency tables:\n"
                    + "\n".join(bd_items)
                )
            return "Breslow-Day diagnostic tests whether subgroup odds ratios are homogeneous across strata."

        if any(w in q_lower for w in ["itcv", "frank", "impact threshold"]):
            itcv_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                itcv = conf.get("frank_itcv")
                impact = conf.get("frank_observed_impact")
                exceeded = conf.get("is_itcv_exceeded", False)
                if itcv is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = " ⚠️ ITCV Exceeded (Inference Invalidated)" if exceeded else " (Below Threshold)"
                    itcv_items.append(f"• **{v_name}** vs **{c_name}**: ITCV = {itcv:.3f}, Observed Impact = {impact:.3f}{flag}")
            if itcv_items:
                return (
                    "**Kenneth Frank (2000) Impact Threshold for a Confounding Variable (ITCV):**\n"
                    "Quantifies the exact correlation product (r_xz * r_yz) an unmeasured confounder must have to nullify empirical significance:\n"
                    + "\n".join(itcv_items)
                )
            return "Frank's ITCV determines the minimum product of confounder correlations needed to overturn statistical significance."

        if any(w in q_lower for w in ["akaike", "aic weight", "evidence ratio", "aicc"]):
            aic_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                daic = conf.get("akaike_delta_aic")
                er = conf.get("akaike_evidence_ratio")
                w_aic = conf.get("akaike_model_weight")
                if w_aic is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = " ⚠️ Decisive Confounder Support (w1 > 0.95)" if w_aic > 0.95 else f" (w1 = {w_aic:.3f})"
                    aic_items.append(f"• **{v_name}** vs **{c_name}**: ΔAIC = {daic:+.2f}, Evidence Ratio = {er:.2f}, Model Weight = {w_aic:.3f}{flag}")
            if aic_items:
                return (
                    "**Burnham-Anderson / Akaike AIC Evidence Ratios & Information Model Weights:**\n"
                    "Computes relative likelihood and posterior model probabilities between unadjusted and confounder-adjusted models:\n"
                    + "\n".join(aic_items)
                )
            return "Akaike model weights quantify the probability that the confounder-adjusted model represents the true data-generating process."

        if any(w in q_lower for w in ["theil-sen", "theil sen", "median slope", "robust regression"]):
            ts_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                ts_slope = stab.get("theil_sen_slope")
                ts_disc = stab.get("theil_sen_discrepancy")
                if ts_slope is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ High Discrepancy ({ts_disc:.1%})" if ts_disc and ts_disc > 0.20 else f" (Discrepancy = {ts_disc:.1%})"
                    ts_items.append(f"• **{v_name}**: Theil-Sen β_TS = {ts_slope:.4f}{flag}")
            if ts_items:
                return (
                    "**Theil-Sen (1950) Robust Non-Parametric Median Slope Estimator:**\n"
                    "High breakdown-point (29.3%) non-parametric median pairwise slope resisting catastrophic outlier distortion:\n"
                    + "\n".join(ts_items)
                )
            return "Theil-Sen robust estimator computes the median of all pairwise slopes to certify resistance against non-Gaussian outliers."

        if any(w in q_lower for w in ["bca", "bias corrected", "accelerated bootstrap", "bca bootstrap"]):
            bca_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                bca_l = stab.get("bca_ci_low")
                bca_h = stab.get("bca_ci_high")
                bca_z = stab.get("bca_zero_crossing", False)
                if bca_l is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Zero-Crossing Detected" if bca_z else " (Stable Non-Zero)"
                    bca_items.append(f"• **{v_name}**: BCa 95% CI = [{bca_l:.4f}, {bca_h:.4f}]{flag}")
            if bca_items:
                return (
                    "**Efron & Tibshirani (1993) Bias-Corrected and Accelerated (BCa) Bootstrap:**\n"
                    "Second-order accurate bootstrap confidence intervals adjusting for median bias (z0) and skewness acceleration (a):\n"
                    + "\n".join(bca_items)
                )
            return "BCa bootstrap intervals provide second-order accurate confidence limits correcting for parameter skew and non-normality."

        if any(w in q_lower for w in ["rosenbaum", "hidden bias", "gamma bound"]) or " gamma " in f" {q_lower} " or "γ" in q_lower:
            rosen_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                g_crit = conf.get("rosenbaum_gamma_crit")
                p_up = conf.get("rosenbaum_p_upper")
                is_sens = conf.get("is_rosenbaum_sensitive", False)
                if g_crit is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Sensitive to Hidden Bias (Γcrit < 1.30)" if is_sens else " (Robust to Hidden Bias Γcrit ≥ 1.30)"
                    rosen_items.append(f"• **{v_name}**: Rosenbaum Γcrit = {g_crit:.2f} (p_upper = {p_up:.3e}){flag}")
            if rosen_items:
                return (
                    "**Paul Rosenbaum (1987, 2002) Sensitivity Bounds (Γ) for Hidden Bias:**\n"
                    "Quantifies the odds ratio differential in treatment assignment that an unmeasured confounder must produce to overturn statistical significance:\n"
                    + "\n".join(rosen_items)
                )
            return "Rosenbaum's Gamma bound determines the degree of hidden bias needed to invalidate causal or empirical conclusions."

        if any(w in q_lower for w in ["white test", "white's test", "white lm", "white heteroscedasticity"]):
            white_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                w_lm = stab.get("white_lm_stat")
                w_p = stab.get("white_lm_p")
                is_w_het = stab.get("is_white_heteroscedastic", False)
                if w_p is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ General Heteroscedasticity Detected" if is_w_het else " (Homoscedastic Errors)"
                    white_items.append(f"• **{v_name}**: White LM = {w_lm:.2f} (p = {w_p:.3e}){flag}")
            if white_items:
                return (
                    "**Halbert White (1980) General Heteroscedasticity Lagrange Multiplier Test:**\n"
                    "Regresses squared residuals on regressors and their squares (LM = n*R² ~ χ²(2)) to detect non-linear variance heteroscedasticity:\n"
                    + "\n".join(white_items)
                )
            return "White's general heteroscedasticity test evaluates non-linear error variance patterns without assuming normality."

        if any(w in q_lower for w in ["manski", "partial identification", "identification bounds", "sharp bounds"]):
            manski_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                m_l = conf.get("manski_bound_low")
                m_h = conf.get("manski_bound_high")
                m_z = conf.get("manski_contains_zero", False)
                if m_l is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Contains Zero (Sign Non-Identified)" if m_z else " (Identified Sign - Excludes Zero)"
                    manski_items.append(f"• **{v_name}**: Manski Bounds = [{m_l:.3f}, {m_h:.3f}]{flag}")
            if manski_items:
                return (
                    "**Charles Manski (1990, 2003) Non-Parametric Partial Identification Bounds:**\n"
                    "Evaluates worst-case causal effect bounds under arbitrary selection bias without imposing parametric distribution assumptions:\n"
                    + "\n".join(manski_items)
                )
            return "Manski non-parametric bounds establish the sharp identification region under worst-case unmeasured selection bias."

        if any(w in q_lower for w in ["altonji", "elder", "taber", "aet", "selection ratio", "selection on unobservables"]):
            aet_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                r_aet = conf.get("aet_selection_ratio")
                is_vuln = conf.get("is_aet_vulnerable", False)
                if r_aet is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = " ⚠️ Highly Vulnerable to Unobservables (R_AET < 1.0)" if is_vuln else " (Robust to Unobservables R_AET ≥ 1.0)"
                    aet_items.append(f"• **{v_name}** vs **{c_name}**: Altonji-Elder-Taber R_AET = {r_aet:.2f}{flag}")
            if aet_items:
                return (
                    "**Altonji, Elder & Taber (2005) Selection Ratio on Unobservables (R_AET):**\n"
                    "Measures how much stronger selection on unmeasured confounders must be relative to observed covariates to attribute the entire effect to selection bias:\n"
                    + "\n".join(aet_items)
                )
            return "Altonji-Elder-Taber selection ratio measures how much stronger selection on unobservables must be relative to observables to nullify the effect."

        if "chow" in q_lower or (any(w in q_lower for w in ["structural break", "parameter constancy", "regime shift", "regime instability"]) and "andrews" not in q_lower and "sup" not in q_lower and "wald" not in q_lower):
            chow_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                f_c = stab.get("chow_f_stat")
                p_c = stab.get("chow_p_value")
                has_b = stab.get("has_structural_break", False)
                if p_c is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Structural Break / Regime Shift Detected" if has_b else " (Parameter Invariance Maintained)"
                    chow_items.append(f"• **{v_name}**: Chow F = {f_c:.2f} (p = {p_c:.3e}){flag}")
            if chow_items:
                return (
                    "**Gregory Chow (1960) Structural Break & Parameter Stability Test:**\n"
                    "Tests parameter equality across partitioned data regimes (median split on x). Rejection detects structural non-stationarity:\n"
                    + "\n".join(chow_items)
                )
            return "Chow test evaluates parameter constancy and structural stability across subsample regimes."

        if any(w in q_lower for w in ["galbraith", "radial outlier", "radial plot", "outlier strata", "higgins h", "excess heterogeneity"]):
            galb_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                h2 = simp.get("higgins_h_squared")
                cnt = simp.get("galbraith_outlier_count", 0)
                outliers = simp.get("galbraith_outlier_subgroups", [])
                if h2 is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    conf = simp.get("confounder_column")
                    outlier_str = f" [Outliers: {', '.join(outliers)}]" if cnt > 0 else " [No Outlier Strata]"
                    galb_items.append(f"• **{v_name}** stratified by **{conf}**: Higgins H² = {h2:.2f}, {cnt} Galbraith outliers{outlier_str}")
            if galb_items:
                return (
                    "**Higgins H² Relative Heterogeneity & Galbraith (1988) Radial Outlier Detection:**\n"
                    "Quantifies proportional variance inflation (H² = Q / (k-1)) and flags anomalous subgroup strata exerting excessive leverage:\n"
                    + "\n".join(galb_items)
                )
            return "Higgins H² and Galbraith radial diagnostics evaluate total heterogeneity inflation and isolate outlying strata driving paradoxes."

        if any(w in q_lower for w in ["woolf", "haldane", "anscombe"]):
            woolf_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                w_p = simp.get("woolf_p")
                w_chi2 = simp.get("woolf_chi2")
                is_het = simp.get("is_woolf_heterogeneous", False)
                if w_p is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    conf = simp.get("confounder_column")
                    flag = " ⚠️ Statistically Significant Heterogeneity (p < 0.05)" if is_het else " (Homogeneous Across Strata)"
                    woolf_items.append(f"• **{v_name}** conditioned on **{conf}**: Woolf χ² = {w_chi2:.2f} (p = {w_p:.3e}){flag}")
            if woolf_items:
                return (
                    "**Barnet Woolf (1955) Stratified Odds Ratio Heterogeneity Test:**\n"
                    "Evaluates cross-stratum odds ratio invariance with Haldane-Anscombe 0.5 continuity correction for sparse cells:\n"
                    + "\n".join(woolf_items)
                )
            return "Woolf's test assesses odds ratio homogeneity across stratified contingency tables using inverse-variance weights."

        if any(w in q_lower for w in ["imbens", "manski ci", "im ci", "partially identified ci"]):
            im_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                im_l = conf.get("imbens_manski_ci_low")
                im_h = conf.get("imbens_manski_ci_high")
                im_cn = conf.get("imbens_manski_cn")
                excl = conf.get("imbens_manski_excludes_zero", False)
                if im_l is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " 🟢 Excludes Zero (Identified Directional Sign)" if excl else " ⚠️ Spans Zero (Partially Identified Sign)"
                    im_items.append(f"• **{v_name}**: Imbens-Manski 95% CI = [{im_l:.3f}, {im_h:.3f}] (c_n = {im_cn:.3f}){flag}")
            if im_items:
                return (
                    "**Guido Imbens & Charles Manski (2004) Partial Identification Confidence Intervals:**\n"
                    "Derives asymptotically valid 95% confidence intervals accounting for non-parametric boundary sampling variability:\n"
                    + "\n".join(im_items)
                )
            return "Imbens-Manski confidence intervals provide exact statistical coverage for partially identified parameters under arbitrary non-random selection."

        if any(w in q_lower for w in ["mallows", "cp criterion", "cp discrepancy", "specification bias"]):
            mal_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                cp = conf.get("mallows_cp")
                disc = conf.get("cp_bias_discrepancy")
                is_bad = conf.get("is_mallows_biased", False)
                if cp is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = " ⚠️ Severe Omitted-Variable Bias (ΔCp > 2.0)" if is_bad else " (Well-Specified Model)"
                    mal_items.append(f"• **{v_name}** omitting **{c_name}**: Mallows' Cp = {cp:.2f} (ΔCp = {disc:+.2f}){flag}")
            if mal_items:
                return (
                    "**Colin Mallows (1973) Cp Model Misspecification & Selection Criterion:**\n"
                    "Compares restricted bivariate models against confounder-adjusted specifications. Cp >> p (ΔCp > 2.0) reveals severe omitted-variable bias:\n"
                    + "\n".join(mal_items)
                )
            return "Mallows' Cp criterion evaluates whether omitting a candidate confounder introduces substantial model bias and prediction variance inflation."

        if any(w in q_lower for w in ["gelman", "carlin", "type s", "type m", "exaggeration", "retrodesign", "winner's curse", "sign error"]):
            type_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                tm = stab.get("type_m_exaggeration_ratio")
                ts = stab.get("type_s_error_prob")
                rp = stab.get("retrodesign_power")
                vuln = stab.get("is_type_m_vulnerable", False)
                if tm is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ High Exaggeration / Winner's Curse Vulnerability" if vuln else " (Stable Effect Size)"
                    type_items.append(f"• **{v_name}**: Type M Exaggeration = {tm:.2f}x (Type S Sign Error = {ts:.1%}, Power = {rp:.1%}){flag}")
            if type_items:
                return (
                    "**Andrew Gelman & John Carlin (2014) Type S (Sign) & Type M (Magnitude) Design Analysis:**\n"
                    "Evaluates the probability of incorrect directional signs (Type S) and factor of effect inflation (Type M) conditional on statistical significance:\n"
                    + "\n".join(type_items)
                )
            return "Gelman-Carlin design analysis quantifies Type S sign error rates and Type M exaggeration ratios to safeguard against the Winner's Curse."

        if any(w in q_lower for w in ["freeman", "tukey", "double arcsine", "double-arcsine"]):
            ft_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                ft_p = simp.get("freeman_tukey_p")
                ft_chi2 = simp.get("freeman_tukey_chi2")
                is_het = simp.get("is_freeman_tukey_heterogeneous", False)
                if ft_p is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    conf = simp.get("confounder_column")
                    flag = " ⚠️ Statistically Significant Heterogeneity (p < 0.05)" if is_het else " (Variance-Stabilized Homogeneity)"
                    ft_items.append(f"• **{v_name}** conditioned on **{conf}**: Freeman-Tukey χ² = {ft_chi2:.2f} (p = {ft_p:.3e}){flag}")
            if ft_items:
                return (
                    "**Freeman & Tukey (1950) Variance-Stabilizing Double-Arcsine Homogeneity Test:**\n"
                    "Applies variance-stabilizing angular transformations (Var(θ_k) = 1 / (4n_k + 2)) to prevent distortion from sparse strata or extreme proportions:\n"
                    + "\n".join(ft_items)
                )
            return "Freeman-Tukey double-arcsine transformation tests proportion homogeneity across strata while stabilizing binomial sampling variance."

        if any(w in q_lower for w in ["kruskal", "wallis", "rank sum", "rank heterogeneity"]):
            kw_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                kw_p = simp.get("kruskal_wallis_p")
                kw_h = simp.get("kruskal_wallis_h")
                is_het = simp.get("is_kruskal_heterogeneous", False)
                if kw_p is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    conf = simp.get("confounder_column")
                    flag = " ⚠️ Statistically Significant Strata Shift (p < 0.05)" if is_het else " (Invariant Distribution Medians)"
                    kw_items.append(f"• **{v_name}** stratified by **{conf}**: Kruskal-Wallis H = {kw_h:.2f} (p = {kw_p:.3e}){flag}")
            if kw_items:
                return (
                    "**Kruskal & Wallis (1952) Non-Parametric Rank-Sum Heterogeneity Test:**\n"
                    "Evaluates whether the distribution medians of the target differ across strata without assuming Gaussian normality:\n"
                    + "\n".join(kw_items)
                )
            return "Kruskal-Wallis non-parametric rank test evaluates whether subgroup target distributions differ significantly across strata."

        if any(w in q_lower for w in ["davidson", "mackinnon", "j-test", "j test", "non-nested", "non nested"]):
            j_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                j_p = conf.get("davidson_mackinnon_p")
                j_t = conf.get("davidson_mackinnon_j_stat")
                is_rej = conf.get("is_j_test_rejected", False)
                if j_p is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = " ⚠️ Original Model Rejected (Confounder Specification Favored, p < 0.05)" if is_rej else " (Original Specification Adequate)"
                    j_items.append(f"• **{v_name}** vs Confounder **{c_name}**: Davidson-MacKinnon t_J = {j_t:.2f} (p = {j_p:.3e}){flag}")
            if j_items:
                return (
                    "**Russell Davidson & James MacKinnon (1981) J-Test for Non-Nested Model Specification:**\n"
                    "Tests bivariate claims against alternative confounder specifications by augmenting the model with predicted values from the rival model:\n"
                    + "\n".join(j_items)
                )
            return "Davidson-MacKinnon J-test tests non-nested model specifications to verify whether candidate confounders capture critical explanatory power."

        if any(w in q_lower for w in ["andrews", "sup w", "sup-w", "supremum wald", "unknown break", "change point"]):
            and_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                w_stat = stab.get("andrews_sup_w")
                frac = stab.get("andrews_break_fraction")
                p_val = stab.get("andrews_p_value")
                has_brk = stab.get("has_unknown_structural_break", False)
                if w_stat is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Unknown Structural Break at τ* = {frac:.1%} (p < 0.05)" if has_brk else f" (Parameter Constant along Regressor, τ* = {frac:.1%})"
                    and_items.append(f"• **{v_name}**: Andrews Sup W = {w_stat:.2f} (p = {p_val:.3e}){flag}")
            if and_items:
                return (
                    "**Donald Andrews (1993) Supremum Wald Test for Unknown Structural Breaks:**\n"
                    "Evaluates parameter invariance across all interior candidate split points τ ∈ [0.15, 0.85] with Hansen (1997) asymptotic p-values:\n"
                    + "\n".join(and_items)
                )
            return "Andrews' Supremum Wald test detects structural breaks and regime shifts at unknown change points along sorted regressors."

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
        lines.append("Our neuro-symbolic multi-agent architecture verifies statistical hypotheses across 10 continuous and categorical stress tests:")
        lines.append("1. **Cochran-Mantel-Haenszel, Breslow-Day & Tarone Homogeneity, Woolf Stratified Test, Freeman-Tukey Double-Arcsine Homogeneity, Kruskal-Wallis Rank Heterogeneity, Higgins I² & H², DerSimonian-Laird τ², Egger/Begg Funnel Tests, Galbraith Radial Outliers, Duval-Tweedie Trim & Fill, Hausman Endogeneity & Vuong Model Selection:** Verifies whether stratified odds ratios reverse, tests cross-stratum invariance with Haldane-Anscombe continuity corrections (Woolf's test), applies variance-stabilizing angular transformations (Freeman-Tukey test), tests non-parametric median shifts across strata (Kruskal-Wallis $H$), quantifies true heterogeneity percentage ($I^2$) and excess variance inflation ($H^2, \\tau^2$), flags leverage-inducing strata via Galbraith radial standardized deviations ($|z_i| \\ge 2.0$), estimates missing strata via Trim & Fill, verifies endogeneity via Hausman specification test, and proves whether stratified conditioning dominates crude models via Vuong's likelihood ratio test.")
        lines.append("2. **VanderWeele E-Values, Cinelli-Hazlett Robustness Values (RV), Critical Confounder Boundary (R²crit), Cohen's f², Mallows' Cp Criterion & Davidson-MacKinnon J-Test:** Establishes mathematical bounds, minimal partial $R^2$ strength, critical confounder boundaries, unique variance effect sizes ($f^2$), Mallows' $C_p$ model misspecification discrepancy ($\\Delta C_p > 2.0$), and Davidson-MacKinnon $J$-tests arbitrating non-nested specification dominance between bivariate claims and rival confounder models.")
        lines.append("3. **Altonji-Elder-Taber (AET 2005) Selection Ratio, Rosenbaum (1987, 2002) Sensitivity Bounds (Γ), Manski (1990, 2003) Non-Parametric Partial Identification & Imbens-Manski (2004) Asymptotically Valid CIs:** Evaluates the relative selection ratio ($R_{\\text{AET}}$) on unobservables required to attribute effects to selection bias, calculates critical odds ratio differential threshold ($\\Gamma_{\\text{crit}}$) required for hidden bias to overturn findings, computes worst-case non-parametric sharp identification intervals $[\\Delta_{\\text{low}}, \\Delta_{\\text{high}}]$ under arbitrary selection, and constructs exact Imbens-Manski confidence intervals accounting for boundary sampling variability.")
        lines.append("4. **Frank's (2000) Confounding Impact Threshold (ITCV) & Burnham-Anderson AIC Information Weights:** Quantifies the exact correlation product threshold an unmeasured confounder must exceed to overturn statistical significance, and computes Akaike model selection weights ($w_1$) arbitrating covariate necessity.")
        lines.append("5. **Ramsey RESET Specification Test & Oster (2019) Selection Bound (δ*):** Evaluates functional form misspecification/omitted non-linearities and calculates proportional selection on unobservables needed to nullify effect estimates.")
        lines.append("6. **Kass-Raftery Bayes Factors & BIC Model Selection:** Computes exact posterior evidence odds ($BF_{10} = \\exp(\\Delta \\text{BIC} / 2)$) arbitrating between bivariate claims and confounder-adjusted multivariate models.")
        lines.append("7. **Baron-Kenny Sobel Mediation Diagnostics:** Disentangles true exogenous confounding from valid intermediate mechanistic transmission ($X \\to Z \\to Y$).")
        lines.append("8. **Theil-Sen Robust Non-Parametric Median Regression & Efron-Tibshirani BCa Bootstrap:** Compares OLS against breakdown-point 29.3% Theil-Sen median slopes and evaluates second-order accurate bias-corrected and accelerated (BCa) bootstrap confidence intervals.")
        lines.append("9. **Gregory Chow (1960) Structural Break Test, Donald Andrews (1993) Supremum Wald Test (Sup W), Vectorized Bootstrap, Cook's Distance, DFFITS, DFBETAS, Breusch-Pagan, White LM, Jarque-Bera, Durbin-Watson & MacKinnon-White HC3:** Evaluates parameter invariance across data partitions (Chow $F$-test), detects structural change points at unknown locations along sorted regressors (Andrews $\\operatorname{Sup} W$), empirical confidence intervals, sign invariance probability ($P \\ge 90\\%$), parameter-specific influence (DFBETAS), overall leverage (Cook's D, DFFITS), linear & non-linear heteroscedasticity (Breusch-Pagan, White LM), residual Gaussian normality, autocorrelation (Durbin-Watson), and heteroscedasticity-consistent robust inference ($HC_3$).")
        lines.append("10. **Andrew Gelman & John Carlin (2014) Type S/M Design Analysis, Quantile Discrepancy Index, Exact Permutation Shuffle & FDR/FWER Correction:** Assesses conditional slope invariance across 25th/50th/75th percentiles, quantifies Type S sign error probabilities and Type M exaggeration ratios to safeguard against the Winner's Curse, validates empirical null distributions, and controls false discovery rates via Benjamini-Hochberg FDR and Bonferroni FWER.")

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
