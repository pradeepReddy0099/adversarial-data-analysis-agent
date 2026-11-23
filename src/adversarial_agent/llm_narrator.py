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
        if (any(w in q_lower for w in ["risk", "overall", "health", "safe"]) or ("score" in q_lower and not any(w in q_lower for w in ["crump", "propensity", "harbord", "score test", "score asymmetry"]))):
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

        if "tarone" in q_lower or (any(w in q_lower for w in ["homogeneity", "interaction", "effect modification"]) and not any(k in q_lower for k in ["freeman", "woolf", "breslow", "fligner", "peto", "brown", "forsythe", "scale"])):
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

        if "dffits" in q_lower or ("studentized residual" in q_lower and not any(k in q_lower for k in ["koenker", "breusch", "bp"])):
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

        if any(w in q_lower for w in ["i-squared", "i^2", "heterogeneity", "cochran"]) and not any(w in q_lower for w in ["woolf", "haldane", "anscombe", "peto", "fligner", "freeman", "kruskal", "rank"]):
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

        if (any(w in q_lower for w in ["breusch", "pagan"]) or "heteroscedastic" in q_lower) and "white" not in q_lower and "koenker" not in q_lower:
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

        if (any(w in q_lower for w in ["watson", "serial correlation", "autocorrelation"]) or ("durbin" in q_lower and not any(k in q_lower for k in ["evans", "cusum", "wu", "hausman"]))):
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

        if any(w in q_lower for w in ["hc3", "heteroscedasticity-robust", "robust standard error", "mackinnon", "sandwich"]) and not any(w in q_lower for w in ["newey", "hac"]):
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

        if any(w in q_lower for w in ["theil-sen", "theil sen", "median slope"]) or ("robust regression" in q_lower and "huber" not in q_lower):
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

        if any(w in q_lower for w in ["imbens", "manski ci", "im ci", "partially identified ci"]) and not any(k in q_lower for k in ["aipw", "doubly robust", "crump"]):
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

        if "peto" in q_lower:
            peto_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                p_or = simp.get("peto_odds_ratio")
                p_z = simp.get("peto_z_stat")
                p_p = simp.get("peto_p_value")
                p_q = simp.get("peto_heterogeneity_q")
                p_qp = simp.get("peto_heterogeneity_p")
                is_het = simp.get("is_peto_heterogeneous", False)
                if p_or is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Peto Strata Heterogeneity (p_het = {p_qp:.3e})" if is_het else f" (Homogeneous Peto Strata, p_het = {p_qp:.3e})"
                    peto_items.append(f"• **{v_name}**: Peto OR = {p_or:.3f} (Z = {p_z:.2f}, p = {p_p:.3e}; Q_Peto = {p_q:.2f}){flag}")
            if peto_items:
                return (
                    "**Peto et al. (1977) Assumption-Free Stratified Odds Ratio & Heterogeneity Test:**\n"
                    "Applies hypergeometric one-step pooled odds ratio estimation with exact finite-sample properties under sparse or unbalanced strata:\n"
                    + "\n".join(peto_items)
                )
            return "Peto's method calculates assumption-free stratified odds ratios and evaluates cross-stratum heterogeneity via the Peto Q statistic."

        if any(w in q_lower for w in ["fligner", "fligner-killeen"]) or ("variance homogeneity" in q_lower and "breusch" not in q_lower and "white" not in q_lower and "koenker" not in q_lower):
            flig_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                f_stat = simp.get("fligner_killeen_stat")
                f_p = simp.get("fligner_killeen_p")
                is_het = simp.get("is_fligner_heterogeneous", False)
                if f_stat is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Heterogeneous Strata Variances (p < 0.05)" if is_het else " (Equal Strata Spread / Homoscedastic)"
                    flig_items.append(f"• **{v_name}**: Fligner-Killeen χ² = {f_stat:.2f} (p = {f_p:.3e}){flag}")
            if flig_items:
                return (
                    "**Fligner & Killeen (1976) Non-Parametric Variance Homogeneity Test:**\n"
                    "Scores ranked absolute deviations from subgroup medians using normal scores to evaluate scale consistency across conditioning strata:\n"
                    + "\n".join(flig_items)
                )
            return "The Fligner-Killeen test assesses whether target variances differ significantly across conditioning subgroups without assuming normality."

        if any(w in q_lower for w in ["dwh", "durbin-wu-hausman", "wu-hausman"]) or ("endogeneity" in q_lower and "hausman" in q_lower and "specification" not in q_lower):
            dwh_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                d_f = conf.get("durbin_wu_hausman_f")
                d_p = conf.get("durbin_wu_hausman_p")
                is_endog = conf.get("is_dwh_endogenous", False)
                if d_f is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = f" ⚠️ Statistically Significant Endogeneity / OVB when omitting '{c_name}' (p < 0.05)" if is_endog else f" (Exogenous Regressor w.r.t. '{c_name}')"
                    dwh_items.append(f"• **{v_name}**: DWH F = {d_f:.2f} (p = {d_p:.3e}){flag}")
            if dwh_items:
                return (
                    "**Durbin-Wu-Hausman (1954, 1973, 1978) Specification Test for Endogeneity:**\n"
                    "Evaluates whether omitting candidate confounders introduces statistically significant endogeneity and bias into the primary predictor:\n"
                    + "\n".join(dwh_items)
                )
            return "The Durbin-Wu-Hausman test determines whether regressor endogeneity and omitted variable bias render bivariate OLS inconsistent."

        if any(w in q_lower for w in ["stock-yogo", "stock yogo", "weak confounder", "first-stage relevance", "first stage f"]):
            sy_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                sy_f = conf.get("stock_yogo_f_stat")
                is_weak = conf.get("is_weak_confounder", False)
                if sy_f is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = " ⚠️ Weak Confounder / Low Identification (F < 10.0)" if is_weak else " (Strong First-Stage Coupling / High Identification, F ≥ 10.0)"
                    sy_items.append(f"• **{v_name}** vs **{c_name}**: First-Stage F = {sy_f:.2f}{flag}")
            if sy_items:
                return (
                    "**Stock & Yogo (2005) First-Stage Instrument / Confounder Relevance Test:**\n"
                    "Applies the maximal bias identification threshold (F ≥ 10.0) to assess coupling strength between regressors and candidate confounders:\n"
                    + "\n".join(sy_items)
                )
            return "Stock-Yogo first-stage relevance evaluates whether candidate confounders exhibit sufficiently strong association (F >= 10.0) with primary regressors."

        if (any(w in q_lower for w in ["cusum", "brown-durbin-evans", "parameter drift"]) or ("recursive residual" in q_lower and "harvey" not in q_lower and "collier" not in q_lower)) and not any(sq in q_lower for sq in ["cusumsq", "square"]):
            cusum_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                s_stat = stab.get("cusum_stat")
                p_val = stab.get("cusum_p_approx")
                has_drift = stab.get("has_cusum_instability", False)
                if s_stat is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ CUSUM Boundary Breached (S > 0.948, Parameter Drift Detected)" if has_drift else " (Stable Recursive Path, S ≤ 0.948)"
                    cusum_items.append(f"• **{v_name}**: CUSUM S = {s_stat:.3f} (p ≈ {p_val:.3e}){flag}")
            if cusum_items:
                return (
                    "**Brown, Durbin, & Evans (1975) Cumulative Sum of Recursive Residuals (CUSUM) Test:**\n"
                    "Standardizes recursive least-squares residuals against linear confidence boundaries (a = 0.948 at α = 0.05) along sorted regressors:\n"
                    + "\n".join(cusum_items)
                )
            return "The Brown-Durbin-Evans CUSUM test tracks cumulative recursive residuals to detect systematic parameter drift and model instability."

        if any(w in q_lower for w in ["koenker", "studentized bp", "studentized breusch"]):
            koen_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                lm_stat = stab.get("koenker_lm_stat")
                p_val = stab.get("koenker_p_value")
                is_het = stab.get("is_koenker_heteroscedastic", False)
                if lm_stat is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Non-Normal Heteroscedasticity (p < 0.05)" if is_het else " (Homoscedastic Residuals)"
                    koen_items.append(f"• **{v_name}**: Koenker Studentized LM = {lm_stat:.2f} (p = {p_val:.3e}){flag}")
            if koen_items:
                return (
                    "**Koenker (1981) Studentized Breusch-Pagan Test for Heteroscedasticity:**\n"
                    "Studentizes squared residual regressions by empirical kurtosis to provide robust heteroscedasticity testing under non-Gaussian distributions:\n"
                    + "\n".join(koen_items)
                )
            return "Koenker's studentized Breusch-Pagan test detects heteroscedasticity while remaining robust to kurtosis and leptokurtic departures from normality."

        if any(w in q_lower for w in ["harbord", "harbord-egger", "score asymmetry"]):
            harb_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                h_t = simp.get("harbord_t_stat")
                h_p = simp.get("harbord_p_value")
                is_asym = simp.get("is_harbord_asymmetric", False)
                if h_t is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Funnel Asymmetry / Small-Stratum Bias (p = {h_p:.3e})" if is_asym else f" (Symmetric Strata, p = {h_p:.3e})"
                    harb_items.append(f"• **{v_name}**: Harbord Score t = {h_t:.2f} (p = {h_p:.3e}){flag}")
            if harb_items:
                return (
                    "**Harbord, Egger, & Sterne (2006) Modified Score Test for Funnel Plot Asymmetry:**\n"
                    "Regresses hypergeometric scores against precision to eliminate artifactual false positives in sparse 2x2 strata:\n"
                    + "\n".join(harb_items)
                )
            return "The Harbord modified score test detects publication/stratum asymmetry in binary contingency tables without false-positive inflation."

        if any(w in q_lower for w in ["brown-forsythe", "brown forsythe", "robust scale", "median variance"]) or ("forsythe" in q_lower):
            bf_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                b_w = simp.get("brown_forsythe_stat")
                b_p = simp.get("brown_forsythe_p")
                is_het = simp.get("is_brown_forsythe_heterogeneous", False)
                if b_w is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = " ⚠️ Heterogeneous Strata Variances (p < 0.05)" if is_het else " (Homoscedastic Strata Spread)"
                    bf_items.append(f"• **{v_name}**: Brown-Forsythe W = {b_w:.2f} (p = {b_p:.3e}){flag}")
            if bf_items:
                return (
                    "**Brown & Forsythe (1974) Robust Scale Homogeneity Test:**\n"
                    "Computes ANOVA on absolute deviations from subgroup medians to test scale equality without Gaussian assumptions:\n"
                    + "\n".join(bf_items)
                )
            return "The Brown-Forsythe test tests whether conditional variance spreads differ significantly across conditioning subgroups using median alignments."

        if any(w in q_lower for w in ["bkw", "belsley", "condition index", "variance decomposition proportion"]) or ("belsley-kuh-welsch" in q_lower):
            bkw_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                b_k = conf.get("bkw_condition_index")
                b_pi = conf.get("bkw_max_variance_prop")
                is_coll = conf.get("is_bkw_collinear", False)
                if b_k is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = f" ⚠️ Severe Collinearity (κ ≥ 15, Max Prop = {b_pi:.1%})" if is_coll else f" (Stable Conditioning, κ = {b_k:.2f})"
                    bkw_items.append(f"• **{v_name}** vs **{c_name}**: BKW Condition Index κ = {b_k:.2f} (Max Var Prop = {b_pi:.1%}){flag}")
            if bkw_items:
                return (
                    "**Belsley, Kuh, & Welsch (1980) Collinearity Condition Index (κ_BKW) & Variance Decomposition:**\n"
                    "Applies singular value decomposition on equilibrated design matrices to isolate degrading near-dependencies:\n"
                    + "\n".join(bkw_items)
                )
            return "The Belsley-Kuh-Welsch (BKW) diagnostic evaluates collinearity condition indices and variance decomposition proportions."

        if any(w in q_lower for w in ["sargan", "hansen j", "sargan-hansen", "overidentifying", "overidentification"]):
            sarg_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                s_j = conf.get("sargan_j_stat")
                s_p = conf.get("sargan_p_value")
                is_rej = conf.get("is_sargan_rejected", False)
                if s_j is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Exclusion Restrictions Violated (p = {s_p:.3e})" if is_rej else f" (Valid Overidentifying Restrictions, p = {s_p:.3e})"
                    sarg_items.append(f"• **{v_name}**: Sargan J = {s_j:.2f} (p = {s_p:.3e}){flag}")
            if sarg_items:
                return (
                    "**Sargan (1958) & Hansen (1982) Overidentifying Restrictions Test (J_Sargan):**\n"
                    "Tests orthogonality between candidate instrument/covariate specifications and 2SLS structural disturbances:\n"
                    + "\n".join(sarg_items)
                )
            return "The Sargan-Hansen test verifies whether overidentifying instruments are uncorrelated with structural disturbances."

        if any(w in q_lower for w in ["harvey-collier", "harvey collier", "linearity test", "recursive residual linearity"]) or ("harvey" in q_lower and "collier" in q_lower):
            hc_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                h_t = stab.get("harvey_collier_t")
                h_p = stab.get("harvey_collier_p")
                has_nl = stab.get("has_harvey_collier_nonlinearity", False)
                if h_t is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Functional Curvature / Non-Linearity (p = {h_p:.3e})" if has_nl else f" (Linear Specification Sound, p = {h_p:.3e})"
                    hc_items.append(f"• **{v_name}**: Harvey-Collier t = {h_t:.2f} (p = {h_p:.3e}){flag}")
            if hc_items:
                return (
                    "**Harvey & Collier (1977) Recursive Residual Linearity Test:**\n"
                    "Evaluates the mean of standardized recursive residuals along sorted regressors to detect curvature and misspecification:\n"
                    + "\n".join(hc_items)
                )
            return "The Harvey-Collier test assesses linear specification correctness based on standardized recursive forecast errors."

        if any(w in q_lower for w in ["huber", "huber m-estimator", "huber slope", "robust m-estimator", "irls slope"]):
            hub_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                h_b = stab.get("huber_m_slope")
                h_d = stab.get("huber_slope_discrepancy")
                is_dist = stab.get("is_huber_outlier_distorted", False)
                if h_b is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ High Outlier Contamination (Δ = {h_d:.1%})" if is_dist else f" (Robust to Outliers, Δ = {h_d:.1%})"
                    hub_items.append(f"• **{v_name}**: Huber Robust β = {h_b:.4f} (Slope Discrepancy Δ = {h_d:.1%}){flag}")
            if hub_items:
                return (
                    "**Huber (1964) Robust M-Estimator Regression Discrepancy:**\n"
                    "Calculates outlier-resistant slope estimates via IRLS (c = 1.345 σ_MAD) and measures percentage discrepancy vs OLS:\n"
                    + "\n".join(hub_items)
                )
            return "Huber's M-estimator provides outlier-resistant slope estimates via iteratively reweighted least squares."

        if any(w in q_lower for w in ["peters", "peters regression", "peters test", "peters-egger"]):
            peters_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                p_t = simp.get("peters_test_t")
                p_p = simp.get("peters_test_p")
                is_asym = simp.get("is_peters_asymmetric", False)
                if p_t is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Funnel Asymmetry / Small-Stratum Bias (p = {p_p:.3e})" if is_asym else f" (Symmetric Strata, p = {p_p:.3e})"
                    peters_items.append(f"• **{v_name}**: Peters Asymmetry t = {p_t:.2f} (p = {p_p:.3e}){flag}")
            if peters_items:
                return (
                    "**Peters et al. (2006) Regression Test for Funnel Plot Asymmetry in Binary Data:**\n"
                    "Regresses unstandardized log-odds ratios against inverse sample size weighted by effective stratum size (eliminating Egger's mathematical artifact):\n"
                    + "\n".join(peters_items)
                )
            return "Peters' regression test detects small-stratum asymmetry in binary 2x2xK tables without spurious correlation between log-OR and its standard error."

        if any(w in q_lower for w in ["kish", "kish design effect", "design effect", "variance ratio", "stratification efficiency"]):
            kish_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                k_deff = simp.get("kish_design_effect")
                s_vr = simp.get("stratified_variance_ratio")
                if k_deff is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    vr_flag = " (Variance Reduction)" if (s_vr is not None and s_vr < 1.0) else " (Variance Inflation)"
                    kish_items.append(f"• **{v_name}**: Kish D_eff = {k_deff:.2f}, Stratified Variance Ratio = {s_vr:.2f}{vr_flag}")
            if kish_items:
                return (
                    "**Leslie Kish (1965) Stratification Design Effect & Variance Ratio:**\n"
                    "Measures stratum sample-size imbalance and the efficiency gain/loss of subgroup stratification vs crude estimation:\n"
                    + "\n".join(kish_items)
                )
            return "Kish's design effect quantifies stratum weighting efficiency and sampling variance modification under subgroup disaggregation."

        if any(w in q_lower for w in ["crump", "propensity overlap", "common support", "overlap deficiency"]):
            crump_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                c_ratio = conf.get("crump_overlap_ratio")
                is_def = conf.get("is_crump_overlap_deficient", False)
                if c_ratio is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = f" ⚠️ Severe Lack of Common Support ({c_ratio:.1%} Off-Support)" if is_def else f" (Adequate Overlap, {c_ratio:.1%} Off-Support)"
                    crump_items.append(f"• **{v_name}** vs **{c_name}**: Crump Off-Support Ratio = {c_ratio:.1%}{flag}")
            if crump_items:
                return (
                    "**Crump, Hotz, Imbens, & Mitnik (2009) Propensity Score Overlap & Common Support:**\n"
                    "Evaluates the percentage of observational units lying outside the optimal common support window [0.10, 0.90]:\n"
                    + "\n".join(crump_items)
                )
            return "The Crump propensity score overlap test diagnoses whether confounder adjustments suffer from severe lack of common support."

        if ("hansen" in q_lower and "sargan" not in q_lower) or any(w in q_lower for w in ["threshold regime", "structural regime", "hansen test", "hansen threshold"]):
            hansen_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                h_f = conf.get("hansen_threshold_f")
                h_p = conf.get("hansen_threshold_p")
                has_reg = conf.get("has_hansen_threshold_regime", False)
                if h_f is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    c_name = conf.get("top_confounder")
                    flag = f" ⚠️ Threshold Regime Shift Detected (p = {h_p:.3e})" if has_reg else f" (Linear Response Adequate, p = {h_p:.3e})"
                    hansen_items.append(f"• **{v_name}** along **{c_name}**: Sup-F = {h_f:.2f} (p = {h_p:.3e}){flag}")
            if hansen_items:
                return (
                    "**Bruce Hansen (2000) Structural Threshold Non-Linearity F-Test:**\n"
                    "Searches over candidate sample splits along the confounder to detect non-linear regime-switching:\n"
                    + "\n".join(hansen_items)
                )
            return "Hansen's threshold test evaluates whether the relationship between regressor and target splits into discrete non-linear regimes across covariate thresholds."

        if any(w in q_lower for w in ["cusumsq", "cusum of squares", "variance constancy", "variance stability", "ploberger"]):
            cusumsq_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                c_w = stab.get("cusumsq_stat")
                c_crit = stab.get("cusumsq_critical_val")
                has_inst = stab.get("has_cusumsq_instability", False)
                if c_w is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Structural Variance Shift / Instability (W > {c_crit:.3f})" if has_inst else f" (Homoscedastic Constancy, W ≤ {c_crit:.3f})"
                    cusumsq_items.append(f"• **{v_name}**: CUSUMSQ W = {c_w:.3f} (crit = {c_crit:.3f}){flag}")
            if cusumsq_items:
                return (
                    "**Brown-Durbin-Evans / Ploberger-Krämer CUSUM of Squares Test:**\n"
                    "Evaluates cumulative sums of squared recursive residuals to detect sudden variance shifts and heteroscedastic shocks along ordered regressors:\n"
                    + "\n".join(cusumsq_items)
                )
            return "The Brown-Durbin-Evans / Ploberger-Krämer CUSUM of squares (CUSUMSQ) test monitors cumulative squared forecast errors to identify abrupt variance changes and parameter instability."

        if any(w in q_lower for w in ["least trimmed squares", "lts", "lts slope", "high-breakdown", "trimmed squares"]) and not any(w in q_lower for w in ["siegel", "repeated medians"]):
            lts_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                l_b = stab.get("lts_slope")
                l_d = stab.get("lts_slope_discrepancy")
                is_dist = stab.get("is_lts_outlier_distorted", False)
                if l_b is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ High-Breakdown Leverage Distortion (Δ = {l_d:.1%})" if is_dist else f" (Resistant to Outliers, Δ = {l_d:.1%})"
                    lts_items.append(f"• **{v_name}**: LTS Robust β = {l_b:.4f} (Slope Discrepancy Δ = {l_d:.1%}){flag}")
            if lts_items:
                return (
                    "**Peter Rousseeuw (1984) Least Trimmed Squares (LTS) High-Breakdown Regression:**\n"
                    "Minimizes the sum of the lowest 75% squared residuals to achieve a 25% breakdown point resistant to leverage clusters:\n"
                    + "\n".join(lts_items)
                )
            return "Least Trimmed Squares (LTS) provides high-breakdown robust regression estimates resistant to masked multi-point leverage clusters."

        if any(w in q_lower for w in ["rucker", "rücker", "arcsine test", "arcsine asymmetry", "rucker test"]):
            rucker_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                r_t = simp.get("rucker_test_t")
                r_p = simp.get("rucker_test_p")
                is_asym = simp.get("is_rucker_asymmetric", False)
                if r_t is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Funnel Asymmetry / Sparse Cell Bias (p = {r_p:.3e})" if is_asym else f" (Symmetric Strata, p = {r_p:.3e})"
                    rucker_items.append(f"• **{v_name}**: Rücker Arcsine Asymmetry t = {r_t:.2f} (p = {r_p:.3e}){flag}")
            if rucker_items:
                return (
                    "**Gerta Rücker et al. (2008) Arcsine Test for Funnel Plot Asymmetry in Binary Data:**\n"
                    "Applies angular variance-stabilizing transformations arcsin(sqrt(p)) to eliminate spurious correlation between log-OR and its precision under H_0:\n"
                    + "\n".join(rucker_items)
                )
            return "Rücker's arcsine test stabilizes stratum variances to detect publication bias and small-study asymmetry without Type I error inflation."

        if any(w in q_lower for w in ["levene", "levene test", "levene scale", "mean-centered scale"]) and not any(w in q_lower for w in ["brown-forsythe", "brown forsythe"]):
            levene_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                l_w = simp.get("levene_stat")
                l_p = simp.get("levene_p")
                is_het = simp.get("is_levene_heterogeneous", False)
                if l_w is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Scale Heterogeneity / Unequal Variances (p = {l_p:.3e})" if is_het else f" (Homoscedastic Strata Spread, p = {l_p:.3e})"
                    levene_items.append(f"• **{v_name}**: Levene Scale Homogeneity W = {l_w:.2f} (p = {l_p:.3e}){flag}")
            if levene_items:
                return (
                    "**Howard Levene's (1960) Classical Mean-Centered Scale Homogeneity Test:**\n"
                    "Evaluates whether target dispersion varies significantly across conditioning strata:\n"
                    + "\n".join(levene_items)
                )
            return "Levene's test certifies whether conditional outcome variances remain constant across conditioning subgroups."

        if any(w in q_lower for w in ["hosmer", "lemeshow", "hosmer-lemeshow", "propensity calibration", "calibration test"]):
            hl_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                hl_c = conf.get("hosmer_lemeshow_chi2")
                hl_p = conf.get("hosmer_lemeshow_p")
                is_misc = conf.get("is_propensity_miscalibrated", False)
                if hl_c is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Severe Propensity Miscalibration (p = {hl_p:.3e})" if is_misc else f" (Well-Calibrated Propensity, p = {hl_p:.3e})"
                    hl_items.append(f"• **{v_name}**: Hosmer-Lemeshow χ² = {hl_c:.2f} (p = {hl_p:.3e}){flag}")
            if hl_items:
                return (
                    "**David Hosmer & Stanley Lemeshow's (1980) Propensity Calibration Test:**\n"
                    "Tests goodness-of-fit deciles for exposure propensity scores to guarantee unmeasured confounding controls are not biased by model misspecification:\n"
                    + "\n".join(hl_items)
                )
            return "Hosmer-Lemeshow tests calibration across deciles to ensure propensity weighting and covariate adjustment are valid."

        if any(w in q_lower for w in ["newey", "newey-west", "hac", "hac se", "hac ratio", "autocorrelation consistent"]):
            hac_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                h_se = conf.get("newey_west_hac_se")
                h_rat = conf.get("hac_se_ratio")
                is_inf = conf.get("is_hac_inflated", False)
                if h_rat is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Autocorrelation Inflation (ρ = {h_rat:.2f} > 1.25)" if is_inf else f" (Robust to Serial Correlation, ρ = {h_rat:.2f})"
                    hac_items.append(f"• **{v_name}**: Newey-West HAC SE = {h_se:.4f} (Ratio ρ_HAC = {h_rat:.2f}){flag}")
            if hac_items:
                return (
                    "**Whitney Newey & Kenneth West's (1987) HAC Robust Standard Error Ratio:**\n"
                    "Applies Bartlett kernel autocovariance weighting to evaluate whether naive regression standard errors underestimate variance due to residual serial correlation:\n"
                    + "\n".join(hac_items)
                )
            return "Newey-West HAC standard errors protect hypothesis testing against residual autocorrelation and heteroscedasticity."

        if any(w in q_lower for w in ["siegel", "repeated medians", "siegel slope", "siegel's"]):
            siegel_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                s_b = stab.get("siegel_slope")
                s_d = stab.get("siegel_slope_discrepancy")
                is_dist = stab.get("is_siegel_outlier_distorted", False)
                if s_b is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ 50% Breakdown Outlier Contamination (Δ = {s_d:.1%})" if is_dist else f" (Resistant to Outliers, Δ = {s_d:.1%})"
                    siegel_items.append(f"• **{v_name}**: Siegel Robust β = {s_b:.4f} (Slope Discrepancy Δ = {s_d:.1%}){flag}")
            if siegel_items:
                return (
                    "**Andrew Siegel's (1982) Repeated Medians High-Breakdown Robust Regression:**\n"
                    "Computes nested medians of pairwise slopes yielding a 50% breakdown point impervious to gross outlier contamination:\n"
                    + "\n".join(siegel_items)
                )
            return "Siegel's Repeated Medians estimator achieves a 50% breakdown point, resisting distortion even when up to half the sample consists of arbitrary outliers."

        if any(w in q_lower for w in ["breusch-godfrey", "breusch godfrey", "higher-order autocorrelation", "bg test", "godfrey"]) and not any(w in q_lower for w in ["breusch-pagan", "breusch pagan"]):
            bg_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                bg_lm = stab.get("breusch_godfrey_lm")
                bg_p = stab.get("breusch_godfrey_p")
                has_ac = stab.get("has_higher_order_autocorrelation", False)
                if bg_lm is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Higher-Order Autocorrelation Detected (p = {bg_p:.3e})" if has_ac else f" (Independent Errors, p = {bg_p:.3e})"
                    bg_items.append(f"• **{v_name}**: Breusch-Godfrey LM = {bg_lm:.2f} (p = {bg_p:.3e}){flag}")
            if bg_items:
                return (
                    "**Trevor Breusch & Leslie Godfrey's (1978) Higher-Order Serial Correlation LM Test:**\n"
                    "Regresses residuals on lagged errors to detect higher-order autoregressive dependence along sorted regressors:\n"
                    + "\n".join(bg_items)
                )
            return "The Breusch-Godfrey LM test checks for higher-order serial correlation that invalidates naive OLS inference."

        if any(w in q_lower for w in ["macaskill", "macaskill test", "sample-size weighted regression", "macaskill asymmetry"]):
            mac_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                m_t = simp.get("macaskill_test_t")
                m_p = simp.get("macaskill_test_p")
                is_asym = simp.get("is_macaskill_asymmetric", False)
                if m_t is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Small-Study / Strata-Size Selection Bias (p = {m_p:.3e})" if is_asym else f" (Symmetric Strata Size, p = {m_p:.3e})"
                    mac_items.append(f"• **{v_name}**: Macaskill Funnel Asymmetry t = {m_t:.2f} (p = {m_p:.3e}){flag}")
            if mac_items:
                return (
                    "**Peter Macaskill et al.'s (2001) Sample-Size Weighted Funnel Regression Test:**\n"
                    "Regresses stratum log-odds ratios on inverse sample size 1/N_k weighted by inverse-variance to detect small-stratum bias without Egger's false-positive inflation:\n"
                    + "\n".join(mac_items)
                )
            return "Macaskill's test evaluates whether stratum effect estimates systematically depend on stratum sample sizes."

        if any(w in q_lower for w in ["bartlett", "bartlett test", "bartlett scale", "bartlett's"]) and not any(w in q_lower for w in ["newey", "kernel"]):
            bart_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                b_stat = simp.get("bartlett_stat")
                b_p = simp.get("bartlett_p")
                is_het = simp.get("is_bartlett_heterogeneous", False)
                if b_stat is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Scale Heterogeneity / Unequal Variances (p = {b_p:.3e})" if is_het else f" (Homoscedastic Strata Spread, p = {b_p:.3e})"
                    bart_items.append(f"• **{v_name}**: Bartlett Scale Homogeneity B = {b_stat:.2f} (p = {b_p:.3e}){flag}")
            if bart_items:
                return (
                    "**Maurice Bartlett's (1937) Likelihood Ratio Scale Homogeneity Test:**\n"
                    "Evaluates the equality of conditional target variances across K conditioning strata:\n"
                    + "\n".join(bart_items)
                )
            return "Bartlett's test evaluates whether residual variances are homogeneous across conditioning strata to protect against Behrens-Fisher distortions."

        if any(w in q_lower for w in ["aipw", "doubly robust", "augmented inverse probability", "robins"]):
            aipw_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                a_tau = conf.get("aipw_treatment_effect")
                a_disc = conf.get("aipw_discrepancy")
                is_conf = conf.get("is_aipw_confounded", False)
                if a_tau is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Selection Confounding (Δ = {a_disc:.1%})" if is_conf else f" (Robust Doubly Identified, Δ = {a_disc:.1%})"
                    aipw_items.append(f"• **{v_name}**: AIPW Doubly Robust τ = {a_tau:.4f} (Discrepancy Δ_AIPW = {a_disc:.1%}){flag}")
            if aipw_items:
                return (
                    "**Robins-Rotnitzky-Zhao (1994) & Imbens (2004) AIPW Doubly Robust Treatment Effect:**\n"
                    "Combines propensity score weighting with outcome regression to ensure consistent causal effect estimation even if one model is misspecified:\n"
                    + "\n".join(aipw_items)
                )
            return "AIPW doubly robust estimation combines propensity weighting with outcome regression models to guard against treatment selection bias."

        if any(w in q_lower for w in ["leamer", "extreme bounds", "eba", "leamer's", "fragility"]):
            leamer_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                l_min = conf.get("leamer_bound_min")
                l_max = conf.get("leamer_bound_max")
                is_frag = conf.get("is_leamer_fragile", False)
                if l_min is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Leamer Fragile (0 ∈ [{l_min:.3f}, {l_max:.3f}])" if is_frag else f" (Leamer Robust, [{l_min:.3f}, {l_max:.3f}] excludes 0)"
                    leamer_items.append(f"• **{v_name}**: Leamer EBA Range = [{l_min:.4f}, {l_max:.4f}]{flag}")
            if leamer_items:
                return (
                    "**Edward Leamer's (1983) Extreme Bounds Analysis (EBA):**\n"
                    "Evaluates the focal regression slope across all subsets of candidate confounders to determine whether empirical findings are fragile to control variable selection:\n"
                    + "\n".join(leamer_items)
                )
            return "Leamer's Extreme Bounds Analysis searches across all confounder subsets to verify if the regression slope sign can be overturned by changing control variables."

        if any(w in q_lower for w in ["runs test", "wald-wolfowitz", "residual independence", "runs_z", "wolfowitz"]):
            runs_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                r_z = stab.get("runs_test_z")
                r_p = stab.get("runs_test_p")
                has_drift = stab.get("has_runs_nonrandomness", False)
                if r_z is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Non-Random Residual Clustering / Drift (p = {r_p:.3e})" if has_drift else f" (Independent Residuals, p = {r_p:.3e})"
                    runs_items.append(f"• **{v_name}**: Wald-Wolfowitz Runs Z = {r_z:.2f} (p = {r_p:.3e}){flag}")
            if runs_items:
                return (
                    "**Abraham Wald & Jacob Wolfowitz's (1940) Non-Parametric Runs Test for Residual Independence:**\n"
                    "Evaluates whether the sequence of regression residual signs exhibits non-random clustering or persistence along sorted regressors:\n"
                    + "\n".join(runs_items)
                )
            return "The Wald-Wolfowitz runs test checks whether regression residual signs are randomly distributed along regressor orderings."

        if any(w in q_lower for w in ["kendall", "tau-b", "kendall's", "concordance discrepancy", "rank concordance"]):
            kend_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                k_tau = stab.get("kendall_tau_b")
                k_disc = stab.get("kendall_discrepancy")
                is_dist = stab.get("is_kendall_distorted", False)
                if k_tau is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Rank Discrepancy / Leverage Distortion (Δ = {k_disc:.1%})" if is_dist else f" (Robust Rank Concordance, Δ = {k_disc:.1%})"
                    kend_items.append(f"• **{v_name}**: Kendall τ_b = {k_tau:.4f} (Discrepancy Δ_Kendall = {k_disc:.1%}){flag}")
            if kend_items:
                return (
                    "**Maurice Kendall's (1938) Tau-b Rank Concordance Discrepancy:**\n"
                    "Evaluates pairwise concordant vs discordant pairs with exact tie corrections to detect gross bivariate leverage and outlier distortion:\n"
                    + "\n".join(kend_items)
                )
            return "Kendall's Tau-b evaluates pairwise rank concordance with exact tie adjustments to expose leverage points inflating linear Pearson correlation."

        if any(w in q_lower for w in ["deeks", "deeks test", "deeks dor", "deeks funnel", "diagnostic odds ratio asymmetry"]):
            deeks_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                d_t = simp.get("deeks_test_t")
                d_p = simp.get("deeks_test_p")
                is_asym = simp.get("is_deeks_asymmetric", False)
                if d_t is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ DOR Funnel Asymmetry / Selection Bias (p = {d_p:.3e})" if is_asym else f" (Symmetric Stratified Precision, p = {d_p:.3e})"
                    deeks_items.append(f"• **{v_name}**: Deeks DOR Funnel t = {d_t:.2f} (p = {d_p:.3e}){flag}")
            if deeks_items:
                return (
                    "**Jon Deeks, Peter Macaskill, & Les Irwig's (2005) Diagnostic Odds Ratio (DOR) Asymmetry Regression Test:**\n"
                    "Regresses stratum log-diagnostic odds ratios on 1/sqrt(ESS_k) weighted by effective sample size ESS_k to detect binary 2x2 table asymmetry without false-positive inflation:\n"
                    + "\n".join(deeks_items)
                )
            return "Deeks' DOR funnel asymmetry test regresses stratum log-odds ratios against effective sample size weights to identify stratified publication or selection bias."

        if any(w in q_lower for w in ["mood", "mood scale", "mood dispersion", "mood's", "median-rank dispersion"]):
            mood_items = []
            for f in findings:
                simp = f.get("audit", {}).get("simpsons_audit", {})
                m_z = simp.get("mood_stat_z")
                m_p = simp.get("mood_p_value")
                is_het = simp.get("is_mood_heterogeneous", False)
                if m_z is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Unequal Rank Dispersion (p = {m_p:.3e})" if is_het else f" (Homoscedastic Median-Rank Spread, p = {m_p:.3e})"
                    mood_items.append(f"• **{v_name}**: Mood Scale Z = {m_z:.2f} (p = {m_p:.3e}){flag}")
            if mood_items:
                return (
                    "**Alexander M. Mood's (1954) Non-Parametric Median-Rank Dispersion Scale Test:**\n"
                    "Evaluates differences in scale dispersion across conditioning strata using squared rank deviations from the median, completely free of Gaussian distribution assumptions:\n"
                    + "\n".join(mood_items)
                )
            return "Mood's test evaluates whether subpopulation dispersions differ across strata based on non-parametric median-centered rank variations."

        if any(w in q_lower for w in ["double machine learning", "dml", "neyman orthogonal", "orthogonalized treatment effect", "chernozhukov"]):
            dml_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                d_tau = conf.get("dml_treatment_effect")
                d_disc = conf.get("dml_discrepancy")
                is_conf = conf.get("is_dml_confounded", False)
                if d_tau is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Confounded Causal Bias (Δ = {d_disc:.1%})" if is_conf else f" (Robust DML Identification, Δ = {d_disc:.1%})"
                    dml_items.append(f"• **{v_name}**: DML τ = {d_tau:.4f} (Discrepancy Δ_DML = {d_disc:.1%}){flag}")
            if dml_items:
                return (
                    "**Chernozhukov et al.'s (2018) Double Machine Learning (DML) Neyman-Orthogonalized Treatment Effect:**\n"
                    "Constructs Neyman-orthogonal partialing-out residuals for treatment and outcome to identify robust causal effects immune to first-stage nuisance model bias:\n"
                    + "\n".join(dml_items)
                )
            return "Double Machine Learning uses Neyman-orthogonal residualization to isolate true treatment effects from confounding covariate influences."

        if any(w in q_lower for w in ["cinelli benchmark", "cinelli multiplier", "extreme confounder multiplier", "cinelli k", "confounder multiplier"]):
            cin_items = []
            for f in findings:
                conf = f.get("audit", {}).get("confounder_audit", {})
                k_mult = conf.get("cinelli_k_multiplier")
                is_frag = conf.get("is_cinelli_fragile", False)
                if k_mult is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Fragile (k = {k_mult:.2f}x < 1.0x — weaker confounder can overturn)" if is_frag else f" (Robust, k = {k_mult:.2f}x >= 1.0x)"
                    cin_items.append(f"• **{v_name}**: Cinelli Extreme Confounder Multiplier k = {k_mult:.2f}x{flag}")
            if cin_items:
                return (
                    "**Carlos Cinelli & Chad Hazlett's (2020) Extreme Confounder Relative Multiplier:**\n"
                    "Calculates how many multiples (k) of the top observed confounder's explanatory power are required to completely overturn the estimated relationship:\n"
                    + "\n".join(cin_items)
                )
            return "The Cinelli-Hazlett extreme confounder multiplier calculates how many times stronger an unobserved confounder must be relative to observed controls to overturn the finding."

        if any(w in q_lower for w in ["goldfeld", "quandt", "goldfeld-quandt", "regime heteroskedasticity", "split-regime"]):
            gq_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                g_f = stab.get("goldfeld_quandt_f")
                g_p = stab.get("goldfeld_quandt_p")
                has_het = stab.get("has_goldfeld_quandt_heteroskedasticity", False)
                if g_f is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Regime-Shift Heteroskedasticity (p = {g_p:.3e})" if has_het else f" (Homoscedastic Regimes, p = {g_p:.3e})"
                    gq_items.append(f"• **{v_name}**: Goldfeld-Quandt F = {g_f:.2f} (p = {g_p:.3e}){flag}")
            if gq_items:
                return (
                    "**Stephen M. Goldfeld & Richard E. Quandt's (1965) Split-Regime Heteroskedasticity Test:**\n"
                    "Omits central observations and evaluates residual variance ratios between lower and upper regressor regimes to detect monotonic heteroskedastic shifts:\n"
                    + "\n".join(gq_items)
                )
            return "The Goldfeld-Quandt test checks whether residual variance changes significantly across distinct regressor regimes."

        if any(w in q_lower for w in ["shapiro-francia", "shapiro francia", "francia", "residual normality test", "w' statistic"]):
            sf_items = []
            for f in findings:
                stab = f.get("audit", {}).get("stability_audit", {})
                s_w = stab.get("shapiro_francia_w")
                s_p = stab.get("shapiro_francia_p")
                is_non_norm = stab.get("is_shapiro_francia_non_normal", False)
                if s_w is not None:
                    v_name = f.get("claim", {}).get("source_variable")
                    flag = f" ⚠️ Non-Gaussian Residuals (p = {s_p:.3e})" if is_non_norm else f" (Gaussian Residuals, p = {s_p:.3e})"
                    sf_items.append(f"• **{v_name}**: Shapiro-Francia W' = {s_w:.4f} (p = {s_p:.3e}){flag}")
            if sf_items:
                return (
                    "**S. S. Shapiro & R. S. Francia's (1972) Normality Test for Regression Residuals:**\n"
                    "Evaluates the squared correlation between ordered OLS residuals and standard normal order statistics to diagnose residual non-Gaussianity:\n"
                    + "\n".join(sf_items)
                )
            return "The Shapiro-Francia test assesses whether regression residuals follow a Gaussian distribution using Blom normal scores."

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
        lines.append("1. **Cochran-Mantel-Haenszel, Breslow-Day & Tarone Homogeneity, Woolf Stratified Test, Freeman-Tukey Double-Arcsine Homogeneity, Peto Odds Ratio & Heterogeneity, Harbord Modified Score Funnel Asymmetry, Peters Regression Asymmetry Test, Gerta Rücker Arcsine Asymmetry Test, Peter Macaskill Funnel Regression Asymmetry Test, Jon Deeks Diagnostic Odds Ratio (DOR) Asymmetry Test, Leslie Kish Stratification Design Effect, Howard Levene Scale Homogeneity, Maurice Bartlett Likelihood Ratio Scale Homogeneity, Alexander Mood Non-Parametric Rank Dispersion Scale Test, Brown-Forsythe Scale Homogeneity, Fligner-Killeen Variance Homogeneity, Kruskal-Wallis Rank Heterogeneity, Higgins I² & H², DerSimonian-Laird τ², Egger/Begg Funnel Tests, Galbraith Radial Outliers, Duval-Tweedie Trim & Fill, Hausman Endogeneity & Vuong Model Selection:** Verifies whether stratified odds ratios reverse, tests cross-stratum invariance with Haldane-Anscombe continuity corrections (Woolf's test), applies variance-stabilizing angular transformations (Freeman-Tukey test), computes assumption-free hypergeometric one-step odds ratios and cross-stratum heterogeneity (Peto's method), detects sparse-stratum funnel plot asymmetry without artifactual Type I error inflation (Harbord's modified score test, Peters' weighted regression test, Rücker's arcsine transformation test, Macaskill's sample-size weighted regression test, and Deeks' effective-sample-size weighted regression test), evaluates stratum sample-size imbalance and variance efficiency (Kish's design effect $D_{\\text{Kish}}$ and stratified variance ratio), evaluates parametric and non-parametric scale spread (Levene mean-centered, Bartlett likelihood ratio, Mood's squared-rank median dispersion test, and Brown-Forsythe median-aligned tests), tests non-parametric variance scale equality (Fligner-Killeen test), tests non-parametric median shifts across strata (Kruskal-Wallis $H$), quantifies true heterogeneity percentage ($I^2$) and excess variance inflation ($H^2, \\tau^2$), flags leverage-inducing strata via Galbraith radial standardized deviations ($|z_i| \\ge 2.0$), estimates missing strata via Trim & Fill, verifies endogeneity via Hausman specification test, and proves whether stratified conditioning dominates crude models via Vuong's likelihood ratio test.")
        lines.append("2. **VanderWeele E-Values, Cinelli-Hazlett Robustness Values (RV), Critical Confounder Boundary (R²crit), Extreme Confounder Multiplier (k), Cohen's f², Mallows' Cp Criterion, Davidson-MacKinnon J-Test, Durbin-Wu-Hausman Endogeneity Test, Stock-Yogo First-Stage Relevance, Belsley-Kuh-Welsch Condition Index (κ), Sargan-Hansen Overidentifying Restrictions (J), Crump Propensity Score Overlap, Bruce Hansen Structural Threshold F-Test, David Hosmer & Stanley Lemeshow Propensity Calibration Test, Whitney Newey & Kenneth West HAC Variance Ratio, Robins-Rotnitzky-Zhao & Imbens AIPW Doubly Robust Treatment Effect, Chernozhukov et al. Double Machine Learning (DML) Orthogonalized Effect, & Edward Leamer Extreme Bounds Analysis (EBA):** Establishes mathematical bounds, minimal partial $R^2$ strength, critical confounder boundaries, extreme confounder benchmark multipliers ($k_{\\text{Cinelli}}$), unique variance effect sizes ($f^2$), Mallows' $C_p$ model misspecification discrepancy ($\\Delta C_p > 2.0$), Davidson-MacKinnon $J$-tests arbitrating non-nested specification dominance between bivariate claims and rival confounder models, Durbin-Wu-Hausman two-stage specification tests detecting omitted variable bias endogeneity ($F_{\\text{DWH}}$), Stock-Yogo first-stage relevance benchmarks ($F \\ge 10.0$), Belsley-Kuh-Welsch singular value condition indices and variance decomposition proportions isolating collinear dependencies ($\\kappa \\ge 15.0$), Sargan-Hansen overidentifying restrictions testing instrument orthogonality ($J_{\\text{Sargan}}$), Crump et al. propensity score common support trimming diagnostics, Hansen (2000) supremum threshold regime-switching tests, Hosmer-Lemeshow decile goodness-of-fit calibration tests certifying exposure model validity, Newey-West HAC kernel standard error adjustments certifying resistance to residual autocorrelation, Augmented Inverse Probability Weighted (AIPW) doubly robust estimators guaranteeing consistent causal effects under dual-misspecification protection, Double Machine Learning (DML) Neyman-orthogonal residualization isolating causal slopes free of nuisance estimation biases, and Leamer's Extreme Bounds Analysis evaluating coefficient fragility across all candidate confounder subsets.")
        lines.append("3. **Altonji-Elder-Taber (AET 2005) Selection Ratio, Rosenbaum (1987, 2002) Sensitivity Bounds (Γ), Manski (1990, 2003) Non-Parametric Partial Identification & Imbens-Manski (2004) Asymptotically Valid CIs:** Evaluates the relative selection ratio ($R_{\\text{AET}}$) on unobservables required to attribute effects to selection bias, calculates critical odds ratio differential threshold ($\\Gamma_{\\text{crit}}$) required for hidden bias to overturn findings, computes worst-case non-parametric sharp identification intervals $[\\Delta_{\\text{low}}, \\Delta_{\\text{high}}]$ under arbitrary selection, and constructs exact Imbens-Manski confidence intervals accounting for boundary sampling variability.")
        lines.append("4. **Frank's (2000) Confounding Impact Threshold (ITCV) & Burnham-Anderson AIC Information Weights:** Quantifies the exact correlation product threshold an unmeasured confounder must exceed to overturn statistical significance, and computes Akaike model selection weights ($w_1$) arbitrating covariate necessity.")
        lines.append("5. **Ramsey RESET Specification Test & Oster (2019) Selection Bound (δ*):** Evaluates functional form misspecification/omitted non-linearities and calculates proportional selection on unobservables needed to nullify effect estimates.")
        lines.append("6. **Kass-Raftery Bayes Factors & BIC Model Selection:** Computes exact posterior evidence odds ($BF_{10} = \\exp(\\Delta \\text{BIC} / 2)$) arbitrating between bivariate claims and confounder-adjusted multivariate models.")
        lines.append("7. **Baron-Kenny Sobel Mediation Diagnostics:** Disentangles true exogenous confounding from valid intermediate mechanistic transmission ($X \\to Z \\to Y$).")
        lines.append("8. **Theil-Sen Robust Non-Parametric Median Regression & Efron-Tibshirani BCa Bootstrap:** Compares OLS against breakdown-point 29.3% Theil-Sen median slopes and evaluates second-order accurate bias-corrected and accelerated (BCa) bootstrap confidence intervals.")
        lines.append("9. **Gregory Chow (1960) Structural Break Test, Donald Andrews (1993) Supremum Wald Test (Sup W), Brown-Durbin-Evans (1975) CUSUM & CUSUMSQ Tests, Koenker (1981) Studentized Breusch-Pagan Test, Stephen Goldfeld & Richard Quandt (1965) Split-Regime Heteroskedasticity F-Test, Harvey & Collier (1977) Recursive Linearity Test, Huber (1964) Robust M-Estimator Regression, Rousseeuw (1984) Least Trimmed Squares (LTS), Andrew Siegel (1982) Repeated Medians High-Breakdown Slope, Trevor Breusch & Leslie Godfrey (1978) Higher-Order Serial Correlation LM Test, Abraham Wald & Jacob Wolfowitz (1940) Runs Test for Residual Randomness, Maurice Kendall (1938) Tau-b Rank Concordance, S. S. Shapiro & R. S. Francia (1972) Residual Normality Test, Vectorized Bootstrap, Cook's Distance, DFFITS, DFBETAS, Breusch-Pagan, White LM, Jarque-Bera, Durbin-Watson & MacKinnon-White HC3:** Evaluates parameter invariance across data partitions (Chow $F$-test), detects structural change points at unknown locations along sorted regressors (Andrews $\\operatorname{Sup} W$), tracks recursive residual parameter drift (Brown-Durbin-Evans CUSUM) and structural variance shifts (CUSUMSQ), tests heteroscedasticity robust to non-normal kurtosis (Koenker studentized test), evaluates regime-dependent variance expansion (Goldfeld-Quandt $F$-test), evaluates functional form linearity through standardized recursive forecast errors (Harvey-Collier $t$-test), quantifies resistance to influential outliers via Huber robust M-regression slopes ($\\Delta_{\\text{Huber}}$), high-breakdown Least Trimmed Squares ($\\Delta_{\\text{LTS}}$), and Siegel's 50% breakdown repeated medians estimator ($\\Delta_{\\text{Siegel}}$), tests higher-order autoregressive errors (Breusch-Godfrey LM test), evaluates non-parametric residual sequence randomness and cyclic drift (Wald-Wolfowitz runs test), tests bivariate ordinal concordance and outlier leverage resistance with exact tie adjustments (Kendall $\\tau_b$), verifies residual Gaussianity with standard normal order scores (Shapiro-Francia $W'$), empirical confidence intervals, sign invariance probability ($P \\ge 90\\%$), parameter-specific influence (DFBETAS), overall leverage (Cook's D, DFFITS), linear & non-linear heteroscedasticity (Breusch-Pagan, White LM), residual Gaussian normality, autocorrelation (Durbin-Watson), and heteroscedasticity-consistent robust inference ($HC_3$).")
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
