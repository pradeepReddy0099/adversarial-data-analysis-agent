"""Interactive Web UI: The Adversarial Debate Arena (Streamlit)."""
from __future__ import annotations

import sys
from pathlib import Path
import json
import pandas as pd
import streamlit as st

# Add src to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from adversarial_agent.core import run_adversarial_pipeline, load_dataset
from adversarial_agent.llm_narrator import LLMNarrator, calculate_risk_score

st.set_page_config(
    page_title="Adversarial Data Analysis Agent",
    page_icon="⚔️",
    layout="wide",
)

st.title("⚔️ Adversarial Data Analysis Agent")
st.caption("A Dialectical Neuro-Symbolic Multi-Agent System for Automated Hypothesis Generation and Statistical Falsification")

st.markdown(
    """
    Standard automated data agents suffer from **confirmation bias** and **sycophancy**, presenting surface-level 
    correlations as causal truths. This system implements an adversarial dialectic:
    * 🔍 **Explorer Agent:** Autonomously mines the dataset to propose candidate empirical hypotheses.
    * ⚖️ **Devil's Advocate Agent:** Actively seeks to **falsify** claims (auditing for **Simpson's Paradox**, CMH odds ratio reversals, confounder collapse, and bootstrap stability).
    * 🏆 **Referee Agent:** Adjudicates the debate with calibrated credibility badges.
    * 🧠 **LLM Narrator & Copilot:** Synthesizes executive risk scorecards and answers user queries grounded in mathematical proofs.
    """
)

# Sidebar: Devil's Advocate Configuration
st.sidebar.header("⚙️ Auditor Hyperparameters")
n_bootstrap = st.sidebar.slider("Bootstrap Resamples (N)", min_value=50, max_value=500, value=150, step=50)
min_subgroup_size = st.sidebar.slider("Min Subgroup Size (N)", min_value=4, max_value=30, value=8, step=2)
confounder_att_threshold = st.sidebar.slider("Confounder Attenuation Threshold (%)", min_value=30, max_value=90, value=60, step=5)

st.write("---")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1. Select Benchmark Dataset")
    data_dir = Path(__file__).resolve().parent / "data"
    benchmarks = {
        "UC Berkeley Admissions (Simpson's Paradox Benchmark)": data_dir / "berkeley_admissions.csv",
        "Commercial Customer Churn (Subgroup Paradox Benchmark)": data_dir / "simpsons_churn.csv",
        "UCI Adult Census Income (Real-World Social Benchmark)": data_dir / "adult_census.csv",
        "Synthetic Enterprise Income & Confounders (Multi-Confounder Benchmark)": data_dir / "income_confounder_benchmark.csv",
    }
    
    choice = st.radio("Choose a benchmark or upload your own:", list(benchmarks.keys()) + ["Upload Custom CSV"])
    
    if choice == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV / Excel file", type=["csv", "xlsx"])
        if uploaded_file:
            temp_path = Path("output") / uploaded_file.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(uploaded_file.getvalue())
            selected_path = temp_path
        else:
            selected_path = None
    else:
        selected_path = benchmarks[choice]

    if selected_path and selected_path.exists():
        df_preview = load_dataset(selected_path)
        st.write(f"**Shape:** `{len(df_preview)} rows, {len(df_preview.columns)} columns`")
        
        target_candidates = ["None"] + list(df_preview.columns)
        default_target_idx = 0
        if "admitted" in df_preview.columns:
            default_target_idx = target_candidates.index("admitted")
        elif "churn" in df_preview.columns:
            default_target_idx = target_candidates.index("churn")
        elif "annual_income" in df_preview.columns:
            default_target_idx = target_candidates.index("annual_income")
        elif "income_above_50k" in df_preview.columns:
            default_target_idx = target_candidates.index("income_above_50k")
            
        selected_target = st.selectbox("Optional Outcome / Target Variable:", target_candidates, index=default_target_idx)
        target_val = None if selected_target == "None" else selected_target
        
        enable_llm = st.checkbox("🧠 Enable Neuro-Symbolic Executive Briefing", value=True)
        llm_provider = st.selectbox("LLM Provider", ["auto", "gemini", "openai", "offline"]) if enable_llm else "offline"

        run_btn = st.button("🚀 Run Adversarial Multi-Agent Audit", type="primary")
    else:
        run_btn = False

with col_right:
    st.subheader("2. Multi-Agent Debate Arena")
    
    if run_btn and selected_path:
        with st.spinner("Agents are debating and executing falsification batteries..."):
            results = run_adversarial_pipeline(
                input_path=selected_path,
                target=target_val,
                out_dir="output/ui_run",
                n_bootstrap=n_bootstrap,
                min_subgroup_size=min_subgroup_size,
                confounder_attenuation_threshold=float(confounder_att_threshold),
            )
            st.session_state["audit_results"] = results
            st.session_state["dataset_name"] = selected_path.name

    if "audit_results" in st.session_state:
        results = st.session_state["audit_results"]
        dataset_name = st.session_state.get("dataset_name", "Dataset")
        findings = results["adversarial_findings"]

        risk = calculate_risk_score(findings)
        st.success(f"Audit Complete! Evaluated {len(findings)} candidate empirical hypotheses.")
        
        # Summary Metric Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        c_robust = sum(1 for f in findings if f["verdict"] == "ROBUST_VERIFIED")
        c_simpson = sum(1 for f in findings if f["verdict"] == "SIMPSONS_PARADOX_ALERT")
        c_spurious = sum(1 for f in findings if "CONFOUNDED" in f["verdict"] or "SPURIOUS" in f["verdict"])
        
        m1.metric("Claims Evaluated", len(findings))
        m2.metric("🟢 Robust Findings", c_robust)
        m3.metric("🟡 Simpson's Paradox", c_simpson)
        m4.metric("🔴 Confounded", c_spurious)
        m5.metric("Decision Risk Score", f"{risk['score']}/100", delta=risk['tier'], delta_color="inverse")
        
        st.write("---")
        
        # Tabs layout for organized navigation
        tab_dossier, tab_viz, tab_briefing, tab_copilot, tab_export = st.tabs([
            "🏆 Referee Dossier",
            "📊 Subgroup Disaggregation",
            "📋 Executive Briefing",
            "💬 AI Copilot Q&A",
            "📥 Export Reports"
        ])

        with tab_dossier:
            # Filter bar for claims
            f_col1, f_col2 = st.columns([2, 1])
            with f_col1:
                verdict_filter = st.multiselect(
                    "Filter by Verdict Type:",
                    ["🟢 ROBUST & VERIFIED", "🟡 SIMPSON'S PARADOX", "🔴 CONFOUNDED", "🔴 UNSTABLE"],
                    default=["🟢 ROBUST & VERIFIED", "🟡 SIMPSON'S PARADOX", "🔴 CONFOUNDED", "🔴 UNSTABLE"]
                )
            with f_col2:
                search_kw = st.text_input("🔍 Search variable name:", placeholder="e.g., gender, support_calls")

            filtered_findings = [
                f for f in findings
                if any(vf.split()[0] in f["verdict_badge"] for vf in verdict_filter)
                and (not search_kw or search_kw.lower() in f["claim"]["source_variable"].lower() or search_kw.lower() in f["claim"]["target_variable"].lower())
            ]

            if not filtered_findings:
                st.info("No findings match the current filter criteria.")

            for f in filtered_findings:
                badge = f["verdict_badge"]
                claim = f["claim"]
                audit = f["audit"]
                
                with st.expander(f"{badge}: {claim['source_variable']} vs {claim['target_variable']}", expanded=("SIMPSON" in badge)):
                    deb_col1, deb_col2 = st.columns(2)
                    
                    with deb_col1:
                        st.markdown("#### 🔍 Explorer Agent Claim")
                        st.info(claim["description"])
                        st.write(f"- **Observed Statistic:** `{claim['statistic_name']}` = `{claim['statistic_value']:.4f}`")
                        st.write(f"- **Significance:** `p = {claim['p_value']:.4e}`")
                        if claim.get("fdr_q_value") is not None:
                            st.write(f"- **Benjamini-Hochberg FDR:** `q = {claim['fdr_q_value']:.4e}`")
                        if claim.get("bonferroni_p_value") is not None:
                            st.write(f"- **Bonferroni FWER:** `p_adj = {claim['bonferroni_p_value']:.4e}`")
                        st.write(f"- **Sample Size:** `N = {claim['sample_size']}`")
                        
                    with deb_col2:
                        st.markdown("#### ⚖️ Devil's Advocate Audit")
                        if audit["falsified"]:
                            st.error(f"**Falsification Triggered:** {audit['falsification_reason']}")
                        else:
                            st.success(f"**Audit Status:** {audit['falsification_reason']}")
                            
                        simpsons = audit.get("simpsons_audit", {})
                        if simpsons.get("cmh_odds_ratio") is not None:
                            st.markdown(f"**Cochran-Mantel-Haenszel Test:** Common OR = `{simpsons['cmh_odds_ratio']:.3f}` (p = `{simpsons.get('cmh_p_value', 0.0):.3e}`)")
                        if simpsons.get("tarone_homogeneity_p") is not None:
                            st.markdown(f"**Tarone Homogeneity Test:** `p = {simpsons['tarone_homogeneity_p']:.3e}` (χ² = `{simpsons.get('tarone_chi2', 0.0):.2f}`)")
                        if simpsons.get("i_squared") is not None:
                            st.markdown(f"**Higgins & Thompson I² Heterogeneity:** `{simpsons['i_squared']:.1f}%` (Cochran's Q = `{simpsons.get('cochrans_q', 0.0):.2f}`)")
                        if simpsons.get("begg_funnel_p") is not None:
                            st.markdown(f"**Begg & Mazumdar Funnel Test:** τ = `{simpsons.get('begg_funnel_tau', 0.0):.2f}` (p = `{simpsons['begg_funnel_p']:.3e}`)")
                        if simpsons.get("e_value") is not None:
                            st.markdown(f"**Causal Sensitivity E-Value:** `{simpsons['e_value']:.2f}`")
                        if simpsons.get("tau_squared") is not None:
                            st.markdown(f"**DerSimonian-Laird Heterogeneity Variance:** `τ² = {simpsons['tau_squared']:.4f}`")
                        if simpsons.get("egger_test_p") is not None:
                            eg_flag = " ⚠️ Asymmetric" if simpsons["egger_test_p"] < 0.05 else " (Symmetric)"
                            st.markdown(f"**Egger Regression Test:** t = `{simpsons.get('egger_test_t', 0.0):.2f}` (p = `{simpsons['egger_test_p']:.3e}`){eg_flag}")
                        if simpsons.get("trim_and_fill_k0") is not None and simpsons.get("trim_and_fill_k0", 0) > 0:
                            st.markdown(f"**Duval-Tweedie Trim & Fill:** `{simpsons['trim_and_fill_k0']}` missing strata imputed (adjusted stat = `{simpsons.get('trim_and_fill_adjusted_stat', 0.0):.4f}`)")
                        if simpsons.get("hausman_p") is not None:
                            h_flag = " ⚠️ Inconsistent / Endogenous" if simpsons.get("is_hausman_inconsistent") else " (Consistent)"
                            st.markdown(f"**Hausman Endogeneity Test:** H = `{simpsons.get('hausman_stat', 0.0):.2f}` (p = `{simpsons['hausman_p']:.3e}`){h_flag}")
                        if simpsons.get("breslow_day_p") is not None:
                            bd_flag = " ⚠️ Heterogeneous" if simpsons["breslow_day_p"] < 0.05 else " (Homogeneous)"
                            st.markdown(f"**Breslow-Day Homogeneity Test:** χ² = `{simpsons.get('breslow_day_chi2', 0.0):.2f}` (p = `{simpsons['breslow_day_p']:.3e}`){bd_flag}")
                        if simpsons.get("vuong_p") is not None:
                            v_flag = " ⚠️ Stratification Significantly Favored" if simpsons.get("is_stratification_favored") else " (Equivalent)"
                            st.markdown(f"**Vuong Model Selection Test:** z = `{simpsons.get('vuong_z', 0.0):.2f}` (p = `{simpsons['vuong_p']:.3e}`){v_flag}")
                        if simpsons.get("higgins_h_squared") is not None and simpsons.get("higgins_h_squared") > 1.0:
                            st.markdown(f"**Higgins H² Heterogeneity:** `{simpsons['higgins_h_squared']:.2f}` (Inflation Factor H = `{np.sqrt(simpsons['higgins_h_squared']):.2f}`)")
                        if simpsons.get("galbraith_outlier_count", 0) > 0:
                            st.markdown(f"**Galbraith Radial Outliers:** `{simpsons['galbraith_outlier_count']}` strata flagged (`{', '.join(simpsons.get('galbraith_outlier_subgroups', []))}`)")
                        if simpsons.get("woolf_p") is not None:
                            w_flag = " ⚠️ Heterogeneous" if simpsons.get("is_woolf_heterogeneous") else " (Homogeneous)"
                            st.markdown(f"**Woolf Odds Ratio Heterogeneity Test:** χ² = `{simpsons.get('woolf_chi2', 0.0):.2f}` (p = `{simpsons['woolf_p']:.3e}`){w_flag}")
                        if simpsons.get("freeman_tukey_p") is not None:
                            ft_flag = " ⚠️ Heterogeneous" if simpsons.get("is_freeman_tukey_heterogeneous") else " (Homogeneous)"
                            st.markdown(f"**Freeman-Tukey Double-Arcsine Homogeneity:** χ² = `{simpsons.get('freeman_tukey_chi2', 0.0):.2f}` (p = `{simpsons['freeman_tukey_p']:.3e}`){ft_flag}")
                        if simpsons.get("kruskal_wallis_p") is not None:
                            kw_flag = " ⚠️ Strata Shift" if simpsons.get("is_kruskal_heterogeneous") else " (Invariant Medians)"
                            st.markdown(f"**Kruskal-Wallis Rank Heterogeneity:** H = `{simpsons.get('kruskal_wallis_h', 0.0):.2f}` (p = `{simpsons['kruskal_wallis_p']:.3e}`){kw_flag}")
                        if simpsons.get("peto_p_value") is not None:
                            peto_flag = " ⚠️ Heterogeneous" if simpsons.get("is_peto_heterogeneous") else " (Homogeneous)"
                            st.markdown(f"**Peto Odds Ratio & Heterogeneity:** OR_Peto = `{simpsons.get('peto_odds_ratio', 0.0):.3f}` (Z = `{simpsons.get('peto_z_stat', 0.0):.2f}`, p = `{simpsons.get('peto_p_value', 0.0):.3e}; Q = {simpsons.get('peto_heterogeneity_q', 0.0):.2f}`, p_het = `{simpsons.get('peto_heterogeneity_p', 0.0):.3e}`){peto_flag}")
                        if simpsons.get("fligner_killeen_p") is not None:
                            flig_flag = " ⚠️ Heterogeneous Variances" if simpsons.get("is_fligner_heterogeneous") else " (Homoscedastic Strata)"
                            st.markdown(f"**Fligner-Killeen Variance Homogeneity:** χ² = `{simpsons.get('fligner_killeen_stat', 0.0):.2f}` (p = `{simpsons.get('fligner_killeen_p', 0.0):.3e}`){flig_flag}")

                        conf_aud = audit.get("confounder_audit", {})
                        if conf_aud.get("e_value") is not None:
                            st.markdown(f"**VanderWeele E-Value:** `{conf_aud['e_value']:.2f}`")
                        if conf_aud.get("aet_selection_ratio") is not None:
                            aet_flag = " ⚠️ High Selection Vulnerability (R_AET < 1.0)" if conf_aud.get("is_aet_vulnerable") else " (Robust R_AET ≥ 1.0)"
                            st.markdown(f"**Altonji-Elder-Taber Ratio:** R_AET = `{conf_aud['aet_selection_ratio']:.2f}`{aet_flag}")
                        if conf_aud.get("imbens_manski_ci_low") is not None:
                            im_flag = " 🟢 Excludes Zero" if conf_aud.get("imbens_manski_excludes_zero") else " ⚠️ Spans Zero"
                            st.markdown(f"**Imbens-Manski 95% Partial Identification CI:** [`{conf_aud['imbens_manski_ci_low']:.3f}`, `{conf_aud.get('imbens_manski_ci_high', 0.0):.3f}`] (c_n = `{conf_aud.get('imbens_manski_cn', 0.0):.3f}`){im_flag}")
                        if conf_aud.get("mallows_cp") is not None:
                            cp_flag = " ⚠️ Severe Omitted Bias" if conf_aud.get("is_mallows_biased") else " (Well-Specified)"
                            st.markdown(f"**Mallows' Cp Model Selection:** Cp = `{conf_aud['mallows_cp']:.2f}` (ΔCp = `{conf_aud.get('cp_bias_discrepancy', 0.0):+.2f}`){cp_flag}")
                        if conf_aud.get("davidson_mackinnon_p") is not None:
                            j_flag = " ⚠️ Confounder Favored (p < 0.05)" if conf_aud.get("is_j_test_rejected") else " (Original Model Adequate)"
                            st.markdown(f"**Davidson-MacKinnon J-Test:** t_J = `{conf_aud.get('davidson_mackinnon_j_stat', 0.0):.2f}` (p = `{conf_aud['davidson_mackinnon_p']:.3e}`){j_flag}")
                        if conf_aud.get("durbin_wu_hausman_p") is not None:
                            dwh_flag = " ⚠️ Regressor Endogeneity (p < 0.05)" if conf_aud.get("is_dwh_endogenous") else " (Exogenous Specification)"
                            st.markdown(f"**Durbin-Wu-Hausman Endogeneity Test:** F = `{conf_aud.get('durbin_wu_hausman_f', 0.0):.2f}` (p = `{conf_aud.get('durbin_wu_hausman_p', 0.0):.3e}`){dwh_flag}")
                        if conf_aud.get("stock_yogo_f_stat") is not None:
                            sy_flag = " ⚠️ Weak Confounder (F < 10)" if conf_aud.get("is_weak_confounder") else " (Strong First-Stage Coupling, F ≥ 10)"
                            st.markdown(f"**Stock-Yogo First-Stage Relevance:** F = `{conf_aud.get('stock_yogo_f_stat', 0.0):.2f}`{sy_flag}")
                        if conf_aud.get("frank_itcv") is not None:
                            itcv_flag = " ⚠️ ITCV Exceeded" if conf_aud.get("is_itcv_exceeded") else " (Below ITCV)"
                            st.markdown(f"**Frank's ITCV Threshold:** ITCV = `{conf_aud['frank_itcv']:.3f}` (Impact = `{conf_aud.get('frank_observed_impact', 0.0):.3f}`){itcv_flag}")
                        if conf_aud.get("akaike_model_weight") is not None:
                            st.markdown(f"**Akaike Model Selection:** w_AIC = `{conf_aud['akaike_model_weight']:.3f}` (ΔAIC = `{conf_aud.get('akaike_delta_aic', 0.0):+.2f}`, ER = `{conf_aud.get('akaike_evidence_ratio', 0.0):.2f}`)")
                        if conf_aud.get("top_confounder_vif") is not None:
                            vif_flag = " ⚠️ High Collinearity" if conf_aud.get("is_collinear") else ""
                            st.markdown(f"**Confounder VIF Collinearity:** `{conf_aud['top_confounder_vif']:.2f}`{vif_flag}")
                        if conf_aud.get("robustness_value") is not None:
                            rcrit_s = f" (R²crit = `{conf_aud.get('cinelli_hazlett_rcrit', 0.0):.2f}`)" if conf_aud.get("cinelli_hazlett_rcrit") is not None else ""
                            st.markdown(f"**Cinelli-Hazlett Robustness Value:** RV_q=1 = `{conf_aud['robustness_value']:.2f}` (RV_α = `{conf_aud.get('robustness_value_alpha', 0.0):.2f}`){rcrit_s}")
                        if conf_aud.get("cohens_f2") is not None:
                            f2_flag = " ⚠️ Negligible" if conf_aud["cohens_f2"] < 0.02 else " (Substantial)"
                            st.markdown(f"**Cohen's f² Partial Effect Size:** f² = `{conf_aud['cohens_f2']:.3f}`{f2_flag}")
                        if conf_aud.get("sobel_p") is not None:
                            st.markdown(f"**Sobel Mediation Test:** `p = {conf_aud['sobel_p']:.3e}` ({conf_aud.get('proportion_mediated', 0.0):.1f}% mediated by `{conf_aud.get('top_confounder')}`)")
                        if conf_aud.get("bayes_factor_bf10") is not None:
                            st.markdown(f"**Kass-Raftery Bayes Factor:** BF_10 = `{conf_aud['bayes_factor_bf10']:.2f}` (ΔBIC = `{conf_aud.get('bic_delta', 0.0):+.2f}`)")
                        if conf_aud.get("ramsey_reset_p") is not None:
                            res_flag = " ⚠️ Functional Misspecification" if conf_aud.get("is_reset_misspecified") else " (Well-Specified)"
                            st.markdown(f"**Ramsey RESET Test:** F = `{conf_aud.get('ramsey_reset_f', 0.0):.2f}` (p = `{conf_aud['ramsey_reset_p']:.3e}`){res_flag}")
                        if conf_aud.get("oster_delta") is not None:
                            ost_flag = " (Robust to unobservables)" if abs(conf_aud["oster_delta"]) >= 1.0 else " ⚠️ Vulnerable"
                            st.markdown(f"**Oster Selection Bound:** `δ* = {conf_aud['oster_delta']:.2f}`{ost_flag}")

                        stab_aud = audit.get("stability_audit", {})
                        if stab_aud.get("sign_invariance_prob") is not None:
                            st.markdown(f"**Sign Invariance Probability:** `P(sign) = {stab_aud['sign_invariance_prob']:.1%}`")
                        if stab_aud.get("max_cooks_distance") is not None and stab_aud.get("max_cooks_distance", 0) > 0:
                            st.markdown(f"**Cook's Distance Influence:** Max `D_i = {stab_aud['max_cooks_distance']:.3f}` ({stab_aud.get('cooks_outlier_count', 0)} points > 4/n, shift = `{stab_aud.get('cooks_statistic_shift', 0.0):.4f}`)")
                        if stab_aud.get("max_dffits") is not None and stab_aud.get("max_dffits", 0) > 0:
                            st.markdown(f"**DFFITS Regression Influence:** Max |DFFITS| = `{stab_aud['max_dffits']:.3f}` ({stab_aud.get('dffits_outlier_count', 0)} points > 2√(p/n))")
                        if stab_aud.get("max_dfbetas") is not None and stab_aud.get("max_dfbetas", 0) > 0:
                            st.markdown(f"**DFBETAS Slope Influence:** Max |DFBETAS| = `{stab_aud['max_dfbetas']:.3f}` ({stab_aud.get('dfbetas_outlier_count', 0)} points > 2/√n)")
                        if stab_aud.get("permutation_p_value") is not None:
                            st.markdown(f"**Permutation Shuffle Test:** `p_perm = {stab_aud['permutation_p_value']:.3e}`")
                        if stab_aud.get("breusch_pagan_p") is not None:
                            bp_flag = " ⚠️ Heteroscedastic" if stab_aud.get("is_heteroscedastic") else " (Homoscedastic)"
                            st.markdown(f"**Breusch-Pagan Test:** LM = `{stab_aud.get('breusch_pagan_lm', 0.0):.2f}` (p = `{stab_aud['breusch_pagan_p']:.3e}`){bp_flag}")
                        if stab_aud.get("hc3_robust_p") is not None:
                            hc3_flag = " ⚠️ Significance Lost" if stab_aud.get("hc3_significance_lost") else " (Robust)"
                            st.markdown(f"**MacKinnon-White HC3 Robust Standard Error:** SE_HC3 = `{stab_aud.get('hc3_robust_se', 0.0):.4f}` (p = `{stab_aud['hc3_robust_p']:.3e}`){hc3_flag}")
                        if stab_aud.get("durbin_watson_stat") is not None:
                            dw_flag = " ⚠️ Serial Correlation" if stab_aud.get("has_serial_correlation") else " (Independent)"
                            st.markdown(f"**Durbin-Watson Serial Test:** DW = `{stab_aud['durbin_watson_stat']:.2f}`{dw_flag}")
                        if stab_aud.get("jarque_bera_p") is not None:
                            jb_flag = " ⚠️ Non-Gaussian" if stab_aud.get("is_non_normal") else " (Normal)"
                            st.markdown(f"**Jarque-Bera Normality Test:** JB = `{stab_aud.get('jarque_bera_stat', 0.0):.2f}` (p = `{stab_aud['jarque_bera_p']:.3e}`){jb_flag}")
                        if stab_aud.get("quantile_discrepancy_index") is not None:
                            qdi_flag_str = " ⚠️ High Discrepancy" if stab_aud.get("qdi_flag") else " (Uniform)"
                            st.markdown(f"**Quantile Discrepancy Index:** QDI = `{stab_aud['quantile_discrepancy_index']:.2f}`{qdi_flag_str}")
                        if stab_aud.get("rank_discrepancy") is not None:
                            st.markdown(f"**Spearman Rank Discrepancy:** `Δr = {stab_aud['rank_discrepancy']:.3f}` (ρ = `{stab_aud.get('spearman_rho', 0.0):.3f}`)")
                        if stab_aud.get("theil_sen_discrepancy") is not None:
                            ts_flag = " ⚠️ High Discrepancy" if stab_aud["theil_sen_discrepancy"] > 0.20 else " (Concordant)"
                            st.markdown(f"**Theil-Sen Robust Slope:** β_TS = `{stab_aud.get('theil_sen_slope', 0.0):.4f}` (Δ_TS = `{stab_aud['theil_sen_discrepancy']:.1%}`){ts_flag}")
                        if stab_aud.get("bca_ci_low") is not None:
                            bca_flag = " ⚠️ Crosses Zero" if stab_aud.get("bca_zero_crossing") else " (Stable Non-Zero)"
                            st.markdown(f"**BCa Bootstrap 95% CI:** [`{stab_aud['bca_ci_low']:.4f}`, `{stab_aud.get('bca_ci_high', 0.0):.4f}`]{bca_flag}")
                        if stab_aud.get("max_leverage") is not None and stab_aud.get("max_leverage", 0) > 0:
                            st.markdown(f"**Outlier Leverage:** Max `h_ii = {stab_aud['max_leverage']:.4f}` ({stab_aud.get('high_leverage_points', 0)} high-leverage points)")
                        if stab_aud.get("chow_p_value") is not None:
                            chow_flag = " ⚠️ Structural Break Detected" if stab_aud.get("has_structural_break") else " (Parameter Invariant)"
                            st.markdown(f"**Chow Structural Break Test:** F = `{stab_aud.get('chow_f_stat', 0.0):.2f}` (p = `{stab_aud['chow_p_value']:.3e}`){chow_flag}")
                        if stab_aud.get("type_m_exaggeration_ratio") is not None:
                            tm_flag = " ⚠️ Winner's Curse Vulnerable" if stab_aud.get("is_type_m_vulnerable") else " (Stable Effect Size)"
                            st.markdown(f"**Gelman-Carlin Design Analysis:** Type M = `{stab_aud['type_m_exaggeration_ratio']:.2f}x` (Type S = `{stab_aud.get('type_s_error_prob', 0.0):.1%}`, Power = `{stab_aud.get('retrodesign_power', 0.0):.1%}`){tm_flag}")
                        if stab_aud.get("andrews_sup_w") is not None:
                            and_flag = f" ⚠️ Unknown Break at τ* = `{stab_aud.get('andrews_break_fraction', 0.0):.1%}`" if stab_aud.get("has_unknown_structural_break") else f" (Parameter Constant along Regressor, τ* = `{stab_aud.get('andrews_break_fraction', 0.0):.1%}`)"
                            st.markdown(f"**Andrews Sup-Wald Unknown Break Test:** Sup W = `{stab_aud['andrews_sup_w']:.2f}` (p = `{stab_aud.get('andrews_p_value', 0.0):.3e}`){and_flag}")
                        if stab_aud.get("cusum_stat") is not None:
                            cusum_flag = " ⚠️ Parameter Drift Detected (S > 0.948)" if stab_aud.get("has_cusum_instability") else " (Stable Recursive Path)"
                            st.markdown(f"**Brown-Durbin-Evans CUSUM Drift:** S = `{stab_aud.get('cusum_stat', 0.0):.3f}` (p ≈ `{stab_aud.get('cusum_p_approx', 0.0):.3e}`){cusum_flag}")
                        if stab_aud.get("koenker_p_value") is not None:
                            koen_flag = " ⚠️ Heteroscedastic" if stab_aud.get("is_koenker_heteroscedastic") else " (Homoscedastic)"
                            st.markdown(f"**Koenker Studentized Breusch-Pagan Test:** LM = `{stab_aud.get('koenker_lm_stat', 0.0):.2f}` (p = `{stab_aud['koenker_p_value']:.3e}`){koen_flag}")

                        if simpsons.get("confounder_column") and simpsons.get("breakdowns"):
                            st.markdown(f"**Subgroup Conditioning on `{simpsons['confounder_column']}`:**")
                            bd_table = pd.DataFrame([{
                                "Subgroup": b["group_name"],
                                "N": b["sample_size"],
                                "Subgroup Stat": f"{b['subgroup_statistic']:+.4f}",
                                "Reversal?": "⚠️ REVERSED" if b["reverses_sign"] else "Consistent"
                            } for b in simpsons["breakdowns"]])
                            st.dataframe(bd_table, hide_index=True)
                    
                    st.markdown("#### 🏆 Referee Adjudication & Verdict")
                    if "SIMPSON" in badge:
                        st.warning(f"**Synthesis:** {f['synthesis_summary']}")
                    elif "ROBUST" in badge:
                        st.success(f"**Synthesis:** {f['synthesis_summary']}")
                    else:
                        st.error(f"**Synthesis:** {f['synthesis_summary']}")
                    
                    rationale = f.get("classification_rationale", {})
                    if rationale.get("why_assigned"):
                        with st.container():
                            st.markdown(
                                f"""
                                <div style="background-color: rgba(30, 41, 59, 0.4); border-left: 4px solid #3b82f6; padding: 10px 14px; margin: 8px 0; border-radius: 4px;">
                                    <p style="margin: 0 0 6px 0; font-weight: bold; color: #93c5fd;">🎯 Why Classified as {rationale.get('category_name', badge)}:</p>
                                    <p style="margin: 0 0 6px 0; font-size: 0.95em;">{rationale['why_assigned']}</p>
                                    <p style="margin: 0 0 4px 0; font-size: 0.85em; color: #cbd5e1;"><b>Decision Rule:</b> <code>{rationale.get('decision_rule', '')}</code></p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            if rationale.get("key_evidence"):
                                st.markdown("**Key Mathematical Evidence:**")
                                for ev in rationale["key_evidence"]:
                                    st.markdown(f"- `{ev}`")
                    
                    st.markdown(f"**💡 Actionable Takeaway:** *{f['actionable_takeaway']}*")

        with tab_viz:
            paradoxical_findings = [f for f in findings if f["audit"].get("simpsons_audit", {}).get("breakdowns")]
            if paradoxical_findings:
                selected_var = st.selectbox("Select variable to visualize subgroup conditioning:", [f["claim"]["source_variable"] for f in paradoxical_findings])
                match_f = next(f for f in paradoxical_findings if f["claim"]["source_variable"] == selected_var)
                simps = match_f["audit"]["simpsons_audit"]
                
                chart_data = []
                chart_data.append({"Category": "Aggregate Global Effect", "Effect Size": simps["global_statistic"]})
                for b in simps["breakdowns"]:
                    chart_data.append({"Category": f"Stratum: {b['group_name']}", "Effect Size": b["subgroup_statistic"]})
                
                df_chart = pd.DataFrame(chart_data)
                st.bar_chart(df_chart.set_index("Category"))
                st.info(f"**Conditioning Variable:** `{simps['confounder_column']}`. Notice the inversion between aggregate and stratified bars!")
            else:
                st.info("No subgroup reversals were detected in this audit.")

            st.write("---")
            st.markdown("#### 🕸️ Causal Directed Acyclic Graph (DAG) & Backdoor Path Inspector")
            all_vars = [f["claim"]["source_variable"] for f in findings]
            sel_dag_var = st.selectbox("Select variable for Causal DAG Inspection:", all_vars, key="dag_select_var")
            f_dag = next(f for f in findings if f["claim"]["source_variable"] == sel_dag_var)
            
            src = f_dag["claim"]["source_variable"]
            tgt = f_dag["claim"]["target_variable"]
            v_badge = f_dag["verdict_badge"]
            
            if "SIMPSON" in v_badge:
                conf = f_dag["audit"]["simpsons_audit"].get("confounder_column", "Confounder")
                st.info(f"**Structural Causal Model:** Backdoor Path Open via `{conf}`. Conditioning $P(Y \\mid do(X)) \\ne P(Y \\mid X)$.")
                st.markdown(
                    f"**Causal Diagram (Pearl's Backdoor Criterion):**\n"
                    f"* Confounder `{conf}` simultaneously influences `{src}` and `{tgt}`.\n"
                    f"* Naive association is an artifact of composition; adjusting for `{conf}` closes the backdoor path."
                )
            elif "CONFOUNDED" in v_badge:
                conf = f_dag["audit"]["confounder_audit"].get("top_confounder", "Covariate")
                att = f_dag["audit"]["confounder_audit"].get("attenuation_percentage", 0.0)
                st.warning(f"**Structural Causal Model:** Association between `{src}` and `{tgt}` attenuates by {att:.1f}% when conditioning on `{conf}`.")
            else:
                st.success(f"**Structural Causal Model:** Verified invariant direct path `{src} \u2192 {tgt}` across tested covariates.")

        with tab_briefing:
            narrator = LLMNarrator(provider=llm_provider)
            briefing = narrator.generate_executive_briefing(findings, dataset_name=dataset_name)
            st.markdown(briefing)

        with tab_copilot:
            st.markdown("#### 💬 Ask ADAA AI Copilot")
            st.caption("Ask questions about risk scores, specific variables, confounders, or leadership actions.")

            st.markdown("**Suggested Quick Inquiries:**")
            sug_col1, sug_col2, sug_col3 = st.columns(3)
            with sug_col1:
                if st.button("📊 Altonji-Elder-Taber Ratio", key="sug_aet"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Altonji-Elder-Taber Ratio"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Altonji-Elder-Taber Ratio", findings, dataset_name=dataset_name)
            with sug_col2:
                if st.button("⚡ Chow Structural Break", key="sug_chow"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Check Chow Structural Break"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Check Chow Structural Break", findings, dataset_name=dataset_name)
            with sug_col3:
                if st.button("🎯 Galbraith Outlier Strata", key="sug_galb"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Detect Galbraith Outlier Strata"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Detect Galbraith Outlier Strata", findings, dataset_name=dataset_name)

            sug_col4, sug_col5, sug_col6 = st.columns(3)
            with sug_col4:
                if st.button("🔬 Gelman-Carlin Type S/M", key="sug_gelman"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Gelman-Carlin Type S and Type M design analysis"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Gelman-Carlin Type S and Type M design analysis", findings, dataset_name=dataset_name)
            with sug_col5:
                if st.button("📐 Imbens-Manski Bounds CI", key="sug_im"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Imbens-Manski partial identification confidence intervals"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Imbens-Manski partial identification confidence intervals", findings, dataset_name=dataset_name)
            with sug_col6:
                if st.button("📈 Mallows' Cp Criterion", key="sug_cp"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Mallows Cp model misspecification criterion"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Mallows Cp model misspecification criterion", findings, dataset_name=dataset_name)

            sug_col7, sug_col8, sug_col9 = st.columns(3)
            with sug_col7:
                if st.button("📐 Freeman-Tukey Arcsine", key="sug_ft"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Freeman-Tukey double-arcsine homogeneity test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Freeman-Tukey double-arcsine homogeneity test", findings, dataset_name=dataset_name)
            with sug_col8:
                if st.button("🔍 Davidson-MacKinnon J-Test", key="sug_jtest"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Davidson-MacKinnon J-test for non-nested models"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Davidson-MacKinnon J-test for non-nested models", findings, dataset_name=dataset_name)
            with sug_col9:
                if st.button("📈 Andrews Sup-Wald Break", key="sug_andrews"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Donald Andrews Supremum Wald test for unknown structural breaks"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Donald Andrews Supremum Wald test for unknown structural breaks", findings, dataset_name=dataset_name)

            sug_col10, sug_col11, sug_col12 = st.columns(3)
            with sug_col10:
                if st.button("🧮 Peto Odds & Heterogeneity", key="sug_peto"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Peto odds ratio and heterogeneity test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Peto odds ratio and heterogeneity test", findings, dataset_name=dataset_name)
            with sug_col11:
                if st.button("⚖️ Durbin-Wu-Hausman Test", key="sug_dwh"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Durbin-Wu-Hausman specification test for endogeneity"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Durbin-Wu-Hausman specification test for endogeneity", findings, dataset_name=dataset_name)
            with sug_col12:
                if st.button("🌊 Brown-Durbin-Evans CUSUM", key="sug_cusum"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Brown-Durbin-Evans CUSUM test for recursive parameter stability"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Brown-Durbin-Evans CUSUM test for recursive parameter stability", findings, dataset_name=dataset_name)

            with st.form(key="copilot_form", clear_on_submit=False):
                user_q = st.text_input(
                    "Enter your question:",
                    placeholder="e.g., Why did gender flag as a Simpson's Paradox?",
                    key="copilot_user_q"
                )
                submit_q = st.form_submit_button("Ask Copilot 🚀")

            if submit_q and user_q:
                with st.spinner("Copilot is analyzing the audited findings..."):
                    narrator = LLMNarrator(provider=llm_provider)
                    answer = narrator.answer_query(user_q, findings, dataset_name=dataset_name)
                    st.session_state["last_copilot_q"] = user_q
                    st.session_state["last_copilot_a"] = answer

            if "last_copilot_a" in st.session_state:
                st.markdown(f"**Q:** *{st.session_state.get('last_copilot_q', '')}*")
                st.info(f"**Copilot:** {st.session_state['last_copilot_a']}")

        with tab_export:
            st.markdown("#### 📥 Download Audit Artifacts")
            html_path = Path(results["reports"]["html"])
            md_path = Path(results["reports"]["markdown"])
            json_path = Path(results["reports"]["json"])

            if html_path.exists():
                st.download_button("📄 Download HTML Report", data=html_path.read_text(encoding="utf-8"), file_name=html_path.name, mime="text/html")
            if md_path.exists():
                st.download_button("📝 Download Markdown Report", data=md_path.read_text(encoding="utf-8"), file_name=md_path.name, mime="text/markdown")
            if json_path.exists():
                st.download_button("📦 Download JSON Findings Dossier", data=json_path.read_text(encoding="utf-8"), file_name=json_path.name, mime="application/json")

            narrator = LLMNarrator(provider=llm_provider)
            briefing_text = narrator.generate_executive_briefing(findings, dataset_name=dataset_name)
            st.download_button("🧠 Download Executive Narrative Briefing (Markdown)", data=briefing_text, file_name="llm_executive_briefing.md", mime="text/markdown")

            summary_rows = []
            for f in findings:
                cl = f.get("claim", {})
                stab = f.get("audit", {}).get("stability_audit", {})
                conf = f.get("audit", {}).get("confounder_audit", {})
                simp = f.get("audit", {}).get("simpsons_audit", {})
                summary_rows.append({
                    "Variable": cl.get("source_variable", ""),
                    "Target": cl.get("target_variable", ""),
                    "Statistic": f"{cl.get('statistic_name', '')} = {cl.get('statistic_value', 0.0):.4f}",
                    "p_value": f"{cl.get('p_value', 0.0):.4e}",
                    "FDR_q_value": f"{cl.get('fdr_q_value', 0.0):.4e}" if cl.get("fdr_q_value") is not None else "N/A",
                    "Bonferroni_p": f"{cl.get('bonferroni_p_value', 0.0):.4e}" if cl.get("bonferroni_p_value") is not None else "N/A",
                    "Verdict": f.get("verdict_badge", ""),
                    "Confidence": f.get("confidence_score", 0.0),
                    "Higgins_I2": f"{simp.get('i_squared', 0.0):.1f}%" if simp.get("i_squared") is not None else "N/A",
                    "Higgins_H_Squared": f"{simp.get('higgins_h_squared', 0.0):.2f}" if simp.get("higgins_h_squared") is not None else "N/A",
                    "Galbraith_Outliers": f"{simp.get('galbraith_outlier_count', 0)}" if simp.get("galbraith_outlier_count") is not None else "N/A",
                    "Woolf_Heterogeneity_p": f"{simp.get('woolf_p', 0.0):.3e}" if simp.get("woolf_p") is not None else "N/A",
                    "Freeman_Tukey_Heterogeneity_p": f"{simp.get('freeman_tukey_p', 0.0):.3e}" if simp.get("freeman_tukey_p") is not None else "N/A",
                    "Kruskal_Wallis_p": f"{simp.get('kruskal_wallis_p', 0.0):.3e}" if simp.get("kruskal_wallis_p") is not None else "N/A",
                    "Peto_Odds_Ratio": f"{simp.get('peto_odds_ratio', 0.0):.3f}" if simp.get("peto_odds_ratio") is not None else "N/A",
                    "Peto_Heterogeneity_p": f"{simp.get('peto_heterogeneity_p', 0.0):.3e}" if simp.get("peto_heterogeneity_p") is not None else "N/A",
                    "Fligner_Killeen_p": f"{simp.get('fligner_killeen_p', 0.0):.3e}" if simp.get("fligner_killeen_p") is not None else "N/A",
                    "DerSimonian_Laird_Tau2": f"{simp.get('tau_squared', 0.0):.4f}" if simp.get("tau_squared") is not None else "N/A",
                    "Trim_and_Fill_k0": f"{simp.get('trim_and_fill_k0', 0)}" if simp.get("trim_and_fill_k0") is not None else "N/A",
                    "Hausman_p": f"{simp.get('hausman_p', 0.0):.3e}" if simp.get("hausman_p") is not None else "N/A",
                    "Breslow_Day_p": f"{simp.get('breslow_day_p', 0.0):.3e}" if simp.get("breslow_day_p") is not None else "N/A",
                    "Vuong_z": f"{simp.get('vuong_z', 0.0):.2f}" if simp.get("vuong_z") is not None else "N/A",
                    "Vuong_p": f"{simp.get('vuong_p', 0.0):.3e}" if simp.get("vuong_p") is not None else "N/A",
                    "Begg_Funnel_p": f"{simp.get('begg_funnel_p', 0.0):.3e}" if simp.get("begg_funnel_p") is not None else "N/A",
                    "Egger_Funnel_p": f"{simp.get('egger_test_p', 0.0):.3e}" if simp.get("egger_test_p") is not None else "N/A",
                    "Cinelli_Hazlett_RV": f"{conf.get('robustness_value', 0.0):.2f}" if conf.get("robustness_value") is not None else "N/A",
                    "Cinelli_Hazlett_Rcrit": f"{conf.get('cinelli_hazlett_rcrit', 0.0):.2f}" if conf.get("cinelli_hazlett_rcrit") is not None else "N/A",
                    "Frank_ITCV": f"{conf.get('frank_itcv', 0.0):.3f}" if conf.get("frank_itcv") is not None else "N/A",
                    "Frank_Observed_Impact": f"{conf.get('frank_observed_impact', 0.0):.3f}" if conf.get("frank_observed_impact") is not None else "N/A",
                    "Rosenbaum_Gamma_Crit": f"{conf.get('rosenbaum_gamma_crit', 0.0):.2f}" if conf.get("rosenbaum_gamma_crit") is not None else "N/A",
                    "Manski_Bound_Low": f"{conf.get('manski_bound_low', 0.0):.3f}" if conf.get("manski_bound_low") is not None else "N/A",
                    "Manski_Bound_High": f"{conf.get('manski_bound_high', 0.0):.3f}" if conf.get("manski_bound_high") is not None else "N/A",
                    "Imbens_Manski_CI_Low": f"{conf.get('imbens_manski_ci_low', 0.0):.3f}" if conf.get("imbens_manski_ci_low") is not None else "N/A",
                    "Imbens_Manski_CI_High": f"{conf.get('imbens_manski_ci_high', 0.0):.3f}" if conf.get("imbens_manski_ci_high") is not None else "N/A",
                    "Mallows_Cp": f"{conf.get('mallows_cp', 0.0):.2f}" if conf.get("mallows_cp") is not None else "N/A",
                    "Mallows_Discrepancy": f"{conf.get('cp_bias_discrepancy', 0.0):+.2f}" if conf.get("cp_bias_discrepancy") is not None else "N/A",
                    "Davidson_MacKinnon_J_Stat": f"{conf.get('davidson_mackinnon_j_stat', 0.0):.2f}" if conf.get("davidson_mackinnon_j_stat") is not None else "N/A",
                    "Davidson_MacKinnon_p": f"{conf.get('davidson_mackinnon_p', 0.0):.3e}" if conf.get("davidson_mackinnon_p") is not None else "N/A",
                    "DWH_Endogeneity_F": f"{conf.get('durbin_wu_hausman_f', 0.0):.2f}" if conf.get("durbin_wu_hausman_f") is not None else "N/A",
                    "DWH_Endogeneity_p": f"{conf.get('durbin_wu_hausman_p', 0.0):.3e}" if conf.get("durbin_wu_hausman_p") is not None else "N/A",
                    "Stock_Yogo_First_Stage_F": f"{conf.get('stock_yogo_f_stat', 0.0):.2f}" if conf.get("stock_yogo_f_stat") is not None else "N/A",
                    "Altonji_Elder_Taber_Ratio": f"{conf.get('aet_selection_ratio', 0.0):.2f}" if conf.get("aet_selection_ratio") is not None else "N/A",
                    "Akaike_Model_Weight": f"{conf.get('akaike_model_weight', 0.0):.3f}" if conf.get("akaike_model_weight") is not None else "N/A",
                    "Cohens_f2": f"{conf.get('cohens_f2', 0.0):.3f}" if conf.get("cohens_f2") is not None else "N/A",
                    "Sobel_p_value": f"{conf.get('sobel_p', 0.0):.3e}" if conf.get("sobel_p") is not None else "N/A",
                    "Bayes_Factor_BF10": f"{conf.get('bayes_factor_bf10', 0.0):.2f}" if conf.get("bayes_factor_bf10") is not None else "N/A",
                    "Ramsey_RESET_p": f"{conf.get('ramsey_reset_p', 0.0):.3e}" if conf.get("ramsey_reset_p") is not None else "N/A",
                    "Oster_Delta": f"{conf.get('oster_delta', 0.0):.2f}" if conf.get("oster_delta") is not None else "N/A",
                    "Sign_Invariance_P": f"{stab.get('sign_invariance_prob', 1.0):.1%}",
                    "Cooks_D_Max": f"{stab.get('max_cooks_distance', 0.0):.3f}",
                    "Max_DFFITS": f"{stab.get('max_dffits', 0.0):.3f}" if stab.get("max_dffits") is not None else "N/A",
                    "Max_DFBETAS": f"{stab.get('max_dfbetas', 0.0):.3f}" if stab.get("max_dfbetas") is not None else "N/A",
                    "Permutation_p": f"{stab.get('permutation_p_value', 0.0):.3e}" if stab.get("permutation_p_value") is not None else "N/A",
                    "Breusch_Pagan_p": f"{stab.get('breusch_pagan_p', 0.0):.3e}" if stab.get("breusch_pagan_p") is not None else "N/A",
                    "White_LM_p": f"{stab.get('white_lm_p', 0.0):.3e}" if stab.get("white_lm_p") is not None else "N/A",
                    "Chow_F_Stat": f"{stab.get('chow_f_stat', 0.0):.2f}" if stab.get("chow_f_stat") is not None else "N/A",
                    "Chow_p_value": f"{stab.get('chow_p_value', 0.0):.3e}" if stab.get("chow_p_value") is not None else "N/A",
                    "Andrews_Sup_W": f"{stab.get('andrews_sup_w', 0.0):.2f}" if stab.get("andrews_sup_w") is not None else "N/A",
                    "Andrews_Break_Fraction": f"{stab.get('andrews_break_fraction', 0.0):.1%}" if stab.get("andrews_break_fraction") is not None else "N/A",
                    "Andrews_Break_p": f"{stab.get('andrews_p_value', 0.0):.3e}" if stab.get("andrews_p_value") is not None else "N/A",
                    "CUSUM_Stat": f"{stab.get('cusum_stat', 0.0):.3f}" if stab.get("cusum_stat") is not None else "N/A",
                    "CUSUM_Drift_Flag": str(stab.get("has_cusum_instability", False)),
                    "Koenker_Studentized_LM": f"{stab.get('koenker_lm_stat', 0.0):.2f}" if stab.get("koenker_lm_stat") is not None else "N/A",
                    "Koenker_p_value": f"{stab.get('koenker_p_value', 0.0):.3e}" if stab.get("koenker_p_value") is not None else "N/A",
                    "HC3_Robust_p": f"{stab.get('hc3_robust_p', 0.0):.3e}" if stab.get("hc3_robust_p") is not None else "N/A",
                    "Durbin_Watson": f"{stab.get('durbin_watson_stat', 0.0):.2f}" if stab.get("durbin_watson_stat") is not None else "N/A",
                    "Jarque_Bera_p": f"{stab.get('jarque_bera_p', 0.0):.3e}" if stab.get("jarque_bera_p") is not None else "N/A",
                    "Quantile_Discrepancy_QDI": f"{stab.get('quantile_discrepancy_index', 0.0):.2f}" if stab.get("quantile_discrepancy_index") is not None else "N/A",
                    "Theil_Sen_Discrepancy": f"{stab.get('theil_sen_discrepancy', 0.0):.1%}" if stab.get("theil_sen_discrepancy") is not None else "N/A",
                    "Type_S_Error_P": f"{stab.get('type_s_error_prob', 0.0):.1%}" if stab.get("type_s_error_prob") is not None else "N/A",
                    "Type_M_Exaggeration_Ratio": f"{stab.get('type_m_exaggeration_ratio', 0.0):.2f}x" if stab.get("type_m_exaggeration_ratio") is not None else "N/A",
                    "BCa_CI_Low": f"{stab.get('bca_ci_low', 0.0):.4f}" if stab.get("bca_ci_low") is not None else "N/A",
                    "BCa_CI_High": f"{stab.get('bca_ci_high', 0.0):.4f}" if stab.get("bca_ci_high") is not None else "N/A",
                    "Confounder_VIF": f"{conf.get('top_confounder_vif', 1.0):.2f}" if conf.get("top_confounder_vif") is not None else "N/A",
                    "Synthesis": f.get("synthesis_summary", ""),
                    "Takeaway": f.get("actionable_takeaway", ""),
                })
            df_summary = pd.DataFrame(summary_rows)
            st.download_button("📊 Download Findings Summary CSV", data=df_summary.to_csv(index=False).encode("utf-8"), file_name="findings_summary.csv", mime="text/csv")
                
    else:
        st.info("👈 Select a benchmark dataset or upload your own table, then press **Run Adversarial Multi-Agent Audit**.")
