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
                        if simpsons.get("harbord_p_value") is not None:
                            harb_flag = " ⚠️ Funnel Asymmetry (p < 0.05)" if simpsons.get("is_harbord_asymmetric") else " (Symmetric Strata)"
                            st.markdown(f"**Harbord Modified Score Asymmetry:** t = `{simpsons.get('harbord_t_stat', 0.0):.2f}` (p = `{simpsons.get('harbord_p_value', 0.0):.3e}`){harb_flag}")
                        if simpsons.get("brown_forsythe_p") is not None:
                            bf_flag = " ⚠️ Heterogeneous Variances" if simpsons.get("is_brown_forsythe_heterogeneous") else " (Homoscedastic Strata)"
                            st.markdown(f"**Brown-Forsythe Scale Homogeneity:** W = `{simpsons.get('brown_forsythe_stat', 0.0):.2f}` (p = `{simpsons.get('brown_forsythe_p', 0.0):.3e}`){bf_flag}")
                        if simpsons.get("peters_test_p") is not None:
                            peters_flag = " ⚠️ Funnel Asymmetry (p < 0.05)" if simpsons.get("is_peters_asymmetric") else " (Symmetric Strata)"
                            st.markdown(f"**Peters Regression Asymmetry Test:** t = `{simpsons.get('peters_test_t', 0.0):.2f}` (p = `{simpsons.get('peters_test_p', 0.0):.3e}`){peters_flag}")
                        if simpsons.get("kish_design_effect") is not None:
                            vr_flag = " (Variance Reduction)" if simpsons.get("stratified_variance_ratio", 1.0) < 1.0 else " (Variance Inflation)"
                            st.markdown(f"**Kish Stratification Design Effect:** D_Kish = `{simpsons.get('kish_design_effect', 0.0):.2f}` (Var Ratio = `{simpsons.get('stratified_variance_ratio', 0.0):.2f}`{vr_flag})")
                        if simpsons.get("rucker_test_p") is not None:
                            ruck_flag = " ⚠️ Arcsine Funnel Asymmetry (p < 0.05)" if simpsons.get("is_rucker_asymmetric") else " (Symmetric Strata Precision)"
                            st.markdown(f"**Rücker Arcsine Asymmetry Test:** t = `{simpsons.get('rucker_test_t', 0.0):.2f}` (p = `{simpsons.get('rucker_test_p', 0.0):.3e}`){ruck_flag}")
                        if simpsons.get("levene_p") is not None:
                            lev_flag = " ⚠️ Unequal Scale Spread (p < 0.05)" if simpsons.get("is_levene_heterogeneous") else " (Homoscedastic Strata Spread)"
                            st.markdown(f"**Levene Scale Homogeneity:** W = `{simpsons.get('levene_stat', 0.0):.2f}` (p = `{simpsons.get('levene_p', 0.0):.3e}`){lev_flag}")
                        if simpsons.get("macaskill_test_p") is not None:
                            mac_flag = " ⚠️ Funnel Asymmetry / Small Strata Bias (p < 0.05)" if simpsons.get("is_macaskill_asymmetric") else " (Symmetric Strata Precision)"
                            st.markdown(f"**Macaskill Weighted Funnel Asymmetry:** t = `{simpsons.get('macaskill_test_t', 0.0):.2f}` (p = `{simpsons.get('macaskill_test_p', 0.0):.3e}`){mac_flag}")
                        if simpsons.get("bartlett_p") is not None:
                            bart_flag = " ⚠️ Unequal Scale Spread (p < 0.05)" if simpsons.get("is_bartlett_heterogeneous") else " (Homoscedastic Strata Spread)"
                            st.markdown(f"**Bartlett Scale Homogeneity:** B = `{simpsons.get('bartlett_stat', 0.0):.2f}` (p = `{simpsons.get('bartlett_p', 0.0):.3e}`){bart_flag}")
                        if simpsons.get("deeks_test_p") is not None:
                            deeks_flag = " ⚠️ DOR Funnel Asymmetry (p < 0.05)" if simpsons.get("is_deeks_asymmetric") else " (Symmetric DOR Precision)"
                            st.markdown(f"**Deeks DOR Funnel Asymmetry:** t = `{simpsons.get('deeks_test_t', 0.0):.2f}` (p = `{simpsons.get('deeks_test_p', 0.0):.3e}`){deeks_flag}")
                        if simpsons.get("mood_p_value") is not None:
                            mood_flag = " ⚠️ Unequal Rank Dispersion (p < 0.05)" if simpsons.get("is_mood_heterogeneous") else " (Homoscedastic Median Spread)"
                            st.markdown(f"**Mood Median-Rank Scale Dispersion:** Z = `{simpsons.get('mood_stat_z', 0.0):.2f}` (p = `{simpsons.get('mood_p_value', 0.0):.3e}`){mood_flag}")
                        if simpsons.get("thompson_sharp_p") is not None:
                            ts_flag = " ⚠️ Random-Effects Funnel Asymmetry (p < 0.05)" if simpsons.get("is_thompson_asymmetric") else " (Symmetric RE Precision)"
                            st.markdown(f"**Thompson-Sharp Funnel Test:** t = `{simpsons.get('thompson_sharp_t', 0.0):.2f}` (p = `{simpsons.get('thompson_sharp_p', 0.0):.3e}`){ts_flag}")
                        if simpsons.get("ansari_bradley_p") is not None:
                            ab_flag = " ⚠️ Unequal Rank-Sum Dispersion (p < 0.05)" if simpsons.get("is_ansari_heterogeneous") else " (Homogeneous Scale Spread)"
                            st.markdown(f"**Ansari-Bradley Scale Dispersion:** W = `{simpsons.get('ansari_bradley_stat', 0.0):.2f}` (p = `{simpsons.get('ansari_bradley_p', 0.0):.3e}`){ab_flag}")
                        if simpsons.get("moreno_test_p") is not None:
                            moreno_flag = " ⚠️ Variance-Dependent Funnel Asymmetry (p < 0.05)" if simpsons.get("is_moreno_asymmetric") else " (Symmetric Precision)"
                            st.markdown(f"**Moreno Funnel Asymmetry Test:** t = `{simpsons.get('moreno_test_t', 0.0):.2f}` (p = `{simpsons.get('moreno_test_p', 0.0):.3e}`){moreno_flag}")
                        if simpsons.get("siegel_tukey_p") is not None:
                            st_flag = " ⚠️ Unequal Rank Extremes Dispersion (p < 0.05)" if simpsons.get("is_siegel_tukey_heterogeneous") else " (Equidispersed Extremes)"
                            st.markdown(f"**Siegel-Tukey Rank Extremes Scale:** W = `{simpsons.get('siegel_tukey_stat', 0.0):.2f}` (p = `{simpsons.get('siegel_tukey_p', 0.0):.3e}`){st_flag}")
                        if simpsons.get("sterne_test_p") is not None:
                            sterne_flag = " ⚠️ Standard-Error Funnel Asymmetry (p < 0.05)" if simpsons.get("is_sterne_asymmetric") else " (Symmetric SE Precision)"
                            st.markdown(f"**Sterne-Egger Funnel Asymmetry:** t = `{simpsons.get('sterne_test_t', 0.0):.2f}` (p = `{simpsons.get('sterne_test_p', 0.0):.3e}`){sterne_flag}")
                        if simpsons.get("klotz_scale_p") is not None:
                            klotz_flag = " ⚠️ Unequal Normal Scores Scale (p < 0.05)" if simpsons.get("is_klotz_heterogeneous") else " (Equidispersed Normal Scores)"
                            st.markdown(f"**Klotz Normal Scores Scale:** Z = `{simpsons.get('klotz_scale_stat', 0.0):.2f}` (p = `{simpsons.get('klotz_scale_p', 0.0):.3e}`){klotz_flag}")
                        if simpsons.get("lfk_index") is not None:
                            lfk_flag = " ⚠️ Major Funnel Asymmetry (|LFK| > 2.0)" if simpsons.get("is_lfk_asymmetric") else " (Symmetric Precision)"
                            st.markdown(f"**LFK Funnel Asymmetry Index:** LFK = `{simpsons['lfk_index']:.2f}`{lfk_flag}")
                        if simpsons.get("conover_p_value") is not None:
                            con_flag = " ⚠️ Unequal Scale Dispersion (p < 0.05)" if simpsons.get("is_conover_heterogeneous") else " (Equidispersed Squared Ranks)"
                            st.markdown(f"**Conover Squared Ranks Scale:** Z = `{simpsons.get('conover_stat_z', 0.0):.2f}` (p = `{simpsons.get('conover_p_value', 0.0):.3e}`){con_flag}")
                        if simpsons.get("pet_test_p") is not None:
                            pet_flag = " ⚠️ Small-Stratum Reporting Bias (p < 0.05)" if simpsons.get("is_pet_asymmetric") else " (Symmetric Precision Limit)"
                            st.markdown(f"**PET-PEESE Meta-Regression:** t_PET = `{simpsons.get('pet_test_t', 0.0):.2f}` (p = `{simpsons.get('pet_test_p', 0.0):.3e}`), t_PEESE = `{simpsons.get('peese_test_t', 0.0):.2f}`{pet_flag}")
                        if simpsons.get("sukhatme_p_value") is not None:
                            sukh_flag = " ⚠️ Unequal Scale Dispersion (p < 0.05)" if simpsons.get("is_sukhatme_heterogeneous") else " (Homogeneous Scale Spread)"
                            st.markdown(f"**Sukhatme Rank Dispersion Scale:** Z = `{simpsons.get('sukhatme_stat_z', 0.0):.2f}` (p = `{simpsons.get('sukhatme_p_value', 0.0):.3e}`){sukh_flag}")
                        if simpsons.get("tang_test_p") is not None:
                            tang_flag = " ⚠️ Small-Study Reporting Bias (p < 0.05)" if simpsons.get("is_tang_asymmetric") else " (Symmetric Precision Limit)"
                            st.markdown(f"**Tang-Liu Funnel Asymmetry:** t = `{simpsons.get('tang_test_t', 0.0):.2f}` (p = `{simpsons.get('tang_test_p', 0.0):.3e}`){tang_flag}")
                        if simpsons.get("capon_p_value") is not None:
                            capon_flag = " ⚠️ LMP Normal Order Scale Dispersion (p < 0.05)" if simpsons.get("is_capon_heterogeneous") else " (Homogeneous Normal Order Spread)"
                            st.markdown(f"**Capon Normal Scores Scale:** Z = `{simpsons.get('capon_stat_z', 0.0):.2f}` (p = `{simpsons.get('capon_p_value', 0.0):.3e}`){capon_flag}")
                        if simpsons.get("pustejovsky_test_p") is not None:
                            pust_flag = " ⚠️ Variance-Stabilized Asymmetry (p < 0.05)" if simpsons.get("is_pustejovsky_asymmetric") else " (Symmetric Precision Limit)"
                            st.markdown(f"**Pustejovsky-Rodgers Funnel Asymmetry:** t = `{simpsons.get('pustejovsky_test_t', 0.0):.2f}` (p = `{simpsons.get('pustejovsky_test_p', 0.0):.3e}`){pust_flag}")
                        if simpsons.get("moses_stat_p") is not None:
                            moses_flag = " ⚠️ Sub-Sample Scale Dispersion (p < 0.05)" if simpsons.get("is_moses_heterogeneous") else " (Homogeneous Sub-Sample Spread)"
                            st.markdown(f"**Moses Sub-Sample Scale:** U = `{simpsons.get('moses_stat_u', 0.0):.1f}` (p = `{simpsons.get('moses_stat_p', 0.0):.3e}`){moses_flag}")
                        if simpsons.get("schwarzer_test_p") is not None:
                            schw_flag = " ⚠️ Arcsine Funnel Asymmetry (p < 0.05)" if simpsons.get("is_schwarzer_asymmetric") else " (Symmetric Arcsine Precision)"
                            st.markdown(f"**Schwarzer Arcsine Funnel Asymmetry:** t = `{simpsons.get('schwarzer_test_t', 0.0):.2f}` (p = `{simpsons.get('schwarzer_test_p', 0.0):.3e}`){schw_flag}")
                        if simpsons.get("westenberg_stat_p") is not None:
                            west_flag = " ⚠️ IQR Scale Dispersion (p < 0.05)" if simpsons.get("is_westenberg_heterogeneous") else " (Homogeneous IQR Spread)"
                            st.markdown(f"**Westenberg IQR Scale:** Z = `{simpsons.get('westenberg_stat_z', 0.0):.2f}` (p = `{simpsons.get('westenberg_stat_p', 0.0):.3e}`){west_flag}")
                        if simpsons.get("tes_p_value") is not None:
                            tes_flag = " ⚠️ Excess Significance Detected (p < 0.05)" if simpsons.get("is_tes_excessive") else " (Expected Statistical Discovery Power)"
                            st.markdown(f"**Ioannidis-Trikalinos TES Excess Test:** A = `{simpsons.get('tes_excess_ratio', 0.0):.2f}` (χ² = `{simpsons.get('tes_chi2', 0.0):.2f}`, p = `{simpsons.get('tes_p_value', 0.0):.3e}`){tes_flag}")
                        if simpsons.get("obrien_stat_f") is not None:
                            ob_flag = " ⚠️ Scale Dispersion Heterogeneity (p < 0.05)" if simpsons.get("is_obrien_heterogeneous") else " (Homogeneous Stratum Variances)"
                            st.markdown(f"**O'Brien Homogeneity of Variance:** F = `{simpsons.get('obrien_stat_f', 0.0):.2f}` (p = `{simpsons.get('obrien_p_value', 0.0):.3e}`){ob_flag}")
                        if simpsons.get("copas_test_p") is not None:
                            copas_flag = " ⚠️ Selection Bias / Funnel Asymmetry (p < 0.05)" if simpsons.get("is_copas_asymmetric") else " (Unbiased Stratum Precision)"
                            st.markdown(f"**Copas Selection Asymmetry:** t = `{simpsons.get('copas_test_t', 0.0):.2f}` (p = `{simpsons.get('copas_test_p', 0.0):.3e}`){copas_flag}")
                        if simpsons.get("cochran_c_p") is not None:
                            cochran_flag = " ⚠️ Disproportionate Variance Outlier (p < 0.05)" if simpsons.get("is_cochran_c_heterogeneous") else " (Homogeneous Stratum Variances)"
                            st.markdown(f"**Cochran's C Variance Test:** C = `{simpsons.get('cochran_c_stat', 0.0):.3f}` (p = `{simpsons.get('cochran_c_p', 0.0):.3e}`){cochran_flag}")
                        if simpsons.get("gail_simon_p") is not None:
                            gs_flag = " ⚠️ Qualitative Reversal Proven (p < 0.05)" if simpsons.get("is_gail_simon_reversal") else " (Quantitative Interaction / Uniform Sign)"
                            st.markdown(f"**Gail-Simon Qualitative Interaction Test:** s_GS = `{simpsons.get('gail_simon_stat', 0.0):.2f}` (p = `{simpsons.get('gail_simon_p', 0.0):.3e}`){gs_flag}")
                        if simpsons.get("hartley_fmax_p") is not None:
                            hartley_flag = " ⚠️ Stratum Variance Heterogeneity (p < 0.05)" if simpsons.get("is_hartley_heterogeneous") else " (Homogeneous Stratum Variances)"
                            st.markdown(f"**Hartley F_max Variance Ratio:** F_max = `{simpsons.get('hartley_fmax_stat', 0.0):.2f}` (p = `{simpsons.get('hartley_fmax_p', 0.0):.3e}`){hartley_flag}")
                        if simpsons.get("piantadosi_byar_green_p") is not None:
                            pbg_flag = " ⚠️ Qualitative Reversal Proven (p < 0.05)" if simpsons.get("is_pbg_reversal") else " (Uniform Direction / Monotone)"
                            st.markdown(f"**Piantadosi-Byar-Green Likelihood-Ratio Interaction:** s_PBG = `{simpsons.get('piantadosi_byar_green_stat', 0.0):.2f}` (p = `{simpsons.get('piantadosi_byar_green_p', 0.0):.3e}`){pbg_flag}")
                        if simpsons.get("layard_p_value") is not None:
                            layard_flag = " ⚠️ Kurtosis-Robust Stratum Variance Heterogeneity (p < 0.05)" if simpsons.get("is_layard_heterogeneous") else " (Homogeneous Stratum Variances)"
                            st.markdown(f"**Layard Kurtosis-Robust Variance Test:** L = `{simpsons.get('layard_stat_l', 0.0):.2f}` (p = `{simpsons.get('layard_p_value', 0.0):.3e}`){layard_flag}")
                        if simpsons.get("pan_score_p") is not None:
                            pan_flag = " ⚠️ Qualitative Reversal Proven (p < 0.05)" if simpsons.get("is_pan_reversal") else " (Uniform Direction / Monotone)"
                            st.markdown(f"**Pan Directional Qualitative Score Test:** S_Pan = `{simpsons.get('pan_score_stat', 0.0):.2f}` (p = `{simpsons.get('pan_score_p', 0.0):.3e}`){pan_flag}")
                        if simpsons.get("samiuddin_p_value") is not None:
                            sam_flag = " ⚠️ Kurtosis-Adjusted Stratum Variance Heterogeneity (p < 0.05)" if simpsons.get("is_samiuddin_heterogeneous") else " (Homogeneous Stratum Variances)"
                            st.markdown(f"**Samiuddin-Atiqullah Scale Homogeneity:** F_Sam = `{simpsons.get('samiuddin_stat_f', 0.0):.2f}` (p = `{simpsons.get('samiuddin_p_value', 0.0):.3e}`){sam_flag}")
                        if simpsons.get("silvapulle_sen_p") is not None:
                            silva_flag = " ⚠️ Qualitative Reversal Proven (p < 0.05)" if simpsons.get("is_silvapulle_reversal") else " (Uniform Direction / Concordant)"
                            st.markdown(f"**Silvapulle-Sen Likelihood-Ratio Inversion:** S_SS = `{simpsons.get('silvapulle_sen_stat', 0.0):.2f}` (p = `{simpsons.get('silvapulle_sen_p', 0.0):.3e}`){silva_flag}")
                        if simpsons.get("miller_jackknife_p_value") is not None:
                            miller_flag = " ⚠️ Distribution-Free Stratum Variance Heterogeneity (p < 0.05)" if simpsons.get("is_miller_heterogeneous") else " (Homogeneous Stratum Variances)"
                            st.markdown(f"**Miller Jackknife Variance Homogeneity:** F_Miller = `{simpsons.get('miller_jackknife_stat_f', 0.0):.2f}` (p = `{simpsons.get('miller_jackknife_p_value', 0.0):.3e}`){miller_flag}")

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
                        if conf_aud.get("bkw_condition_index") is not None:
                            bkw_flag = " ⚠️ Severe Collinearity (κ ≥ 15)" if conf_aud.get("is_bkw_collinear") else " (Stable Conditioning)"
                            st.markdown(f"**Belsley-Kuh-Welsch Condition Index:** κ = `{conf_aud.get('bkw_condition_index', 0.0):.2f}` (Max Var Prop = `{conf_aud.get('bkw_max_variance_prop', 0.0):.1%}`){bkw_flag}")
                        if conf_aud.get("sargan_p_value") is not None:
                            sarg_flag = " ⚠️ Restrictions Violated (p < 0.05)" if conf_aud.get("is_sargan_rejected") else " (Valid Overidentifying Restrictions)"
                            st.markdown(f"**Sargan-Hansen Overidentification:** J = `{conf_aud.get('sargan_j_stat', 0.0):.2f}` (p = `{conf_aud.get('sargan_p_value', 0.0):.3e}`){sarg_flag}")
                        if conf_aud.get("crump_overlap_ratio") is not None:
                            crump_flag = f" ⚠️ Severe Lack of Common Support ({conf_aud['crump_overlap_ratio']:.1%} Off-Support)" if conf_aud.get("is_crump_overlap_deficient") else f" (Adequate Overlap, {conf_aud['crump_overlap_ratio']:.1%} Off-Support)"
                            st.markdown(f"**Crump Propensity Score Overlap:** `{conf_aud['crump_overlap_ratio']:.1%}`{crump_flag}")
                        if conf_aud.get("hansen_threshold_p") is not None:
                            hansen_flag = f" ⚠️ Threshold Regime Shift (p = {conf_aud['hansen_threshold_p']:.3e})" if conf_aud.get("has_hansen_threshold_regime") else f" (Linear Response Adequate, p = {conf_aud['hansen_threshold_p']:.3e})"
                            st.markdown(f"**Hansen Structural Threshold Test:** Sup F = `{conf_aud.get('hansen_threshold_f', 0.0):.2f}` (p = `{conf_aud['hansen_threshold_p']:.3e}`){hansen_flag}")
                        if conf_aud.get("hosmer_lemeshow_p") is not None:
                            hl_flag = f" ⚠️ Propensity Model Miscalibrated (p = {conf_aud['hosmer_lemeshow_p']:.3e})" if conf_aud.get("is_propensity_miscalibrated") else f" (Well-Calibrated Propensity, p = {conf_aud['hosmer_lemeshow_p']:.3e})"
                            st.markdown(f"**Hosmer-Lemeshow Calibration:** χ² = `{conf_aud.get('hosmer_lemeshow_chi2', 0.0):.2f}` (p = `{conf_aud['hosmer_lemeshow_p']:.3e}`){hl_flag}")
                        if conf_aud.get("hac_se_ratio") is not None:
                            hac_flag = f" ⚠️ Autocorrelation Inflation (ρ = {conf_aud['hac_se_ratio']:.2f} > 1.25)" if conf_aud.get("is_hac_inflated") else f" (Robust to Serial Correlation, ρ = {conf_aud['hac_se_ratio']:.2f})"
                            st.markdown(f"**Newey-West HAC Variance Ratio:** ρ_HAC = `{conf_aud['hac_se_ratio']:.2f}` (HAC SE = `{conf_aud.get('newey_west_hac_se', 0.0):.4f}`){hac_flag}")
                        if conf_aud.get("aipw_discrepancy") is not None:
                            aipw_flag = f" ⚠️ Selection Confounded (Δ = {conf_aud['aipw_discrepancy']:.1%})" if conf_aud.get("is_aipw_confounded") else f" (Robust Doubly Identified, Δ = {conf_aud['aipw_discrepancy']:.1%})"
                            st.markdown(f"**AIPW Doubly Robust Treatment Effect:** τ_AIPW = `{conf_aud.get('aipw_treatment_effect', 0.0):.4f}` (Discrepancy Δ_AIPW = `{conf_aud['aipw_discrepancy']:.1%}`){aipw_flag}")
                        if conf_aud.get("leamer_bound_min") is not None:
                            leamer_flag = " ⚠️ Fragile (0 ∈ EBA)" if conf_aud.get("is_leamer_fragile") else " (Robust to Covariate Selection)"
                            st.markdown(f"**Leamer Extreme Bounds Analysis (EBA):** β ∈ `[{conf_aud.get('leamer_bound_min', 0.0):.4f}, {conf_aud.get('leamer_bound_max', 0.0):.4f}]`{leamer_flag}")
                        if conf_aud.get("dml_discrepancy") is not None:
                            dml_flag = f" ⚠️ Confounded Causal Bias (Δ = {conf_aud['dml_discrepancy']:.1%})" if conf_aud.get("is_dml_confounded") else f" (Robust DML Identified, Δ = {conf_aud['dml_discrepancy']:.1%})"
                            st.markdown(f"**Double Machine Learning (DML) Effect:** β_DML = `{conf_aud.get('dml_treatment_effect', 0.0):.4f}` (Discrepancy Δ_DML = `{conf_aud['dml_discrepancy']:.1%}`){dml_flag}")
                        if conf_aud.get("cinelli_k_multiplier") is not None:
                            cin_flag = " ⚠️ Fragile (k < 1.0x)" if conf_aud.get("is_cinelli_fragile") else " (Robust, k >= 1.0x)"
                            st.markdown(f"**Cinelli Extreme Confounder Multiplier:** k = `{conf_aud['cinelli_k_multiplier']:.2f}x`{cin_flag}")
                        if conf_aud.get("rubin_b_score") is not None:
                            rub_flag = " ⚠️ Severe Covariate Imbalance (B ≥ 25% or R ∉ [0.5, 2.0])" if conf_aud.get("is_rubin_imbalanced") else " (Sufficient Balance)"
                            st.markdown(f"**Rubin's Covariate Balance:** B = `{conf_aud['rubin_b_score']:.1f}%`, R = `{conf_aud.get('rubin_r_ratio', 0.0):.2f}`{rub_flag}")
                        if conf_aud.get("brunner_munzel_p") is not None:
                            bm_flag = " (Significant Relative Effect)" if conf_aud.get("has_brunner_munzel_effect") else " ⚠️ No Stochastic Effect (p ≥ 0.05)"
                            st.markdown(f"**Brunner-Munzel Relative Effect:** W = `{conf_aud.get('brunner_munzel_w', 0.0):.2f}` (p = `{conf_aud.get('brunner_munzel_p', 0.0):.3e}`){bm_flag}")
                        if conf_aud.get("smd_covariate_balance") is not None:
                            smd_flag = " ⚠️ Severe Covariate Imbalance (SMD > 0.10 or D² > 0.25)" if conf_aud.get("is_smd_imbalanced") else " (Balanced Covariates)"
                            st.markdown(f"**Rosenbaum-Rubin SMD & Mahalanobis:** SMD = `{conf_aud['smd_covariate_balance']:.3f}`, D² = `{conf_aud.get('mahalanobis_distance', 0.0):.3f}`{smd_flag}")
                        if conf_aud.get("imbens_normalized_diff") is not None:
                            imb_flag = " ⚠️ Positivity Deficit (Δ_Norm > 0.25)" if conf_aud.get("is_imbens_diff_excessive") else " (Adequate Support)"
                            st.markdown(f"**Imbens-Rubin Normalized Diff:** Δ_Norm = `{conf_aud['imbens_normalized_diff']:.3f}`{imb_flag}")
                        if conf_aud.get("austin_variance_ratio") is not None:
                            aust_flag = " ⚠️ Severe Covariate Variance Imbalance (VR < 0.5 or > 2.0)" if conf_aud.get("is_austin_imbalanced") else " (Balanced Covariate Spread)"
                            st.markdown(f"**Austin Covariate Variance Ratio:** VR = `{conf_aud.get('austin_variance_ratio', 0.0):.2f}`{aust_flag}")
                        if conf_aud.get("stuart_prognostic_discrepancy") is not None:
                            stuart_flag = " ⚠️ Severe Baseline Prognostic Disparity (Δ_prog > 0.25)" if conf_aud.get("is_prognostic_discrepant") else " (Similar Prognostic Profiles)"
                            st.markdown(f"**Stuart Prognostic Score Discrepancy:** Δ_prog = `{conf_aud.get('stuart_prognostic_discrepancy', 0.0):.3f}`{stuart_flag}")
                        if conf_aud.get("hansen_bowers_p") is not None:
                            hb_flag = " ⚠️ Multivariate Imbalance (p < 0.05)" if conf_aud.get("is_hansen_bowers_imbalanced") else " (Omnibus Balance Confirmed)"
                            st.markdown(f"**Hansen-Bowers Omnibus Balance:** χ² = `{conf_aud.get('hansen_bowers_chi2', 0.0):.2f}` (p = `{conf_aud.get('hansen_bowers_p', 0.0):.3e}`){hb_flag}")
                        if conf_aud.get("caliper_discordance_rate") is not None:
                            cal_flag = " ⚠️ Severe Caliper Discordance (> 15% off-support)" if conf_aud.get("is_caliper_discordant") else " (Adequate Support Overlap)"
                            st.markdown(f"**Rosenbaum Caliper Discordance:** `{conf_aud.get('caliper_discordance_rate', 0.0):.1%}` (ε = `{conf_aud.get('caliper_threshold', 0.0):.3f}`){cal_flag}")
                        if conf_aud.get("gps_balance_p") is not None:
                            gps_flag = " ⚠️ Generalized Propensity Imbalance (p < 0.05)" if conf_aud.get("is_gps_imbalanced") else " (Generalized Propensity Balance Confirmed)"
                            st.markdown(f"**Imai-van Dyk GPS Covariate Balance:** r_GPS = `{conf_aud.get('gps_balance_r', 0.0):.3f}` (t = `{conf_aud.get('gps_balance_t', 0.0):.2f}`, p = `{conf_aud.get('gps_balance_p', 0.0):.3e}`){gps_flag}")
                        if conf_aud.get("austin_ks_p_value") is not None:
                            ks_flag = " ⚠️ Distributional Imbalance (D_KS > 0.10, p < 0.05)" if conf_aud.get("is_austin_ks_imbalanced") else " (Satisfactory Distributional Overlap)"
                            st.markdown(f"**Austin Kolmogorov-Smirnov Balance:** D_KS = `{conf_aud.get('austin_ks_stat', 0.0):.3f}` (p = `{conf_aud.get('austin_ks_p_value', 0.0):.3e}`){ks_flag}")
                        if conf_aud.get("l1_multivariate_imbalance") is not None:
                            l1_flag = " ⚠️ Severe Multivariate Imbalance (L1 > 0.25)" if conf_aud.get("is_l1_imbalanced") else " (Sufficient Common Support)"
                            st.markdown(f"**Iacus-King-Porro L1 Imbalance:** L1 = `{conf_aud.get('l1_multivariate_imbalance', 0.0):.3f}`{l1_flag}")
                        if conf_aud.get("rubin_parallelism_p") is not None:
                            rp_flag = " ⚠️ Non-Parallel Surfaces / Effect Heterogeneity (p < 0.05)" if conf_aud.get("is_rubin_nonparallel") else " (Parallel Covariate Slopes)"
                            st.markdown(f"**Rubin Surface Non-Parallelism:** F = `{conf_aud.get('rubin_parallelism_f', 0.0):.2f}` (p = `{conf_aud.get('rubin_parallelism_p', 0.0):.3e}`){rp_flag}")
                        if conf_aud.get("eqq_mean_diff") is not None:
                            eqq_flag = " ⚠️ Severe Quantile Imbalance (eQQ > 0.15)" if conf_aud.get("is_eqq_imbalanced") else " (Satisfactory Empirical Q-Q Overlap)"
                            st.markdown(f"**Ho-Imai-King-Stuart eQQ Balance:** Mean Diff = `{conf_aud.get('eqq_mean_diff', 0.0):.3f}` (Max = `{conf_aud.get('eqq_max_diff', 0.0):.3f}`){eqq_flag}")
                        if conf_aud.get("lin_interaction_p") is not None:
                            lin_flag = " ⚠️ Agnostic Effect Heterogeneity (p < 0.05)" if conf_aud.get("is_lin_heterogeneous") else " (Homogeneous Covariate Slopes)"
                            st.markdown(f"**Winston Lin Centered Interaction:** t = `{conf_aud.get('lin_interaction_t', 0.0):.2f}` (p = `{conf_aud.get('lin_interaction_p', 0.0):.3e}`){lin_flag}")
                        if conf_aud.get("rr_variance_ratio") is not None:
                            rr_flag = " ⚠️ Residual Variance Imbalance (VR outside [0.8, 1.25])" if conf_aud.get("is_rr_variance_imbalanced") else " (Balanced Second-Order Residual Variance)"
                            st.markdown(f"**Rosenbaum-Rubin Residual Variance:** VR = `{conf_aud.get('rr_variance_ratio', 0.0):.2f}` (p = `{conf_aud.get('rr_variance_p', 0.0):.3e}`){rr_flag}")
                        if conf_aud.get("iw_quadratic_diff") is not None:
                            iw_flag = " ⚠️ Severe Quadratic Imbalance (Δ_Z² > 0.25)" if conf_aud.get("is_iw_quadratic_excessive") else " (Adequate Quadratic Support Overlap)"
                            st.markdown(f"**Imbens-Wooldridge Quadratic Balance:** Δ_Z² = `{conf_aud.get('iw_quadratic_diff', 0.0):.3f}`{iw_flag}")
                        if conf_aud.get("hodges_lehmann_p") is not None:
                            hl_flag = " ⚠️ Covariate Location Shift (p < 0.05)" if conf_aud.get("is_hl_imbalanced") else " (Balanced Non-Parametric Location)"
                            st.markdown(f"**Rosenbaum Hodges-Lehmann Shift:** Δ_HL = `{conf_aud.get('hodges_lehmann_shift', 0.0):.3f}` (Z = `{conf_aud.get('hodges_lehmann_z', 0.0):.2f}`, p = `{conf_aud.get('hodges_lehmann_p', 0.0):.3e}`){hl_flag}")
                        if conf_aud.get("frank_rir_pct") is not None:
                            rir_flag = f" ⚠️ Extreme Causal Fragility (< 15%, {conf_aud.get('frank_rir_count', 0)} cases)" if conf_aud.get("is_rir_fragile") else f" (Robust Replacement Buffer, {conf_aud.get('frank_rir_count', 0)} cases)"
                            st.markdown(f"**Frank Robustness of Inference to Replacement (RIR):** RIR = `{conf_aud.get('frank_rir_pct', 0.0):.1f}%`{rir_flag}")
                        if conf_aud.get("imbens_contour_q") is not None:
                            imb_flag = " ⚠️ Sensitive to Weak Unmeasured Confounder (q < 0.05)" if conf_aud.get("is_imbens_sensitive") else " (Robust to Moderate Confounding)"
                            st.markdown(f"**Imbens Sensitivity Contour:** q = `{conf_aud.get('imbens_contour_q', 0.0):.3f}` (p = `{conf_aud.get('imbens_contour_p', 0.0):.3e}`){imb_flag}")
                        if conf_aud.get("austin_stuart_masd") is not None:
                            masd_flag = " ⚠️ Residual IPTW Imbalance (MASD > 0.10)" if conf_aud.get("is_austin_stuart_imbalanced") else " (Balanced under IPTW)"
                            st.markdown(f"**Austin-Stuart IPTW Balance:** MASD = `{conf_aud.get('austin_stuart_masd', 0.0):.3f}` (p = `{conf_aud.get('austin_stuart_p', 0.0):.3e}`){masd_flag}")
                        if conf_aud.get("kallus_gamma_crit") is not None:
                            kallus_flag = " ⚠️ Sensitive to Bounded Confounding (Γ_crit < 1.30)" if conf_aud.get("is_kallus_sensitive") else " (Minimax Robust, Γ_crit ≥ 1.30)"
                            st.markdown(f"**Kallus Minimax Confounding Bounds:** Γ_crit = `{conf_aud.get('kallus_gamma_crit', 0.0):.2f}` (Bias_max = `{conf_aud.get('kallus_bias_max', 0.0):.3f}`, β ∈ `[{conf_aud.get('kallus_bound_low', 0.0):.3f}, {conf_aud.get('kallus_bound_high', 0.0):.3f}]`){kallus_flag}")
                        if conf_aud.get("cbps_overid_p") is not None:
                            cbps_flag = " ⚠️ Moment Balance Rejected (p < 0.05)" if conf_aud.get("is_cbps_rejected") else " (Exact Moment Balance Verified)"
                            st.markdown(f"**Imai-Ratkovic CBPS Balance Test:** J = `{conf_aud.get('cbps_overid_j', 0.0):.2f}` (p = `{conf_aud.get('cbps_overid_p', 0.0):.3e}`){cbps_flag}")
                        if conf_aud.get("rosenbaum_silber_delta_crit") is not None:
                            silber_flag = " ⚠️ Fragile to Hidden Bias Amplification (Δ_crit < 2.0)" if conf_aud.get("is_silber_sensitive") else " (Robust to Bias Amplification, Δ_crit ≥ 2.0)"
                            st.markdown(f"**Rosenbaum-Silber Hidden Bias Amplification:** Δ_crit = `{conf_aud.get('rosenbaum_silber_delta_crit', 0.0):.2f}` (at Λ = `{conf_aud.get('rosenbaum_silber_lambda', 2.0):.1f}`){silber_flag}")
                        if conf_aud.get("fogarty_gamma_crit") is not None:
                            fogarty_flag = " ⚠️ Sensitive to Studentized Confounding (Γ_crit < 1.30)" if conf_aud.get("is_fogarty_sensitive") else " (Studentized Robust, Γ_crit ≥ 1.30)"
                            st.markdown(f"**Fogarty Studentized Sensitivity Bounds:** Γ_crit = `{conf_aud.get('fogarty_gamma_crit', 0.0):.2f}` (t_sens = `{conf_aud.get('fogarty_studentized_t', 0.0):.2f}`){fogarty_flag}")
                        if conf_aud.get("rosenbaum_attr_gamma_crit") is not None:
                            attr_flag = " ⚠️ Fragile to Attributable Hidden Bias (Γ_attr < 1.25)" if conf_aud.get("is_rosenbaum_attr_sensitive") else " (Robust Attributable Effect, Γ_attr ≥ 1.25)"
                            st.markdown(f"**Rosenbaum Attributable Sensitivity:** Γ_attr = `{conf_aud.get('rosenbaum_attr_gamma_crit', 0.0):.2f}` (p_upper = `{conf_aud.get('rosenbaum_attr_p_upper', 0.0):.3e}`){attr_flag}")
                        if conf_aud.get("zubizarreta_sbw_cv") is not None:
                            sbw_flag = " ⚠️ Severe Balancing Weight Dispersion (CV > 1.50)" if conf_aud.get("is_zubizarreta_imbalanced") else " (Balanced Covariate Weight Stability)"
                            st.markdown(f"**Zubizarreta Stable Balancing Weights:** CV_w = `{conf_aud.get('zubizarreta_sbw_cv', 0.0):.2f}` (Eff Sample = `{conf_aud.get('zubizarreta_eff_sample_pct', 0.0):.1f}%`){sbw_flag}")
                        if conf_aud.get("rosenbaum_design_sensitivity_gamma") is not None:
                            des_flag = " ⚠️ Fragile Asymptotic Design (Γ_tilde < 1.30)" if conf_aud.get("is_design_fragile") else " (Asymptotically Robust Design, Γ_tilde ≥ 1.30)"
                            st.markdown(f"**Rosenbaum Design Sensitivity Bounds:** Γ_tilde = `{conf_aud.get('rosenbaum_design_sensitivity_gamma', 0.0):.2f}` (μ/σ = `{conf_aud.get('rosenbaum_standardized_effect_mu_sigma', 0.0):.2f}`){des_flag}")
                        if conf_aud.get("hosman_hansen_holland_delta") is not None:
                            hhh_flag = " ⚠️ Sensitive to Modest Partial Confounding (δ < 0.15)" if conf_aud.get("is_hosman_sensitive") else " (Robust to Partial Confounding, δ ≥ 0.15)"
                            st.markdown(f"**Hosman-Hansen-Holland Sensitivity Relation:** δ_HHH = `{conf_aud.get('hosman_hansen_holland_delta', 0.0):.3f}` (scale = `{conf_aud.get('hosman_hansen_holland_scale', 0.0):.3f}`){hhh_flag}")
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
                        if stab_aud.get("harvey_collier_p") is not None:
                            hc_flag = " ⚠️ Functional Curvature (p < 0.05)" if stab_aud.get("has_harvey_collier_nonlinearity") else " (Linear Specification Sound)"
                            st.markdown(f"**Harvey-Collier Linearity Test:** t = `{stab_aud.get('harvey_collier_t', 0.0):.2f}` (p = `{stab_aud.get('harvey_collier_p', 0.0):.3e}`){hc_flag}")
                        if stab_aud.get("huber_slope_discrepancy") is not None:
                            hub_flag = " ⚠️ High Outlier Contamination" if stab_aud.get("is_huber_outlier_distorted") else " (Robust to Outliers)"
                            st.markdown(f"**Huber Robust M-Slope Discrepancy:** β_Huber = `{stab_aud.get('huber_m_slope', 0.0):.4f}` (Δ = `{stab_aud.get('huber_slope_discrepancy', 0.0):.1%}`){hub_flag}")
                        if stab_aud.get("cusumsq_stat") is not None:
                            cusumsq_flag = f" ⚠️ Variance Instability (W > {stab_aud.get('cusumsq_critical_val', 0.0):.3f})" if stab_aud.get("has_cusumsq_instability") else f" (Variance Constancy, W ≤ {stab_aud.get('cusumsq_critical_val', 0.0):.3f})"
                            st.markdown(f"**CUSUM of Squares Test:** W = `{stab_aud.get('cusumsq_stat', 0.0):.3f}` (crit = `{stab_aud.get('cusumsq_critical_val', 0.0):.3f}`){cusumsq_flag}")
                        if stab_aud.get("lts_slope_discrepancy") is not None:
                            lts_flag = " ⚠️ High-Breakdown Leverage Distortion" if stab_aud.get("is_lts_outlier_distorted") else " (Resistant to Outliers)"
                            st.markdown(f"**Least Trimmed Squares (LTS) Slope:** β_LTS = `{stab_aud.get('lts_slope', 0.0):.4f}` (Δ_LTS = `{stab_aud.get('lts_slope_discrepancy', 0.0):.1%}`){lts_flag}")
                        if stab_aud.get("siegel_slope_discrepancy") is not None:
                            sieg_flag = " ⚠️ 50% Breakdown Outlier Contamination" if stab_aud.get("is_siegel_outlier_distorted") else " (Robust to Gross Outliers)"
                            st.markdown(f"**Siegel Repeated Medians Robust Slope:** β_Siegel = `{stab_aud.get('siegel_slope', 0.0):.4f}` (Δ_Siegel = `{stab_aud.get('siegel_slope_discrepancy', 0.0):.1%}`){sieg_flag}")
                        if stab_aud.get("breusch_godfrey_p") is not None:
                            bg_flag = f" ⚠️ Higher-Order Autocorrelation (p = {stab_aud['breusch_godfrey_p']:.3e})" if stab_aud.get("has_higher_order_autocorrelation") else f" (Independent Residuals, p = {stab_aud['breusch_godfrey_p']:.3e})"
                            st.markdown(f"**Breusch-Godfrey Serial Correlation Test:** LM = `{stab_aud.get('breusch_godfrey_lm', 0.0):.2f}` (p = `{stab_aud['breusch_godfrey_p']:.3e}`){bg_flag}")
                        if stab_aud.get("runs_test_p") is not None:
                            runs_flag = f" ⚠️ Non-Random Residual Drift (p = {stab_aud['runs_test_p']:.3e})" if stab_aud.get("has_runs_nonrandomness") else f" (Independent Residuals, p = {stab_aud['runs_test_p']:.3e})"
                            st.markdown(f"**Wald-Wolfowitz Runs Test:** Z = `{stab_aud.get('runs_test_z', 0.0):.2f}` (p = `{stab_aud['runs_test_p']:.3e}`){runs_flag}")
                        if stab_aud.get("kendall_discrepancy") is not None:
                            kend_flag = " ⚠️ Leverage Distortion / Rank Discrepancy" if stab_aud.get("is_kendall_distorted") else " (Robust Concordance)"
                            st.markdown(f"**Kendall Tau-b Rank Concordance:** τ_b = `{stab_aud.get('kendall_tau_b', 0.0):.4f}` (Δ_Kendall = `{stab_aud.get('kendall_discrepancy', 0.0):.1%}`){kend_flag}")
                        if stab_aud.get("goldfeld_quandt_p") is not None:
                            gq_flag = f" ⚠️ Regime Heteroskedasticity (p = {stab_aud['goldfeld_quandt_p']:.3e})" if stab_aud.get("has_goldfeld_quandt_heteroskedasticity") else f" (Homoscedastic Regimes, p = {stab_aud['goldfeld_quandt_p']:.3e})"
                            st.markdown(f"**Goldfeld-Quandt Heteroskedasticity:** F = `{stab_aud.get('goldfeld_quandt_f', 0.0):.2f}` (p = `{stab_aud['goldfeld_quandt_p']:.3e}`){gq_flag}")
                        if stab_aud.get("shapiro_francia_p") is not None:
                            sf_flag = f" ⚠️ Non-Gaussian Residuals (p = {stab_aud['shapiro_francia_p']:.3e})" if stab_aud.get("is_shapiro_francia_non_normal") else f" (Gaussian Residuals, p = {stab_aud['shapiro_francia_p']:.3e})"
                            st.markdown(f"**Shapiro-Francia Residual Normality:** W' = `{stab_aud.get('shapiro_francia_w', 0.0):.4f}` (p = `{stab_aud['shapiro_francia_p']:.3e}`){sf_flag}")
                        if stab_aud.get("dagostino_pearson_p") is not None:
                            dp_flag = f" ⚠️ Non-Gaussian Skew/Kurtosis (p = {stab_aud['dagostino_pearson_p']:.3e})" if stab_aud.get("is_dagostino_non_normal") else f" (Gaussian Skew/Kurtosis, p = {stab_aud['dagostino_pearson_p']:.3e})"
                            st.markdown(f"**D'Agostino-Pearson Omnibus Normality:** K² = `{stab_aud.get('dagostino_pearson_k2', 0.0):.2f}` (p = `{stab_aud['dagostino_pearson_p']:.3e}`){dp_flag}")
                        if stab_aud.get("glejser_p_value") is not None:
                            gl_flag = f" ⚠️ Scale Dispersion Scales with X (p = {stab_aud['glejser_p_value']:.3e})" if stab_aud.get("has_glejser_heteroskedasticity") else f" (Homoscedastic Error Spread, p = {stab_aud['glejser_p_value']:.3e})"
                            st.markdown(f"**Glejser Heteroskedasticity Test:** t = `{stab_aud.get('glejser_t_stat', 0.0):.2f}` (p = `{stab_aud['glejser_p_value']:.3e}`){gl_flag}")
                        if stab_aud.get("anderson_darling_stat") is not None:
                            ad_flag = f" ⚠️ Heavy Tail Non-Gaussianity (A² > {stab_aud.get('anderson_darling_crit', 0.75):.3f})" if stab_aud.get("is_anderson_non_normal") else f" (Normal Empirical CDF, A² = {stab_aud['anderson_darling_stat']:.3f})"
                            st.markdown(f"**Anderson-Darling EDF Normality:** A² = `{stab_aud['anderson_darling_stat']:.3f}` (5% crit = `{stab_aud.get('anderson_darling_crit', 0.75):.3f}`){ad_flag}")
                        if stab_aud.get("harvey_lm_p") is not None:
                            harv_flag = f" ⚠️ Multiplicative Heteroskedasticity (p = {stab_aud['harvey_lm_p']:.3e})" if stab_aud.get("has_harvey_heteroskedasticity") else f" (Homoscedastic Log Variance, p = {stab_aud['harvey_lm_p']:.3e})"
                            st.markdown(f"**Harvey Multiplicative Heteroskedasticity:** LM = `{stab_aud.get('harvey_lm_stat', 0.0):.2f}` (p = `{stab_aud['harvey_lm_p']:.3e}`){harv_flag}")
                        if stab_aud.get("lilliefors_p_value") is not None:
                            lill_flag = f" ⚠️ Composite Non-Gaussian Residuals (p = {stab_aud['lilliefors_p_value']:.3e})" if stab_aud.get("is_lilliefors_non_normal") else f" (Normal Empirical Residuals, p = {stab_aud['lilliefors_p_value']:.3e})"
                            st.markdown(f"**Lilliefors Composite Normality:** D = `{stab_aud.get('lilliefors_stat', 0.0):.3f}` (p = `{stab_aud['lilliefors_p_value']:.3e}`){lill_flag}")
                        if stab_aud.get("park_p_value") is not None:
                            park_flag = f" ⚠️ Log-Linear Heteroskedasticity (p = {stab_aud['park_p_value']:.3e})" if stab_aud.get("has_park_heteroskedasticity") else f" (Homoscedastic Log Error Variance, p = {stab_aud['park_p_value']:.3e})"
                            st.markdown(f"**Park Log-Linear Heteroskedasticity:** t = `{stab_aud.get('park_t_stat', 0.0):.2f}` (p = `{stab_aud['park_p_value']:.3e}`){park_flag}")
                        if stab_aud.get("cramer_von_mises_p") is not None:
                            cvm_flag = f" ⚠️ Non-Gaussian Residuals (p = {stab_aud['cramer_von_mises_p']:.3e})" if stab_aud.get("is_cvm_non_normal") else f" (Normal Empirical Residuals, p = {stab_aud['cramer_von_mises_p']:.3e})"
                            st.markdown(f"**Cramér-von Mises Normality:** W² = `{stab_aud.get('cramer_von_mises_w2', 0.0):.3f}` (p = `{stab_aud['cramer_von_mises_p']:.3e}`){cvm_flag}")
                        if stab_aud.get("cook_weisberg_p") is not None:
                            cw_flag = f" ⚠️ Fitted-Value Heteroskedasticity (p = {stab_aud['cook_weisberg_p']:.3e})" if stab_aud.get("has_cook_weisberg_heteroskedasticity") else f" (Homoscedastic Fitted Spread, p = {stab_aud['cook_weisberg_p']:.3e})"
                            st.markdown(f"**Cook-Weisberg Heteroskedasticity:** S = `{stab_aud.get('cook_weisberg_score', 0.0):.2f}` (p = `{stab_aud['cook_weisberg_p']:.3e}`){cw_flag}")
                        if stab_aud.get("kuiper_p_value") is not None:
                            kuiper_flag = f" ⚠️ Invariant Non-Gaussian Residuals (V* = {stab_aud.get('kuiper_v_star', 0.0):.3f})" if stab_aud.get("is_kuiper_non_normal") else f" (Invariant Gaussian Residuals, p = {stab_aud['kuiper_p_value']:.3e})"
                            st.markdown(f"**Kuiper Invariant EDF Normality:** V = `{stab_aud.get('kuiper_v_stat', 0.0):.3f}` (V* = `{stab_aud.get('kuiper_v_star', 0.0):.3f}`, p = `{stab_aud['kuiper_p_value']:.3e}`){kuiper_flag}")
                        if stab_aud.get("max_covratio_dev") is not None:
                            covr_flag = f" ⚠️ Precision Degradation ({stab_aud.get('covratio_outlier_count', 0)} influential)" if stab_aud.get("is_covratio_influential") else " (Stable Covariance Precision)"
                            st.markdown(f"**Belsley COVRATIO Covariance Influence:** Max |COVRATIO - 1| = `{stab_aud.get('max_covratio_dev', 0.0):.3f}`{covr_flag}")
                        if stab_aud.get("hosking_l_p") is not None:
                            hosk_flag = f" ⚠️ Non-Gaussian L-Moments (p = {stab_aud['hosking_l_p']:.3e})" if stab_aud.get("is_hosking_non_normal") else f" (Gaussian L-Moments Fit, p = {stab_aud['hosking_l_p']:.3e})"
                            st.markdown(f"**Hosking L-Moments Ratio Normality:** L = `{stab_aud.get('hosking_l_stat', 0.0):.2f}` (p = `{stab_aud['hosking_l_p']:.3e}`){hosk_flag}")
                        if stab_aud.get("max_atkinson_distance") is not None:
                            atk_flag = f" ⚠️ Leverage-Amplified Influence ({stab_aud.get('atkinson_outlier_count', 0)} influential)" if stab_aud.get("is_atkinson_influential") else " (Bounded Atkinson Influence)"
                            st.markdown(f"**Atkinson Modified Influence:** Max A = `{stab_aud.get('max_atkinson_distance', 0.0):.2f}`{atk_flag}")
                        if stab_aud.get("shapiro_wilk_p") is not None:
                            sw_flag = f" ⚠️ Non-Gaussian Residuals (p = {stab_aud['shapiro_wilk_p']:.3e})" if stab_aud.get("is_shapiro_wilk_non_normal") else f" (Normal Gaussian Fit, p = {stab_aud['shapiro_wilk_p']:.3e})"
                            st.markdown(f"**Shapiro-Wilk Omnibus Normality:** W = `{stab_aud.get('shapiro_wilk_w', 0.0):.4f}` (p = `{stab_aud['shapiro_wilk_p']:.3e}`){sw_flag}")
                        if stab_aud.get("max_hadi_influence") is not None:
                            hadi_flag = f" ⚠️ Potential Influence Outliers ({stab_aud.get('hadi_outlier_count', 0)} influential)" if stab_aud.get("is_hadi_influential") else " (Bounded Potential Influence)"
                            st.markdown(f"**Hadi Potential Influence:** Max H² = `{stab_aud.get('max_hadi_influence', 0.0):.2f}`{hadi_flag}")
                        if stab_aud.get("doornik_hansen_p") is not None:
                            dh_flag = f" ⚠️ Non-Gaussian Residuals (p = {stab_aud['doornik_hansen_p']:.3e})" if stab_aud.get("is_doornik_hansen_non_normal") else f" (Normal Gaussian Fit, p = {stab_aud['doornik_hansen_p']:.3e})"
                            st.markdown(f"**Doornik-Hansen Omnibus Normality:** E_DH = `{stab_aud.get('doornik_hansen_stat', 0.0):.2f}` (p = `{stab_aud['doornik_hansen_p']:.3e}`){dh_flag}")
                        if stab_aud.get("max_welsch_distance") is not None:
                            welsch_flag = f" ⚠️ Leverage-Weighted Distance Outliers ({stab_aud.get('welsch_outlier_count', 0)} influential)" if stab_aud.get("is_welsch_influential") else " (Bounded Welsch Distance)"
                            st.markdown(f"**Welsch Distance Influence:** Max W = `{stab_aud.get('max_welsch_distance', 0.0):.2f}`{welsch_flag}")
                        if stab_aud.get("gel_gastwirth_p") is not None:
                            rjb_flag = f" ⚠️ Non-Gaussian Residuals (p = {stab_aud['gel_gastwirth_p']:.3e})" if stab_aud.get("is_rjb_non_normal") else f" (Normal Gaussian Residuals, p = {stab_aud['gel_gastwirth_p']:.3e})"
                            st.markdown(f"**Gel-Gastwirth Robust Jarque-Bera (RJB):** RJB = `{stab_aud.get('gel_gastwirth_rjb', 0.0):.2f}` (p = `{stab_aud['gel_gastwirth_p']:.3e}`){rjb_flag}")
                        if stab_aud.get("max_pena_influence") is not None:
                            pena_flag = f" ⚠️ Projection Subspace Outliers ({stab_aud.get('pena_outlier_count', 0)} influential)" if stab_aud.get("is_pena_influential") else " (Bounded Peña Influence)"
                            st.markdown(f"**Daniel Peña Influence Diagnostic:** Max P = `{stab_aud.get('max_pena_influence', 0.0):.3f}`{pena_flag}")
                        if stab_aud.get("szroeter_p_value") is not None:
                            sz_flag = f" ⚠️ Monotonic Variance Expansion (p = {stab_aud['szroeter_p_value']:.3e})" if stab_aud.get("is_szroeter_heteroskedastic") else " (Rank Homoskedastic)"
                            st.markdown(f"**Szroeter Monotonic Homoskedasticity:** Q = `{stab_aud.get('szroeter_stat_q', 0.0):.3f}` (Z = `{stab_aud.get('szroeter_stat_z', 0.0):.2f}`, p = `{stab_aud['szroeter_p_value']:.3e}`){sz_flag}")
                        if stab_aud.get("max_ap_deviation") is not None:
                            ap_flag = f" ⚠️ Ellipsoid Volume Shrinkage ({stab_aud.get('ap_outlier_count', 0)} influential)" if stab_aud.get("is_ap_influential") else " (Bounded Ellipsoid Shrinkage)"
                            st.markdown(f"**Andrews-Pregibon Ellipsoid Influence:** Max Δ_AP = `{stab_aud.get('max_ap_deviation', 0.0):.3f}`{ap_flag}")
                        if stab_aud.get("anscombe_glynn_p") is not None:
                            ag_flag = f" ⚠️ Significant Residual Kurtosis (p = {stab_aud['anscombe_glynn_p']:.3e})" if stab_aud.get("is_anscombe_glynn_non_normal") else " (Gaussian Error Kurtosis)"
                            st.markdown(f"**Anscombe-Glynn Kurtosis Test:** Z_AG = `{stab_aud.get('anscombe_glynn_z', 0.0):.2f}` (p = `{stab_aud['anscombe_glynn_p']:.3e}`){ag_flag}")
                        if stab_aud.get("scale_inflation_ratio_qn") is not None:
                            qn_flag = f" ⚠️ Scale Inflation by Outliers (SIR = {stab_aud['scale_inflation_ratio_qn']:.2f}x > 1.30)" if stab_aud.get("is_qn_inflated") else " (Stable Robust Scale)"
                            st.markdown(f"**Rousseeuw-Croux Q_n Scale Estimator:** Q_n = `{stab_aud.get('rousseeuw_croux_qn', 0.0):.4f}` (SIR = `{stab_aud.get('scale_inflation_ratio_qn', 0.0):.2f}x`){qn_flag}")
                        if stab_aud.get("epps_pulley_p") is not None:
                            ep_flag = f" ⚠️ Significant Fourier Non-Normality (p = {stab_aud['epps_pulley_p']:.3e})" if stab_aud.get("is_epps_pulley_non_normal") else " (Gaussian Characteristic Function Validated)"
                            st.markdown(f"**Epps-Pulley Characteristic Function Test:** T_EP = `{stab_aud.get('epps_pulley_stat', 0.0):.3f}` (p = `{stab_aud['epps_pulley_p']:.3e}`){ep_flag}")
                        if stab_aud.get("max_chatterjee_hadi_influence") is not None:
                            ch_flag = f" ⚠️ Compounding Leverage-Residual Outlier ({stab_aud.get('chatterjee_hadi_outlier_count', 0)} influential)" if stab_aud.get("is_chatterjee_hadi_influential") else " (Bounded Joint Influence)"
                            st.markdown(f"**Chatterjee-Hadi Unified Influence Measure:** Max H^2 = `{stab_aud.get('max_chatterjee_hadi_influence', 0.0):.3f}`{ch_flag}")
                        if stab_aud.get("baringhaus_henze_p") is not None:
                            bh_flag = f" ⚠️ Significant Fourier Non-Normality (p = {stab_aud['baringhaus_henze_p']:.3e})" if stab_aud.get("is_baringhaus_henze_non_normal") else " (Gaussian Characteristic Function Validated)"
                            st.markdown(f"**Baringhaus-Henze-Zirkler Normality Test:** T_BHZ = `{stab_aud.get('baringhaus_henze_stat', 0.0):.3f}` (p = `{stab_aud['baringhaus_henze_p']:.3e}`){bh_flag}")
                        if stab_aud.get("max_gray_ling_influence") is not None:
                            gl_flag = f" ⚠️ Joint Leverage-Residual Outlier ({stab_aud.get('gray_ling_outlier_count', 0)} influential)" if stab_aud.get("is_gray_ling_influential") else " (Bounded Joint Influence)"
                            st.markdown(f"**Gray-Ling Leverage-Residual Influence:** Max LR = `{stab_aud.get('max_gray_ling_influence', 0.0):.3f}`{gl_flag}")
                        if stab_aud.get("vasicek_entropy_p") is not None:
                            vas_flag = f" ⚠️ Significant Entropy Departure (p = {stab_aud['vasicek_entropy_p']:.3e})" if stab_aud.get("is_vasicek_non_normal") else " (Gaussian Residual Entropy Validated)"
                            st.markdown(f"**Vasicek Sample Entropy Normality:** K_mn = `{stab_aud.get('vasicek_entropy_stat', 0.0):.3f}` (p = `{stab_aud['vasicek_entropy_p']:.3e}`){vas_flag}")
                        if stab_aud.get("rousseeuw_zomeren_max_rdz") is not None:
                            rdz_flag = f" ⚠️ High-Leverage Robust Distance Outlier ({stab_aud.get('rousseeuw_zomeren_outlier_count', 0)} influential)" if stab_aud.get("is_rousseeuw_zomeren_influential") else " (Bounded Robust Distance)"
                            st.markdown(f"**Rousseeuw-van Zomeren Robust Distance:** Max RDZ = `{stab_aud.get('rousseeuw_zomeren_max_rdz', 0.0):.3f}`{rdz_flag}")

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

            sug_col13, sug_col14, sug_col15 = st.columns(3)
            with sug_col13:
                if st.button("📐 Harbord Score Asymmetry", key="sug_harbord"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Harbord modified score test for funnel plot asymmetry"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Harbord modified score test for funnel plot asymmetry", findings, dataset_name=dataset_name)
            with sug_col14:
                if st.button("🔍 BKW Condition Index", key="sug_bkw"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Belsley-Kuh-Welsch (BKW) collinearity condition index"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Belsley-Kuh-Welsch (BKW) collinearity condition index", findings, dataset_name=dataset_name)
            with sug_col15:
                if st.button("📈 Huber Robust M-Slope", key="sug_huber"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Huber M-estimator robust regression discrepancy"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Huber M-estimator robust regression discrepancy", findings, dataset_name=dataset_name)

            sug_col16, sug_col17, sug_col18 = st.columns(3)
            with sug_col16:
                if st.button("📊 Peters Asymmetry Test", key="sug_peters"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Peters regression test for funnel plot asymmetry in binary data"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Peters regression test for funnel plot asymmetry in binary data", findings, dataset_name=dataset_name)
            with sug_col17:
                if st.button("📈 Hansen Threshold Regime", key="sug_hansen"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Bruce Hansen structural threshold non-linearity test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Bruce Hansen structural threshold non-linearity test", findings, dataset_name=dataset_name)
            with sug_col18:
                if st.button("🛡️ Least Trimmed Squares (LTS)", key="sug_lts"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Least Trimmed Squares (LTS) high-breakdown robust slope"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Least Trimmed Squares (LTS) high-breakdown robust slope", findings, dataset_name=dataset_name)

            sug_col19, sug_col20, sug_col21 = st.columns(3)
            with sug_col19:
                if st.button("📐 Rücker Arcsine Asymmetry", key="sug_rucker"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Rücker arcsine test for funnel plot asymmetry in binary data"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Rücker arcsine test for funnel plot asymmetry in binary data", findings, dataset_name=dataset_name)
            with sug_col20:
                if st.button("🔍 Hosmer-Lemeshow Calibration", key="sug_hosmer"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Hosmer-Lemeshow propensity model calibration test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Hosmer-Lemeshow propensity model calibration test", findings, dataset_name=dataset_name)
            with sug_col21:
                if st.button("📈 Siegel Repeated Medians", key="sug_siegel"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Siegel Repeated Medians high-breakdown robust slope"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Siegel Repeated Medians high-breakdown robust slope", findings, dataset_name=dataset_name)

            sug_col22, sug_col23, sug_col24 = st.columns(3)
            with sug_col22:
                if st.button("📊 Macaskill Asymmetry Test", key="sug_macaskill"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Peter Macaskill sample-size weighted regression funnel test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Peter Macaskill sample-size weighted regression funnel test", findings, dataset_name=dataset_name)
            with sug_col23:
                if st.button("🔬 AIPW Doubly Robust Effect", key="sug_aipw"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Robins-Rotnitzky-Zhao and Imbens AIPW doubly robust treatment effect"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Robins-Rotnitzky-Zhao and Imbens AIPW doubly robust treatment effect", findings, dataset_name=dataset_name)
            with sug_col24:
                if st.button("📉 Leamer Extreme Bounds", key="sug_leamer"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Edward Leamer Extreme Bounds Analysis (EBA)"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Edward Leamer Extreme Bounds Analysis (EBA)", findings, dataset_name=dataset_name)

            sug_col25, sug_col26, sug_col27 = st.columns(3)
            with sug_col25:
                if st.button("📊 Deeks DOR Funnel Test", key="sug_deeks"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Jon Deeks diagnostic odds ratio asymmetry test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Jon Deeks diagnostic odds ratio asymmetry test", findings, dataset_name=dataset_name)
            with sug_col26:
                if st.button("🤖 Double Machine Learning (DML)", key="sug_dml"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Chernozhukov Double Machine Learning (DML) causal effect"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Chernozhukov Double Machine Learning (DML) causal effect", findings, dataset_name=dataset_name)
            with sug_col27:
                if st.button("📐 Goldfeld-Quandt Heteroskedasticity", key="sug_gq"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Stephen Goldfeld and Richard Quandt split-regime heteroskedasticity test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Stephen Goldfeld and Richard Quandt split-regime heteroskedasticity test", findings, dataset_name=dataset_name)

            sug_col28, sug_col29, sug_col30 = st.columns(3)
            with sug_col28:
                if st.button("📊 Thompson-Sharp Funnel Test", key="sug_ts"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Simon Thompson and Stephen Sharp random-effects funnel asymmetry test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Simon Thompson and Stephen Sharp random-effects funnel asymmetry test", findings, dataset_name=dataset_name)
            with sug_col29:
                if st.button("⚖️ Rubin's Covariate Balance", key="sug_rubin"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Donald Rubin propensity score covariate balance diagnostics"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Donald Rubin propensity score covariate balance diagnostics", findings, dataset_name=dataset_name)
            with sug_col30:
                if st.button("📐 D'Agostino-Pearson Normality", key="sug_dp"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Ralph D'Agostino and E. S. Pearson omnibus normality test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Ralph D'Agostino and E. S. Pearson omnibus normality test", findings, dataset_name=dataset_name)

            sug_col31, sug_col32, sug_col33 = st.columns(3)
            with sug_col31:
                if st.button("📊 Moreno Funnel Test", key="sug_moreno"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Moreno variance-weighted funnel plot asymmetry test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Moreno variance-weighted funnel plot asymmetry test", findings, dataset_name=dataset_name)
            with sug_col32:
                if st.button("⚖️ Rosenbaum-Rubin SMD Balance", key="sug_smd"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Rosenbaum and Rubin standardized mean difference and Mahalanobis balance"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Rosenbaum and Rubin standardized mean difference and Mahalanobis balance", findings, dataset_name=dataset_name)
            with sug_col33:
                if st.button("📐 Anderson-Darling Normality", key="sug_ad"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Anderson-Darling empirical distribution function residual normality test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Anderson-Darling empirical distribution function residual normality test", findings, dataset_name=dataset_name)

            sug_col34, sug_col35, sug_col36 = st.columns(3)
            with sug_col34:
                if st.button("📊 Sterne-Egger Funnel Test", key="sug_sterne"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Sterne and Egger standard error funnel plot asymmetry test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Sterne and Egger standard error funnel plot asymmetry test", findings, dataset_name=dataset_name)
            with sug_col35:
                if st.button("⚖️ Austin Variance Ratio Balance", key="sug_austin"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Peter Austin covariate variance ratio balance metric"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Peter Austin covariate variance ratio balance metric", findings, dataset_name=dataset_name)
            with sug_col36:
                if st.button("📐 Lilliefors Composite Normality", key="sug_lill"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Hubert Lilliefors composite normality goodness of fit test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Hubert Lilliefors composite normality goodness of fit test", findings, dataset_name=dataset_name)

            sug_col37, sug_col38, sug_col39 = st.columns(3)
            with sug_col37:
                if st.button("📊 LFK Funnel Asymmetry Index", key="sug_lfk"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Luis Furuya-Kanamori and Suhail Doi LFK index of funnel plot asymmetry"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Luis Furuya-Kanamori and Suhail Doi LFK index of funnel plot asymmetry", findings, dataset_name=dataset_name)
            with sug_col38:
                if st.button("⚖️ Hansen-Bowers Omnibus Balance", key="sug_hb"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Ben Hansen and Jake Bowers omnibus covariate balance test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Ben Hansen and Jake Bowers omnibus covariate balance test", findings, dataset_name=dataset_name)
            with sug_col39:
                if st.button("📐 Cramér-von Mises Normality", key="sug_cvm"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Harald Cramér and Richard von Mises empirical distribution function residual normality test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Harald Cramér and Richard von Mises empirical distribution function residual normality test", findings, dataset_name=dataset_name)

            sug_col40, sug_col41, sug_col42 = st.columns(3)
            with sug_col40:
                if st.button("📊 PET-PEESE Meta-Regression", key="sug_pet_peese"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Stanley and Doucouliagos PET-PEESE meta-regression falsification test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Stanley and Doucouliagos PET-PEESE meta-regression falsification test", findings, dataset_name=dataset_name)
            with sug_col41:
                if st.button("⚖️ Imai-van Dyk GPS Balance", key="sug_gps"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Kosuke Imai and David van Dyk generalized propensity score balance test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Kosuke Imai and David van Dyk generalized propensity score balance test", findings, dataset_name=dataset_name)
            with sug_col42:
                if st.button("📐 Kuiper Invariant Normality", key="sug_kuiper"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Nicolaas Kuiper invariant empirical distribution function residual normality test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Nicolaas Kuiper invariant empirical distribution function residual normality test", findings, dataset_name=dataset_name)

            sug_col43, sug_col44, sug_col45 = st.columns(3)
            with sug_col43:
                if st.button("📊 Tang-Liu Funnel Test", key="sug_tang"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Tang and Liu linear regression funnel asymmetry test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Tang and Liu linear regression funnel asymmetry test", findings, dataset_name=dataset_name)
            with sug_col44:
                if st.button("⚖️ Iacus-King-Porro L1", key="sug_l1"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Stefano Iacus Gary King and Giuseppe Porro L1 multivariate imbalance metric"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Stefano Iacus Gary King and Giuseppe Porro L1 multivariate imbalance metric", findings, dataset_name=dataset_name)
            with sug_col45:
                if st.button("📐 Hosking L-Moments", key="sug_hosking"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain J. R. M. Hosking L-moments ratio residual normality test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain J. R. M. Hosking L-moments ratio residual normality test", findings, dataset_name=dataset_name)

            sug_col46, sug_col47, sug_col48 = st.columns(3)
            with sug_col46:
                if st.button("📉 Pustejovsky-Rodgers Funnel", key="sug_pust"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Pustejovsky and Rodgers variance-stabilized funnel asymmetry test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Pustejovsky and Rodgers variance-stabilized funnel asymmetry test", findings, dataset_name=dataset_name)
            with sug_col47:
                if st.button("⚖️ eQQ Covariate Balance", key="sug_eqq"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Ho Imai King Stuart empirical qq quantile balance metric"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Ho Imai King Stuart empirical qq quantile balance metric", findings, dataset_name=dataset_name)
            with sug_col48:
                if st.button("📐 Shapiro-Wilk Normality", key="sug_shapiro"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Samuel Shapiro and Martin Wilk omnibus residual normality test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Samuel Shapiro and Martin Wilk omnibus residual normality test", findings, dataset_name=dataset_name)

            sug_col49, sug_col50, sug_col51 = st.columns(3)
            with sug_col49:
                if st.button("📉 Schwarzer Arcsine Funnel", key="sug_schw"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Guido Schwarzer arcsine transformed funnel asymmetry test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Guido Schwarzer arcsine transformed funnel asymmetry test", findings, dataset_name=dataset_name)
            with sug_col50:
                if st.button("⚖️ Rosenbaum-Rubin Variance", key="sug_rr"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Paul Rosenbaum and Donald Rubin residual variance ratio balance test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Paul Rosenbaum and Donald Rubin residual variance ratio balance test", findings, dataset_name=dataset_name)
            with sug_col51:
                if st.button("📐 Doornik-Hansen Normality", key="sug_dh"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Doornik and Hansen omnibus residual normality test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Doornik and Hansen omnibus residual normality test", findings, dataset_name=dataset_name)

            sug_col52, sug_col53, sug_col54 = st.columns(3)
            with sug_col52:
                if st.button("📊 Ioannidis-Trikalinos TES", key="sug_tes"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain John Ioannidis and Thomas Trikalinos test for excess significance"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain John Ioannidis and Thomas Trikalinos test for excess significance", findings, dataset_name=dataset_name)
            with sug_col53:
                if st.button("🛡️ Frank Causal RIR", key="sug_rir"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Kenneth Frank robustness of inference to replacement"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Kenneth Frank robustness of inference to replacement", findings, dataset_name=dataset_name)
            with sug_col54:
                if st.button("🎯 Daniel Peña Influence", key="sug_pena"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Daniel Peña measure of influence in regression"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Daniel Peña measure of influence in regression", findings, dataset_name=dataset_name)

            sug_col55, sug_col56, sug_col57 = st.columns(3)
            with sug_col55:
                if st.button("📉 Copas Selection Funnel", key="sug_copas"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain John Copas and Frank Shi heuristic selection model funnel asymmetry test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain John Copas and Frank Shi heuristic selection model funnel asymmetry test", findings, dataset_name=dataset_name)
            with sug_col56:
                if st.button("🛡️ Imbens Sensitivity Contour", key="sug_imbens"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Guido Imbens parametric sensitivity contour boundary test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Guido Imbens parametric sensitivity contour boundary test", findings, dataset_name=dataset_name)
            with sug_col57:
                if st.button("🎯 Andrews-Pregibon Influence", key="sug_ap"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain David Andrews and Daryl Pregibon data ellipsoid volume influence diagnostic"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain David Andrews and Daryl Pregibon data ellipsoid volume influence diagnostic", findings, dataset_name=dataset_name)

            sug_col58, sug_col59, sug_col60 = st.columns(3)
            with sug_col58:
                if st.button("🔬 Gail-Simon Reversal Test", key="sug_gs"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Mitchell Gail and Richard Simon test for qualitative interactions"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Mitchell Gail and Richard Simon test for qualitative interactions", findings, dataset_name=dataset_name)
            with sug_col59:
                if st.button("⚖️ Kallus Minimax Confounding", key="sug_kallus"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Nathan Kallus minimax confounding bias bounds"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Nathan Kallus minimax confounding bias bounds", findings, dataset_name=dataset_name)
            with sug_col60:
                if st.button("📐 Rousseeuw-Croux Qn Scale", key="sug_qn"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Peter Rousseeuw and Christophe Croux Qn robust scale estimator"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Peter Rousseeuw and Christophe Croux Qn robust scale estimator", findings, dataset_name=dataset_name)

            sug_col61, sug_col62, sug_col63 = st.columns(3)
            with sug_col61:
                if st.button("🔄 Piantadosi-Byar-Green Interaction", key="sug_pbg"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Piantadosi Byar and Green likelihood ratio qualitative interaction test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Piantadosi Byar and Green likelihood ratio qualitative interaction test", findings, dataset_name=dataset_name)
            with sug_col62:
                if st.button("🛡️ Rosenbaum-Silber Amplification", key="sug_silber"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Paul Rosenbaum and Jeffrey Silber amplification of sensitivity to hidden bias"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Paul Rosenbaum and Jeffrey Silber amplification of sensitivity to hidden bias", findings, dataset_name=dataset_name)
            with sug_col63:
                if st.button("📈 Epps-Pulley Characteristic Normality", key="sug_ep"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Epps and Pulley empirical characteristic function normality test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Epps and Pulley empirical characteristic function normality test", findings, dataset_name=dataset_name)

            sug_col64, sug_col65, sug_col66 = st.columns(3)
            with sug_col64:
                if st.button("🎯 Pan Qualitative Score", key="sug_pan"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Guohua Pan directional qualitative interaction score test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Guohua Pan directional qualitative interaction score test", findings, dataset_name=dataset_name)
            with sug_col65:
                if st.button("⚖️ Zubizarreta Stable Weights", key="sug_sbw"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Jose Zubizarreta stable balancing weights diagnostic"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Jose Zubizarreta stable balancing weights diagnostic", findings, dataset_name=dataset_name)
            with sug_col66:
                if st.button("🔬 Baringhaus-Henze-Zirkler Normality", key="sug_bhz"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Baringhaus Henze and Zirkler empirical characteristic function normality test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Baringhaus Henze and Zirkler empirical characteristic function normality test", findings, dataset_name=dataset_name)

            sug_col67, sug_col68, sug_col69 = st.columns(3)
            with sug_col67:
                if st.button("📐 Silvapulle-Sen Inversion", key="sug_silva"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Silvapulle and Sen likelihood ratio inversion test for qualitative interaction"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Silvapulle and Sen likelihood ratio inversion test for qualitative interaction", findings, dataset_name=dataset_name)
            with sug_col68:
                if st.button("🛡️ Rosenbaum Design Sensitivity", key="sug_design"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Paul Rosenbaum design sensitivity bound Gamma tilde"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Paul Rosenbaum design sensitivity bound Gamma tilde", findings, dataset_name=dataset_name)
            with sug_col69:
                if st.button("🔬 Vasicek Entropy Normality", key="sug_vasicek"):
                    narrator = LLMNarrator(provider=llm_provider)
                    st.session_state["last_copilot_q"] = "Explain Oldrich Vasicek sample entropy residual normality test"
                    st.session_state["last_copilot_a"] = narrator.answer_query("Explain Oldrich Vasicek sample entropy residual normality test", findings, dataset_name=dataset_name)

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
                    "Harbord_Score_t": f"{simp.get('harbord_t_stat', 0.0):.2f}" if simp.get("harbord_t_stat") is not None else "N/A",
                    "Harbord_p_value": f"{simp.get('harbord_p_value', 0.0):.3e}" if simp.get("harbord_p_value") is not None else "N/A",
                    "Brown_Forsythe_p": f"{simp.get('brown_forsythe_p', 0.0):.3e}" if simp.get("brown_forsythe_p") is not None else "N/A",
                    "Peters_Score_t": f"{simp.get('peters_test_t', 0.0):.2f}" if simp.get("peters_test_t") is not None else "N/A",
                    "Peters_p_value": f"{simp.get('peters_test_p', 0.0):.3e}" if simp.get("peters_test_p") is not None else "N/A",
                    "Kish_Design_Effect": f"{simp.get('kish_design_effect', 0.0):.2f}" if simp.get("kish_design_effect") is not None else "N/A",
                    "Stratified_Variance_Ratio": f"{simp.get('stratified_variance_ratio', 0.0):.2f}" if simp.get("stratified_variance_ratio") is not None else "N/A",
                    "Rucker_Arcsine_t": f"{simp.get('rucker_test_t', 0.0):.2f}" if simp.get("rucker_test_t") is not None else "N/A",
                    "Rucker_Arcsine_p": f"{simp.get('rucker_test_p', 0.0):.3e}" if simp.get("rucker_test_p") is not None else "N/A",
                    "Levene_Scale_W": f"{simp.get('levene_stat', 0.0):.2f}" if simp.get("levene_stat") is not None else "N/A",
                    "Levene_Scale_p": f"{simp.get('levene_p', 0.0):.3e}" if simp.get("levene_p") is not None else "N/A",
                    "Macaskill_Funnel_t": f"{simp.get('macaskill_test_t', 0.0):.2f}" if simp.get("macaskill_test_t") is not None else "N/A",
                    "Macaskill_Funnel_p": f"{simp.get('macaskill_test_p', 0.0):.3e}" if simp.get("macaskill_test_p") is not None else "N/A",
                    "Bartlett_Scale_B": f"{simp.get('bartlett_stat', 0.0):.2f}" if simp.get("bartlett_stat") is not None else "N/A",
                    "Bartlett_Scale_p": f"{simp.get('bartlett_p', 0.0):.3e}" if simp.get("bartlett_p") is not None else "N/A",
                    "Deeks_Funnel_t": f"{simp.get('deeks_test_t', 0.0):.2f}" if simp.get("deeks_test_t") is not None else "N/A",
                    "Deeks_Funnel_p": f"{simp.get('deeks_test_p', 0.0):.3e}" if simp.get("deeks_test_p") is not None else "N/A",
                    "Mood_Scale_Z": f"{simp.get('mood_stat_z', 0.0):.2f}" if simp.get("mood_stat_z") is not None else "N/A",
                    "Mood_Scale_p": f"{simp.get('mood_p_value', 0.0):.3e}" if simp.get("mood_p_value") is not None else "N/A",
                    "Thompson_Sharp_t": f"{simp.get('thompson_sharp_t', 0.0):.2f}" if simp.get("thompson_sharp_t") is not None else "N/A",
                    "Thompson_Sharp_p": f"{simp.get('thompson_sharp_p', 0.0):.3e}" if simp.get("thompson_sharp_p") is not None else "N/A",
                    "Ansari_Bradley_W": f"{simp.get('ansari_bradley_stat', 0.0):.2f}" if simp.get("ansari_bradley_stat") is not None else "N/A",
                    "Ansari_Bradley_p": f"{simp.get('ansari_bradley_p', 0.0):.3e}" if simp.get("ansari_bradley_p") is not None else "N/A",
                    "Moreno_Funnel_t": f"{simp.get('moreno_test_t', 0.0):.2f}" if simp.get("moreno_test_t") is not None else "N/A",
                    "Moreno_Funnel_p": f"{simp.get('moreno_test_p', 0.0):.3e}" if simp.get("moreno_test_p") is not None else "N/A",
                    "Siegel_Tukey_W": f"{simp.get('siegel_tukey_stat', 0.0):.2f}" if simp.get("siegel_tukey_stat") is not None else "N/A",
                    "Siegel_Tukey_p": f"{simp.get('siegel_tukey_p', 0.0):.3e}" if simp.get("siegel_tukey_p") is not None else "N/A",
                    "Sterne_Funnel_t": f"{simp.get('sterne_test_t', 0.0):.2f}" if simp.get("sterne_test_t") is not None else "N/A",
                    "Sterne_Funnel_p": f"{simp.get('sterne_test_p', 0.0):.3e}" if simp.get("sterne_test_p") is not None else "N/A",
                    "Klotz_Scale_Z": f"{simp.get('klotz_scale_stat', 0.0):.2f}" if simp.get("klotz_scale_stat") is not None else "N/A",
                    "Klotz_Scale_p": f"{simp.get('klotz_scale_p', 0.0):.3e}" if simp.get("klotz_scale_p") is not None else "N/A",
                    "LFK_Index": f"{simp.get('lfk_index', 0.0):.2f}" if simp.get("lfk_index") is not None else "N/A",
                    "Conover_Scale_Z": f"{simp.get('conover_stat_z', 0.0):.2f}" if simp.get("conover_stat_z") is not None else "N/A",
                    "Conover_Scale_p": f"{simp.get('conover_p_value', 0.0):.3e}" if simp.get("conover_p_value") is not None else "N/A",
                    "PET_Test_t": f"{simp.get('pet_test_t', 0.0):.2f}" if simp.get("pet_test_t") is not None else "N/A",
                    "PET_Test_p": f"{simp.get('pet_test_p', 0.0):.3e}" if simp.get("pet_test_p") is not None else "N/A",
                    "Tang_Funnel_t": f"{simp.get('tang_test_t', 0.0):.2f}" if simp.get("tang_test_t") is not None else "N/A",
                    "Tang_Funnel_p": f"{simp.get('tang_test_p', 0.0):.3e}" if simp.get("tang_test_p") is not None else "N/A",
                    "PEESE_Test_t": f"{simp.get('peese_test_t', 0.0):.2f}" if simp.get("peese_test_t") is not None else "N/A",
                    "PEESE_Test_p": f"{simp.get('peese_test_p', 0.0):.3e}" if simp.get("peese_test_p") is not None else "N/A",
                    "Sukhatme_Scale_Z": f"{simp.get('sukhatme_stat_z', 0.0):.2f}" if simp.get("sukhatme_stat_z") is not None else "N/A",
                    "Sukhatme_Scale_p": f"{simp.get('sukhatme_p_value', 0.0):.3e}" if simp.get("sukhatme_p_value") is not None else "N/A",
                    "Capon_Scale_Z": f"{simp.get('capon_stat_z', 0.0):.2f}" if simp.get("capon_stat_z") is not None else "N/A",
                    "Capon_Scale_p": f"{simp.get('capon_p_value', 0.0):.3e}" if simp.get("capon_p_value") is not None else "N/A",
                    "Pustejovsky_Funnel_t": f"{simp.get('pustejovsky_test_t', 0.0):.2f}" if simp.get("pustejovsky_test_t") is not None else "N/A",
                    "Pustejovsky_Funnel_p": f"{simp.get('pustejovsky_test_p', 0.0):.3e}" if simp.get("pustejovsky_test_p") is not None else "N/A",
                    "Moses_Scale_U": f"{simp.get('moses_stat_u', 0.0):.1f}" if simp.get("moses_stat_u") is not None else "N/A",
                    "Moses_Scale_p": f"{simp.get('moses_stat_p', 0.0):.3e}" if simp.get("moses_stat_p") is not None else "N/A",
                    "Schwarzer_Funnel_t": f"{simp.get('schwarzer_test_t', 0.0):.2f}" if simp.get("schwarzer_test_t") is not None else "N/A",
                    "Schwarzer_Funnel_p": f"{simp.get('schwarzer_test_p', 0.0):.3e}" if simp.get("schwarzer_test_p") is not None else "N/A",
                    "Westenberg_Scale_Z": f"{simp.get('westenberg_stat_z', 0.0):.2f}" if simp.get("westenberg_stat_z") is not None else "N/A",
                    "Westenberg_Scale_p": f"{simp.get('westenberg_stat_p', 0.0):.3e}" if simp.get("westenberg_stat_p") is not None else "N/A",
                    "TES_Excess_Ratio": f"{simp.get('tes_excess_ratio', 0.0):.2f}" if simp.get("tes_excess_ratio") is not None else "N/A",
                    "TES_Excess_p": f"{simp.get('tes_p_value', 0.0):.3e}" if simp.get("tes_p_value") is not None else "N/A",
                    "OBrien_Scale_F": f"{simp.get('obrien_stat_f', 0.0):.2f}" if simp.get("obrien_stat_f") is not None else "N/A",
                    "OBrien_Scale_p": f"{simp.get('obrien_p_value', 0.0):.3e}" if simp.get("obrien_p_value") is not None else "N/A",
                    "Copas_Selection_t": f"{simp.get('copas_test_t', 0.0):.2f}" if simp.get("copas_test_t") is not None else "N/A",
                    "Copas_Selection_p": f"{simp.get('copas_test_p', 0.0):.3e}" if simp.get("copas_test_p") is not None else "N/A",
                    "Cochran_C_Stat": f"{simp.get('cochran_c_stat', 0.0):.3f}" if simp.get("cochran_c_stat") is not None else "N/A",
                    "Cochran_C_p": f"{simp.get('cochran_c_p', 0.0):.3e}" if simp.get("cochran_c_p") is not None else "N/A",
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
                    "BKW_Condition_Index": f"{conf.get('bkw_condition_index', 0.0):.2f}" if conf.get("bkw_condition_index") is not None else "N/A",
                    "Sargan_J_Stat": f"{conf.get('sargan_j_stat', 0.0):.2f}" if conf.get("sargan_j_stat") is not None else "N/A",
                    "Sargan_p_value": f"{conf.get('sargan_p_value', 0.0):.3e}" if conf.get("sargan_p_value") is not None else "N/A",
                    "Crump_Overlap_Ratio": f"{conf.get('crump_overlap_ratio', 0.0):.1%}" if conf.get("crump_overlap_ratio") is not None else "N/A",
                    "Hansen_Threshold_F": f"{conf.get('hansen_threshold_f', 0.0):.2f}" if conf.get("hansen_threshold_f") is not None else "N/A",
                    "Hansen_Threshold_p": f"{conf.get('hansen_threshold_p', 0.0):.3e}" if conf.get("hansen_threshold_p") is not None else "N/A",
                    "Hosmer_Lemeshow_p": f"{conf.get('hosmer_lemeshow_p', 0.0):.3e}" if conf.get("hosmer_lemeshow_p") is not None else "N/A",
                    "Newey_West_HAC_Ratio": f"{conf.get('hac_se_ratio', 0.0):.2f}" if conf.get("hac_se_ratio") is not None else "N/A",
                    "AIPW_Treatment_Effect": f"{conf.get('aipw_treatment_effect', 0.0):.4f}" if conf.get("aipw_treatment_effect") is not None else "N/A",
                    "AIPW_Discrepancy": f"{conf.get('aipw_discrepancy', 0.0):.1%}" if conf.get("aipw_discrepancy") is not None else "N/A",
                    "Leamer_Bound_Min": f"{conf.get('leamer_bound_min', 0.0):.4f}" if conf.get("leamer_bound_min") is not None else "N/A",
                    "Leamer_Bound_Max": f"{conf.get('leamer_bound_max', 0.0):.4f}" if conf.get("leamer_bound_max") is not None else "N/A",
                    "DML_Treatment_Effect": f"{conf.get('dml_treatment_effect', 0.0):.4f}" if conf.get("dml_treatment_effect") is not None else "N/A",
                    "DML_Discrepancy": f"{conf.get('dml_discrepancy', 0.0):.1%}" if conf.get("dml_discrepancy") is not None else "N/A",
                    "Cinelli_k_Multiplier": f"{conf.get('cinelli_k_multiplier', 0.0):.2f}x" if conf.get("cinelli_k_multiplier") is not None else "N/A",
                    "Rubin_B_Score": f"{conf.get('rubin_b_score', 0.0):.1f}%" if conf.get("rubin_b_score") is not None else "N/A",
                    "Rubin_R_Ratio": f"{conf.get('rubin_r_ratio', 0.0):.2f}" if conf.get("rubin_r_ratio") is not None else "N/A",
                    "Brunner_Munzel_W": f"{conf.get('brunner_munzel_w', 0.0):.2f}" if conf.get("brunner_munzel_w") is not None else "N/A",
                    "Brunner_Munzel_p": f"{conf.get('brunner_munzel_p', 0.0):.3e}" if conf.get("brunner_munzel_p") is not None else "N/A",
                    "Rosenbaum_Rubin_SMD": f"{conf.get('smd_covariate_balance', 0.0):.3f}" if conf.get("smd_covariate_balance") is not None else "N/A",
                    "Mahalanobis_Distance": f"{conf.get('mahalanobis_distance', 0.0):.3f}" if conf.get("mahalanobis_distance") is not None else "N/A",
                    "Imbens_Normalized_Diff": f"{conf.get('imbens_normalized_diff', 0.0):.3f}" if conf.get("imbens_normalized_diff") is not None else "N/A",
                    "Austin_Variance_Ratio": f"{conf.get('austin_variance_ratio', 0.0):.2f}" if conf.get("austin_variance_ratio") is not None else "N/A",
                    "Stuart_Prognostic_Discrepancy": f"{conf.get('stuart_prognostic_discrepancy', 0.0):.3f}" if conf.get("stuart_prognostic_discrepancy") is not None else "N/A",
                    "Hansen_Bowers_Chi2": f"{conf.get('hansen_bowers_chi2', 0.0):.2f}" if conf.get("hansen_bowers_chi2") is not None else "N/A",
                    "Hansen_Bowers_p": f"{conf.get('hansen_bowers_p', 0.0):.3e}" if conf.get("hansen_bowers_p") is not None else "N/A",
                    "Caliper_Discordance_Rate": f"{conf.get('caliper_discordance_rate', 0.0):.1%}" if conf.get("caliper_discordance_rate") is not None else "N/A",
                    "GPS_Balance_r": f"{conf.get('gps_balance_r', 0.0):.3f}" if conf.get("gps_balance_r") is not None else "N/A",
                    "GPS_Balance_p": f"{conf.get('gps_balance_p', 0.0):.3e}" if conf.get("gps_balance_p") is not None else "N/A",
                    "Austin_KS_Stat": f"{conf.get('austin_ks_stat', 0.0):.3f}" if conf.get("austin_ks_stat") is not None else "N/A",
                    "Austin_KS_p": f"{conf.get('austin_ks_p_value', 0.0):.3e}" if conf.get("austin_ks_p_value") is not None else "N/A",
                    "L1_Multivariate_Imbalance": f"{conf.get('l1_multivariate_imbalance', 0.0):.3f}" if conf.get("l1_multivariate_imbalance") is not None else "N/A",
                    "Rubin_Parallelism_F": f"{conf.get('rubin_parallelism_f', 0.0):.2f}" if conf.get("rubin_parallelism_f") is not None else "N/A",
                    "Rubin_Parallelism_p": f"{conf.get('rubin_parallelism_p', 0.0):.3e}" if conf.get("rubin_parallelism_p") is not None else "N/A",
                    "eQQ_Mean_Diff": f"{conf.get('eqq_mean_diff', 0.0):.3f}" if conf.get("eqq_mean_diff") is not None else "N/A",
                    "eQQ_Max_Diff": f"{conf.get('eqq_max_diff', 0.0):.3f}" if conf.get("eqq_max_diff") is not None else "N/A",
                    "Lin_Interaction_t": f"{conf.get('lin_interaction_t', 0.0):.2f}" if conf.get("lin_interaction_t") is not None else "N/A",
                    "Lin_Interaction_p": f"{conf.get('lin_interaction_p', 0.0):.3e}" if conf.get("lin_interaction_p") is not None else "N/A",
                    "Rosenbaum_Rubin_VR": f"{conf.get('rr_variance_ratio', 0.0):.2f}" if conf.get("rr_variance_ratio") is not None else "N/A",
                    "Rosenbaum_Rubin_p": f"{conf.get('rr_variance_p', 0.0):.3e}" if conf.get("rr_variance_p") is not None else "N/A",
                    "Imbens_Wooldridge_Quad_Diff": f"{conf.get('iw_quadratic_diff', 0.0):.3f}" if conf.get("iw_quadratic_diff") is not None else "N/A",
                    "Hodges_Lehmann_Shift": f"{conf.get('hodges_lehmann_shift', 0.0):.3f}" if conf.get("hodges_lehmann_shift") is not None else "N/A",
                    "Hodges_Lehmann_p": f"{conf.get('hodges_lehmann_p', 0.0):.3e}" if conf.get("hodges_lehmann_p") is not None else "N/A",
                    "Frank_RIR_Pct": f"{conf.get('frank_rir_pct', 0.0):.1f}%" if conf.get("frank_rir_pct") is not None else "N/A",
                    "Frank_RIR_Count": f"{conf.get('frank_rir_count', 0)}" if conf.get("frank_rir_count") is not None else "N/A",
                    "Imbens_Contour_q": f"{conf.get('imbens_contour_q', 0.0):.3f}" if conf.get("imbens_contour_q") is not None else "N/A",
                    "Imbens_Contour_p": f"{conf.get('imbens_contour_p', 0.0):.3e}" if conf.get("imbens_contour_p") is not None else "N/A",
                    "Austin_Stuart_MASD": f"{conf.get('austin_stuart_masd', 0.0):.3f}" if conf.get("austin_stuart_masd") is not None else "N/A",
                    "Austin_Stuart_p": f"{conf.get('austin_stuart_p', 0.0):.3e}" if conf.get("austin_stuart_p") is not None else "N/A",
                    "Kallus_Gamma_Crit": f"{conf.get('kallus_gamma_crit', 0.0):.2f}" if conf.get("kallus_gamma_crit") is not None else "N/A",
                    "Kallus_Bias_Max": f"{conf.get('kallus_bias_max', 0.0):.3f}" if conf.get("kallus_bias_max") is not None else "N/A",
                    "CBPS_Overid_J": f"{conf.get('cbps_overid_j', 0.0):.2f}" if conf.get("cbps_overid_j") is not None else "N/A",
                    "CBPS_Overid_p": f"{conf.get('cbps_overid_p', 0.0):.3e}" if conf.get("cbps_overid_p") is not None else "N/A",
                    "Gail_Simon_Stat": f"{simp.get('gail_simon_stat', 0.0):.2f}" if simp.get("gail_simon_stat") is not None else "N/A",
                    "Gail_Simon_p": f"{simp.get('gail_simon_p', 0.0):.3e}" if simp.get("gail_simon_p") is not None else "N/A",
                    "Hartley_Fmax_Stat": f"{simp.get('hartley_fmax_stat', 0.0):.2f}" if simp.get("hartley_fmax_stat") is not None else "N/A",
                    "Hartley_Fmax_p": f"{simp.get('hartley_fmax_p', 0.0):.3e}" if simp.get("hartley_fmax_p") is not None else "N/A",
                    "Anscombe_Glynn_Z": f"{stab.get('anscombe_glynn_z', 0.0):.2f}" if stab.get("anscombe_glynn_z") is not None else "N/A",
                    "Anscombe_Glynn_p": f"{stab.get('anscombe_glynn_p', 0.0):.3e}" if stab.get("anscombe_glynn_p") is not None else "N/A",
                    "Rousseeuw_Croux_Qn": f"{stab.get('rousseeuw_croux_qn', 0.0):.4f}" if stab.get("rousseeuw_croux_qn") is not None else "N/A",
                    "Scale_Inflation_Ratio_Qn": f"{stab.get('scale_inflation_ratio_qn', 0.0):.2f}" if stab.get("scale_inflation_ratio_qn") is not None else "N/A",
                    "Piantadosi_Byar_Green_Stat": f"{simp.get('piantadosi_byar_green_stat', 0.0):.2f}" if simp.get("piantadosi_byar_green_stat") is not None else "N/A",
                    "Piantadosi_Byar_Green_p": f"{simp.get('piantadosi_byar_green_p', 0.0):.3e}" if simp.get("piantadosi_byar_green_p") is not None else "N/A",
                    "Layard_Stat_L": f"{simp.get('layard_stat_l', 0.0):.2f}" if simp.get("layard_stat_l") is not None else "N/A",
                    "Layard_p_value": f"{simp.get('layard_p_value', 0.0):.3e}" if simp.get("layard_p_value") is not None else "N/A",
                    "Rosenbaum_Silber_Delta_Crit": f"{conf.get('rosenbaum_silber_delta_crit', 0.0):.2f}" if conf.get("rosenbaum_silber_delta_crit") is not None else "N/A",
                    "Rosenbaum_Silber_Lambda": f"{conf.get('rosenbaum_silber_lambda', 0.0):.1f}" if conf.get("rosenbaum_silber_lambda") is not None else "N/A",
                    "Fogarty_Gamma_Crit": f"{conf.get('fogarty_gamma_crit', 0.0):.2f}" if conf.get("fogarty_gamma_crit") is not None else "N/A",
                    "Fogarty_Studentized_t": f"{conf.get('fogarty_studentized_t', 0.0):.2f}" if conf.get("fogarty_studentized_t") is not None else "N/A",
                    "Epps_Pulley_Stat": f"{stab.get('epps_pulley_stat', 0.0):.3f}" if stab.get("epps_pulley_stat") is not None else "N/A",
                    "Epps_Pulley_p": f"{stab.get('epps_pulley_p', 0.0):.3e}" if stab.get("epps_pulley_p") is not None else "N/A",
                    "Chatterjee_Hadi_Max_H2": f"{stab.get('max_chatterjee_hadi_influence', 0.0):.3f}" if stab.get("max_chatterjee_hadi_influence") is not None else "N/A",
                    "Chatterjee_Hadi_Outlier_Count": f"{stab.get('chatterjee_hadi_outlier_count', 0)}" if stab.get("chatterjee_hadi_outlier_count") is not None else "N/A",
                    "Pan_Score_Stat": f"{simp.get('pan_score_stat', 0.0):.2f}" if simp.get("pan_score_stat") is not None else "N/A",
                    "Pan_Score_p": f"{simp.get('pan_score_p', 0.0):.3e}" if simp.get("pan_score_p") is not None else "N/A",
                    "Samiuddin_Stat_F": f"{simp.get('samiuddin_stat_f', 0.0):.2f}" if simp.get("samiuddin_stat_f") is not None else "N/A",
                    "Samiuddin_p_value": f"{simp.get('samiuddin_p_value', 0.0):.3e}" if simp.get("samiuddin_p_value") is not None else "N/A",
                    "Rosenbaum_Attr_Gamma_Crit": f"{conf.get('rosenbaum_attr_gamma_crit', 0.0):.2f}" if conf.get("rosenbaum_attr_gamma_crit") is not None else "N/A",
                    "Rosenbaum_Attr_p_upper": f"{conf.get('rosenbaum_attr_p_upper', 0.0):.3e}" if conf.get("rosenbaum_attr_p_upper") is not None else "N/A",
                    "Zubizarreta_SBW_CV": f"{conf.get('zubizarreta_sbw_cv', 0.0):.2f}" if conf.get("zubizarreta_sbw_cv") is not None else "N/A",
                    "Zubizarreta_Eff_Sample_Pct": f"{conf.get('zubizarreta_eff_sample_pct', 0.0):.1f}%" if conf.get("zubizarreta_eff_sample_pct") is not None else "N/A",
                    "Baringhaus_Henze_Stat": f"{stab.get('baringhaus_henze_stat', 0.0):.3f}" if stab.get("baringhaus_henze_stat") is not None else "N/A",
                    "Baringhaus_Henze_p": f"{stab.get('baringhaus_henze_p', 0.0):.3e}" if stab.get("baringhaus_henze_p") is not None else "N/A",
                    "Gray_Ling_Max_Influence": f"{stab.get('max_gray_ling_influence', 0.0):.3f}" if stab.get("max_gray_ling_influence") is not None else "N/A",
                    "Gray_Ling_Outlier_Count": f"{stab.get('gray_ling_outlier_count', 0)}" if stab.get("gray_ling_outlier_count") is not None else "N/A",
                    "Silvapulle_Sen_Stat": f"{simp.get('silvapulle_sen_stat', 0.0):.2f}" if simp.get("silvapulle_sen_stat") is not None else "N/A",
                    "Silvapulle_Sen_p": f"{simp.get('silvapulle_sen_p', 0.0):.3e}" if simp.get("silvapulle_sen_p") is not None else "N/A",
                    "Miller_Jackknife_F": f"{simp.get('miller_jackknife_stat_f', 0.0):.2f}" if simp.get("miller_jackknife_stat_f") is not None else "N/A",
                    "Miller_Jackknife_p": f"{simp.get('miller_jackknife_p_value', 0.0):.3e}" if simp.get("miller_jackknife_p_value") is not None else "N/A",
                    "Rosenbaum_Design_Gamma": f"{conf.get('rosenbaum_design_sensitivity_gamma', 0.0):.2f}" if conf.get("rosenbaum_design_sensitivity_gamma") is not None else "N/A",
                    "Hosman_HHH_Delta": f"{conf.get('hosman_hansen_holland_delta', 0.0):.3f}" if conf.get("hosman_hansen_holland_delta") is not None else "N/A",
                    "Vasicek_Entropy_Kmn": f"{stab.get('vasicek_entropy_stat', 0.0):.3f}" if stab.get("vasicek_entropy_stat") is not None else "N/A",
                    "Vasicek_Entropy_p": f"{stab.get('vasicek_entropy_p', 0.0):.3e}" if stab.get("vasicek_entropy_p") is not None else "N/A",
                    "Rousseeuw_Zomeren_RDZ": f"{stab.get('rousseeuw_zomeren_max_rdz', 0.0):.3f}" if stab.get("rousseeuw_zomeren_max_rdz") is not None else "N/A",
                    "Rousseeuw_Zomeren_Outliers": f"{stab.get('rousseeuw_zomeren_outlier_count', 0)}" if stab.get("rousseeuw_zomeren_outlier_count") is not None else "N/A",
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
                    "CUSUMSQ_Stat": f"{stab.get('cusumsq_stat', 0.0):.3f}" if stab.get("cusumsq_stat") is not None else "N/A",
                    "CUSUMSQ_Critical_Val": f"{stab.get('cusumsq_critical_val', 0.0):.3f}" if stab.get("cusumsq_critical_val") is not None else "N/A",
                    "Koenker_Studentized_LM": f"{stab.get('koenker_lm_stat', 0.0):.2f}" if stab.get("koenker_lm_stat") is not None else "N/A",
                    "Koenker_p_value": f"{stab.get('koenker_p_value', 0.0):.3e}" if stab.get("koenker_p_value") is not None else "N/A",
                    "Harvey_Collier_t": f"{stab.get('harvey_collier_t', 0.0):.2f}" if stab.get("harvey_collier_t") is not None else "N/A",
                    "Harvey_Collier_p": f"{stab.get('harvey_collier_p', 0.0):.3e}" if stab.get("harvey_collier_p") is not None else "N/A",
                    "Huber_M_Slope": f"{stab.get('huber_m_slope', 0.0):.4f}" if stab.get("huber_m_slope") is not None else "N/A",
                    "Huber_Slope_Discrepancy": f"{stab.get('huber_slope_discrepancy', 0.0):.1%}" if stab.get("huber_slope_discrepancy") is not None else "N/A",
                    "LTS_Slope": f"{stab.get('lts_slope', 0.0):.4f}" if stab.get("lts_slope") is not None else "N/A",
                    "LTS_Slope_Discrepancy": f"{stab.get('lts_slope_discrepancy', 0.0):.1%}" if stab.get("lts_slope_discrepancy") is not None else "N/A",
                    "Siegel_Slope": f"{stab.get('siegel_slope', 0.0):.4f}" if stab.get("siegel_slope") is not None else "N/A",
                    "Siegel_Slope_Discrepancy": f"{stab.get('siegel_slope_discrepancy', 0.0):.1%}" if stab.get("siegel_slope_discrepancy") is not None else "N/A",
                    "Breusch_Godfrey_p": f"{stab.get('breusch_godfrey_p', 0.0):.3e}" if stab.get("breusch_godfrey_p") is not None else "N/A",
                    "Runs_Test_Z": f"{stab.get('runs_test_z', 0.0):.2f}" if stab.get("runs_test_z") is not None else "N/A",
                    "Runs_Test_p": f"{stab.get('runs_test_p', 0.0):.3e}" if stab.get("runs_test_p") is not None else "N/A",
                    "Kendall_Tau_b": f"{stab.get('kendall_tau_b', 0.0):.4f}" if stab.get("kendall_tau_b") is not None else "N/A",
                    "Kendall_Discrepancy": f"{stab.get('kendall_discrepancy', 0.0):.1%}" if stab.get("kendall_discrepancy") is not None else "N/A",
                    "Goldfeld_Quandt_F": f"{stab.get('goldfeld_quandt_f', 0.0):.2f}" if stab.get("goldfeld_quandt_f") is not None else "N/A",
                    "Goldfeld_Quandt_p": f"{stab.get('goldfeld_quandt_p', 0.0):.3e}" if stab.get("goldfeld_quandt_p") is not None else "N/A",
                    "Shapiro_Francia_W": f"{stab.get('shapiro_francia_w', 0.0):.4f}" if stab.get("shapiro_francia_w") is not None else "N/A",
                    "Shapiro_Francia_p": f"{stab.get('shapiro_francia_p', 0.0):.3e}" if stab.get("shapiro_francia_p") is not None else "N/A",
                    "DAgostino_Pearson_K2": f"{stab.get('dagostino_pearson_k2', 0.0):.2f}" if stab.get("dagostino_pearson_k2") is not None else "N/A",
                    "DAgostino_Pearson_p": f"{stab.get('dagostino_pearson_p', 0.0):.3e}" if stab.get("dagostino_pearson_p") is not None else "N/A",
                    "Glejser_t": f"{stab.get('glejser_t_stat', 0.0):.2f}" if stab.get("glejser_t_stat") is not None else "N/A",
                    "Glejser_p": f"{stab.get('glejser_p_value', 0.0):.3e}" if stab.get("glejser_p_value") is not None else "N/A",
                    "Anderson_Darling_A2": f"{stab.get('anderson_darling_stat', 0.0):.3f}" if stab.get("anderson_darling_stat") is not None else "N/A",
                    "Harvey_LM_Stat": f"{stab.get('harvey_lm_stat', 0.0):.2f}" if stab.get("harvey_lm_stat") is not None else "N/A",
                    "Harvey_LM_p": f"{stab.get('harvey_lm_p', 0.0):.3e}" if stab.get("harvey_lm_p") is not None else "N/A",
                    "Lilliefors_D": f"{stab.get('lilliefors_stat', 0.0):.3f}" if stab.get("lilliefors_stat") is not None else "N/A",
                    "Lilliefors_p": f"{stab.get('lilliefors_p_value', 0.0):.3e}" if stab.get("lilliefors_p_value") is not None else "N/A",
                    "Park_Heteroskedasticity_t": f"{stab.get('park_t_stat', 0.0):.2f}" if stab.get("park_t_stat") is not None else "N/A",
                    "Park_Heteroskedasticity_p": f"{stab.get('park_p_value', 0.0):.3e}" if stab.get("park_p_value") is not None else "N/A",
                    "Cramer_von_Mises_W2": f"{stab.get('cramer_von_mises_w2', 0.0):.3f}" if stab.get("cramer_von_mises_w2") is not None else "N/A",
                    "Cramer_von_Mises_p": f"{stab.get('cramer_von_mises_p', 0.0):.3e}" if stab.get("cramer_von_mises_p") is not None else "N/A",
                    "Cook_Weisberg_Score": f"{stab.get('cook_weisberg_score', 0.0):.2f}" if stab.get("cook_weisberg_score") is not None else "N/A",
                    "Cook_Weisberg_p": f"{stab.get('cook_weisberg_p', 0.0):.3e}" if stab.get("cook_weisberg_p") is not None else "N/A",
                    "Kuiper_V_Stat": f"{stab.get('kuiper_v_stat', 0.0):.3f}" if stab.get("kuiper_v_stat") is not None else "N/A",
                    "Kuiper_V_Star": f"{stab.get('kuiper_v_star', 0.0):.3f}" if stab.get("kuiper_v_star") is not None else "N/A",
                    "Kuiper_p": f"{stab.get('kuiper_p_value', 0.0):.3e}" if stab.get("kuiper_p_value") is not None else "N/A",
                    "Max_COVRATIO_Dev": f"{stab.get('max_covratio_dev', 0.0):.3f}" if stab.get("max_covratio_dev") is not None else "N/A",
                    "COVRATIO_Outlier_Count": f"{stab.get('covratio_outlier_count', 0)}" if stab.get("covratio_outlier_count") is not None else "N/A",
                    "Hosking_L_Stat": f"{stab.get('hosking_l_stat', 0.0):.2f}" if stab.get("hosking_l_stat") is not None else "N/A",
                    "Hosking_L_p": f"{stab.get('hosking_l_p', 0.0):.3e}" if stab.get("hosking_l_p") is not None else "N/A",
                    "Max_Atkinson_Distance": f"{stab.get('max_atkinson_distance', 0.0):.2f}" if stab.get("max_atkinson_distance") is not None else "N/A",
                    "Atkinson_Outlier_Count": f"{stab.get('atkinson_outlier_count', 0)}" if stab.get("atkinson_outlier_count") is not None else "N/A",
                    "Shapiro_Wilk_W": f"{stab.get('shapiro_wilk_w', 0.0):.4f}" if stab.get("shapiro_wilk_w") is not None else "N/A",
                    "Shapiro_Wilk_p": f"{stab.get('shapiro_wilk_p', 0.0):.3e}" if stab.get("shapiro_wilk_p") is not None else "N/A",
                    "Max_Hadi_Influence": f"{stab.get('max_hadi_influence', 0.0):.2f}" if stab.get("max_hadi_influence") is not None else "N/A",
                    "Hadi_Outlier_Count": f"{stab.get('hadi_outlier_count', 0)}" if stab.get("hadi_outlier_count") is not None else "N/A",
                    "Doornik_Hansen_E": f"{stab.get('doornik_hansen_stat', 0.0):.2f}" if stab.get("doornik_hansen_stat") is not None else "N/A",
                    "Doornik_Hansen_p": f"{stab.get('doornik_hansen_p', 0.0):.3e}" if stab.get("doornik_hansen_p") is not None else "N/A",
                    "Max_Welsch_Distance": f"{stab.get('max_welsch_distance', 0.0):.2f}" if stab.get("max_welsch_distance") is not None else "N/A",
                    "Welsch_Outlier_Count": f"{stab.get('welsch_outlier_count', 0)}" if stab.get("welsch_outlier_count") is not None else "N/A",
                    "Gel_Gastwirth_RJB": f"{stab.get('gel_gastwirth_rjb', 0.0):.2f}" if stab.get("gel_gastwirth_rjb") is not None else "N/A",
                    "Gel_Gastwirth_p": f"{stab.get('gel_gastwirth_p', 0.0):.3e}" if stab.get("gel_gastwirth_p") is not None else "N/A",
                    "Max_Pena_Influence": f"{stab.get('max_pena_influence', 0.0):.3f}" if stab.get("max_pena_influence") is not None else "N/A",
                    "Pena_Outlier_Count": f"{stab.get('pena_outlier_count', 0)}" if stab.get("pena_outlier_count") is not None else "N/A",
                    "Szroeter_Stat_Q": f"{stab.get('szroeter_stat_q', 0.0):.3f}" if stab.get("szroeter_stat_q") is not None else "N/A",
                    "Szroeter_p": f"{stab.get('szroeter_p_value', 0.0):.3e}" if stab.get("szroeter_p_value") is not None else "N/A",
                    "Max_AP_Deviation": f"{stab.get('max_ap_deviation', 0.0):.3f}" if stab.get("max_ap_deviation") is not None else "N/A",
                    "AP_Outlier_Count": f"{stab.get('ap_outlier_count', 0)}" if stab.get("ap_outlier_count") is not None else "N/A",
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
