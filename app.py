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
                        if simpsons.get("e_value") is not None:
                            st.markdown(f"**Causal Sensitivity E-Value:** `{simpsons['e_value']:.2f}`")

                        conf_aud = audit.get("confounder_audit", {})
                        if conf_aud.get("e_value") is not None:
                            st.markdown(f"**VanderWeele E-Value:** `{conf_aud['e_value']:.2f}`")
                        if conf_aud.get("top_confounder_vif") is not None:
                            vif_flag = " ⚠️ High Collinearity" if conf_aud.get("is_collinear") else ""
                            st.markdown(f"**Confounder VIF Collinearity:** `{conf_aud['top_confounder_vif']:.2f}`{vif_flag}")

                        stab_aud = audit.get("stability_audit", {})
                        if stab_aud.get("sign_invariance_prob") is not None:
                            st.markdown(f"**Sign Invariance Probability:** `P(sign) = {stab_aud['sign_invariance_prob']:.1%}`")
                        if stab_aud.get("max_cooks_distance") is not None and stab_aud.get("max_cooks_distance", 0) > 0:
                            st.markdown(f"**Cook's Distance Influence:** Max `D_i = {stab_aud['max_cooks_distance']:.3f}` ({stab_aud.get('cooks_outlier_count', 0)} points > 4/n, shift = `{stab_aud.get('cooks_statistic_shift', 0.0):.4f}`)")
                        if stab_aud.get("rank_discrepancy") is not None:
                            st.markdown(f"**Spearman Rank Discrepancy:** `Δr = {stab_aud['rank_discrepancy']:.3f}` (ρ = `{stab_aud.get('spearman_rho', 0.0):.3f}`)")
                        if stab_aud.get("max_leverage") is not None and stab_aud.get("max_leverage", 0) > 0:
                            st.markdown(f"**Outlier Leverage:** Max `h_ii = {stab_aud['max_leverage']:.4f}` ({stab_aud.get('high_leverage_points', 0)} high-leverage points)")

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
                summary_rows.append({
                    "Variable": cl.get("source_variable", ""),
                    "Target": cl.get("target_variable", ""),
                    "Statistic": f"{cl.get('statistic_name', '')} = {cl.get('statistic_value', 0.0):.4f}",
                    "p_value": f"{cl.get('p_value', 0.0):.4e}",
                    "FDR_q_value": f"{cl.get('fdr_q_value', 0.0):.4e}" if cl.get("fdr_q_value") is not None else "N/A",
                    "Verdict": f.get("verdict_badge", ""),
                    "Confidence": f.get("confidence_score", 0.0),
                    "Sign_Invariance_P": f"{stab.get('sign_invariance_prob', 1.0):.1%}",
                    "Cooks_D_Max": f"{stab.get('max_cooks_distance', 0.0):.3f}",
                    "Confounder_VIF": f"{conf.get('top_confounder_vif', 1.0):.2f}" if conf.get("top_confounder_vif") is not None else "N/A",
                    "Synthesis": f.get("synthesis_summary", ""),
                    "Takeaway": f.get("actionable_takeaway", ""),
                })
            df_summary = pd.DataFrame(summary_rows)
            st.download_button("📊 Download Findings Summary CSV", data=df_summary.to_csv(index=False).encode("utf-8"), file_name="findings_summary.csv", mime="text/csv")
                
    else:
        st.info("👈 Select a benchmark dataset or upload your own table, then press **Run Adversarial Multi-Agent Audit**.")
