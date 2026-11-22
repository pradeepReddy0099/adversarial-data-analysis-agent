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
            for f in findings:
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

        with tab_briefing:
            narrator = LLMNarrator(provider=llm_provider)
            briefing = narrator.generate_executive_briefing(findings, dataset_name=dataset_name)
            st.markdown(briefing)

        with tab_copilot:
            st.markdown("#### 💬 Ask ADAA AI Copilot")
            st.caption("Ask questions about risk scores, specific variables, confounders, or leadership actions.")
            user_q = st.text_input("Enter your question:", placeholder="e.g., Why did customer support calls reverse across tiers?", key="copilot_user_q")
            if user_q:
                narrator = LLMNarrator(provider=llm_provider)
                answer = narrator.answer_query(user_q, findings, dataset_name=dataset_name)
                st.info(f"**Copilot:** {answer}")

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
                
    else:
        st.info("👈 Select a benchmark dataset or upload your own table, then press **Run Adversarial Multi-Agent Audit**.")
