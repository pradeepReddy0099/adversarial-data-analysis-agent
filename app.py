"""Interactive Web UI: The Adversarial Debate Arena (Streamlit)."""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Add src to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from adversarial_agent.core import run_adversarial_pipeline, load_dataset
from adversarial_agent.llm_narrator import LLMNarrator

st.set_page_config(
    page_title="Adversarial Data Analysis Agent",
    page_icon="⚔️",
    layout="wide",
)

st.title("⚔️ Adversarial Data Analysis Agent")
st.caption("A Dialectical Multi-Agent System for Automated Hypothesis Generation and Statistical Falsification")

st.markdown(
    """
    Standard automated data agents suffer from **confirmation bias** and **sycophancy**, presenting surface-level 
    correlations as causal truths. This system implements an adversarial debate:
    * 🔍 **Explorer Agent:** Autonomously mines the dataset to propose candidate empirical hypotheses.
    * ⚖️ **Devil's Advocate Agent:** Actively seeks to **falsify** claims (auditing for **Simpson's Paradox**, confounder collapse, and stability).
    * 🏆 **Referee Agent:** Adjudicates the debate with calibrated credibility badges.
    """
)

st.write("---")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1. Select Benchmark Dataset")
    data_dir = Path(__file__).resolve().parent / "data"
    benchmarks = {
        "UC Berkeley Admissions (Simpson's Paradox Benchmark)": data_dir / "berkeley_admissions.csv",
        "Commercial Customer Churn (Subgroup Paradox Benchmark)": data_dir / "simpsons_churn.csv",
        "UCI Adult Census Income (Real-World Social Benchmark)": data_dir / "adult_census.csv",
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
            
        selected_target = st.selectbox("Optional Outcome / Target Variable:", target_candidates, index=default_target_idx)
        target_val = None if selected_target == "None" else selected_target
        
        enable_llm = st.checkbox("🧠 Enable LLM Executive Narrator (Gemini / OpenAI / Offline)", value=True)
        llm_provider = st.selectbox("LLM Provider", ["auto", "gemini", "openai", "offline"]) if enable_llm else "offline"

        run_btn = st.button("🚀 Run Adversarial Multi-Agent Audit", type="primary")
    else:
        run_btn = False

with col_right:
    st.subheader("2. Multi-Agent Debate Arena")
    if run_btn and selected_path:
        with st.spinner("Agents are debating and stress-testing hypotheses..."):
            results = run_adversarial_pipeline(
                input_path=selected_path,
                target=target_val,
                out_dir="output/ui_run",
            )
            findings = results["adversarial_findings"]

        st.success(f"Audit Complete! Evaluated {len(findings)} candidate empirical hypotheses.")
        
        # Summary Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        c_robust = sum(1 for f in findings if f["verdict"] == "ROBUST_VERIFIED")
        c_simpson = sum(1 for f in findings if f["verdict"] == "SIMPSONS_PARADOX_ALERT")
        c_spurious = sum(1 for f in findings if f["verdict"] == "SPURIOUS_OR_CONFOUNDED")
        
        m1.metric("Claims Evaluated", len(findings))
        m2.metric("🟢 Robust Findings", c_robust)
        m3.metric("🟡 Simpson's Paradox", c_simpson)
        m4.metric("🔴 Spurious / Confounded", c_spurious)
        
        st.write("---")
        
        if enable_llm:
            with st.spinner("LLM Narrator is synthesizing executive briefing..."):
                narrator = LLMNarrator(provider=llm_provider)
                briefing = narrator.generate_executive_briefing(findings, dataset_name=selected_path.name)
                st.markdown(briefing)
                st.write("---")

                st.subheader("💬 Ask ADAA Copilot About Findings")
                user_q = st.text_input("Ask a question about confounders, paradoxes, or recommendations:")
                if user_q:
                    with st.spinner("Copilot is analyzing findings dossier..."):
                        answer = narrator.answer_query(user_q, findings, dataset_name=selected_path.name)
                        st.info(f"**Copilot:** {answer}")
                st.write("---")
        
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
                
    else:
        st.info("👈 Select a benchmark dataset or upload your own table, then press **Run Adversarial Multi-Agent Audit**.")
