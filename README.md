# Adversarial Data Analysis Agent ⚔️
### Automated Hypothesis Generation and Statistical Falsification in Tabular Data

[![CI Pipeline](https://github.com/pradeepReddy0099/adversarial-data-analysis-agent/actions/workflows/run-pipeline.yml/badge.svg)](https://github.com/pradeepReddy0099/adversarial-data-analysis-agent/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pradeepReddy0099/adversarial-data-analysis-agent/blob/main/notebooks/ADAA_Interactive_Demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Course: CS 534 WPI](https://img.shields.io/badge/WPI-CS%20534-crimson.svg)](https://www.wpi.edu/)

---

## 📌 Executive Summary
Standard automated data analysis agents (Auto-EDA and LLM code interpreters) operate under an uncritical "data-fitting" paradigm: they detect surface correlations ($r$) and report them as actionable truths. In real-world data, these naive associations frequently mislead decision-makers due to **Simpson’s Paradox** (where an aggregate trend completely reverses across subgroups), unobserved confounding variables, and outlier leverage.

The **Adversarial Data Analysis Agent (ADAA)** replaces single-pass pipelines with a **dialectical multi-agent system** grounded in Karl Popper’s principle of **scientific falsification**:
1. 🔍 **The Explorer Agent:** Autonomously explores tabular data, extracts empirical patterns, and generates candidate hypotheses.
2. ⚖️ **The Devil's Advocate (Auditor) Agent:** Specifically seeks to **disprove and falsify** the Explorer's claims via four automated stress tests:
   - **Simpson's Paradox Falsifier:** Stratifies data across candidate confounders to detect subgroup sign reversals.
   - **Confounder & Partial Correlation Audit:** Measures correlation attenuation when conditioning on third variables.
   - **Bootstrap & Stability Testing:** Resamples data (B = 150) to check if findings collapse under perturbation.
   - **Target Leakage Detection:** Flags trivial identity overlaps with the target label.
3. 🏆 **The Referee Agent:** Arbitrates the debate and issues calibrated confidence ratings:
   - 🟢 **ROBUST & VERIFIED:** Withstood all falsification batteries.
   - 🟡 **SIMPSON'S PARADOX ALERT:** Critical sign reversal discovered across subpopulations.
   - 🔴 **SPURIOUS / CONFOUNDED:** Driven by omitted confounders, outliers, or target leakage.

---

## ⚡ Run Directly in Google Colab (1-Click)

Run the full adversarial analysis and interactive Streamlit UI in the cloud without installing anything on your local computer:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pradeepReddy0099/adversarial-data-analysis-agent/blob/main/notebooks/ADAA_Interactive_Demo.ipynb)

---

## 🚀 Local Installation & Execution

### 1. Clone & Install
```bash
git clone https://github.com/pradeepReddy0099/adversarial-data-analysis-agent.git
cd adversarial-data-analysis-agent

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Test Suite
Confirm all multi-agent stress tests pass:
```bash
python tests/test_falsification.py
```

### 3. Command Line Interface (CLI)
Run an adversarial audit on any dataset:
```bash
# 1. Audit the canonical UC Berkeley admissions benchmark (1973):
python cli.py data/berkeley_admissions.csv --target admitted --out output/berkeley

# 2. Audit the commercial SaaS churn benchmark:
python cli.py data/simpsons_churn.csv --target churn --out output/churn

# 3. Audit the UCI Adult Census income benchmark:
python cli.py data/adult_census.csv --target income_above_50k --out output/adult
```

### 4. Interactive Web UI ("Debate Arena")
Launch the Streamlit interactive dashboard:
```bash
streamlit run app.py
```

---

## 📊 Benchmark Results

| Benchmark Dataset | Rows | Global Finding (What Naive Agents See) | Adversarial Audit Verdict | Ground Truth Reality |
| :--- | :---: | :--- | :---: | :--- |
| **UC Berkeley Admissions (1973)** | 4,526 | Gender bias favoring males (ANOVA $F = 94.08$, $p < 10^{-20}$) | 🟡 **SIMPSON'S PARADOX** | Aggregate trend completely reverses across departments: women had equal or higher admission rates in 4 of 6 departments. |
| **Commercial Churn Benchmark** | 900 | Support calls reduce churn ($r = -0.27$, $p < 10^{-15}$) | 🟡 **SIMPSON'S PARADOX** | Within *every individual tier*, more calls increased churn ($r = +0.20$); Enterprise accounts had high call volume and lower churn. |
| **UCI Adult Census Income** | 1,200 | Working more hours raises income ($r = +0.225$) | 🔴 **CONFOUNDED** | Effect collapses by **61.4%** when controlling for education level; working hours alone is not a reliable lever. |
| **UCI Adult Census Income** | 1,200 | Higher education raises income ($r = +0.373$) | 🟢 **ROBUST & VERIFIED** | Survives all 4 stress tests; robust across all occupations and demographic strata. |

---

## 🏗️ Project Architecture

```text
adversarial-data-analysis-agent/
├── app.py                          # Streamlit "Debate Arena" Interactive UI
├── cli.py                          # Standalone Command-Line Runner
├── requirements.txt                # Core dependencies
├── src/adversarial_agent/
│   ├── explorer.py                 # Explorer Agent (Hypothesis generation)
│   ├── auditor.py                  # Devil's Advocate Agent (4 Falsification Batteries)
│   ├── referee.py                  # Referee Agent (Verdict classification & guidance)
│   ├── core.py                     # Pipeline Orchestrator
│   ├── profiling.py                # Schema profiler
│   └── reporting.py                # HTML, Markdown, and JSON report generator
├── data/
│   ├── berkeley_admissions.csv     # 1973 UC Berkeley Admissions (4,526 records)
│   ├── simpsons_churn.csv          # SaaS Churn with planted paradox (900 records)
│   └── adult_census.csv            # UCI Adult Census Income benchmark (1,200 records)
├── notebooks/
│   └── ADAA_Interactive_Demo.ipynb # 1-Click Interactive Google Colab Notebook
├── tests/
│   └── test_falsification.py       # Full test suite
└── .github/workflows/
    └── run-pipeline.yml            # Automated GitHub Actions test runner
```

---

## 📜 Citation & Academic Submission

This project was developed for **CS 534: Introduction to Artificial Intelligence (Fall 2026)** at **Worcester Polytechnic Institute (WPI)** under **Dr. Dmitry Korkin**.
