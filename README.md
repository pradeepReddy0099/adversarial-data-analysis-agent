# Adversarial Data Analysis Agent ⚔️
### Automated Hypothesis Generation and Statistical Falsification in Tabular Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Course: CS 534 WPI](https://img.shields.io/badge/WPI-CS%20534-crimson.svg)](https://www.wpi.edu/)

---

## 📌 Executive Summary
Standard automated data analysis agents (Auto-EDA and LLM data interpreters) operate under a naive "data-fitting" paradigm: they detect surface-level correlations ($r$) and report them as actionable truths. In real-world data, these naive associations frequently mislead decision-makers due to **Simpson’s Paradox** (where an aggregate trend completely reverses across subgroups), unobserved confounding variables, and outlier leverage.

The **Adversarial Data Analysis Agent** replaces uncritical single-pass pipelines with a **dialectical multi-agent system** grounded in Karl Popper’s principle of **scientific falsification**:
1. 🔍 **The Explorer Agent:** Autonomously explores tabular data, extracts empirical patterns, and generates candidate hypotheses.
2. ⚖️ **The Devil's Advocate (Auditor) Agent:** Specifically seeks to **disprove and falsify** the Explorer's claims via automated stress tests:
   - **Simpson's Paradox Falsifier:** Stratifies data across candidate confounders to detect subgroup sign reversals.
   - **Confounder & Partial Correlation Audit:** Measures correlation attenuation when conditioning on third variables.
   - **Bootstrap & Stability Testing:** Resamples data to check if findings collapse under perturbation.
   - **Target Leakage Detection:** Flags trivial identity overlaps with the target label.
3. 🏆 **The Referee Agent:** Arbitrates the debate and issues calibrated confidence ratings:
   - 🟢 **ROBUST & VERIFIED:** Withstood all falsification batteries.
   - 🟡 **SIMPSON'S PARADOX ALERT:** Critical sign reversal discovered across subpopulations.
   - 🔴 **SPURIOUS / CONFOUNDED:** Driven by omitted confounders, outliers, or target leakage.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/pradeepReddy0099/Adversarial-Data-Analysis-Agent.git
cd Adversarial-Data-Analysis-Agent

# Recommended virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Command Line Interface (CLI)
Run an adversarial audit on any CSV or Excel file:
```bash
# Audit the canonical UC Berkeley admissions benchmark:
python cli.py data/berkeley_admissions.csv --target admitted --out output/berkeley

# Audit the commercial churn dataset:
python cli.py data/simpsons_churn.csv --target churn --out output/churn
```

### 3. Interactive Web UI ("Debate Arena")
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 📊 Benchmark Experiments

| Benchmark Dataset | Rows | Global Finding (What Naive Agents See) | Adversarial Audit Verdict | Ground Truth Explanation |
| :--- | :---: | :--- | :---: | :--- |
| **UC Berkeley Admissions (1973)** | 4,526 | Gender bias favoring males (ANOVA $F = 94.08$, $p < 10^{-20}$) | 🟡 **SIMPSON'S PARADOX** | Aggregate trend completely reverses across departments (4 of 6 departments favored female applicants; women applied to more competitive departments). |
| **Simpsons Churn Benchmark** | 900 | Support calls reduce churn ($r = -0.27$, $p < 10^{-15}$) | 🟡 **SIMPSON'S PARADOX** | Within *every individual tier*, more calls increased churn ($r = +0.20$); Startup accounts had high base churn and fewer calls. |
| **Synthetic Income & Education** | 1,000 | Experience strongly predicts income ($r = 0.65$) | 🟢 **ROBUST & VERIFIED** | Association holds consistently across all age and demographic subgroups. |

---

## 🏛️ Project Architecture

```
adversarial-data-analysis-agent/
├── app.py                      # Interactive Streamlit "Debate Arena" UI
├── cli.py                      # Command Line Interface runner
├── requirements.txt            # Project dependencies
├── docs/
│   ├── PROJECT_PROPOSAL.md     # Approved 1-page CS 534 project proposal
│   └── RESEARCH_PAPER_DRAFT.md # 5-page research paper for Beta/Final submission
├── data/
│   ├── berkeley_admissions.csv # Historic 1973 UC Berkeley dataset (4,526 rows)
│   ├── simpsons_churn.csv      # Commercial churn benchmark with planted paradox
│   └── generate_benchmarks.py  # Benchmark data generation scripts
├── src/
│   └── adversarial_agent/
│       ├── __init__.py
│       ├── core.py             # Pipeline orchestration & Claim structures
│       ├── explorer.py         # Autonomous hypothesis discovery
│       ├── auditor.py          # Devil's Advocate falsification batteries
│       ├── referee.py          # Calibrated arbitration & verdict badges
│       ├── profiling.py        # Tabular data profiler
│       └── reporting.py        # HTML and Markdown report generation
└── tests/
    ├── test_explorer.py
    ├── test_auditor.py
    └── test_end_to_end.py
```

---

## 📖 Academic Paper & Course Details
* **Course:** CS 534 — Introduction to Artificial Intelligence (Fall 2026)
* **Institution:** Worcester Polytechnic Institute (WPI)
* **Instructor:** Dr. Dmitry Korkin
* **Documentation:** Full 1-page proposal and 5-page research paper draft are located in [`docs/`](docs/).

## 📜 License
MIT License. Free for academic and research use.
