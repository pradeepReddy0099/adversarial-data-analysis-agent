# Project Proposal: Adversarial Data Analysis Agent

**Title:** Adversarial Data Analysis Agent: Automated Hypothesis Generation and Statistical Falsification in Tabular Data  
**Course:** CS 534 — Introduction to Artificial Intelligence, Worcester Polytechnic Institute  
**Instructor:** Dr. Dmitry Korkin  
**Team Members:** *[Add Team Member Names & WPI Emails]*  

---

### 1. Problem Description (1 paragraph)
Modern automated data exploration tools and AI-driven data analysis agents primarily operate under an uncritical "data-fitting" paradigm: they detect surface-level statistical correlations (e.g., Pearson’s $r$, feature importance scores) and present them to users as actionable empirical insights. However, in real-world observational tabular data, aggregate statistical trends are frequently spurious or misleading due to **Simpson’s Paradox**—wherein an observed aggregate trend completely reverses or vanishes when conditioned on sub-populations—as well as unobserved confounding variables, outlier leverage, and target leakage. Current automated agents suffer from confirmation bias and sycophancy, uncritically confirming user preconceptions without subjecting candidate hypotheses to rigorous empirical skepticism. This creates a critical reliability gap between automated correlation discovery and valid scientific inference.

### 2. Significance & Importance of the Problem Supported by the Literature (2 paragraphs)
The distinction between raw statistical association and robust empirical evidence is foundational to modern artificial intelligence and data science. In classical statistics and epidemiology, failure to account for Simpson’s Paradox (Simpson, 1951) and confounding bias (Pearl, 2009) has historically produced disastrous policy and medical conclusions. As autonomous AI agents take on larger roles in business analytics, clinical research, and scientific discovery, the risk of automated agents reporting spurious correlations as causal facts grows exponentially. Recent research highlights that Large Language Models and automated data pipelines exhibit significant "sycophancy" and confirmation bias, favoring user-pleasing explanations over statistically calibrated doubt (Sharma et al., 2023; Perez et al., 2022).

To address these vulnerabilities, researchers have begun exploring multi-agent debate and adversarial verification architectures, which pit distinct agents with opposing objectives against each other to minimize hallucinations and factual drift (Du et al., 2023; Liang et al., 2023). Translating this adversarial principle to tabular data analysis operationalizes Karl Popper’s criterion of scientific falsification (Popper, 1959): an empirical hypothesis should only be trusted if it actively withstands concerted, automated attempts to disprove it. Building a data analysis agent capable of adversarial self-auditing directly targets the reproducibility and trustworthiness crisis in automated AI analytics.

### 3. Proposed AI Methodology (1 paragraph)
We propose the **Adversarial Data Analysis Agent**, a dialectical multi-agent system comprising three specialized agents: (1) **The Explorer Agent**, which ingests structured tabular data, computes exploratory statistics, and generates candidate analytical hypotheses and predictive models; (2) **The Devil’s Advocate (Auditor) Agent**, whose objective is to actively falsify the Explorer’s claims by running an automated battery of statistical stress tests—including subgroup stratification (searching for Simpson's Paradox reversals), partial-correlation confounder checks, and bootstrap outlier sensitivity tests; and (3) **The Referee Agent**, which evaluates the strength of the counter-evidence to classify findings into calibrated credibility tiers (*Robust Finding*, *Subgroup-Dependent*, or *Spurious/Confounded*). This multi-agent game-theoretic formulation moves beyond standard single-pass pipelines into dynamic, self-correcting scientific reasoning.

### 4. Data Sources (2 paragraphs)
To benchmark and validate our agent against both genuine insights and known statistical illusions, we evaluate the system across three standard real-world datasets and one controlled synthetic benchmark:
1. **The UC Berkeley Admissions Benchmark (Bickel et al., 1975):** The canonical historic dataset demonstrating Simpson’s Paradox, where an apparent aggregate gender bias reverses when conditioned on departmental admission rates.
2. **IBM Telco Customer Churn Dataset (Kaggle):** A high-dimensional real-world commercial dataset containing 7,043 customer records across contractual, demographic, and behavioral features, providing rich opportunities for confounding and subgroup analysis.
3. **UCI Adult Census Income Dataset (Kohavi, 1996):** A benchmark containing 48,842 instances with complex inter-feature dependencies, used to test the agent’s robustness against multicollinearity and subgroup variance.
4. **Controlled Synthetic Benchmark:** A parameterized synthetic dataset with injected, known confounders and deliberate Simpson’s Paradox flips, serving as our mathematical ground truth to quantitatively evaluate the agent's precision and recall in detecting spurious relationships.

---

### Key References
* Bickel, P. J., Hammel, E. A., & O'Connell, J. W. (1975). Sex bias in graduate admissions: Data from Berkeley. *Science*, 187(4175), 398-404.
* Du, Y., et al. (2023). Improving factuality and reasoning in language models through multiagent debate. *arXiv preprint arXiv:2305.14325*.
* Liang, T., et al. (2023). Encouraging divergent thinking in large language models through multi-agent debate. *arXiv preprint arXiv:2305.19118*.
* Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
* Perez, E., et al. (2022). Discovering language model behaviors with model-written evaluations. *arXiv preprint arXiv:2212.09251*.
* Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.
* Sharma, M., et al. (2023). Towards understanding sycophancy in language models. *arXiv preprint arXiv:2310.13548*.
* Simpson, E. H. (1951). The interpretation of interaction in contingency tables. *Journal of the Royal Statistical Society: Series B*, 13(2), 238-241.
