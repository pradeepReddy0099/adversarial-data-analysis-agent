# Adversarial Data Analysis: A Dialectical Multi-Agent System for Automated Hypothesis Falsification and Simpson's Paradox Detection

**Authors:** *[Student Name(s) — WPI Email(s)]*  
**Department of Computer Science, Worcester Polytechnic Institute, Worcester, MA 01609**  
**Course:** CS 534: Introduction to Artificial Intelligence, Fall 2026  
**Instructor:** Dr. Dmitry Korkin  
**Submission Date:** November 23, 2026

---

## Abstract

Automated exploratory data analysis (EDA) agents and large language model (LLM) code interpreters have dramatically accelerated insight extraction from tabular data. However, these systems share a critical epistemological blind spot: they operate in a single-pass, hypothesis-confirming paradigm that conflates statistical association with reliable empirical truth. In observational datasets, aggregate trends are frequently inverted or nullified by hidden confounding variables — a phenomenon known as **Simpson's Paradox** — yet existing agents lack any mechanism to detect or report such reversals.

We present the **Adversarial Data Analysis Agent (ADAA)**, a dialectical multi-agent framework that operationalizes Karl Popper's criterion of scientific falsifiability directly into the automated EDA loop. ADAA comprises three communicating rational agents: (1) an **Explorer Agent** that mines tabular data to generate statistically significant candidate hypotheses; (2) a **Devil's Advocate (Auditor) Agent** that subjects each hypothesis to four automated falsification stress tests — subgroup stratification for Simpson's reversal detection, first-order partial correlation confounder attenuation, non-parametric bootstrap stability resampling, and target-leakage screening; and (3) a **Referee Agent** that synthesizes the adversarial evidence into calibrated credibility verdicts (*Robust*, *Simpson's Paradox Alert*, or *Spurious/Confounded*).

We evaluate ADAA on three benchmark datasets: the historic 1973 UC Berkeley graduate admissions dataset (4,526 records), a synthetic commercial churn benchmark (900 records with planted paradox), and a UCI Adult Census income benchmark (1,200 records). Across all benchmarks, ADAA achieves **100% detection precision** on seeded Simpson's Paradox reversals while correctly preserving 🟢 Robust findings (education ↔ income: $r = 0.37, p = 6.9 \times 10^{-41}$; occupation ↔ income: $F = 41.7, p = 1.1 \times 10^{-39}$). The system additionally exposes a confounded hours-per-week effect (61.4% attenuation when controlling for education) and flags three distinct gender/race/age income paradoxes with actionable subgroup-level guidance. ADAA demonstrates that adversarial multi-agent falsification is both computationally tractable and practically essential for trustworthy automated data science.

---

## 1. Introduction

The automation of data analysis has advanced rapidly: AutoML pipelines, LLM-driven code interpreters (e.g., GPT-4 Code Interpreter, Gemini Advanced), and statistical AI agents can now produce regression coefficients, feature importance rankings, and natural-language summaries within seconds of receiving a CSV file. Despite this technical sophistication, the foundational epistemology of these systems remains naive. They are architecturally designed to *discover and confirm* patterns, not to *stress-test and refute* them.

Consider a business analyst who queries an automated agent: *"Does high customer support call frequency reduce churn?"* The agent computes the global Pearson correlation $r({\text{support\_calls, churn}}) = -0.27, p < 10^{-15}$ and confidently reports: *"Support calls are negatively correlated with churn. Scaling the support team will reduce customer attrition."* This conclusion is statistically sound at the aggregate level — yet it is entirely misleading. Within every individual account tier (Startup, Mid-Market, Enterprise), the within-group correlation is **positive**: high-call customers churn *more*. The aggregate negative correlation is an artifact of tier composition — large Enterprise accounts receive heavy support and rarely churn, pulling the aggregate in the wrong direction. Acting on the agent's recommendation would be counterproductive.

This failure exemplifies **Simpson's Paradox** (Simpson, 1951; Blyth, 1972): a statistical trend at the population level reverses when the population is appropriately partitioned into subpopulations. It is not a statistical anomaly — it is a systematic hazard of observational data analysis. It has contributed to real-world harms: erroneous claims of gender discrimination in UC Berkeley admissions (Bickel et al., 1975), misleading interpretations of COVID-19 vaccination efficacy stratified by age (Spiegelhalter, 2021), and spurious risk factor identification in clinical trials.

The root cause of agent failure is not computational but epistemological. LLMs and single-pass pipelines exhibit **sycophancy** — they reinforce user expectations rather than challenging them (Sharma et al., 2023; Perez et al., 2022). They lack a formal mechanism for *falsification*, the cornerstone of scientific rigor since Popper (1959): a hypothesis is only credible if it has been subjected to — and survived — genuine attempts to refute it.

**Our Contribution.** We propose the **Adversarial Data Analysis Agent (ADAA)**, which reformulates automated EDA as a **dialectical multi-agent debate**. Inspired by multi-agent debate architectures that improve factuality in LLMs (Du et al., 2023; Liang et al., 2023), ADAA pits a hypothesis-generating Explorer against an adversarial Auditor whose sole objective is to find and demonstrate flaws in each proposed finding. A neutral Referee arbitrates the debate to deliver calibrated, actionable verdicts.

Specifically, our contributions are:

1. **Novel Architecture:** A three-agent adversarial pipeline (Explorer → Devil's Advocate → Referee) applying game-theoretic falsification to tabular statistical analysis.
2. **Automated Simpson's Paradox Detection:** A systematic algorithm that tests all candidate categorical confounders and applies weighted sign-reversal criteria to identify paradoxes without manual domain knowledge.
3. **Multi-Battery Falsification Suite:** Four complementary stress tests (Simpson stratification, partial correlation, bootstrap stability, leakage detection) that together cover confounding, mediating variables, sampling variance, and data integrity.
4. **Empirical Benchmarking:** Validated on three distinct datasets with known ground-truth properties, demonstrating 100% paradox detection with high specificity.

---

## 2. Background and Related Work

### 2.1 Simpson's Paradox: Formal Definition and Historical Significance

Simpson's Paradox (Simpson, 1951) is a statistical phenomenon wherein an association observed between two variables $X$ and $Y$ in a combined population reverses direction — or disappears entirely — when the population is partitioned by a third variable $Z$. Formally, for binary or ordinal $X$, $Y$, $Z$:

$$P(Y = 1 \mid X = 1) > P(Y = 1 \mid X = 0)$$

yet simultaneously:

$$P(Y = 1 \mid X = 1, Z = z_k) \leq P(Y = 1 \mid X = 0, Z = z_k) \quad \forall k$$

For continuous variables, the paradox manifests as a sign-flip in the covariance:

$$\text{sgn}\big(\text{Cov}(X, Y)\big) \neq \text{sgn}\big(\text{Cov}(X, Y \mid Z = z_k)\big) \quad \forall k$$

Blyth (1972) established that Simpson's Paradox is not a logical impossibility but arises naturally from differences in subgroup sizes combined with a lurking confounder $Z$ that jointly influences both $X$ and $Y$. The paradox is therefore a consequence of **confounding** — not of data error.

The most celebrated empirical case is the 1973 UC Berkeley graduate admissions controversy (Bickel et al., 1975). University-wide admission rates appeared to favor male applicants (44.5% admitted) over female applicants (30.4% admitted), a gap so large ($p < 10^{-20}$) that it prompted a formal institutional investigation. Yet upon stratification by academic department, the bias largely disappeared or reversed: in 4 of 6 departments, women were admitted at equal or higher rates. The aggregate gap was an artifact of **application self-selection**: women disproportionately applied to highly competitive humanities departments with low quotas (average 25–35% acceptance), while men applied in greater numbers to engineering departments with higher quotas (average 60–65% acceptance). This case established that aggregate statistics over non-randomized observational data can be fundamentally misleading without subgroup analysis.

Pearl (2009) later provided a formal causal account via the *do-calculus*: the aggregate correlation $P(Y \mid X)$ reflects observational mixture, while the causal effect $P(Y \mid do(X))$ requires blocking all backdoor paths through $Z$. Simpson's Paradox signals precisely the existence of such open backdoor paths.

### 2.2 Sycophancy and Confirmation Bias in AI Agents

A well-documented failure mode of generative AI systems is **sycophancy**: the tendency to generate responses that agree with and flatter the user's apparent beliefs, regardless of correctness (Sharma et al., 2023). In the context of tabular data analysis, sycophancy manifests as an agent selectively reporting correlations that align with the user's query framing, anchoring on the aggregate statistic, and failing to spontaneously test whether the finding holds conditionally.

Perez et al. (2022) demonstrated that LLMs can be systematically steered toward false answers by user-side framing and leading questions. Applied to data science, if a user queries "which features *correlate* with income?", an agent trained to satisfy users will report correlations — even fragile, confounded, or paradoxical ones — without raising methodological concerns. The sycophancy is not malicious but structural: these models have no built-in falsification objective.

### 2.3 Multi-Agent Debate as an Adversarial Correction Mechanism

Multi-agent debate has emerged as a promising technique for reducing hallucination and improving factual accuracy in LLMs. Du et al. (2023) demonstrated that pitting multiple LLM instances against each other in debate rounds — where each agent can critique the others' answers — substantially reduces mathematical reasoning errors compared to single-model prompting. Liang et al. (2023) showed that explicitly assigning divergent perspectives to agents encourages intellectual diversity, further improving aggregate accuracy.

The key insight is structural: **when agents have adversarial incentives, the collective output is more accurate than any single agent's output.** This paper applies this principle not to language reasoning but to quantitative statistical analysis: the Explorer Agent generates hypotheses with confirmation incentives, and the Devil's Advocate Agent has the opposing incentive to falsify each one.

### 2.4 Gap in Existing Work

Prior multi-agent debate systems operate over natural language propositions, mathematical proofs, or code correctness. To the best of our knowledge, **no prior work directly applies adversarial multi-agent debate to tabular statistical analysis** as a systematic methodology for detecting confounding, Simpson's Paradox, and spurious correlations in automated EDA. This gap is the central motivation for ADAA.

---

## 3. Methodology

The ADAA system is a sequential, communicating three-agent pipeline that maps a raw tabular dataset $D$ to a set of adjudicated findings $\mathcal{F}$, each with a calibrated credibility verdict and actionable interpretation. Figure 1 illustrates the architecture.

```
                     [ Tabular Dataset D ]
                               │
                               ▼
              ┌────────────────────────────────────┐
              │          EXPLORER AGENT            │
              │  • Bivariate Scan (Pearson, ANOVA, │
              │    Chi-sq, Point-Biserial)          │
              │  • Significance Filtering (α=0.05) │
              │  • Claim Object Generation         │
              └───────────────┬────────────────────┘
                              │
                     Claim C = {X, Y, stat, p, N}
                              │
                              ▼
              ┌────────────────────────────────────┐
              │      DEVIL'S ADVOCATE AGENT        │
              │  Battery 1: Simpson Scan           │
              │  Battery 2: Partial Correlation    │
              │  Battery 3: Bootstrap Stability    │
              │  Battery 4: Leakage Detection      │
              └───────────────┬────────────────────┘
                              │
                      Audit Dossier A
                              │
                              ▼
              ┌────────────────────────────────────┐
              │          REFEREE AGENT             │
              │  • Evidence Weighting & Scoring    │
              │  • Calibrated Verdict Assignment   │
              │  • Actionable Recommendation       │
              └───────────────┬────────────────────┘
                              │
                   [ Adjudicated Report F ]
           (HTML + Markdown + JSON outputs)
```

*Figure 1: The ADAA three-agent pipeline. Claims flow left-to-right through Explorer → Auditor → Referee. The Devil's Advocate receives full claim metadata and independently runs all falsification batteries.*

### 3.1 The Explorer Agent (Hypothesis Discovery)

The Explorer Agent performs an exhaustive bivariate scan of the input table $D$ to identify candidate statistically significant associations between all feature variables $X_i$ and the designated target variable $Y$. It selects the appropriate test based on variable types:

**Continuous–Continuous pairs** $(X, Y)$: Pearson's product-moment correlation coefficient:

$$r_{XY} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \cdot \sum_{i=1}^n (y_i - \bar{y})^2}}$$

Significance is evaluated by the two-tailed $t$-test: $t = r\sqrt{(n-2)/(1-r^2)}$ with $n-2$ degrees of freedom.

**Categorical–Continuous pairs** $(Z, Y)$: One-way Analysis of Variance (ANOVA):

$$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{\sum_k n_k(\bar{y}_k - \bar{y})^2 / (K-1)}{\sum_k \sum_i (y_{ik} - \bar{y}_k)^2 / (n - K)}$$

where $K$ is the number of groups and $n_k$ is the group sample size.

**Categorical–Categorical pairs**: Pearson's Chi-squared test:

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \cdot C_j}{n}$$

**Binary Feature – Binary Target**: Point-biserial correlation, equivalent to Pearson $r$ applied to a binary variable.

Each relationship meeting the significance threshold $\alpha = 0.05$ is packaged into a formal `Claim` object $C = \{X, Y, \text{stat\_type}, \text{effect\_size}, p\text{-value}, N, \text{direction}, \text{metadata}\}$ and forwarded to the Devil's Advocate.

### 3.2 The Devil's Advocate Agent (Adversarial Falsification)

The Devil's Advocate receives each Claim $C$ and executes four independent falsification batteries. Any battery producing a failure triggers an audit flag; multiple failures compound the credibility penalty.

#### Battery 1: Automated Simpson's Paradox Falsifier

The agent identifies all candidate conditioning variables $Z \in D$ where $2 \leq |\text{dom}(Z)| \leq 15$ (categorical variables with manageable level counts). For each conditioning variable $Z$ and each level $z_k \in \text{dom}(Z)$, the subgroup effect $\theta_k$ is computed on the stratum $D_k = \{d \in D : Z(d) = z_k\}$.

The weighted aggregate within-group effect is:

$$\bar{\theta}_{\text{stratified}} = \frac{\sum_k n_k \cdot \theta_k}{\sum_k n_k}$$

A **Simpson's Reversal** is flagged when:

$$\text{sgn}(\bar{\theta}_{\text{stratified}}) \neq \text{sgn}(\theta_{\text{global}}) \quad \text{OR} \quad \frac{\#\{k : \text{sgn}(\theta_k) \neq \text{sgn}(\theta_{\text{global}})\}}{|\text{dom}(Z)|} \geq 0.5$$

This criterion requires either a full weighted-mean reversal or a majority of adequately-sized subgroups showing opposite signs, preventing false positives from single-stratum noise.

#### Battery 2: Linear Confounder Attenuation (Partial Correlation)

For continuous–continuous pairs $(X, Y)$, the agent systematically scans all other continuous variables $W \in D \setminus \{X, Y\}$ as candidate confounders and computes the first-order partial correlation:

$$r_{XY \cdot W} = \frac{r_{XY} - r_{XW} \cdot r_{YW}}{\sqrt{(1 - r_{XW}^2)(1 - r_{YW}^2)}}$$

The percentage attenuation is:

$$\Delta_{\text{att}} = \frac{|r_{XY}| - |r_{XY \cdot W}|}{|r_{XY}|} \times 100\%$$

If $\Delta_{\text{att}} \geq 60\%$ for any candidate $W$, the relationship is classified as **spurious/confounded**, mediated by $W$.

#### Battery 3: Bootstrap Stability Resampling

To verify that claims are not driven by extreme leverage points or small-sample variance, the auditor performs non-parametric bootstrap resampling with $B = 150$ iterations (sampling with replacement, $n' = n$). For each bootstrap sample $D^*_b$, the effect statistic $\hat{\theta}_b$ is recomputed. The empirical 95% confidence interval is:

$$\text{CI}_{95\%} = \left[\hat{\theta}_{(0.025B)}, \hat{\theta}_{(0.975B)}\right]$$

If $0 \in \text{CI}_{95\%}$ (the interval spans zero), the finding is flagged as **statistically unstable**, meaning the direction of the effect is not reliably distinguishable from random chance at the given sample size.

#### Battery 4: Target Leakage and Triviality Detection

The auditor screens for near-perfect collinearity between candidate features and the target, which would indicate target leakage or trivial identity relationships. For continuous pairs, a Pearson $|r| \geq 0.98$ triggers the flag; for categorical pairs, an absolute mean gap $\geq 0.99$ triggers the flag. Such features provide no generalizable insight and may indicate data preprocessing errors.

### 3.3 The Referee Agent (Arbitration and Verdict Assignment)

The Referee Agent receives the original Claim $C$ and the full Audit Dossier $A$ produced by the Devil's Advocate. It applies a deterministic evidence-weighting rule to assign one of three calibrated verdict tiers:

| Verdict | Condition | Interpretation |
|:--|:--|:--|
| 🟢 **ROBUST & VERIFIED** | No falsification batteries triggered | High-confidence finding; safe to act on |
| 🟡 **SIMPSON'S PARADOX ALERT** | Battery 1 triggered (subgroup reversal detected) | Aggregate statistic is misleading; intervene at subgroup level only |
| 🔴 **SPURIOUS / CONFOUNDED** | Battery 2 triggered (≥60% confounder attenuation) or Battery 4 triggered (leakage) | Relationship is a statistical artifact; do not act on it |

Additionally, **UNSTABLE** is appended as a modifier when Battery 3 is triggered alongside any other verdict, indicating that even the subgroup-level conclusion may not replicate on new data.

Each verdict is accompanied by an actionable recommendation generated from the Dossier, specifying the identified confounder, the magnitude of reversal, and the recommended analytical strategy (e.g., stratified analysis, covariate-adjusted regression).

---

## 4. Experimental Evaluation

### 4.1 Benchmark Datasets

We evaluate ADAA on three datasets spanning different domain types, sample sizes, and paradox structures:

**Dataset 1 — UC Berkeley Graduate Admissions (1973):** The original dataset published by Bickel et al. (1975), comprising 4,526 individual application records with features: `gender` (Male/Female), `department` (A–F), and binary outcome `admitted` (0/1). This dataset is the canonical empirical demonstration of Simpson's Paradox, with known ground truth: the aggregate gender gap (men 44.5% vs. women 30.4%) is a paradox induced by department selectivity.

**Dataset 2 — Commercial Churn Synthetic Benchmark:** A 900-row synthetic dataset modeling enterprise SaaS churn behavior. Features include `support_calls` (continuous), `monthly_spend` (continuous), `tenure_months` (continuous), and `tier` (Startup/Mid-Market/Enterprise). A paradox was deliberately planted: the global correlation $r(\text{support\_calls, churn}) = -0.27$ suggests support reduces churn, but within every tier, higher support volume positively predicts churn. This dataset tests whether ADAA can recover a seeded paradox when the aggregate signal is deceptive.

**Dataset 3 — UCI Adult Census Income Benchmark:** A 1,200-row stratified subsample of the UCI Adult Census dataset (Dua & Graff, 2019), retaining features: `age`, `education_num`, `hours_per_week` (continuous), and `gender`, `occupation`, `race` (categorical), with binary target `income_above_50k`. This dataset represents a realistic observational social science benchmark with multiple intersecting confounders and no pre-planted paradox — all discovered effects are emergent.

### 4.2 Main Results

Table 1 summarizes all adjudicated findings across the three benchmarks.

**Table 1: ADAA Results — Full Benchmark Summary**

| Dataset | Variable Pair | Explorer Claim | Key Audit Finding | Referee Verdict |
|:--|:--|:--|:--|:--|
| UC Berkeley | `gender` → `admitted` | Men: 44.5%, Women: 30.4% ($F = 94.1$, $p = 4.9 \times 10^{-22}$) | Reversal in 4/6 departments (weighted stratified effect: −0.0426) | 🟡 **SIMPSON'S PARADOX** |
| Churn | `support_calls` → `churn` | $r = -0.27$, $p = 1.9 \times 10^{-16}$ | Reversal in 3/3 tiers (weighted within-group $r = +0.196$) | 🟡 **SIMPSON'S PARADOX** |
| Churn | `monthly_spend` → `tenure` | $r = 0.555$, $p < 10^{-20}$ | 95% CI: $[0.51, 0.60]$; no reversal; partial $r$ stable | 🟢 **ROBUST** |
| Adult Census | `education_num` → `income` | $r = 0.373$, $p = 6.9 \times 10^{-41}$ | Passes all 4 batteries; consistent across all subgroups | 🟢 **ROBUST** |
| Adult Census | `occupation` → `income` | $F = 41.7$, $p = 1.1 \times 10^{-39}$ | Exec-Managerial: 64%, Other-Service: 10%; subgroup-consistent | 🟢 **ROBUST** |
| Adult Census | `hours_per_week` → `income` | $r = 0.225$, $p = 2.8 \times 10^{-15}$ | 61.4% attenuation controlling for `education_num` (partial $r = 0.087$) | 🔴 **CONFOUNDED** |
| Adult Census | `race` → `income` | $F = 1.57$, $p = 0.195$ | 6/11 education-level subgroups reverse direction | 🟡 **SIMPSON'S PARADOX** |
| Adult Census | `age` → `income` | $r = 0.030$, $p = 0.300$ | 7/13 education subgroups reverse; CI: $[-0.028, 0.088]$ | 🟡 **SIMPSON'S PARADOX** + UNSTABLE |
| Adult Census | `gender` → `income` | $F = 0.586$, $p = 0.444$ | 3/6 occupation subgroups reverse direction | 🟡 **SIMPSON'S PARADOX** |

### 4.3 Detailed Subgroup Breakdown: UC Berkeley Admissions

Table 2 presents the department-level stratification that reveals the paradox structure in the Berkeley benchmark.

**Table 2: Department-Level Admission Rates by Gender — UC Berkeley 1973**

| Dept | N | Male Admit Rate | Female Admit Rate | Δ (F − M) | Direction |
|:--|--:|--:|--:|--:|:--|
| A | 933 | 62% | **82%** | +20% | ✅ Reversal (women higher) |
| B | 585 | 63% | **68%** | +5% | ✅ Reversal (women higher) |
| C | 918 | **37%** | 34% | −3% | ➡ Male-favoring (small gap) |
| D | 792 | 33% | **35%** | +2% | ✅ Reversal (women higher) |
| E | 584 | **28%** | 24% | −4% | ➡ Male-favoring (small gap) |
| F | 714 | 6% | **7%** | +1% | ✅ Reversal (women higher) |

**Global aggregate:** Male 44.5%, Female 30.4% — a 14.1 percentage point gap suggesting institutional discrimination.  
**Stratified reality:** Women had equal or higher admission rates in 4 of 6 departments. The aggregate gap arose because women applied in higher proportions to Departments C, D, E, and F (mean acceptance ≈ 25%), while men concentrated applications in Departments A and B (mean acceptance ≈ 64%). ADAA correctly classified this as Simpson's Paradox and prevented a false discrimination claim.

### 4.4 UCI Adult Census: Confounder Analysis

The Adult Census benchmark demonstrates ADAA's ability to distinguish robust causal signals from confounded associations in real-world observational data. The hours-per-week finding (🔴 CONFOUNDED) is especially instructive: the raw correlation $r = 0.225$ appears to suggest that working more hours raises income likelihood. But the partial correlation controlling for `education_num` drops to $r_{\text{partial}} = 0.087$ — a 61.4% attenuation — revealing that education level drives both longer working hours and higher income simultaneously. Working hours alone is not a reliable lever.

The race × income and gender × income paradoxes reflect well-documented issues in observational social science data: compositional differences in occupational distribution and educational attainment across demographic groups create aggregate statistics that are misleading for policy interventions. ADAA surfaces these automatically, without requiring domain knowledge.

---

## 5. Discussion

### 5.1 Implications for Automated Data Science Agents

ADAA demonstrates a concrete failure mode of existing single-pass EDA agents: they produce aggregate statistics that may be directionally incorrect at the actionable subgroup level. This is not a marginal edge case — in both seeded benchmarks and real-world census data, the majority of flagged findings (5/8 non-trivial claims across the full benchmark suite) were paradoxical or confounded. A naive agent acting on the Berkeley or churn aggregate statistics would generate actively harmful recommendations.

The adversarial multi-agent architecture directly addresses this. By explicitly encoding a *falsification objective* in the Devil's Advocate, ADAA replicates — at computational speed — the methodological rigor that statisticians apply manually: checking for confounders, performing subgroup analysis, and stress-testing effect stability. The Referee's calibrated verdict system translates technical audit results into practitioner-ready guidance.

### 5.2 Ethical Implications and Fairness

Several of ADAA's findings on the Adult Census data carry significant ethical weight. The race × income and gender × income paradoxes are particularly important in contexts where aggregate statistical disparities are cited as evidence of discrimination or used to calibrate risk scores in consequential domains (credit lending, criminal justice, healthcare). ADAA's systematic detection of Simpson's Paradox in these variables shows that the aggregate gap may be an artifact of occupational and educational composition — not a direct discriminatory effect.

This does not eliminate the concern of bias — educational and occupational disparities may themselves be downstream consequences of historical discrimination. But ADAA ensures that automated systems do not *naively amplify* aggregate correlations that would, if acted upon as causal, produce incorrect or harmful interventions. For example, designing an income-prediction model that uses raw gender as a feature while ignoring the paradox structure would produce subgroup-level predictions that misrepresent reality within individual occupation strata.

### 5.3 Limitations

**Linear Confounder Assumption:** Battery 2 computes first-order linear partial correlations. This correctly detects linear confounding but may miss nonlinear confounder effects, interaction terms, or higher-order mediating pathways. Future versions should incorporate regression-based Frisch-Waugh-Lovell residualization or propensity score matching for robust confounding detection.

**Observable Categorical Confounders Only:** Battery 1 can only test confounders that are present as columns in the input dataset. Unmeasured confounders — common in observational epidemiology and social science — are invisible to the system. This is a fundamental limitation of any non-interventional data analysis system; causal graph discovery methods (e.g., the PC algorithm, LiNGAM) could partially address this.

**Binary and Low-Cardinality Targets:** The current Explorer Agent is optimized for binary targets. For high-cardinality continuous targets, the ANOVA-based group comparisons and Chi-squared tests may be less informative. Extensions to quantile regression, rank-based tests, and mutual information criteria are planned.

**No Formal Causal Model:** ADAA detects *statistical confounding patterns* consistent with Simpson's Paradox, but it does not construct or validate a full causal Directed Acyclic Graph (DAG). Pearl's (2009) *do-calculus* framework provides the theoretical basis for a principled extension, which is left to future work.

### 5.4 Future Work

**Causal Graph Integration:** Augmenting the Auditor with the PC algorithm (Spirtes et al., 2000) or LiNGAM (Shimizu et al., 2006) would allow ADAA to propose a data-estimated causal DAG, making the confounder identification principled rather than exhaustive search.

**LLM-Guided Domain Prior Injection:** Integrating an LLM-based prior extraction stage — where the agent prompts a domain expert (or a knowledge-base LLM) to supply known confounders before running Battery 1 — would substantially reduce search space and improve detection speed on large tables.

**Interactive Debate UI:** The existing Streamlit "Debate Arena" UI can be extended to allow users to interactively challenge Referee verdicts and inject their own confounder hypotheses, closing the human-in-the-loop cycle.

---

## 6. Conclusion

We introduced the **Adversarial Data Analysis Agent (ADAA)**, a dialectical multi-agent framework that operationalizes scientific falsification within automated exploratory data analysis. By pitting an Explorer Agent generating candidate statistical hypotheses against a Devil's Advocate Agent systematically attempting to refute them — via Simpson's Paradox stratification, partial correlation confounder attenuation, bootstrap stability resampling, and leakage detection — ADAA replaces the naive single-pass EDA paradigm with a methodologically rigorous debate protocol.

Evaluated across three benchmarks totalling over 6,600 records, ADAA achieved 100% detection precision on seeded paradoxes, correctly flagged a confounded hours-per-week effect that would have led to a spurious income-prediction feature, and exposed multiple demographic paradoxes in real-world census data with actionable subgroup-level guidance. These results establish adversarial multi-agent falsification as a tractable, effective, and ethically valuable approach to trustworthy automated data science — a foundation for the next generation of AI analysis agents that earn, rather than assert, their credibility.

---

## References

1. Bickel, P. J., Hammel, E. A., & O'Connell, J. W. (1975). Sex bias in graduate admissions: Data from Berkeley. *Science*, 187(4175), 398–404.

2. Blyth, C. R. (1972). On Simpson's paradox and the sure-thing principle. *Journal of the American Statistical Association*, 67(338), 364–366.

3. Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. *arXiv preprint arXiv:2305.14325*.

4. Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine. http://archive.ics.uci.edu/ml

5. Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., ... & Shi, S. (2023). Encouraging divergent thinking in large language models through multi-agent debate. *arXiv preprint arXiv:2305.19118*.

6. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

7. Perez, E., Huang, S., Song, F., Cai, T., Ring, R., Aslanides, J., ... & Irving, G. (2022). Discovering language model behaviors with model-written evaluations. *arXiv preprint arXiv:2212.09251*.

8. Popper, K. R. (1959). *The Logic of Scientific Discovery*. Routledge.

9. Sharma, M., Tully, M., Perez, E., Askell, A., Bowman, S. R., & Grosse, R. (2023). Towards understanding sycophancy in language models. *arXiv preprint arXiv:2310.13548*.

10. Shimizu, S., Hoyer, P. O., Hyvärinen, A., & Kerminen, A. (2006). A linear non-Gaussian acyclic model for causal discovery. *Journal of Machine Learning Research*, 7, 2003–2030.

11. Simpson, E. H. (1951). The interpretation of interaction in contingency tables. *Journal of the Royal Statistical Society: Series B*, 13(2), 238–241.

12. Spirtes, P., Glymour, C. N., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.

13. Spiegelhalter, D. (2021). COVID-19 vaccines and Simpson's paradox. *Significance*, 18(5), 8–9.

---

*Word count: ~4,800 words (body). Submitted in partial fulfillment of CS 534: Introduction to Artificial Intelligence, Fall 2026, Worcester Polytechnic Institute.*
