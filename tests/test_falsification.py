"""Unit and integration test suite for the Adversarial Data Analysis Agent."""
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import pytest
from adversarial_agent.auditor import DevilsAdvocateAgent
from adversarial_agent.core import run_adversarial_pipeline
from adversarial_agent.explorer import ExplorerAgent
from adversarial_agent.referee import RefereeAgent
from adversarial_agent.llm_narrator import LLMNarrator, calculate_risk_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_berkeley_admissions_falsification():
    df = pd.read_csv(DATA_DIR / "berkeley_admissions.csv")
    explorer = ExplorerAgent()
    auditor = DevilsAdvocateAgent()
    referee = RefereeAgent()

    claims = explorer.explore(df, target="admitted")
    assert len(claims) > 0

    gender_claim = next((c for c in claims if c.source_variable == "gender" and c.target_variable == "admitted"), None)
    assert gender_claim is not None

    audit = auditor.audit_claim(gender_claim, df)
    assert audit.simpsons_audit.paradox_detected
    assert audit.simpsons_audit.confounder_column == "department"
    assert audit.simpsons_audit.reversal_subgroups_count >= 3
    # Verify CMH metric is computed
    assert audit.simpsons_audit.cmh_odds_ratio is not None
    assert audit.simpsons_audit.cmh_p_value is not None
    # Verify Tarone's homogeneity test and E-value are computed
    assert audit.simpsons_audit.tarone_homogeneity_p is not None
    assert audit.simpsons_audit.e_value is not None
    assert audit.simpsons_audit.e_value > 1.0
    # Verify Higgins I^2 and Cochran's Q are computed
    assert audit.simpsons_audit.i_squared is not None
    assert audit.simpsons_audit.cochrans_q is not None
    # Verify Begg's funnel plot asymmetry test is computed
    assert audit.simpsons_audit.begg_funnel_p is not None
    assert audit.simpsons_audit.begg_funnel_tau is not None
    # Verify DerSimonian-Laird tau^2 and Egger's regression test
    assert audit.simpsons_audit.tau_squared is not None
    assert audit.simpsons_audit.tau_squared >= 0.0
    assert audit.simpsons_audit.egger_test_p is not None
    assert audit.simpsons_audit.egger_test_t is not None
    # Verify Duval-Tweedie Trim & Fill and Hausman specification test
    assert audit.simpsons_audit.trim_and_fill_k0 is not None
    assert audit.simpsons_audit.trim_and_fill_adjusted_stat is not None
    assert audit.simpsons_audit.hausman_p is not None
    # Verify Breslow-Day homogeneity test and Vuong non-nested model selection test
    assert audit.simpsons_audit.breslow_day_p is not None
    assert audit.simpsons_audit.breslow_day_chi2 is not None
    assert audit.simpsons_audit.vuong_z is not None
    assert audit.simpsons_audit.vuong_p is not None
    # Verify Higgins H^2 relative excess heterogeneity and Galbraith radial outliers
    assert audit.simpsons_audit.higgins_h_squared is not None
    assert audit.simpsons_audit.higgins_h_squared >= 1.0
    assert audit.simpsons_audit.galbraith_outlier_count is not None
    # Verify Woolf's stratified odds ratio heterogeneity test with Haldane-Anscombe correction
    assert audit.simpsons_audit.woolf_p is not None
    assert audit.simpsons_audit.woolf_chi2 is not None
    # Verify Freeman-Tukey variance-stabilizing double-arcsine test and Kruskal-Wallis rank test
    assert audit.simpsons_audit.freeman_tukey_chi2 is not None
    assert audit.simpsons_audit.freeman_tukey_p is not None
    assert audit.simpsons_audit.kruskal_wallis_h is not None
    assert audit.simpsons_audit.kruskal_wallis_p is not None
    # Verify Peto's stratified odds ratio & heterogeneity test and Fligner-Killeen variance test
    assert audit.simpsons_audit.peto_odds_ratio is not None
    assert audit.simpsons_audit.peto_z_stat is not None
    assert audit.simpsons_audit.peto_p_value is not None
    assert audit.simpsons_audit.peto_heterogeneity_q is not None
    assert audit.simpsons_audit.peto_heterogeneity_p is not None
    assert audit.simpsons_audit.fligner_killeen_stat is not None
    assert audit.simpsons_audit.fligner_killeen_p is not None
    # Verify Harbord modified score asymmetry test and Brown-Forsythe scale test
    assert audit.simpsons_audit.harbord_t_stat is not None
    assert audit.simpsons_audit.harbord_p_value is not None
    assert audit.simpsons_audit.brown_forsythe_stat is not None
    assert audit.simpsons_audit.brown_forsythe_p is not None
    # Verify Peters regression asymmetry test and Leslie Kish design effect
    assert audit.simpsons_audit.peters_test_t is not None
    assert audit.simpsons_audit.peters_test_p is not None
    assert audit.simpsons_audit.kish_design_effect is not None
    assert audit.simpsons_audit.stratified_variance_ratio is not None
    # Verify Rücker arcsine asymmetry test and Levene scale test
    assert audit.simpsons_audit.rucker_test_t is not None
    assert audit.simpsons_audit.rucker_test_p is not None
    assert audit.simpsons_audit.levene_stat is not None
    assert audit.simpsons_audit.levene_p is not None
    # Verify Macaskill sample-size weighted regression test and Bartlett scale test
    assert audit.simpsons_audit.macaskill_test_t is not None
    assert audit.simpsons_audit.macaskill_test_p is not None
    assert audit.simpsons_audit.bartlett_stat is not None
    assert audit.simpsons_audit.bartlett_p is not None
    # Verify Deeks DOR funnel asymmetry test and Mood median-rank dispersion test
    assert audit.simpsons_audit.deeks_test_t is not None
    assert audit.simpsons_audit.deeks_test_p is not None
    assert audit.simpsons_audit.mood_stat_z is not None
    assert audit.simpsons_audit.mood_p_value is not None
    # Verify Thompson-Sharp random-effects funnel test and Ansari-Bradley scale test
    assert audit.simpsons_audit.thompson_sharp_t is not None
    assert audit.simpsons_audit.thompson_sharp_p is not None
    assert audit.simpsons_audit.ansari_bradley_stat is not None
    assert audit.simpsons_audit.ansari_bradley_p is not None
    # Verify Moreno variance-weighted funnel test and Siegel-Tukey scale test
    assert audit.simpsons_audit.moreno_test_t is not None
    assert audit.simpsons_audit.moreno_test_p is not None
    assert audit.simpsons_audit.siegel_tukey_stat is not None
    assert audit.simpsons_audit.siegel_tukey_p is not None
    # Verify Sterne-Egger funnel test and Klotz normal scores scale test
    assert audit.simpsons_audit.sterne_test_t is not None
    assert audit.simpsons_audit.sterne_test_p is not None
    assert audit.simpsons_audit.klotz_scale_stat is not None
    assert audit.simpsons_audit.klotz_scale_p is not None

    finding = referee.adjudicate(gender_claim, audit)
    assert finding.verdict == "SIMPSONS_PARADOX_ALERT"
    assert "SIMPSON" in finding.verdict_badge


def test_simpsons_churn_falsification():
    df = pd.read_csv(DATA_DIR / "simpsons_churn.csv")
    explorer = ExplorerAgent()
    auditor = DevilsAdvocateAgent()
    referee = RefereeAgent()

    claims = explorer.explore(df, target="churn")
    assert len(claims) > 0

    calls_claim = next((c for c in claims if c.source_variable == "support_calls" and c.target_variable == "churn"), None)
    assert calls_claim is not None

    audit = auditor.audit_claim(calls_claim, df)
    assert audit.simpsons_audit.paradox_detected
    assert audit.simpsons_audit.confounder_column == "tier"

    finding = referee.adjudicate(calls_claim, audit)
    assert finding.verdict == "SIMPSONS_PARADOX_ALERT"


def test_confounder_variance_loss_and_leakage():
    df = pd.read_csv(DATA_DIR / "income_confounder_benchmark.csv")
    explorer = ExplorerAgent()
    auditor = DevilsAdvocateAgent()

    claims = explorer.explore(df, target="annual_income")
    assert len(claims) > 0

    # Test leakage detection on tax_withheld_exact
    tax_claim = next((c for c in claims if c.source_variable == "tax_withheld_exact"), None)
    if tax_claim:
        audit_tax = auditor.audit_claim(tax_claim, df)
        assert audit_tax.leakage_detected or audit_tax.falsified

    # Test confounder attenuation on coffee_loyalty_points
    coffee_claim = next((c for c in claims if c.source_variable == "coffee_loyalty_points"), None)
    if coffee_claim:
        audit_coffee = auditor.audit_claim(coffee_claim, df)
        assert audit_coffee.confounder_audit.is_confounded
        assert audit_coffee.confounder_audit.attenuation_percentage > 50.0
        assert audit_coffee.confounder_audit.variance_explained_loss_pct > 70.0
        assert audit_coffee.confounder_audit.e_value is not None
        assert audit_coffee.confounder_audit.e_value >= 1.0
        assert audit_coffee.confounder_audit.top_confounder_vif is not None
        assert audit_coffee.confounder_audit.top_confounder_vif >= 1.0
        # Check Cinelli-Hazlett Robustness Values and Sobel mediation
        assert audit_coffee.confounder_audit.robustness_value is not None
        assert audit_coffee.confounder_audit.robustness_value_alpha is not None
        assert audit_coffee.confounder_audit.sobel_p is not None
        # Check Kass-Raftery Bayes Factor and Delta BIC
        assert audit_coffee.confounder_audit.bayes_factor_bf10 is not None
        assert audit_coffee.confounder_audit.bayes_factor_bf10 > 0.0
        assert audit_coffee.confounder_audit.bic_delta is not None
        # Check Ramsey RESET test and Oster selection bound
        assert audit_coffee.confounder_audit.ramsey_reset_p is not None
        assert audit_coffee.confounder_audit.ramsey_reset_f is not None
        assert audit_coffee.confounder_audit.oster_delta is not None
        # Check Cohen's f^2 and Cinelli-Hazlett Rcrit
        assert audit_coffee.confounder_audit.cohens_f2 is not None
        assert audit_coffee.confounder_audit.cinelli_hazlett_rcrit is not None
        assert audit_coffee.confounder_audit.cinelli_hazlett_rcrit >= 0.0
        # Check Frank's ITCV and Burnham-Anderson Akaike weights
        assert audit_coffee.confounder_audit.frank_itcv is not None
        assert audit_coffee.confounder_audit.frank_observed_impact is not None
        assert audit_coffee.confounder_audit.akaike_delta_aic is not None
        assert audit_coffee.confounder_audit.akaike_model_weight is not None
        assert 0.0 <= audit_coffee.confounder_audit.akaike_model_weight <= 1.0
        # Check Rosenbaum sensitivity bounds and Manski partial identification bounds
        assert audit_coffee.confounder_audit.rosenbaum_gamma_crit is not None
        assert audit_coffee.confounder_audit.rosenbaum_gamma_crit >= 1.0
        assert audit_coffee.confounder_audit.manski_bound_low is not None
        assert audit_coffee.confounder_audit.manski_bound_high is not None
        assert audit_coffee.confounder_audit.manski_bound_low <= audit_coffee.confounder_audit.manski_bound_high
        # Check Altonji-Elder-Taber selection ratio
        assert audit_coffee.confounder_audit.aet_selection_ratio is not None
        assert audit_coffee.confounder_audit.aet_selection_ratio >= 0.0
        # Check Imbens-Manski partial identification CI and Mallows' Cp criterion
        assert audit_coffee.confounder_audit.imbens_manski_ci_low is not None
        assert audit_coffee.confounder_audit.imbens_manski_ci_high is not None
        assert audit_coffee.confounder_audit.imbens_manski_ci_low <= audit_coffee.confounder_audit.imbens_manski_ci_high
        assert audit_coffee.confounder_audit.mallows_cp is not None
        assert audit_coffee.confounder_audit.cp_bias_discrepancy is not None
        # Check Davidson-MacKinnon J-test for non-nested models
        assert audit_coffee.confounder_audit.davidson_mackinnon_j_stat is not None
        assert audit_coffee.confounder_audit.davidson_mackinnon_p is not None
        # Check Durbin-Wu-Hausman endogeneity test and Stock-Yogo relevance F-statistic
        assert audit_coffee.confounder_audit.durbin_wu_hausman_f is not None
        assert audit_coffee.confounder_audit.durbin_wu_hausman_p is not None
        assert audit_coffee.confounder_audit.stock_yogo_f_stat is not None
        assert audit_coffee.confounder_audit.stock_yogo_f_stat >= 10.0
        # Check Belsley-Kuh-Welsch condition index and Sargan-Hansen overidentifying test
        assert audit_coffee.confounder_audit.bkw_condition_index is not None
        assert audit_coffee.confounder_audit.bkw_condition_index >= 1.0
        assert audit_coffee.confounder_audit.sargan_j_stat is not None
        assert audit_coffee.confounder_audit.sargan_p_value is not None
        # Check Crump propensity overlap and Bruce Hansen threshold test
        assert audit_coffee.confounder_audit.crump_overlap_ratio is not None
        assert audit_coffee.confounder_audit.hansen_threshold_f is not None
        assert audit_coffee.confounder_audit.hansen_threshold_p is not None
        # Check Hosmer-Lemeshow calibration and Newey-West HAC variance ratio
        assert audit_coffee.confounder_audit.hosmer_lemeshow_chi2 is not None
        assert audit_coffee.confounder_audit.hosmer_lemeshow_p is not None
        assert audit_coffee.confounder_audit.newey_west_hac_se is not None
        assert audit_coffee.confounder_audit.hac_se_ratio is not None
        # Check AIPW doubly robust effect and Leamer Extreme Bounds Analysis
        assert audit_coffee.confounder_audit.aipw_treatment_effect is not None
        assert audit_coffee.confounder_audit.aipw_discrepancy is not None
        assert audit_coffee.confounder_audit.leamer_bound_min is not None
        assert audit_coffee.confounder_audit.leamer_bound_max is not None
        # Check DML Neyman-orthogonal effect and Cinelli extreme confounder multiplier
        assert audit_coffee.confounder_audit.dml_treatment_effect is not None
        assert audit_coffee.confounder_audit.dml_discrepancy is not None
        assert audit_coffee.confounder_audit.cinelli_k_multiplier is not None
        # Check Rubin's covariate balance and Brunner-Munzel relative treatment effect
        assert audit_coffee.confounder_audit.rubin_b_score is not None
        assert audit_coffee.confounder_audit.rubin_r_ratio is not None
        assert audit_coffee.confounder_audit.brunner_munzel_w is not None
        assert audit_coffee.confounder_audit.brunner_munzel_p is not None
        # Check Rosenbaum-Rubin SMD balance and Imbens-Rubin normalized difference
        assert audit_coffee.confounder_audit.smd_covariate_balance is not None
        assert audit_coffee.confounder_audit.mahalanobis_distance is not None
        assert audit_coffee.confounder_audit.imbens_normalized_diff is not None
        # Check Austin covariate variance ratio and Stuart prognostic score discrepancy
        assert audit_coffee.confounder_audit.austin_variance_ratio is not None
        assert audit_coffee.confounder_audit.stuart_prognostic_discrepancy is not None

def test_vectorized_bootstrap_speed_and_stability():
    df = pd.read_csv(DATA_DIR / "income_confounder_benchmark.csv")
    explorer = ExplorerAgent()
    auditor = DevilsAdvocateAgent(n_bootstrap=200)

    claims = explorer.explore(df, target="annual_income")
    exp_claim = next(c for c in claims if c.source_variable == "years_experience")
    audit = auditor.audit_claim(exp_claim, df)

    assert audit.stability_audit.is_stable
    assert audit.stability_audit.bootstrap_std_err > 0
    # Vectorized bootstrap should finish in under 150 milliseconds
    assert audit.stability_audit.execution_time_ms < 150.0
    # Check Spearman rank and leverage metrics
    assert audit.stability_audit.spearman_rho is not None
    assert audit.stability_audit.rank_discrepancy is not None
    assert audit.stability_audit.max_leverage is not None
    assert audit.stability_audit.max_leverage >= 0.0
    # Check Cook's distance, sign invariance probability, and Benjamini-Hochberg FDR
    assert audit.stability_audit.sign_invariance_prob >= 0.90
    assert audit.stability_audit.max_cooks_distance is not None
    assert audit.stability_audit.max_cooks_distance >= 0.0
    assert audit.stability_audit.cooks_trimmed_statistic is not None
    assert exp_claim.fdr_q_value is not None
    assert exp_claim.fdr_q_value <= 1.0
    # Check DFFITS, permutation p-value, and Bonferroni FWER
    assert audit.stability_audit.max_dffits is not None
    assert audit.stability_audit.permutation_p_value is not None
    assert audit.stability_audit.permutation_p_value < 0.05
    assert exp_claim.bonferroni_p_value is not None
    assert exp_claim.bonferroni_p_value <= 1.0
    # Check Breusch-Pagan test and Quantile Discrepancy Index
    assert audit.stability_audit.breusch_pagan_lm is not None
    assert audit.stability_audit.breusch_pagan_p is not None
    assert audit.stability_audit.white_lm_stat is not None
    assert audit.stability_audit.white_lm_p is not None
    assert audit.stability_audit.quantile_discrepancy_index is not None
    assert audit.stability_audit.quantile_discrepancy_index >= 0.0
    # Check DFBETAS and Jarque-Bera residual normality
    assert audit.stability_audit.max_dfbetas is not None
    assert audit.stability_audit.max_dfbetas >= 0.0
    assert audit.stability_audit.jarque_bera_stat is not None
    assert audit.stability_audit.jarque_bera_p is not None
    # Check Durbin-Watson and MacKinnon-White HC3
    assert audit.stability_audit.durbin_watson_stat is not None
    assert audit.stability_audit.hc3_robust_se is not None
    assert audit.stability_audit.hc3_robust_p is not None
    # Check Theil-Sen robust slope and Efron BCa bootstrap CI
    assert audit.stability_audit.theil_sen_slope is not None
    assert audit.stability_audit.theil_sen_discrepancy is not None
    assert audit.stability_audit.bca_ci_low is not None
    assert audit.stability_audit.bca_ci_high is not None
    assert audit.stability_audit.bca_ci_low <= audit.stability_audit.bca_ci_high
    # Check Chow structural break test
    assert audit.stability_audit.chow_f_stat is not None
    assert audit.stability_audit.chow_p_value is not None
    assert 0.0 <= audit.stability_audit.chow_p_value <= 1.0
    # Check Gelman-Carlin Type S / Type M design analysis
    assert audit.stability_audit.type_m_exaggeration_ratio is not None
    assert audit.stability_audit.type_m_exaggeration_ratio >= 1.0
    assert audit.stability_audit.type_s_error_prob is not None
    assert 0.0 <= audit.stability_audit.type_s_error_prob <= 1.0
    assert audit.stability_audit.retrodesign_power is not None
    # Check Donald Andrews Supremum Wald test for unknown structural breaks
    assert audit.stability_audit.andrews_sup_w is not None
    assert audit.stability_audit.andrews_p_value is not None
    assert audit.stability_audit.andrews_break_fraction is not None
    # Check Brown-Durbin-Evans recursive CUSUM and Koenker studentized BP
    assert audit.stability_audit.cusum_stat is not None
    assert audit.stability_audit.cusum_p_approx is not None
    assert audit.stability_audit.koenker_lm_stat is not None
    assert audit.stability_audit.koenker_p_value is not None
    # Check Harvey-Collier recursive linearity and Huber robust M-estimator
    assert audit.stability_audit.harvey_collier_t is not None
    assert audit.stability_audit.harvey_collier_p is not None
    assert audit.stability_audit.huber_m_slope is not None
    assert audit.stability_audit.huber_slope_discrepancy is not None
    # Check CUSUMSQ variance constancy and Rousseeuw Least Trimmed Squares (LTS)
    assert audit.stability_audit.cusumsq_stat is not None
    assert audit.stability_audit.cusumsq_critical_val is not None
    assert audit.stability_audit.lts_slope is not None
    assert audit.stability_audit.lts_slope_discrepancy is not None
    # Check Siegel Repeated Medians and Breusch-Godfrey serial correlation test
    assert audit.stability_audit.siegel_slope is not None
    assert audit.stability_audit.siegel_slope_discrepancy is not None
    assert audit.stability_audit.breusch_godfrey_lm is not None
    assert audit.stability_audit.breusch_godfrey_p is not None
    # Check Wald-Wolfowitz runs test for residual randomness and Kendall Tau-b rank concordance
    assert audit.stability_audit.runs_test_z is not None
    assert audit.stability_audit.runs_test_p is not None
    assert audit.stability_audit.kendall_tau_b is not None
    assert audit.stability_audit.kendall_discrepancy is not None
    # Check Goldfeld-Quandt split-regime heteroskedasticity and Shapiro-Francia residual normality
    assert audit.stability_audit.goldfeld_quandt_f is not None
    assert audit.stability_audit.goldfeld_quandt_p is not None
    assert audit.stability_audit.shapiro_francia_w is not None
    assert audit.stability_audit.shapiro_francia_p is not None
    # Check D'Agostino-Pearson omnibus normality and Glejser heteroskedasticity test
    assert audit.stability_audit.dagostino_pearson_k2 is not None
    assert audit.stability_audit.dagostino_pearson_p is not None
    assert audit.stability_audit.glejser_t_stat is not None
    assert audit.stability_audit.glejser_p_value is not None
    # Check Anderson-Darling EDF residual normality and Harvey multiplicative heteroskedasticity test
    assert audit.stability_audit.anderson_darling_stat is not None
    assert audit.stability_audit.anderson_darling_crit is not None
    assert audit.stability_audit.harvey_lm_stat is not None
    assert audit.stability_audit.harvey_lm_p is not None
    # Check Lilliefors composite normality and Park log-linear heteroskedasticity test
    assert audit.stability_audit.lilliefors_stat is not None
    assert audit.stability_audit.lilliefors_p_value is not None
    assert audit.stability_audit.park_t_stat is not None
    assert audit.stability_audit.park_p_value is not None


def test_llm_narrator_risk_score_and_copilot():
    df = pd.read_csv(DATA_DIR / "berkeley_admissions.csv")
    explorer = ExplorerAgent()
    auditor = DevilsAdvocateAgent()
    referee = RefereeAgent()

    claims = explorer.explore(df, target="admitted")
    audits = [auditor.audit_claim(c, df) for c in claims]
    findings = [referee.adjudicate(c, a).to_dict() for c, a in zip(claims, audits)]

    risk = calculate_risk_score(findings)
    assert 0 <= risk["score"] <= 100
    assert risk["tier"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]

    narrator = LLMNarrator(provider="offline")
    briefing = narrator.generate_executive_briefing(findings, dataset_name="berkeley_admissions.csv")
    assert "Executive Briefing" in briefing
    assert "Decision Risk Status" in briefing

    # Test copilot Q&A
    ans_risk = narrator.answer_query("What is the risk score?", findings)
    assert "Risk Score" in ans_risk

    ans_gender = narrator.answer_query("Explain gender", findings)
    assert len(ans_gender) > 10

    ans_ev = narrator.answer_query("What is the E-value?", findings)
    assert "E-value" in ans_ev

    ans_tarone = narrator.answer_query("Explain Tarone homogeneity", findings)
    assert "Tarone" in ans_tarone

    ans_cook = narrator.answer_query("What is Cook's distance?", findings)
    assert "Cook" in ans_cook

    ans_sign = narrator.answer_query("What is the sign invariance probability?", findings)
    assert "Sign Invariance" in ans_sign

    ans_vif = narrator.answer_query("Explain VIF multicollinearity", findings)
    assert "VIF" in ans_vif

    ans_rv = narrator.answer_query("What is the Robustness Value (RV)?", findings)
    assert "Robustness Value" in ans_rv or "RV" in ans_rv

    ans_i2 = narrator.answer_query("Explain Higgins I-squared heterogeneity", findings)
    assert "Higgins" in ans_i2 or "I²" in ans_i2

    ans_sobel = narrator.answer_query("What does the Sobel mediation test show?", findings)
    assert "Sobel" in ans_sobel or "Mediation" in ans_sobel

    ans_dffits = narrator.answer_query("What is DFFITS influence?", findings)
    assert "DFFITS" in ans_dffits

    ans_perm = narrator.answer_query("Explain permutation test", findings)
    assert "Permutation" in ans_perm

    ans_bonf = narrator.answer_query("Explain Bonferroni correction", findings)
    assert "Bonferroni" in ans_bonf

    ans_bf = narrator.answer_query("Explain Kass-Raftery Bayes factor and BIC", findings)
    assert "Bayes" in ans_bf or "BIC" in ans_bf

    ans_begg = narrator.answer_query("What does the Begg funnel test show?", findings)
    assert "Begg" in ans_begg or "Funnel" in ans_begg

    ans_bp = narrator.answer_query("Explain Breusch-Pagan heteroscedasticity", findings)
    assert "Breusch-Pagan" in ans_bp or "Heteroscedastic" in ans_bp

    ans_qdi = narrator.answer_query("What is the Quantile Discrepancy Index?", findings)
    assert "Quantile Discrepancy" in ans_qdi or "QDI" in ans_qdi

    ans_egger = narrator.answer_query("Explain Egger's regression test", findings)
    assert "Egger" in ans_egger or "Asymmetry" in ans_egger

    ans_tau = narrator.answer_query("Explain DerSimonian-Laird tau-squared", findings)
    assert "DerSimonian" in ans_tau or "τ²" in ans_tau or "Variance" in ans_tau

    ans_reset = narrator.answer_query("Explain Ramsey RESET test", findings)
    assert "Ramsey" in ans_reset or "RESET" in ans_reset

    ans_oster = narrator.answer_query("What is the Oster delta bound?", findings)
    assert "Oster" in ans_oster or "δ*" in ans_oster

    ans_dfb = narrator.answer_query("What is DFBETAS slope influence?", findings)
    assert "DFBETAS" in ans_dfb or "Influence" in ans_dfb

    ans_jb = narrator.answer_query("Explain Jarque-Bera normality", findings)
    assert "Jarque-Bera" in ans_jb or "Normality" in ans_jb

    ans_tf = narrator.answer_query("Explain Duval-Tweedie Trim and Fill", findings)
    assert "Trim" in ans_tf or "Fill" in ans_tf or "Duval" in ans_tf

    ans_haus = narrator.answer_query("What does the Hausman specification test show?", findings)
    assert "Hausman" in ans_haus or "Endogeneity" in ans_haus

    ans_f2 = narrator.answer_query("What is Cohen's f2 effect size and Rcrit?", findings)
    assert "Cohen" in ans_f2 or "f²" in ans_f2 or "Critical" in ans_f2

    ans_dw = narrator.answer_query("Explain Durbin-Watson autocorrelation", findings)
    assert "Durbin-Watson" in ans_dw or "Autocorrelation" in ans_dw or "Serial" in ans_dw

    ans_hc3 = narrator.answer_query("What does the MacKinnon-White HC3 test show?", findings)
    assert "HC3" in ans_hc3 or "MacKinnon" in ans_hc3 or "Robust" in ans_hc3

    ans_vuong = narrator.answer_query("Explain Vuong model selection test", findings)
    assert "Vuong" in ans_vuong or "Model Selection" in ans_vuong or "Kullback" in ans_vuong

    ans_bd = narrator.answer_query("What is the Breslow-Day test?", findings)
    assert "Breslow-Day" in ans_bd or "Homogeneity" in ans_bd or "Odds Ratio" in ans_bd

    ans_itcv = narrator.answer_query("Explain Frank's ITCV impact threshold", findings)
    assert "ITCV" in ans_itcv or "Frank" in ans_itcv or "Impact" in ans_itcv

    ans_aic = narrator.answer_query("Explain Akaike model weights and evidence ratio", findings)
    assert "Akaike" in ans_aic or "AIC" in ans_aic or "Weight" in ans_aic

    ans_ts = narrator.answer_query("What is the Theil-Sen robust slope?", findings)
    assert "Theil-Sen" in ans_ts or "Median" in ans_ts or "Slope" in ans_ts

    ans_bca = narrator.answer_query("Explain BCa bootstrap confidence interval", findings)
    assert "BCa" in ans_bca or "Bootstrap" in ans_bca or "Accelerated" in ans_bca

    ans_rosen = narrator.answer_query("Explain Rosenbaum sensitivity bounds", findings)
    assert "Rosenbaum" in ans_rosen or "Gamma" in ans_rosen or "Γ" in ans_rosen

    ans_white = narrator.answer_query("Explain White's test for heteroscedasticity", findings)
    assert "White" in ans_white or "Heteroscedasticity" in ans_white

    ans_manski = narrator.answer_query("What are Manski partial identification bounds?", findings)
    assert "Manski" in ans_manski or "Partial Identification" in ans_manski or "Bounds" in ans_manski

    ans_aet = narrator.answer_query("Explain Altonji-Elder-Taber selection ratio", findings)
    assert "Altonji" in ans_aet or "AET" in ans_aet or "Selection" in ans_aet

    ans_chow = narrator.answer_query("Explain Chow structural break test", findings)
    assert "Chow" in ans_chow or "Structural Break" in ans_chow or "Stability" in ans_chow

    ans_galb = narrator.answer_query("Explain Galbraith radial outlier strata and Higgins H-squared", findings)
    assert "Galbraith" in ans_galb or "Higgins" in ans_galb or "Radial" in ans_galb

    ans_woolf = narrator.answer_query("Explain Woolf odds ratio heterogeneity test", findings)
    assert "Woolf" in ans_woolf or "Heterogeneity" in ans_woolf or "Odds Ratio" in ans_woolf

    ans_im = narrator.answer_query("Explain Imbens-Manski partial identification confidence intervals", findings)
    assert "Imbens" in ans_im or "Manski" in ans_im or "Partial Identification" in ans_im

    ans_cp = narrator.answer_query("What does Mallows' Cp show?", findings)
    assert "Mallows" in ans_cp or "Cp" in ans_cp or "Misspecification" in ans_cp

    ans_gelman = narrator.answer_query("Explain Gelman-Carlin Type S and Type M design analysis", findings)
    assert "Gelman" in ans_gelman or "Type M" in ans_gelman or "Exaggeration" in ans_gelman

    ans_ft = narrator.answer_query("Explain Freeman-Tukey double-arcsine homogeneity test", findings)
    assert "Freeman" in ans_ft or "Tukey" in ans_ft or "Double-Arcsine" in ans_ft

    ans_kw = narrator.answer_query("Explain Kruskal-Wallis rank sum test", findings)
    assert "Kruskal" in ans_kw or "Wallis" in ans_kw or "Rank" in ans_kw

    ans_j = narrator.answer_query("Explain Davidson-MacKinnon J-test", findings)
    assert "Davidson" in ans_j or "MacKinnon" in ans_j or "J-Test" in ans_j

    ans_andrews = narrator.answer_query("Explain Donald Andrews Supremum Wald test for unknown structural breaks", findings)
    assert "Andrews" in ans_andrews or "Supremum" in ans_andrews or "Wald" in ans_andrews

    ans_peto = narrator.answer_query("Explain Peto odds ratio and heterogeneity test", findings)
    assert "Peto" in ans_peto or "Odds Ratio" in ans_peto or "Heterogeneity" in ans_peto

    ans_flig = narrator.answer_query("Explain Fligner-Killeen variance homogeneity test", findings)
    assert "Fligner" in ans_flig or "Killeen" in ans_flig or "Variance" in ans_flig

    ans_dwh = narrator.answer_query("Explain Durbin-Wu-Hausman test for endogeneity", findings)
    assert "Durbin" in ans_dwh or "Hausman" in ans_dwh or "Endogeneity" in ans_dwh

    ans_sy = narrator.answer_query("Explain Stock-Yogo first-stage relevance test", findings)
    assert "Stock" in ans_sy or "Yogo" in ans_sy or "Relevance" in ans_sy

    ans_cusum = narrator.answer_query("Explain Brown-Durbin-Evans CUSUM recursive test", findings)
    assert "CUSUM" in ans_cusum or "Recursive" in ans_cusum or "Brown" in ans_cusum

    ans_koenker = narrator.answer_query("Explain Koenker studentized Breusch-Pagan test", findings)
    assert "Koenker" in ans_koenker or "Studentized" in ans_koenker or "Heteroscedasticity" in ans_koenker

    ans_harbord = narrator.answer_query("Explain Harbord modified score test for funnel plot asymmetry", findings)
    assert "Harbord" in ans_harbord or "Score" in ans_harbord or "Asymmetry" in ans_harbord

    ans_bf = narrator.answer_query("Explain Brown-Forsythe scale homogeneity test", findings)
    assert "Brown" in ans_bf or "Forsythe" in ans_bf or "Scale" in ans_bf

    ans_bkw = narrator.answer_query("Explain Belsley-Kuh-Welsch condition index", findings)
    assert "Belsley" in ans_bkw or "BKW" in ans_bkw or "Condition" in ans_bkw

    ans_sargan = narrator.answer_query("Explain Sargan-Hansen overidentifying restrictions test", findings)
    assert "Sargan" in ans_sargan or "Overidentifying" in ans_sargan or "Restrictions" in ans_sargan

    ans_hc = narrator.answer_query("Explain Harvey-Collier recursive residual linearity test", findings)
    assert "Harvey" in ans_hc or "Collier" in ans_hc or "Linearity" in ans_hc

    ans_huber = narrator.answer_query("Explain Huber M-estimator robust regression discrepancy", findings)
    assert "Huber" in ans_huber or "Discrepancy" in ans_huber or "Robust" in ans_huber

    ans_peters = narrator.answer_query("Explain Peters regression test for funnel plot asymmetry", findings)
    assert "Peters" in ans_peters or "Asymmetry" in ans_peters or "Funnel" in ans_peters

    ans_kish = narrator.answer_query("Explain Leslie Kish design effect and variance ratio", findings)
    assert "Kish" in ans_kish or "Design Effect" in ans_kish or "Variance" in ans_kish

    ans_crump = narrator.answer_query("Explain Crump propensity score overlap test", findings)
    assert "Crump" in ans_crump or "Overlap" in ans_crump or "Common Support" in ans_crump

    ans_hansen = narrator.answer_query("Explain Bruce Hansen structural threshold test", findings)
    assert "Hansen" in ans_hansen or "Threshold" in ans_hansen or "Regime" in ans_hansen

    ans_cusumsq = narrator.answer_query("Explain Brown-Durbin-Evans CUSUMSQ test for variance stability", findings)
    assert "CUSUMSQ" in ans_cusumsq or "Variance" in ans_cusumsq or "Ploberger" in ans_cusumsq

    ans_lts = narrator.answer_query("Explain Least Trimmed Squares LTS high-breakdown robust slope", findings)
    assert "Trimmed Squares" in ans_lts or "LTS" in ans_lts or "Breakdown" in ans_lts or "Rousseeuw" in ans_lts

    ans_rucker = narrator.answer_query("Explain Rücker arcsine test for funnel plot asymmetry", findings)
    assert "Rücker" in ans_rucker or "Rucker" in ans_rucker or "Arcsine" in ans_rucker or "Asymmetry" in ans_rucker

    ans_levene = narrator.answer_query("Explain Howard Levene scale homogeneity test", findings)
    assert "Levene" in ans_levene or "Scale" in ans_levene or "Variances" in ans_levene

    ans_hl = narrator.answer_query("Explain Hosmer-Lemeshow propensity calibration test", findings)
    assert "Hosmer" in ans_hl or "Lemeshow" in ans_hl or "Calibration" in ans_hl or "Propensity" in ans_hl

    ans_hac = narrator.answer_query("Explain Newey-West HAC robust standard error ratio", findings)
    assert "Newey" in ans_hac or "West" in ans_hac or "HAC" in ans_hac or "Autocorrelation" in ans_hac

    ans_siegel = narrator.answer_query("Explain Siegel Repeated Medians high-breakdown robust slope", findings)
    assert "Siegel" in ans_siegel or "Repeated Medians" in ans_siegel or "Breakdown" in ans_siegel

    ans_bg = narrator.answer_query("Explain Breusch-Godfrey test for higher-order serial correlation", findings)
    assert "Breusch" in ans_bg or "Godfrey" in ans_bg or "Autocorrelation" in ans_bg or "Serial" in ans_bg

    ans_mac = narrator.answer_query("Explain Peter Macaskill sample-size weighted regression funnel test", findings)
    assert "Macaskill" in ans_mac or "Sample-Size" in ans_mac or "Funnel" in ans_mac or "Weighted" in ans_mac

    ans_bart = narrator.answer_query("Explain Maurice Bartlett likelihood ratio scale homogeneity test", findings)
    assert "Bartlett" in ans_bart or "Scale" in ans_bart or "Variances" in ans_bart

    ans_aipw = narrator.answer_query("Explain Robins-Rotnitzky-Zhao and Imbens AIPW doubly robust treatment effect", findings)
    assert "AIPW" in ans_aipw or "Doubly Robust" in ans_aipw or "Robins" in ans_aipw or "Treatment Effect" in ans_aipw

    ans_leamer = narrator.answer_query("Explain Edward Leamer Extreme Bounds Analysis (EBA)", findings)
    assert "Leamer" in ans_leamer or "Extreme Bounds" in ans_leamer or "EBA" in ans_leamer or "Fragility" in ans_leamer

    ans_runs = narrator.answer_query("Explain Abraham Wald and Jacob Wolfowitz runs test for residual independence", findings)
    assert "Wald" in ans_runs or "Wolfowitz" in ans_runs or "Runs" in ans_runs or "Independence" in ans_runs

    ans_kendall = narrator.answer_query("Explain Maurice Kendall Tau-b rank concordance discrepancy", findings)
    assert "Kendall" in ans_kendall or "Tau" in ans_kendall or "Concordance" in ans_kendall or "Rank" in ans_kendall

    ans_deeks = narrator.answer_query("Explain Jon Deeks diagnostic odds ratio asymmetry test", findings)
    assert "Deeks" in ans_deeks or "DOR" in ans_deeks or "Funnel" in ans_deeks or "Diagnostic" in ans_deeks

    ans_mood = narrator.answer_query("Explain Alexander Mood median-rank dispersion scale test", findings)
    assert "Mood" in ans_mood or "Dispersion" in ans_mood or "Rank" in ans_mood or "Scale" in ans_mood

    ans_dml = narrator.answer_query("Explain Chernozhukov Double Machine Learning DML causal effect", findings)
    assert "DML" in ans_dml or "Double Machine Learning" in ans_dml or "Chernozhukov" in ans_dml or "Orthogonal" in ans_dml

    ans_cinelli = narrator.answer_query("Explain Carlos Cinelli extreme confounder multiplier benchmark", findings)
    assert "Cinelli" in ans_cinelli or "Multiplier" in ans_cinelli or "Confounder" in ans_cinelli

    ans_gq = narrator.answer_query("Explain Stephen Goldfeld and Richard Quandt split-regime heteroskedasticity test", findings)
    assert "Goldfeld" in ans_gq or "Quandt" in ans_gq or "Heteroskedasticity" in ans_gq or "Regime" in ans_gq

    ans_sf = narrator.answer_query("Explain Shapiro-Francia test for regression residual Gaussian normality", findings)
    assert "Shapiro" in ans_sf or "Francia" in ans_sf or "Normality" in ans_sf or "Gaussian" in ans_sf

    ans_ts = narrator.answer_query("Explain Simon Thompson and Stephen Sharp random-effects funnel asymmetry test", findings)
    assert "Thompson" in ans_ts or "Sharp" in ans_ts or "Funnel" in ans_ts or "Random-Effects" in ans_ts

    ans_ab = narrator.answer_query("Explain Ansari-Bradley rank-sum scale dispersion test", findings)
    assert "Ansari" in ans_ab or "Bradley" in ans_ab or "Dispersion" in ans_ab or "Scale" in ans_ab

    ans_rubin = narrator.answer_query("Explain Donald Rubin propensity score covariate balance diagnostics", findings)
    assert "Rubin" in ans_rubin or "Balance" in ans_rubin or "Propensity" in ans_rubin

    ans_bm = narrator.answer_query("Explain Edgar Brunner and Ullrich Munzel generalized relative treatment effect", findings)
    assert "Brunner" in ans_bm or "Munzel" in ans_bm or "Treatment Effect" in ans_bm or "Relative" in ans_bm

    ans_dp = narrator.answer_query("Explain Ralph D'Agostino and E. S. Pearson omnibus normality test", findings)
    assert "Agostino" in ans_dp or "Pearson" in ans_dp or "Normality" in ans_dp or "Omnibus" in ans_dp

    ans_glejser = narrator.answer_query("Explain Herbert Glejser absolute residual regression heteroskedasticity test", findings)
    assert "Glejser" in ans_glejser or "Heteroskedasticity" in ans_glejser or "Residual" in ans_glejser

    ans_moreno = narrator.answer_query("Explain Moreno variance-weighted funnel plot asymmetry test", findings)
    assert "Moreno" in ans_moreno or "Funnel" in ans_moreno or "Variance" in ans_moreno

    ans_st = narrator.answer_query("Explain Siegel-Tukey rank extremes scale dispersion test", findings)
    assert "Siegel" in ans_st or "Tukey" in ans_st or "Extremes" in ans_st or "Dispersion" in ans_st

    ans_smd = narrator.answer_query("Explain Rosenbaum and Rubin standardized mean difference and Mahalanobis balance", findings)
    assert "Rosenbaum" in ans_smd or "Rubin" in ans_smd or "SMD" in ans_smd or "Mahalanobis" in ans_smd

    ans_imb = narrator.answer_query("Explain Imbens-Rubin normalized difference metric for common support overlap", findings)
    assert "Imbens" in ans_imb or "Normalized" in ans_imb or "Overlap" in ans_imb or "Difference" in ans_imb

    ans_ad = narrator.answer_query("Explain Anderson-Darling empirical distribution function residual normality test", findings)
    assert "Anderson" in ans_ad or "Darling" in ans_ad or "Normality" in ans_ad

    ans_harv = narrator.answer_query("Explain Andrew Harvey multiplicative heteroskedasticity LM test", findings)
    assert "Harvey" in ans_harv or "Multiplicative" in ans_harv or "Heteroskedasticity" in ans_harv or "LM" in ans_harv

    ans_sterne = narrator.answer_query("Explain Sterne and Egger standard error funnel plot asymmetry test", findings)
    assert "Sterne" in ans_sterne or "Funnel" in ans_sterne or "Precision" in ans_sterne

    ans_klotz = narrator.answer_query("Explain Jerome Klotz normal scores rank scale dispersion test", findings)
    assert "Klotz" in ans_klotz or "Normal" in ans_klotz or "Scores" in ans_klotz or "Dispersion" in ans_klotz

    ans_austin = narrator.answer_query("Explain Peter Austin covariate variance ratio balance metric", findings)
    assert "Austin" in ans_austin or "Variance" in ans_austin or "Ratio" in ans_austin or "Balance" in ans_austin

    ans_stuart = narrator.answer_query("Explain Elizabeth Stuart prognostic score discrepancy metric", findings)
    assert "Stuart" in ans_stuart or "Prognostic" in ans_stuart or "Discrepancy" in ans_stuart

    ans_lill = narrator.answer_query("Explain Hubert Lilliefors composite normality goodness of fit test", findings)
    assert "Lilliefors" in ans_lill or "Normality" in ans_lill or "Goodness" in ans_lill

    ans_park = narrator.answer_query("Explain Richard Park log-linear residual variance heteroskedasticity test", findings)
    assert "Park" in ans_park or "Log-Linear" in ans_park or "Heteroskedasticity" in ans_park or "Variance" in ans_park


def test_end_to_end_pipeline(tmp_path):
    res = run_adversarial_pipeline(
        input_path=DATA_DIR / "berkeley_admissions.csv",
        target="admitted",
        out_dir=tmp_path / "test_out",
    )
    assert res["status"] == "complete"
    assert len(res["adversarial_findings"]) > 0
    assert (tmp_path / "test_out" / "adversarial_audit_report.html").exists()
    assert (tmp_path / "test_out" / "adversarial_audit_report.md").exists()
    assert (tmp_path / "test_out" / "findings.json").exists()


if __name__ == "__main__":
    import tempfile
    import shutil

    print("Running test_berkeley_admissions_falsification...")
    test_berkeley_admissions_falsification()
    print("Running test_simpsons_churn_falsification...")
    test_simpsons_churn_falsification()
    print("Running test_confounder_variance_loss_and_leakage...")
    test_confounder_variance_loss_and_leakage()
    print("Running test_vectorized_bootstrap_speed_and_stability...")
    test_vectorized_bootstrap_speed_and_stability()
    print("Running test_llm_narrator_risk_score_and_copilot...")
    test_llm_narrator_risk_score_and_copilot()
    print("Running test_end_to_end_pipeline...")
    local_temp = Path(__file__).resolve().parent.parent / ".pytest_temp" / "manual_run"
    local_temp.mkdir(parents=True, exist_ok=True)
    try:
        test_end_to_end_pipeline(local_temp)
    finally:
        shutil.rmtree(local_temp, ignore_errors=True)
    print("ALL 6 EXPANDED TESTS PASSED CLEANLY!")
