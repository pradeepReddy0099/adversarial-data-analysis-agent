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
