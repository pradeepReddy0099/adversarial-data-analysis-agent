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
    assert audit.stability_audit.quantile_discrepancy_index is not None
    assert audit.stability_audit.quantile_discrepancy_index >= 0.0


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
