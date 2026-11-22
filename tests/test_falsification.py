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


if __name__ == "__main__":
    import tempfile
    import shutil

    print("Running test_berkeley_admissions_falsification...")
    test_berkeley_admissions_falsification()
    print("Running test_simpsons_churn_falsification...")
    test_simpsons_churn_falsification()
    print("Running test_end_to_end_pipeline...")
    local_temp = Path(__file__).resolve().parent.parent / ".pytest_temp" / "manual_run"
    local_temp.mkdir(parents=True, exist_ok=True)
    try:
        test_end_to_end_pipeline(local_temp)
    finally:
        shutil.rmtree(local_temp, ignore_errors=True)
    print("ALL TESTS IN NEW REPO PASSED!")
