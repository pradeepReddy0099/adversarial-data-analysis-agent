"""Core orchestration engine for the Adversarial Data Analysis Agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import pandas as pd

from adversarial_agent.auditor import DevilsAdvocateAgent
from adversarial_agent.explorer import ExplorerAgent
from adversarial_agent.profiling import profile_table
from adversarial_agent.referee import RefereeAgent
from adversarial_agent.reporting import build_report


@dataclass
class AgentConfig:
    random_seed: int = 42
    n_bootstrap: int = 150
    significance_level: float = 0.05
    min_subgroup_size: int = 8
    confounder_attenuation_threshold: float = 60.0
    version: str = "1.1.0"

    def to_dict(self) -> dict:
        return asdict(self)


def load_dataset(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    if p.suffix.lower() in [".csv", ".tsv"]:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        return pd.read_csv(p, sep=sep)
    elif p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported file format: {p.suffix}")


def run_adversarial_pipeline(
    input_path: str | Path,
    target: Optional[str] = None,
    out_dir: str | Path = "output",
    seed: int = 42,
    n_bootstrap: int = 150,
    min_subgroup_size: int = 8,
    confounder_attenuation_threshold: float = 60.0,
) -> Dict[str, Any]:
    """Execute the full adversarial multi-agent pipeline."""
    config = AgentConfig(
        random_seed=seed,
        n_bootstrap=n_bootstrap,
        min_subgroup_size=min_subgroup_size,
        confounder_attenuation_threshold=confounder_attenuation_threshold,
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    profile = profile_table(df)

    # 1. Explorer Agent
    explorer = ExplorerAgent(significance_level=config.significance_level)
    claims = explorer.explore(df, target=target)

    # 2. Devil's Advocate (Auditor) Agent
    auditor = DevilsAdvocateAgent(
        n_bootstrap=config.n_bootstrap,
        min_subgroup_size=config.min_subgroup_size,
        random_seed=seed,
        confounder_attenuation_threshold=config.confounder_attenuation_threshold,
    )
    audits = [auditor.audit_claim(c, df) for c in claims]

    # 3. Referee Agent
    referee = RefereeAgent()
    findings = referee.adjudicate_all(claims, audits)

    results = {
        "status": "complete",
        "input_file": str(input_path),
        "target": target,
        "config": config.to_dict(),
        "profile": profile,
        "claims_count": len(claims),
        "adversarial_findings": [f.to_dict() for f in findings],
    }

    # Generate Markdown and HTML Reports
    report_paths = build_report(results, out_path)
    results["reports"] = report_paths
    return results
