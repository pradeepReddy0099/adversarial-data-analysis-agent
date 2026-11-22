"""Adversarial Data Analysis Agent package.

A dialectical multi-agent framework for automated hypothesis generation and
statistical falsification (Simpson's Paradox, Confounders, Bootstrap Stability).
"""

from adversarial_agent.auditor import AuditReport, DevilsAdvocateAgent
from adversarial_agent.core import AgentConfig, run_adversarial_pipeline
from adversarial_agent.explorer import Claim, ExplorerAgent
from adversarial_agent.llm_narrator import LLMNarrator
from adversarial_agent.referee import AdjudicatedFinding, RefereeAgent

__version__ = "1.1.0"

__all__ = [
    "AdjudicatedFinding",
    "AgentConfig",
    "AuditReport",
    "Claim",
    "DevilsAdvocateAgent",
    "ExplorerAgent",
    "LLMNarrator",
    "RefereeAgent",
    "run_adversarial_pipeline",
]
