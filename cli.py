"""Command Line Interface for the Adversarial Data Analysis Agent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from adversarial_agent.core import run_adversarial_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="⚔️ Adversarial Data Analysis Agent: Automated Hypothesis Generation & Statistical Falsification"
    )
    parser.add_argument("input_file", type=str, help="Path to input CSV, TSV, or Excel file")
    parser.add_argument("--target", type=str, default=None, help="Optional outcome/target variable to audit")
    parser.add_argument("--out", type=str, default="output", help="Directory where reports will be saved")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    print(f"\n==================================================================")
    print(f"⚔️  ADVERSARIAL DATA ANALYSIS AGENT")
    print(f"==================================================================")
    print(f"Ingesting dataset: {args.input_file}")
    if args.target:
        print(f"Target column:    {args.target}")
    print(f"Output folder:    {args.out}")
    print(f"Executing multi-agent dialectic: Explorer -> Devil's Advocate -> Referee...")

    results = run_adversarial_pipeline(
        input_path=args.input_file,
        target=args.target,
        out_dir=args.out,
        seed=args.seed,
    )

    findings = results["adversarial_findings"]
    print(f"\nCompleted! Evaluated {len(findings)} candidate empirical claims.")
    print(f"------------------------------------------------------------------")
    for f in findings:
        print(f"\n[{f['verdict_badge']}] {f['claim']['source_variable']} -> {f['claim']['target_variable']}")
        print(f"  • Explorer: {f['claim']['description']}")
        print(f"  • Devil's Advocate: {f['audit']['falsification_reason']}")
        print(f"  • Referee: {f['synthesis_summary']}")
        print(f"  • Takeaway: {f['actionable_takeaway']}")

    print(f"\n==================================================================")
    print(f"Audit reports successfully generated:")
    print(f"  - HTML Report:     {results['reports']['html']}")
    print(f"  - Markdown Report: {results['reports']['markdown']}")
    print(f"  - JSON Findings:   {results['reports']['json']}")
    print(f"==================================================================\n")


if __name__ == "__main__":
    main()
