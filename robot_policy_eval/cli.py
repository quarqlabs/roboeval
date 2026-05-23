from __future__ import annotations

import argparse

from .loaders import load_eval_config
from .runner import EvalRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local robot policy evals.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an eval config.")
    run_parser.add_argument("config", help="Path to eval config JSON.")
    run_parser.add_argument("--output-dir", default="runs/latest", help="Directory for eval artifacts.")

    args = parser.parse_args()
    if args.command == "run":
        config = load_eval_config(args.config)
        report = EvalRunner(
            policies=config.policies,
            scenarios=config.scenarios,
            success_criteria=config.success_criteria,
            baseline_policy=config.baseline_policy,
        ).run()
        report.save(args.output_dir)
        print(f"Eval complete. Report: {args.output_dir}/report.md")
        print(f"Regressions: {len(report.regressions)}")
        print(f"Failure cases: {len(report.failure_cases)}")
