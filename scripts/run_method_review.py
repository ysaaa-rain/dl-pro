"""Run each fixed stage with immutable command, code/config and exit records."""

import argparse
from dlpro.method_review import DEFAULT_CONFIG, load_config, logged_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=["prepare", "baseline", "novelty", "train", "evaluate", "diagnose", "candidate_audit", "cluster_audit", "summarize", "integrity", "test"], required=True)
    parser.add_argument("--attempt", default="")
    parser.add_argument("--report-output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    commands = {
        "prepare": ["scripts/prepare_method_review.py", "--config", args.config],
        "baseline": ["scripts/reproduce_method_baselines.py", "--config", args.config],
        "novelty": ["scripts/benchmark_novelty.py", "--config", args.config],
        "train": ["scripts/train_allocation_review.py", "--config", args.config],
        "evaluate": ["scripts/evaluate_allocation_review.py", "--config", args.config],
        "diagnose": ["scripts/diagnose_method_review.py", "--config", args.config],
        "candidate_audit": ["scripts/audit_novelty_candidates.py", "--config", args.config],
        "cluster_audit": ["scripts/cluster_review_candidates.py", "--config", args.config],
        "summarize": ["scripts/summarize_method_review.py", "--config", args.config],
        "integrity": ["scripts/audit_method_review.py", "--config", args.config],
        "test": ["-m", "pytest", "-q"],
    }
    if args.report_output:
        if args.stage != "summarize":
            parser.error("--report-output only applies to summarize")
        commands["summarize"] += ["--output", args.report_output]
    logged_run(cfg, args.stage + args.attempt, commands[args.stage])


if __name__ == "__main__":
    main()
