"""
Generic wrapper template for baselines in the UCF workspace.
This script is intentionally lightweight and only uses standard library so it can run in a smoke test.
It demonstrates: logging, deterministic seed, writing metrics JSON, and printing repo info.

Usage (smoke test):
  python run_template.py --baseline TEST --out results/sample_metrics.json
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time


def get_git_commit(path):
    try:
        p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path, capture_output=True, text=True, check=True)
        return p.stdout.strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="template", help="Baseline name")
    parser.add_argument("--out", type=str, default="results/metrics_template.json", help="Output metrics JSON path")
    parser.add_argument("--workdir", type=str, default=os.path.abspath(os.path.join(os.getcwd(), os.pardir, "baselines")), help="Baselines folder")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    info = {
        "baseline": args.baseline,
        "timestamp": int(time.time()),
        "python": sys.version.replace('\n', ' '),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "baselines_folder": args.workdir,
        "example_metric": 0.0,
    }

    # try to find a matching repo and commit
    candidate = os.path.join(args.workdir, args.baseline)
    if os.path.isdir(candidate):
        info["git_commit"] = get_git_commit(candidate)
    else:
        info["git_commit"] = None

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("Wrote sample metrics to:", args.out)


if __name__ == "__main__":
    main()
