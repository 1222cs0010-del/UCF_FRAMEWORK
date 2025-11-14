"""
Wrapper for the Distilling Step-by-Step baseline.
This is a lightweight wrapper that documents how to invoke the baseline and provides a smoke-mode.
"""
import argparse
import json
import os
import subprocess


def run_smoke(baselines_dir, out_path):
    # Call the generic template to produce a small metrics file
    tpl = os.path.join(os.path.dirname(__file__), "run_template.py")
    subprocess.check_call(["python", tpl, "--baseline", "distilling-step-by-step", "--out", out_path, "--workdir", baselines_dir])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines_dir", default=os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "baselines")))
    parser.add_argument("--out", default="results/distill_metrics.json")
    parser.add_argument("--mode", choices=["smoke"], default="smoke")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.mode == "smoke":
        run_smoke(args.baselines_dir, args.out)
    else:
        raise SystemExit("Only smoke mode supported by wrapper template")
