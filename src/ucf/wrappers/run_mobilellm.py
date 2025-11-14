"""
Wrapper for MobileLLM baseline.
"""
import argparse
import os
import subprocess


def run_smoke(baselines_dir, out_path):
    tpl = os.path.join(os.path.dirname(__file__), "run_template.py")
    subprocess.check_call(["python", tpl, "--baseline", "MobileLLM", "--out", out_path, "--workdir", baselines_dir])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines_dir", default=os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "baselines")))
    parser.add_argument("--out", default="results/mobilellm_metrics.json")
    parser.add_argument("--mode", choices=["smoke"], default="smoke")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.mode == "smoke":
        run_smoke(args.baselines_dir, args.out)
    else:
        raise SystemExit("Only smoke mode supported by wrapper template")
