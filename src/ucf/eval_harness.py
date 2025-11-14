"""
Evaluation harness to read simple JSON metric outputs from wrappers and compute aggregated report.
This is intentionally small — it's a starting point for adding standardized metrics (accuracy, fairness gaps, latency, memory).
"""
import json
import os
from glob import glob


def aggregate_metrics(results_folder="results"):
    out = {}
    paths = glob(os.path.join(results_folder, "*.json"))
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = os.path.splitext(os.path.basename(p))[0]
            out[name] = data
        except Exception as e:
            out[os.path.basename(p)] = {"error": str(e)}
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/aggregate_report.json")
    args = parser.parse_args()

    agg = aggregate_metrics(args.results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print("Wrote aggregated report:", args.out)
