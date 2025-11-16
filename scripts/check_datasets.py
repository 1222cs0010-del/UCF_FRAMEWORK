#!/usr/bin/env python3
"""
Check presence of required datasets for UCF pipeline.
Attempts to load each dataset via `UCFDataLoader` and writes a JSON report.
"""
import json
import time
import sys
from pathlib import Path

# Ensure project root is importable when script is executed from `scripts/`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ucf_data_utils import UCFDataLoader
from datasets import load_dataset

REQUIRED_DATASETS = [
    # use full dataset ids where possible (preferred list)
    "QuixiAI/Code-290k-ShareGPT-Vicuna",
    "vicgalle/alpaca-gpt4",
    "gsm8k",
    "ChilleD/SVAMP",
    "hendrycks/competition_math",
    "Open-Orca/OpenOrca",
]


def check_dataset(name: str, max_samples: int = 500):
    """Try loading a small sample of the named dataset and return status info."""
    start = time.time()
    info = {"name": name, "status": "unknown", "samples": 0, "error": None, "time_s": 0.0}
    try:
        # If given a HF-style dataset id (contains '/'), try loading directly
        if "/" in name or name in ["gsm8k", "ChilleD/SVAMP"]:
            # Use project root as cache directory to avoid relying on DEFAULT_CACHE_DIR
            ds = load_dataset(name, split="train", cache_dir=str(PROJECT_ROOT))
            if max_samples:
                try:
                    ds = ds.select(range(min(max_samples, len(ds))))
                except Exception:
                    # streaming or unknown length
                    pass
            info.update({"status": "ok", "samples": len(ds) if hasattr(ds, '__len__') else max_samples, "time_s": time.time() - start})
            return info

        # Fallback to UCFDataLoader convenience loaders for legacy short names
        if name == "sharegpt":
            data = UCFDataLoader.load_sharegpt(max_samples=max_samples)
        elif name in ["alpaca-gpt4", "alpaca_gpt4", "alpacagpt4"]:
            data = UCFDataLoader.load_alpaca_gpt4(max_samples=max_samples)
        elif name == "bold":
            data = UCFDataLoader.load_bold(max_samples=max_samples)
        elif name == "gsm8k":
            data = UCFDataLoader.load_gsm8k(max_samples=max_samples)
        elif name == "svamp":
            data = UCFDataLoader.load_svamp(max_samples=max_samples)
        elif name == "c4":
            data = UCFDataLoader.load_c4(max_samples=max_samples)
        elif name == "massive":
            data = UCFDataLoader.load_massive(max_samples=max_samples)
        else:
            info.update({"status": "unsupported", "time_s": time.time() - start})
            return info

        info.update({"status": "ok", "samples": len(data), "time_s": time.time() - start})
    except Exception as e:
        info.update({"status": "error", "error": str(e), "time_s": time.time() - start})

    return info


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check datasets for UCF pipeline")
    parser.add_argument("--max-samples", type=int, default=500, help="Maximum samples to load per dataset (default: 500)")
    args = parser.parse_args()

    out_dir = Path("./pipeline_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"checked_at": time.time(), "results": []}

    for ds in REQUIRED_DATASETS:
        print(f"Checking dataset: {ds} (max_samples={args.max_samples}) ...")
        res = check_dataset(ds, max_samples=args.max_samples)
        print(f"  -> {res['status']}, samples={res.get('samples',0)}, time={res['time_s']:.2f}s")
        report["results"].append(res)

    report_file = out_dir / "dataset_check.json"
    with report_file.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to: {report_file}")


if __name__ == '__main__':
    main()
