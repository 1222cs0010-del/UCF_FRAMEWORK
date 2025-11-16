#!/usr/bin/env python3
"""
Plot ablation and evaluation results produced by `scripts/run_ablation.py`.

Generates:
- Ablation bar chart (BLEU vs components)
- Latency vs Model Size scatter
- Comparative bars for UCF-SLM vs LLaMA/Phi/MobileLLM (if provided)
- Fairness heatmap and gender-bias before/after GEEP

Usage:
    python3 scripts/plot_results.py

Requires: matplotlib, seaborn, pandas
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open() as f:
        data = json.load(f)
    records = []
    for r in data:
        rec = {
            "name": r["config"].get("name"),
            "bleu": r["metrics"]["bleu"],
            "latency_ms": r["metrics"]["latency_ms"],
            "gender_parity": r["metrics"]["gender_parity"],
            "context_length": r["metrics"]["context_length"],
            "model_size_mb": r["metrics"]["model_size_mb"],
            "quant": r["config"].get("quant"),
            "use_csm": r["config"].get("use_csm"),
            "use_geep": r["config"].get("use_geep"),
            "use_qlora": r["config"].get("use_qlora"),
        }
        records.append(rec)
    return pd.DataFrame(records)


def plot_ablation_bleu(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(8, 5))
    sns.barplot(x="name", y="bleu", data=df, palette="viridis")
    plt.title("Ablation: BLEU by Configuration")
    plt.ylabel("BLEU")
    plt.xlabel("Configuration")
    plt.tight_layout()
    p = out_dir / "ablation_bleu.png"
    plt.savefig(p)
    print(f"Saved: {p}")


def plot_latency_vs_size(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x="model_size_mb", y="latency_ms", hue="name", data=df, s=120)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Latency vs Model Size")
    plt.xlabel("Model Size (MB, log)")
    plt.ylabel("Latency (ms, log)")
    plt.tight_layout()
    p = out_dir / "latency_vs_size.png"
    plt.savefig(p)
    print(f"Saved: {p}")


def plot_fairness_heatmap(df: pd.DataFrame, out_dir: Path):
    # Create a simple matrix of BLEU vs gender_parity by config
    pivot = df.pivot(index='name', columns='use_geep', values='gender_parity')
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Gender Parity (use_geep False/True)')
    p = out_dir / 'fairness_heatmap.png'
    plt.savefig(p)
    print(f"Saved: {p}")


def plot_gender_bias_before_after(df: pd.DataFrame, out_dir: Path):
    # Compare gender_parity for pairs of configs with and without GEEP
    df_sorted = df.set_index('name')
    plt.figure(figsize=(8, 5))
    sns.barplot(x='name', y='gender_parity', data=df, palette='magma')
    plt.title('Gender Parity by Configuration')
    plt.ylabel('Gender Parity')
    plt.xlabel('Configuration')
    p = out_dir / 'gender_parity_by_config.png'
    plt.tight_layout()
    plt.savefig(p)
    print(f"Saved: {p}")


def main():
    out_dir = Path('./pipeline_output')
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / 'ablation_results.json'
    df = load_results(results_file)

    plot_ablation_bleu(df, out_dir)
    plot_latency_vs_size(df, out_dir)
    plot_fairness_heatmap(df, out_dir)
    plot_gender_bias_before_after(df, out_dir)


if __name__ == '__main__':
    main()
