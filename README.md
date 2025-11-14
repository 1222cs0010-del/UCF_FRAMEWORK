# Unified Conversion Framework (UCF)

Efficient, Coherent, Fair, and Conversational AI on Edge Devices.

## Project Structure

```
D:\ucf_framework/
├── baselines/                          # Cloned SOTA baselines
│   ├── distilling-step-by-step/        # Knowledge distillation (ACL 2023)
│   ├── llm-awq/                        # Quantization (MLSys 2024)
│   ├── streaming-llm/                  # Long-context streaming (EMNLP/ICLR)
│   ├── qlora/                          # Quantized LoRA fine-tuning (NeurIPS 2023)
│   └── MobileLLM/                      # Mobile LLM optimization (ICML)
├── src/
│   └── ucf/
│       ├── __init__.py
│       ├── wrappers/                   # Baseline integration wrappers
│       │   ├── run_template.py         # Generic template
│       │   ├── run_distilling.py       # Distillation wrapper
│       │   ├── run_awq.py              # Quantization wrapper
│       │   ├── run_qlora.py            # QLoRA wrapper
│       │   ├── run_streaming.py        # StreamingLLM wrapper
│       │   └── run_mobilellm.py        # MobileLLM wrapper
│       └── eval_harness.py             # Evaluation aggregator
├── scripts/
│   └── create_conda_env.ps1            # Windows conda setup helper
├── experiments/
│   └── sample_config.yaml              # Sample experiment config
├── results/                            # Output metrics and reports
├── data/                               # Datasets (placeholder)
├── ckpts/                              # Checkpoints (placeholder)
├── environment.yml                     # Base conda environment
├── environment_ucf.yml                 # Extended env (generated)
├── requirements_ucf.txt                # Aggregated pip requirements
├── README_UCF.md                       # Research strategy and publication plan
└── README.md                           # This file
```

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/krishnamurthi-ramesh/ucf_framework.git
cd ucf_framework
```

### 2. Create Conda Environment (Windows)

```powershell
# In PowerShell, from D:\ucf_framework
conda env create -f environment.yml
conda activate ucf

# Install PyTorch for CUDA 12.8 (adjust to your CUDA version)
conda install -y pytorch pytorch-cuda=12.8 -c pytorch -c nvidia

# Install common pip packages
pip install --no-cache-dir accelerate transformers datasets peft sentencepiece evaluate
```

### 3. Verify GPU Setup

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
nvidia-smi
```

### 4. Run Smoke Tests

```bash
python src/ucf/wrappers/run_distilling.py --mode smoke --out results/distill_metrics.json
python src/ucf/wrappers/run_awq.py --mode smoke --out results/awq_metrics.json
python src/ucf/eval_harness.py --results results --out results/agg_report.json
```

## Research Goals

- **Efficient**: Quantization (AWQ, QLoRA) and distillation for edge-device deployment
- **Coherent**: StreamingLLM for long-context generation with preserved quality
- **Fair**: Parameter-efficient debiasing (GEEP) without catastrophic forgetting
- **Conversational**: Unified framework for fine-tuning and inference on edge devices (RTX 2000 Ada 16GB)

## Workstation Specifications

- **Model**: HP Z4 Tower G5 Business Workstation
- **CPU**: Intel Xeon W5-2565X (18 cores, 3.2–4.8 GHz)
- **RAM**: 64 GB DDR5 4800 MHz
- **GPU**: NVIDIA RTX 2000 Ada 16 GB

## Key Baselines

1. **Distilling Step-by-Step** (ACL 2023): Knowledge distillation with fewer parameters and data
2. **AWQ** (MLSys 2024): Activation-aware weight quantization for LLM compression
3. **QLoRA** (NeurIPS 2023): Efficient fine-tuning of quantized LLMs
4. **StreamingLLM** (EMNLP 2023): Efficient inference on infinite-length inputs
5. **MobileLLM** (ICML 2024): Mobile-optimized LLM architectures

## Publication Targets

See `README_UCF.md` for:
- Detailed research strategy and experimental design
- Target conferences (A*/A tier): NeurIPS, ICLR, ACL, EMNLP, MLSys
- Timeline and submission strategy
- Evaluation metrics and ablation study design

## Environment Setup Details

### Python Version
- Python 3.10 (recommended for broad library support)

### Key Dependencies
- `transformers` >= 4.30
- `accelerate` (for distributed training and mixed precision)
- `peft` (parameter-efficient fine-tuning)
- `datasets` (data loading and preprocessing)
- `sentencepiece` (tokenization)
- `evaluate` (metrics)

### Windows-Specific Notes
- Use Miniconda for ease of PyTorch GPU setup
- Some packages (bitsandbytes, AWQ native kernels) may require special Windows wheels or WSL2
- For maximum compatibility with native GPU code, consider WSL2 + Ubuntu

## Troubleshooting

### Conda not found after installation
- Restart PowerShell to load the conda shell integration from `conda init`
- Verify with `conda --version`

### PyTorch / CUDA mismatch
- Run `nvidia-smi` to confirm your CUDA version (e.g., 12.8)
- Install matching PyTorch: `conda install pytorch pytorch-cuda=12.8 -c pytorch -c nvidia`

### bitsandbytes / AWQ native extensions fail on Windows
- Option 1: Use WSL2 + Ubuntu for full native GPU toolchain
- Option 2: Install prebuilt wheels (I can provide links)

## Next Steps

1. Create and activate the conda environment
2. Run smoke tests to verify setup
3. Implement per-baseline integration (actual CLI invocation, not just template)
4. Conduct small-scale ablation studies
5. Prepare paper outline and experiments for top-tier venues

## Contributing

This is a research workspace. Please:
- Log all experiments in `experiments/` with YAML configs
- Store results in `results/` with a consistent JSONL/CSV format
- Document any modifications to baselines in commit messages
- Use semantic versioning for releases

---

For detailed strategy, timeline, and publication plans, see `README_UCF.md`.
