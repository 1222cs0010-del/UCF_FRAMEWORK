# UCF Framework - Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- 16GB+ RAM
- CUDA 11.8+ (for GPU acceleration, optional but recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/1222cs0010-del/UCF_FRAMEWORK.git
cd UCF_FRAMEWORK
```

### 2. Set Up Virtual Environment

#### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda

```bash
conda env create -f environment.yml
conda activate ucf
```

### 3. Install PyTorch

For GPU support (CUDA 12.x):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU only:
```bash
pip install torch torchvision torchaudio
```

### 4. Verify Installation

```bash
python3 -c "
import torch
import transformers
import datasets
print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ Datasets:', datasets.__version__)
print('✓ CUDA Available:', torch.cuda.is_available())
"
```

---

## 📋 Project Structure

```
UCF_FRAMEWORK/
├── ucf_pipeline_final.py          # Main pipeline orchestrator
├── ucf_data_utils.py              # Data loading and preprocessing
├── ucf_core/                      # Core pipeline implementation
├── baselines/                     # SOTA baseline implementations
│   ├── distilling-step-by-step/   # Knowledge distillation
│   ├── llm-awq/                   # Quantization
│   ├── streaming-llm/             # Long-context inference
│   ├── qlora/                     # Parameter-efficient tuning
│   ├── geep/                      # Fairness optimization
│   └── MobileLLM/                 # Mobile optimization
├── configs/                       # Configuration files
├── scripts/                       # Utility scripts
├── pipeline_output/               # Pipeline execution results
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── START_HERE.md                  # Getting started guide
├── DISTILLATION_CODE_ANALYSIS.md  # Code review and analysis
└── README.md                      # Original README
```

---

## 🔧 Main Components

### 1. Pipeline Orchestrator (`ucf_pipeline_final.py`)

The main entry point with 5 stages:

```bash
# Run full pipeline with default settings
python3 ucf_pipeline_final.py

# Run with specific dataset
python3 ucf_pipeline_final.py --dataset sharegpt --samples 1000

# Run specific stages only
python3 ucf_pipeline_final.py --stages 1,3,5
```

**Stages:**
1. **Load Input** - Load and preprocess data
2. **CA-KD Distilling** - Knowledge distillation with teacher-student models
3. **CSM Streaming** - Context-aware streaming for long sequences
4. **Quantization** - Model compression (AWQ + GPTQ)
5. **Fairness** - Debiasing and fairness evaluation

### 2. Data Utilities (`ucf_data_utils.py`)

Handles data loading and preprocessing:

```python
from ucf_data_utils import UCFDataLoader

loader = UCFDataLoader()
data = loader.load_svamp(split='train', limit=1000)
```

### 3. Configuration (`configs/`)

Edit configuration files to customize:

```yaml
# configs/default.yaml
pipeline:
  teacher_model: "meta-llama/Llama-2-7b-hf"
  student_model: "microsoft/Phi-3-mini-4k-instruct"
  batch_size: 32
  num_epochs: 3
```

---

## 📊 Available Datasets

The framework supports multiple datasets:

| Dataset | Size | Domain | Load Method |
|---------|------|--------|-------------|
| ShareGPT | ~50K | General Q&A | Automatic |
| SVAMP | 1K | Math Problems | Automatic |
| GSM8K | 8.8K | Grade School Math | Automatic |
| Open Orca | ~500K | Multi-task | Automatic |
| C4 | Large | Web Text | Streaming |

**Load specific dataset:**

```bash
python3 ucf_pipeline_final.py --dataset gsm8k
python3 ucf_pipeline_final.py --dataset svamp --samples 500
```

---

## 🎯 Models Supported

### Teacher Models (for distillation)
- `meta-llama/Llama-2-7b-hf` - 7B parameters
- `meta-llama/Llama-2-13b-hf` - 13B parameters

### Student Models (for optimization)
- `microsoft/Phi-3-mini-4k-instruct` - 3.8B parameters
- `google/t5-v1_1-base` - 220M parameters (distillation)

### Quantization Targets
- AWQ: 4-bit quantization
- GPTQ: 4-bit/8-bit quantization

---

## 🚦 Running Examples

### Example 1: Quick Demo (1 minute)

```bash
python3 ucf_pipeline_final.py --dataset sharegpt --samples 100 --quick-mode
```

### Example 2: Full Pipeline with SVAMP

```bash
python3 ucf_pipeline_final.py --dataset svamp --samples 1000
```

### Example 3: Quantization Only

```bash
python3 ucf_pipeline_final.py --stages 4 --dataset sharegpt
```

### Example 4: Custom Configuration

```bash
python3 ucf_pipeline_final.py \
  --config configs/custom.yaml \
  --dataset gsm8k \
  --output pipeline_output_custom
```

---

## 📈 Monitoring Progress

The pipeline outputs metrics to `pipeline_output/`:

```bash
# View results
cat pipeline_output/pipeline_result.json | python -m json.tool

# Check specific metrics
grep "BLEU\|Gender Parity" pipeline_output/final_response.txt
```

**Key Metrics:**
- **BLEU Score** - Text generation quality (0-1, higher is better)
- **Gender Parity** - Fairness metric (0-1, closer to 0.5 is better)
- **Inference Time** - Model speed (seconds)
- **Memory Usage** - GPU/CPU memory consumption

---

## 🔍 Troubleshooting

### Issue: "PyTorch not installed"

```bash
# Solution: Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "CUDA out of memory"

```bash
# Solutions:
# 1. Reduce batch size
python3 ucf_pipeline_final.py --batch-size 8

# 2. Use CPU instead
CUDA_VISIBLE_DEVICES="" python3 ucf_pipeline_final.py

# 3. Use smaller models
python3 ucf_pipeline_final.py --student-model "microsoft/Phi-3-mini-4k-instruct"
```

### Issue: "Dataset not found"

```bash
# Solutions:
# 1. Check internet connection
# 2. Clear cache
rm -rf ~/.cache/huggingface

# 3. Manually download
huggingface-cli download meta-llama/Llama-2-7b-hf
```

### Issue: Pipeline stages show 0.00s timing

**Expected behavior** - Some stages may complete instantly if dependencies are not installed or in fallback mode. This is normal and documented in `DISTILLATION_CODE_ANALYSIS.md`.

---

## 📚 Advanced Usage

### Custom Training Configuration

Edit `configs/unified_experiment.yaml`:

```yaml
training:
  learning_rate: 5e-5
  num_epochs: 10
  warmup_steps: 500
  weight_decay: 0.01
```

### Pipeline Modes

```bash
# Smoke test (minimal, <1 min)
python3 ucf_pipeline_final.py --mode smoke

# Standard (full pipeline)
python3 ucf_pipeline_final.py --mode standard

# Debug (verbose logging)
python3 ucf_pipeline_final.py --mode debug
```

### Baseline Integration

Each baseline is independently testable:

```bash
# Test distillation baseline
python3 baselines/distilling-step-by-step/run.py

# Test quantization baseline
python3 baselines/llm-awq/run.py

# Test streaming baseline
python3 baselines/streaming-llm/run.py
```

---

## 📖 Documentation

- **START_HERE.md** - First-time setup guide
- **DISTILLATION_CODE_ANALYSIS.md** - Detailed code review
- **QUICK_START.md** - Quick reference
- **README_UCF.md** - Research strategy and publication plan

---

## 🛠 Development

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_pipeline.py -v
```

### Code Quality

```bash
# Format code
black ucf_*.py

# Check types
mypy ucf_*.py

# Lint
pylint ucf_*.py
```

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🤝 Support

For issues, questions, or suggestions:

1. Check existing issues: https://github.com/1222cs0010-del/UCF_FRAMEWORK/issues
2. Create new issue with detailed description
3. Include error logs and system information

---

## 📞 Contact

- **Repository**: https://github.com/1222cs0010-del/UCF_FRAMEWORK
- **Issues**: https://github.com/1222cs0010-del/UCF_FRAMEWORK/issues

---

**Last Updated**: November 16, 2025  
**Framework Version**: 1.0.0  
**Status**: Ready for Deployment ✓
