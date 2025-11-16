#!/usr/bin/env bash
set -euo pipefail

# AWQ Demo Script — Quick inference with quantized model
# This script demonstrates how to run AWQ with a pre-quantized model

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="awq_env"
BASELINES_DIR="$REPO_ROOT/baselines"

echo "=== AWQ Baseline Demo ==="
echo "Environment: $ENV_NAME"
echo "Repository: $BASELINES_DIR/llm-awq"
echo ""

# Set LD_LIBRARY_PATH for PyTorch libraries (required for native kernels)
export LD_LIBRARY_PATH="/home/cse-sdpl/miniconda/envs/$ENV_NAME/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# Change to AWQ directory
cd "$BASELINES_DIR/llm-awq"

# Example 1: Check AWQ installation and validate imports
echo "Step 1: Validating AWQ installation..."
conda run -n $ENV_NAME bash -lc "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH && python - <<'PYEOF'
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')

try:
    import awq_inference_engine
    print('✅ AWQ inference engine loaded')
except Exception as e:
    print('❌ AWQ inference engine:', e)
    sys.exit(1)

print('\n✅ All imports successful!')
PYEOF"

echo ""
echo "Step 2: AWQ usage examples..."
echo ""
echo "To run AWQ quantization/inference on a model, use the entry point:"
echo ""
echo "  conda run -n $ENV_NAME python -m awq.entry \\"
echo "    --model_path meta-llama/Llama-2-7b-hf \\"
echo "    --w_bit 4 --q_group_size 128 \\"
echo "    --tasks wikitext"
echo ""
echo "Common options:"
echo "  --model_path PATH          Path to HuggingFace model"
echo "  --w_bit {3,4}              Quantization bit width"
echo "  --q_group_size N           Quantization group size (128, 256, etc.)"
echo "  --load_awq PATH            Load pre-computed AWQ search results"
echo "  --load_quant PATH          Load quantized model weights"
echo "  --tasks TASK               Evaluation task (wikitext, mmlu, etc.)"
echo "  --q_backend {fake,real}    Quantization backend (fake=pseudo, real=actual)"
echo ""
echo "Step 3: Try a small model with pre-quantized weights..."
echo ""
echo "  # (Requires downloading AWQ model zoo; see README)"
echo "  git clone https://huggingface.co/datasets/mit-han-lab/awq-model-zoo awq_cache"
echo ""
echo "Then run:"
echo "  conda run -n $ENV_NAME python -m awq.entry \\"
echo "    --model_path meta-llama/Llama-2-7b \\"
echo "    --w_bit 4 --q_group_size 128 \\"
echo "    --load_awq awq_cache/llama2-7b-w4-g128.pt \\"
echo "    --q_backend fake --tasks wikitext"
echo ""
echo "For more details, see:"
echo "  - $BASELINES_DIR/llm-awq/README.md"
echo "  - $REPO_ROOT/BASELINES_SETUP.md (AWQ section)"
echo ""
