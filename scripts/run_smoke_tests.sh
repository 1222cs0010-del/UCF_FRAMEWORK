#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Running smoke import tests for each baseline env"

run_in_env() {
  ENV_NAME=$1
  PYCODE=$2
  echo "--- testing env: $ENV_NAME ---"
  conda run -n "$ENV_NAME" python - <<PY
try:
    $PYCODE
    print('OK')
except Exception as e:
    import traceback
    traceback.print_exc()
    raise
PY
}

echo "Testing AWQ (awq_env)"
run_in_env awq_env "import awq; print('awq ok')"

echo "Testing Distilling (distill)"
run_in_env distill "import transformers; print('transformers', transformers.__version__)"

echo "Testing StreamingLLM (streaming_env)"
run_in_env streaming_env "import streaming_llm; print('streaming_llm ok')"

echo "Testing QLoRA (qlora_env)"
run_in_env qlora_env "import importlib; print('qlora importable?', importlib.util.find_spec('qlora'))"

echo "Testing MobileLLM (mobilellm_env)"
run_in_env mobilellm_env "import importlib; print('mobilellm importable?', importlib.util.find_spec('MobileLLM'))"

echo "Smoke tests finished"
