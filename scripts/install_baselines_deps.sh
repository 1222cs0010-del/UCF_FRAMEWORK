#!/usr/bin/env bash
set -euo pipefail
# Single installer for UCF baseline repositories
# - Detects/creates `ucf_env` conda env
# - Installs core Python packages into `ucf_env`
# - Installs per-baseline Python deps where available
# - Skips AWQ native kernel compilation (writes manual instructions)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME=${ENV_NAME:-ucf_env}
LOG_FILE="$REPO_ROOT/logs/install_baselines_$(date +%Y%m%d_%H%M%S).log"
DRY_RUN=0
FORCE_REINSTALL=0
SKIP_AWQ_KERNELS=1

usage() {
  cat <<EOF
Usage: $0 [--dry-run] [--force] [--no-skip-awq-kernels]

Options:
  --dry-run               Show actions without performing installs
  --force                 Reinstall packages (pip install --force-reinstall)
  --no-skip-awq-kernels   Attempt AWQ kernel build (may require manual steps)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE_REINSTALL=1; shift ;;
    --no-skip-awq-kernels) SKIP_AWQ_KERNELS=0; shift ;;
    --per-baseline) PER_BASELINE=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

mkdir -p "$REPO_ROOT/logs"
echo "Logging to $LOG_FILE"

run() {
  echo "+ $*" | tee -a "$LOG_FILE"
  if [ "$DRY_RUN" -eq 0 ]; then
    eval "$*" 2>&1 | tee -a "$LOG_FILE"
  fi
}

conda_exists() {
  command -v conda >/dev/null 2>&1
}

ensure_conda_env() {
  if ! conda_exists; then
    echo "ERROR: conda not found in PATH. Install Miniconda/mamba or run scripts/install_miniconda_and_env.sh" | tee -a "$LOG_FILE"
    return 1
  fi
  if conda env list | awk '{print $1}' | grep -xq "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' exists" | tee -a "$LOG_FILE"
  else
    echo "Creating conda env '$ENV_NAME' with python=3.10" | tee -a "$LOG_FILE"
    run "conda create -y -n $ENV_NAME python=3.10"
  fi
}

pip_opts="-q"
if [ "$FORCE_REINSTALL" -eq 1 ]; then
  pip_opts="$pip_opts --force-reinstall"
fi

install_core_deps() {
  echo "\n== Installing core Python packages into conda env '$ENV_NAME' ==" | tee -a "$LOG_FILE"
  REQ_FILE="$REPO_ROOT/configs/requirements_all.txt"
  run "conda run -n $ENV_NAME python -m pip install --upgrade pip setuptools wheel"
  if [ -f "$REQ_FILE" ]; then
    run "conda run -n $ENV_NAME python -m pip install $pip_opts -r \"$REQ_FILE\""
  else
    echo "Unified requirements file not found at $REQ_FILE; falling back to individual packages" | tee -a "$LOG_FILE"
    core_pkgs=(transformers datasets accelerate bitsandbytes sentencepiece wandb scikit-learn scipy psutil tqdm evaluate)
    pkg_list="${core_pkgs[*]}"
    run "conda run -n $ENV_NAME python -m pip install $pip_opts $pkg_list"
  fi
  echo "Note: install torch via Conda to match your CUDA / PyTorch compatibility; see README." | tee -a "$LOG_FILE"
}

install_qlora() {
  QLORA_DIR="$REPO_ROOT/baselines/qlora"
  if [ -f "$QLORA_DIR/requirements.txt" ]; then
    echo "\n== Installing QLoRA requirements ==" | tee -a "$LOG_FILE"
    run "conda run -n $ENV_NAME python -m pip install $pip_opts -r '$QLORA_DIR/requirements.txt'"
    # Only attempt editable install if the repo is a Python project
    if [ -f "$QLORA_DIR/setup.py" ] || [ -f "$QLORA_DIR/pyproject.toml" ]; then
      echo "Installing QLoRA package in editable mode" | tee -a "$LOG_FILE"
      run "conda run -n $ENV_NAME python -m pip install $pip_opts -e '$QLORA_DIR'"
    else
      echo "QLoRA does not expose a Python package (no setup.py/pyproject.toml); skipping editable install" | tee -a "$LOG_FILE"
    fi
  else
    echo "QLoRA requirements.txt not found; skipping" | tee -a "$LOG_FILE"
  fi
}

install_streamingllm() {
  S_DIR="$REPO_ROOT/baselines/streaming-llm"
  if [ -f "$S_DIR/setup.py" ]; then
    echo "\n== Installing StreamingLLM deps and package ==" | tee -a "$LOG_FILE"
    echo "Checking for existing PyTorch in conda env '$ENV_NAME'..." | tee -a "$LOG_FILE"
    if conda run -n $ENV_NAME python -c "import importlib; importlib.import_module('torch')" >/dev/null 2>&1; then
      echo "PyTorch detected in '$ENV_NAME' — skipping pip install of torch/vision/torchaudio" | tee -a "$LOG_FILE"
    else
      echo "PyTorch not detected. Attempting conda install of PyTorch + torchaudio (recommended)" | tee -a "$LOG_FILE"
      # Try conda install to get a matching CUDA-enabled wheel first
      if conda install -y -n $ENV_NAME pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch >/dev/null 2>&1; then
        echo "Conda-installed PyTorch + torchaudio successfully." | tee -a "$LOG_FILE"
      else
        echo "Conda install failed; falling back to pip install of torch (may cause incompatibilities)" | tee -a "$LOG_FILE"
        run "conda run -n $ENV_NAME python -m pip install $pip_opts torch torchvision torchaudio"
      fi
    fi
    run "conda run -n $ENV_NAME python -m pip install $pip_opts transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece"
    run "conda run -n $ENV_NAME bash -lc 'cd \"$S_DIR\" && python setup.py develop'"
  else
    echo "StreamingLLM setup.py not found; skipping" | tee -a "$LOG_FILE"
  fi
}

install_distilling() {
  D_DIR="$REPO_ROOT/baselines/distilling-step-by-step"
  echo "\n== Installing Distilling Step-by-Step minimal deps ==" | tee -a "$LOG_FILE"
  run "conda run -n $ENV_NAME python -m pip install $pip_opts 'git+https://github.com/huggingface/transformers@v4.24.0' datasets sentencepiece 'protobuf==3.20.*' tensorboardX"
  # The repo is script-based; no package install.
  echo "Note: Distilling Step-by-Step is script-based; follow its README to run experiments." | tee -a "$LOG_FILE"
}

install_mobilellm() {
  M_DIR="$REPO_ROOT/baselines/MobileLLM"
  echo "\n== Installing MobileLLM requirements (if present) ==" | tee -a "$LOG_FILE"
  if [ -f "$M_DIR/requirement.txt" ]; then
    run "conda run -n $ENV_NAME python -m pip install $pip_opts -r '$M_DIR/requirement.txt'"
  elif [ -f "$M_DIR/requirements.txt" ]; then
    run "conda run -n $ENV_NAME python -m pip install $pip_opts -r '$M_DIR/requirements.txt'"
  else
    echo "MobileLLM requirements file not found; skipping" | tee -a "$LOG_FILE"
  fi
}

install_awq() {
  A_DIR="$REPO_ROOT/baselines/llm-awq"
  echo "\n== Installing AWQ Python package (editable) - Kernel build skipped by default ==" | tee -a "$LOG_FILE"
  if [ -f "$A_DIR/pyproject.toml" ] || [ -f "$A_DIR/setup.py" ]; then
    run "conda run -n $ENV_NAME python -m pip install $pip_opts -e '$A_DIR'" || echo "[WARN] AWQ editable install failed"
    if [ "$SKIP_AWQ_KERNELS" -eq 1 ]; then
      echo "AWQ native kernels not built. See: $A_DIR/awq/kernels/README or the file $REPO_ROOT/baselines/llm-awq/AWQ_KERNELS_MANUAL.txt" | tee -a "$LOG_FILE"
      cat > "$REPO_ROOT/baselines/llm-awq/AWQ_KERNELS_MANUAL.txt" <<'MAN'
AWQ native kernels manual steps (run manually):

1) Edit pyproject.toml if required (see repo README for edge devices)
2) Set TORCH_CUDA_ARCH_LIST appropriately for your GPU, e.g.:
   export TORCH_CUDA_ARCH_LIST="compute_86,code=sm_86"
3) Build kernels:
   cd awq/kernels
   python setup.py install
4) Install flash-attn wheel if needed:
   pip install flash-attn --no-build-isolation

Run these steps in a shell where your CUDA toolkit and compiler match your PyTorch installation.
MAN
    else
      echo "AWQ kernel build requested. Attempting kernel build now (may require CUDA/toolchain)." | tee -a "$LOG_FILE"
      run "conda run -n $ENV_NAME bash -lc 'cd \"$A_DIR/awq/kernels\" && python setup.py install'" || echo "[WARN] AWQ kernel build failed"
    fi
  else
    echo "AWQ repo not found at $A_DIR; skipping" | tee -a "$LOG_FILE"
  fi
}

create_per_baseline_envs() {
  echo "\n== Creating per-baseline conda envs and installing their deps ==" | tee -a "$LOG_FILE"
  # Distilling env
  if [ -f "$REPO_ROOT/configs/envs/distilling_env.yaml" ]; then
    run "conda env create -f $REPO_ROOT/configs/envs/distilling_env.yaml --force || conda env update -f $REPO_ROOT/configs/envs/distilling_env.yaml"
  else
    run "conda create -y -n distill python=3.10"
  fi
  # AWQ env
  if [ -f "$REPO_ROOT/configs/envs/awq_env.yaml" ]; then
    run "conda env create -f $REPO_ROOT/configs/envs/awq_env.yaml --force || conda env update -f $REPO_ROOT/configs/envs/awq_env.yaml"
  else
    run "conda create -y -n awq_env python=3.10"
  fi
  # Streaming env
  if [ -f "$REPO_ROOT/configs/envs/streaming_env.yaml" ]; then
    run "conda env create -f $REPO_ROOT/configs/envs/streaming_env.yaml --force || conda env update -f $REPO_ROOT/configs/envs/streaming_env.yaml"
  else
    run "conda create -y -n streaming_env python=3.10"
  fi
  # QLoRA env
  if [ -f "$REPO_ROOT/configs/envs/qlora_env.yaml" ]; then
    run "conda env create -f $REPO_ROOT/configs/envs/qlora_env.yaml --force || conda env update -f $REPO_ROOT/configs/envs/qlora_env.yaml"
  else
    run "conda create -y -n qlora_env python=3.10"
  fi
  # MobileLLM env
  if [ -f "$REPO_ROOT/configs/envs/mobilellm_env.yaml" ]; then
    run "conda env create -f $REPO_ROOT/configs/envs/mobilellm_env.yaml --force || conda env update -f $REPO_ROOT/configs/envs/mobilellm_env.yaml"
  else
    run "conda create -y -n mobilellm_env python=3.10"
  fi

  # Install per-repo pip requirements into their envs when found
  if [ -f "$REPO_ROOT/baselines/llm-awq/requirements.txt" ] || [ -f "$REPO_ROOT/baselines/llm-awq/pyproject.toml" ]; then
    echo "Installing AWQ python requirements into awq_env" | tee -a "$LOG_FILE"
    run "conda run -n awq_env python -m pip install $pip_opts -r '$REPO_ROOT/baselines/llm-awq/requirements.txt'" || true
    run "conda run -n awq_env python -m pip install $pip_opts -e '$REPO_ROOT/baselines/llm-awq'" || true
  fi
  if [ -f "$REPO_ROOT/baselines/distilling-step-by-step/requirements.txt" ]; then
    echo "Installing Distilling requirements into distill env" | tee -a "$LOG_FILE"
    run "conda run -n distill python -m pip install $pip_opts -r '$REPO_ROOT/baselines/distilling-step-by-step/requirements.txt'" || true
  fi
  if [ -f "$REPO_ROOT/baselines/streaming-llm/requirements.txt" ]; then
    echo "Installing StreamingLLM requirements into streaming_env" | tee -a "$LOG_FILE"
    run "conda run -n streaming_env python -m pip install $pip_opts -r '$REPO_ROOT/baselines/streaming-llm/requirements.txt'" || true
    # attempt editable install if packaging present
    if [ -f "$REPO_ROOT/baselines/streaming-llm/pyproject.toml" ] || [ -f "$REPO_ROOT/baselines/streaming-llm/setup.py" ]; then
      run "conda run -n streaming_env python -m pip install $pip_opts -e '$REPO_ROOT/baselines/streaming-llm'" || true
    fi
  fi
  if [ -f "$REPO_ROOT/baselines/qlora/requirements.txt" ]; then
    echo "Installing QLoRA requirements into qlora_env" | tee -a "$LOG_FILE"
    run "conda run -n qlora_env python -m pip install $pip_opts -r '$REPO_ROOT/baselines/qlora/requirements.txt'" || true
  fi
  if [ -f "$REPO_ROOT/baselines/MobileLLM/requirement.txt" ] || [ -f "$REPO_ROOT/baselines/MobileLLM/requirements.txt" ]; then
    echo "Installing MobileLLM requirements into mobilellm_env" | tee -a "$LOG_FILE"
    if [ -f "$REPO_ROOT/baselines/MobileLLM/requirement.txt" ]; then
      run "conda run -n mobilellm_env python -m pip install $pip_opts -r '$REPO_ROOT/baselines/MobileLLM/requirement.txt'" || true
    else
      run "conda run -n mobilellm_env python -m pip install $pip_opts -r '$REPO_ROOT/baselines/MobileLLM/requirements.txt'" || true
    fi
  fi
}

main() {
  ensure_conda_env || exit 1
  install_core_deps
  install_qlora
  install_streamingllm
  install_distilling
  install_mobilellm
  install_awq

  echo "\nAll requested install steps finished. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
}

main
