#!/usr/bin/env bash
set -euo pipefail

# install Miniconda locally and create the ucf_env environment from YAML
# Usage: ./scripts/install_miniconda_and_env.sh [install-prefix]
# Example: ./scripts/install_miniconda_and_env.sh $HOME/miniconda

PREFIX=${1:-$HOME/miniconda}
ENV_YAML="/home/$USER/Desktop/ucf_framework/configs/envs/ucf_env.yaml"
TMPSH="/tmp/miniconda_installer.sh"

echo "Installing Miniconda to $PREFIX"
if [ -d "$PREFIX" ]; then
  echo "Prefix $PREFIX already exists, skipping Miniconda install"
else
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$TMPSH"
  bash "$TMPSH" -b -p "$PREFIX"
  rm -f "$TMPSH"
fi

# Use the conda binary directly to avoid requiring 'conda init' in the caller shell
CONDA_BIN="$PREFIX/bin/conda"
if [ ! -x "$CONDA_BIN" ]; then
  echo "Conda binary not found at $CONDA_BIN" >&2
  exit 2
fi

echo "Creating/updating Conda environment from $ENV_YAML"
$CONDA_BIN env create -f "$ENV_YAML" || $CONDA_BIN env update -f "$ENV_YAML"

echo "Environment created. To activate it in your shell, run:"
echo "  source $PREFIX/etc/profile.d/conda.sh"
echo "  conda activate ucf_env"

echo "Or use conda directly without activating via:"
echo "  $CONDA_BIN run -n ucf_env pytest -q"

echo "Done."
