#!/usr/bin/env bash
set -euo pipefail

# Create a Python venv and install minimal requirements for the UCF project
# Usage: ./scripts/setup_venv.sh [venv-path]
# Example: ./scripts/setup_venv.sh .venv

VENV_PATH=${1:-.venv}
REQ_FILE="/home/$USER/Desktop/ucf_framework/requirements.txt"

python3 -m venv "$VENV_PATH"
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
if [ -f "$REQ_FILE" ]; then
  pip install -r "$REQ_FILE"
else
  echo "Requirements file not found at $REQ_FILE"
  exit 1
fi

echo "Virtualenv created at $VENV_PATH and packages installed. Activate with: source $VENV_PATH/bin/activate"
