#!/usr/bin/env bash
# Simple helper to create conda env using environment file
set -e
if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-env-yaml>"
  exit 1
fi
ENV_FILE="$1"
conda env create -f "$ENV_FILE" || conda env update -f "$ENV_FILE"
echo "Environment created/updated from $ENV_FILE"
