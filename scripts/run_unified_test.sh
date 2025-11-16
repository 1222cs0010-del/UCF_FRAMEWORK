#!/bin/bash
# Quick test script for Unified Conversion Framework

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "UCF Unified Framework Test"
echo "=========================================="
echo ""

# Test 1: Run smoke test on all baselines
echo "Test 1: Running smoke tests on all baselines..."
python run_ucf.py \
  --baselines distilling awq geep \
  --mode smoke \
  --output_dir ./results/test_run

echo ""
echo "Test 1 complete!"
echo ""

# Test 2: Run with config file
echo "Test 2: Running with configuration file..."
if [ -f "configs/unified_experiment.yaml" ]; then
    python run_ucf.py \
      --config configs/unified_experiment.yaml \
      --mode smoke \
      --output_dir ./results/config_test
    echo "Test 2 complete!"
else
    echo "Config file not found, skipping Test 2"
fi

echo ""
echo "=========================================="
echo "All tests complete!"
echo "Results saved in ./results/"
echo "=========================================="

