#!/usr/bin/env bash
set -e

# Simple orchestrator to run core benchmarks.
# Extend this as tiny-backprop matures.

echo "[benchmarks] Running Transformer memory/time benchmark..."
python -m benchmarks.mem_vs_time_transformer --trials 2 --modes naive tiny checkpoint || echo "Transformer benchmark failed"

echo "[benchmarks] Running ResNet memory/time benchmark..."
python -m benchmarks.mem_vs_time_resnet --trials 2 --modes naive tiny || echo "ResNet benchmark failed"

echo "[benchmarks] Done."
