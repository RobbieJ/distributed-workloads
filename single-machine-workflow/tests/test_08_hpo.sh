#!/usr/bin/env bash
# Smoke test for HPO with Optuna.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers.sh"

echo "=== Test: HPO with Optuna ==="

OUTPUT_DIR="/tmp/test_output_hpo"
rm -rf "$OUTPUT_DIR"

# Run HPO in test mode (2 trials, 3 epochs)
OUTPUT=$(python scripts/08_hpo_optuna.py \
    --test-mode \
    --output-dir "$OUTPUT_DIR" 2>&1)
assert_exit_code "HPO script runs successfully" 0 $?

# Verify outputs
assert_file_exists "$OUTPUT_DIR/best_model.pt" "PyTorch model saved"
assert_file_not_empty "$OUTPUT_DIR/best_model.pt" "PyTorch model is not empty"
assert_file_exists "$OUTPUT_DIR/model.onnx" "ONNX model exported"
assert_file_not_empty "$OUTPUT_DIR/model.onnx" "ONNX model is not empty"
assert_contains "$OUTPUT" "ONNX model validation passed" "ONNX validation passes"
assert_contains "$OUTPUT" "completed successfully" "Script reports success"

# Cleanup
rm -rf "$OUTPUT_DIR"

print_summary
