#!/usr/bin/env bash
# Smoke test for DeepSpeed Fine-Tuning.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers.sh"

echo "=== Test: DeepSpeed Fine-Tuning ==="

OUTPUT_DIR="/tmp/test_output_deepspeed"
rm -rf "$OUTPUT_DIR"

# Run DeepSpeed in test mode
OUTPUT=$(python scripts/06_deepspeed_finetune.py \
    --test-mode \
    --model-name ibm-granite/granite-3.0-1b-a400m-base \
    --output-dir "$OUTPUT_DIR" 2>&1)
assert_exit_code "DeepSpeed script runs successfully" 0 $?
assert_contains "$OUTPUT" "completed successfully" "Script reports success"

# Cleanup
rm -rf "$OUTPUT_DIR"

print_summary
