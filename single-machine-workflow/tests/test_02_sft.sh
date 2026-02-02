#!/usr/bin/env bash
# Smoke test for SFT Fine-Tuning.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers.sh"

echo "=== Test: SFT Fine-Tuning ==="

OUTPUT_DIR="/tmp/test_output_sft"
rm -rf "$OUTPUT_DIR"

# Run SFT in test mode (1 epoch, 5 steps, small model)
OUTPUT=$(python scripts/02_sft_llm.py \
    --test-mode \
    --model-name-or-path ibm-granite/granite-3.0-1b-a400m-base \
    --output-dir "$OUTPUT_DIR" 2>&1)
assert_exit_code "SFT script runs successfully" 0 $?
assert_contains "$OUTPUT" "completed successfully" "Script reports success"

# Cleanup
rm -rf "$OUTPUT_DIR"

print_summary
