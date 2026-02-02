#!/usr/bin/env bash
# Smoke test for DreamBooth Training.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers.sh"

echo "=== Test: DreamBooth Training ==="

OUTPUT_DIR="/tmp/test_output_dreambooth"
rm -rf "$OUTPUT_DIR"

# Run DreamBooth in test mode (10 steps)
OUTPUT=$(python scripts/07_dreambooth.py \
    --test-mode \
    --output-dir "$OUTPUT_DIR" 2>&1)
assert_exit_code "DreamBooth script runs successfully" 0 $?
assert_contains "$OUTPUT" "completed successfully" "Script reports success"

# Check pipeline was saved
if [ -d "$OUTPUT_DIR" ]; then
    assert_file_exists "$OUTPUT_DIR/model_index.json" "Pipeline model_index.json exists"
    log_pass "Pipeline directory created"
else
    log_fail "Pipeline directory not created"
fi

# Cleanup
rm -rf "$OUTPUT_DIR"

print_summary
