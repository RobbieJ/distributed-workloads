#!/usr/bin/env bash
# Smoke test for Feast Feature Store demo.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers.sh"

echo "=== Test: Feast Feature Store ==="

OUTPUT_DIR="/tmp/test_output_feast"
rm -rf "$OUTPUT_DIR" /tmp/test_feast_repo

# Run the script in test mode
OUTPUT=$(python scripts/01_feast_feature_store.py \
    --test-mode \
    --output-dir "$OUTPUT_DIR" \
    --repo-dir /tmp/test_feast_repo 2>&1)
assert_exit_code "Feast script runs successfully" 0 $?

# Verify outputs
assert_file_exists "$OUTPUT_DIR/driver_stats_training.jsonl" "Training JSONL file created"
assert_file_not_empty "$OUTPUT_DIR/driver_stats_training.jsonl" "Training JSONL file is not empty"
assert_contains "$OUTPUT" "completed successfully" "Script reports success"

# Cleanup
rm -rf "$OUTPUT_DIR" /tmp/test_feast_repo

print_summary
