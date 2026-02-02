#!/usr/bin/env bash
# Smoke test for SFT + Feast RAG Training.
# Requires Milvus running and data ingested.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers.sh"

echo "=== Test: SFT + Feast RAG Training ==="

OUTPUT_DIR="/tmp/test_output_sft_feast_rag"
CACHE_DIR="/tmp/test_rag_dataset_cache"
rm -rf "$OUTPUT_DIR" "$CACHE_DIR"

# Run SFT + RAG in test mode
OUTPUT=$(python scripts/05_sft_feast_rag.py \
    --test-mode \
    --feast-repo-dir ./data/feast_rag_repo \
    --dataset-cache "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" 2>&1)
assert_exit_code "SFT Feast RAG runs" 0 $?
assert_contains "$OUTPUT" "Forward pass successful" "Forward pass works"
assert_contains "$OUTPUT" "completed successfully" "Script reports success"

# Cleanup
rm -rf "$OUTPUT_DIR" "$CACHE_DIR"

print_summary
