#!/usr/bin/env bash
# Smoke test for Feast RAG Data Ingestion.
# Requires Milvus running: docker compose -f docker-compose.yml up -d
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers.sh"

echo "=== Test: Feast RAG Data Ingestion ==="

FEAST_DIR="/tmp/test_feast_rag_repo"
rm -rf "$FEAST_DIR"

# Run ingest in test mode
OUTPUT=$(python scripts/04_feast_rag_ingest.py \
    --test-mode \
    --feast-repo-dir "$FEAST_DIR" 2>&1)
assert_exit_code "Feast RAG ingest runs" 0 $?

assert_file_exists "$FEAST_DIR/data/wiki_dpr.parquet" "Parquet file created"
assert_file_not_empty "$FEAST_DIR/data/wiki_dpr.parquet" "Parquet file is not empty"
assert_file_exists "$FEAST_DIR/feature_store.yaml" "Feature store config exists"
assert_contains "$OUTPUT" "completed successfully" "Script reports success"

# Cleanup
rm -rf "$FEAST_DIR"

print_summary
