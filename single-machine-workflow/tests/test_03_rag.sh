#!/usr/bin/env bash
# Smoke test for RAG examples.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/helpers.sh"

echo "=== Test: RAG Examples ==="

# Test Sentence Transformers RAG (lighter weight, test first)
echo "--- Testing Sentence Transformers RAG ---"
OUTPUT=$(python scripts/03_rag_sentence_transformers.py \
    --test-mode \
    --query "tell me about cat mummies" 2>&1)
assert_exit_code "Sentence Transformers RAG runs" 0 $?
assert_contains "$OUTPUT" "completed successfully" "ST RAG reports success"
assert_contains "$OUTPUT" "Retrieved knowledge" "ST RAG retrieves context"

# Test HuggingFace RAG
echo "--- Testing HuggingFace RAG ---"
OUTPUT=$(python scripts/03_rag_huggingface.py \
    --test-mode \
    --query "what is the name of the tiniest cat" 2>&1)
assert_exit_code "HuggingFace RAG runs" 0 $?
assert_contains "$OUTPUT" "completed successfully" "HF RAG reports success"
assert_contains "$OUTPUT" "Answer:" "HF RAG generates an answer"

print_summary
