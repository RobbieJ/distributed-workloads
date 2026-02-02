#!/usr/bin/env bash
# Run all smoke tests and report summary.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL_PASS=0
TOTAL_FAIL=0
RESULTS=()

run_test() {
    local name="$1"
    local script="$2"
    echo ""
    echo "============================================"
    echo "  Running: $name"
    echo "============================================"
    if bash "$script"; then
        RESULTS+=("PASS: $name")
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        RESULTS+=("FAIL: $name")
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
}

# Run tests in order
run_test "Feast Feature Store" "$SCRIPT_DIR/test_01_feast.sh"
run_test "SFT Fine-Tuning" "$SCRIPT_DIR/test_02_sft.sh"
run_test "RAG Examples" "$SCRIPT_DIR/test_03_rag.sh"
run_test "Feast RAG Ingest" "$SCRIPT_DIR/test_04_feast_rag_ingest.sh"
run_test "SFT + Feast RAG" "$SCRIPT_DIR/test_05_sft_feast_rag.sh"
run_test "DeepSpeed Fine-Tuning" "$SCRIPT_DIR/test_06_deepspeed.sh"
run_test "DreamBooth Training" "$SCRIPT_DIR/test_07_dreambooth.sh"
run_test "HPO with Optuna" "$SCRIPT_DIR/test_08_hpo.sh"

# Summary
echo ""
echo "============================================"
echo "  OVERALL TEST SUMMARY"
echo "============================================"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo ""
echo "  Passed: $TOTAL_PASS"
echo "  Failed: $TOTAL_FAIL"
echo "  Total:  $((TOTAL_PASS + TOTAL_FAIL))"
echo "============================================"

if [ "$TOTAL_FAIL" -gt 0 ]; then
    exit 1
fi
