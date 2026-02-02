#!/usr/bin/env bash
# Shared test helper functions for smoke tests.

set -euo pipefail

PASS_COUNT=0
FAIL_COUNT=0

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo -e "${GREEN}PASS${NC}: $1"
}

log_fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo -e "${RED}FAIL${NC}: $1"
}

assert_exit_code() {
    local description="$1"
    local expected="$2"
    local actual="$3"
    if [ "$actual" -eq "$expected" ]; then
        log_pass "$description (exit code $actual)"
    else
        log_fail "$description (expected exit code $expected, got $actual)"
    fi
}

assert_file_exists() {
    local filepath="$1"
    local description="${2:-File exists: $filepath}"
    if [ -f "$filepath" ]; then
        log_pass "$description"
    else
        log_fail "$description"
    fi
}

assert_file_not_empty() {
    local filepath="$1"
    local description="${2:-File not empty: $filepath}"
    if [ -s "$filepath" ]; then
        log_pass "$description"
    else
        log_fail "$description"
    fi
}

assert_contains() {
    local text="$1"
    local pattern="$2"
    local description="${3:-Output contains '$pattern'}"
    if echo "$text" | grep -q "$pattern"; then
        log_pass "$description"
    else
        log_fail "$description"
    fi
}

print_summary() {
    echo ""
    echo "=============================="
    echo "  Test Summary"
    echo "=============================="
    echo -e "  ${GREEN}Passed${NC}: $PASS_COUNT"
    echo -e "  ${RED}Failed${NC}: $FAIL_COUNT"
    echo "=============================="
    if [ "$FAIL_COUNT" -gt 0 ]; then
        exit 1
    fi
}

cleanup() {
    echo "Cleaning up test artifacts..."
    rm -rf /tmp/test_output_* 2>/dev/null || true
}
