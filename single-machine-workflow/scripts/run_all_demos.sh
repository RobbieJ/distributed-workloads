#!/usr/bin/env bash
# Run all demo scripts with --test-mode and produce a combined report.
#
# Usage:
#   bash scripts/run_all_demos.sh              # run all demos (auto-starts infra)
#   bash scripts/run_all_demos.sh --only 01,08  # run only specific demos
#
# Infrastructure (MinIO, etcd, Milvus) is automatically started if needed.
# Each demo's stdout/stderr is teed to logs/<script>.log.
# At the end, scripts/report_all_demos.py aggregates every metrics.json.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/robbie/training-test/venv/bin/python}"
LOGDIR="./logs"
mkdir -p "$LOGDIR"

# When running natively (not in Docker), infrastructure is on localhost
export MILVUS_HOST="${MILVUS_HOST:-localhost}"
export MILVUS_PORT="${MILVUS_PORT:-19530}"

ONLY=""

for arg in "$@"; do
  case "$arg" in
    --only=*)     ONLY="${arg#--only=}" ;;
    --only)       shift; ONLY="$1" ;;
  esac
done

BOLD='\033[1m'
GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
RESET='\033[0m'

divider() {
  echo ""
  echo -e "${BOLD}════════════════════════════════════════════════════════════════${RESET}"
  echo -e "${BOLD}  $1${RESET}"
  echo -e "${BOLD}════════════════════════════════════════════════════════════════${RESET}"
  echo ""
}

# ── Dependency checks ─────────────────────────────────────────────────────

check_docker() {
  if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: docker is not installed or not in PATH${RESET}" >&2
    return 1
  fi
  if ! docker info &> /dev/null; then
    echo -e "${RED}Error: docker daemon is not running${RESET}" >&2
    return 1
  fi
  return 0
}

check_python_deps() {
  local missing=()
  for mod in feast pymilvus torch transformers datasets trl peft accelerate \
             sentence_transformers diffusers optuna onnx onnxruntime; do
    if ! "$PYTHON" -c "import $mod" 2>/dev/null; then
      missing+=("$mod")
    fi
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing Python packages: ${missing[*]}${RESET}"
    echo "Installing missing packages..."
    "$PYTHON" -m pip install "${missing[@]}" --quiet 2>&1
  fi
}

ensure_infrastructure() {
  local milvus_health="http://${MILVUS_HOST}:9091/healthz"
  local minio_health="http://${MILVUS_HOST}:9000/minio/health/live"

  # Check if Milvus is already healthy
  if curl -sf "$milvus_health" > /dev/null 2>&1; then
    echo -e "${GREEN}Infrastructure already running (Milvus healthy).${RESET}"
    return 0
  fi

  echo "Infrastructure not running. Starting MinIO, etcd, Milvus..."

  if ! check_docker; then
    echo -e "${RED}Cannot start infrastructure: Docker unavailable.${RESET}" >&2
    return 1
  fi

  docker compose -f docker-compose.yml up -d 2>&1

  echo "Waiting for Milvus to be healthy (up to 120s)..."
  if ! timeout 120 bash -c "while ! curl -sf $milvus_health > /dev/null 2>&1; do sleep 5; done"; then
    echo -e "${RED}Error: Milvus failed to become healthy within 120s.${RESET}" >&2
    echo "Checking service status..."
    docker compose -f docker-compose.yml ps 2>&1
    docker compose -f docker-compose.yml logs --tail=20 milvus 2>&1
    return 1
  fi

  echo -e "${GREEN}Milvus is ready.${RESET}"

  # Verify MinIO is also healthy
  if ! curl -sf "$minio_health" > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: MinIO health check failed, but continuing...${RESET}"
  fi

  return 0
}

# ── Pre-flight checks ─────────────────────────────────────────────────────

divider "Pre-flight Checks"

echo -n "Python: "
"$PYTHON" --version 2>&1

echo -n "CUDA: "
"$PYTHON" -c "import torch; print(f'available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" 2>&1 || echo "not available"

echo -n "Docker: "
if check_docker; then
  echo "OK"
else
  echo "unavailable (infra-dependent demos may fail)"
fi

echo -n "Python deps: "
check_python_deps
echo "OK"

echo -n "Infrastructure: "
if ! ensure_infrastructure; then
  echo -e "${RED}Failed to start infrastructure. Infra-dependent demos will fail.${RESET}"
  INFRA_OK=false
else
  INFRA_OK=true
fi

should_run() {
  local id="$1"
  if [ -n "$ONLY" ]; then
    echo "$ONLY" | tr ',' '\n' | grep -qw "$id"
    return $?
  fi
  return 0
}

# ── Suite Roadmap ──────────────────────────────────────────────────────────

divider "Demo Suite Roadmap"
echo -e "  ${BOLD}Foundations${RESET}"
echo "    01  Feast Feature Store    - Register ML features"
echo "    02  SFT Fine-Tuning        - Fine-tune LLM with LoRA"
echo ""
echo -e "  ${BOLD}Retrieval-Augmented Generation${RESET}"
echo "    03a HuggingFace RAG        - DPR + FAISS pipeline"
echo "    03b Sentence Trans. RAG    - Semantic search pipeline"
echo "    04  Feast RAG Ingest       - Embeddings -> Milvus vector DB"
echo ""
echo -e "  ${BOLD}Advanced Training${RESET}"
echo "    05  SFT + Feast RAG        - Combined features + RAG + SFT"
echo "    06  DeepSpeed ZeRO-2       - Memory-optimized fine-tuning"
echo "    07  DreamBooth             - Stable Diffusion concept learning"
echo ""
echo -e "  ${BOLD}Capstone${RESET}"
echo "    08  HPO with Optuna        - Automatic hyperparameter search"
echo ""

# ── Demo registry ──────────────────────────────────────────────────────────
# Format: id|script|extra_args|needs_infra
DEMOS=(
  "01|scripts/01_feast_feature_store.py||false"
  "02|scripts/02_sft_llm.py||false"
  "03a|scripts/03_rag_huggingface.py||false"
  "03b|scripts/03_rag_sentence_transformers.py||false"
  "04|scripts/04_feast_rag_ingest.py||true"
  "05|scripts/05_sft_feast_rag.py||true"
  "06|scripts/06_deepspeed_finetune.py||false"
  "07|scripts/07_dreambooth.py||false"
  "08|scripts/08_hpo_optuna.py||false"
)

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a RESULTS=()

for entry in "${DEMOS[@]}"; do
  IFS='|' read -r id script extra needs_infra <<< "$entry"
  name=$(basename "$script" .py)

  if ! should_run "$id"; then
    SKIP_COUNT=$((SKIP_COUNT + 1))
    RESULTS+=("SKIP|$id|$name|filtered by --only")
    continue
  fi

  # If demo needs infra and infra failed to start, fail it (don't silently skip)
  if [ "$needs_infra" = "true" ] && [ "$INFRA_OK" = "false" ]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    RESULTS+=("FAIL|$id|$name|infrastructure unavailable")
    continue
  fi

  divider "[$id] $name"

  log="$LOGDIR/${name}.log"
  set +e
  "$PYTHON" "$script" --test-mode $extra 2>&1 | tee "$log"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [ $exit_code -eq 0 ]; then
    PASS_COUNT=$((PASS_COUNT + 1))
    RESULTS+=("PASS|$id|$name|exit 0")
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    RESULTS+=("FAIL|$id|$name|exit $exit_code")
  fi
done

# ── Combined Report ────────────────────────────────────────────────────────

divider "Combined Demo Report"

for r in "${RESULTS[@]}"; do
  IFS='|' read -r status id name detail <<< "$r"
  case "$status" in
    PASS) color="$GREEN" ;;
    FAIL) color="$RED" ;;
    SKIP) color="$YELLOW" ;;
  esac
  printf "  [${color}%-4s${RESET}] %s  %s (%s)\n" "$status" "$id" "$name" "$detail"
done

echo ""
total=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo -e "  ${GREEN}$PASS_COUNT passed${RESET}, ${RED}$FAIL_COUNT failed${RESET}, ${YELLOW}$SKIP_COUNT skipped${RESET} (${total} total)"
echo ""

# ── Aggregate metrics.json files ───────────────────────────────────────────

if [ -f "scripts/report_all_demos.py" ]; then
  divider "Metrics Summary"
  "$PYTHON" scripts/report_all_demos.py 2>&1
fi

# Exit with failure if any demo failed
[ $FAIL_COUNT -eq 0 ]
