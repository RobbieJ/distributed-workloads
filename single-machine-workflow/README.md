# Single-Machine ML Workflow for DGX Spark

Docker Compose-based ML training and serving examples, migrated from the Kubernetes-based [distributed-workloads](https://github.com/opendatahub-io/distributed-workloads) repository. Targets the NVIDIA DGX Spark (GB10 GPU, ARM64, CUDA 13.0, 128GB unified memory).

## Prerequisites

- NVIDIA DGX Spark (or any NVIDIA GPU with Docker support)
- Docker with NVIDIA Container Toolkit
- Docker Compose v2
- HuggingFace account (for gated models)

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your HF_TOKEN

# 2. Build training image
make build

# 3. Start infrastructure (MinIO + Milvus)
make up

# 4. Run an example
make run-hpo                    # HPO with Optuna (fastest)
make run-feast                  # Feast feature store demo
make run-sft ARGS="--test-mode" # SFT fine-tuning (test mode)

# 5. Run smoke tests
make test
```

## Examples

| # | Script | Source Example | Description |
|---|--------|---------------|-------------|
| 1 | `01_feast_feature_store.py` | kfto-feast | Feast feature store setup, feature retrieval, JSONL export |
| 2 | `02_sft_llm.py` | kfto-sft-llm | SFT fine-tuning with TRL SFTTrainer + LoRA |
| 3a | `03_rag_huggingface.py` | rag-llm | HuggingFace RAG (DPR + FAISS) |
| 3b | `03_rag_sentence_transformers.py` | rag-llm | Sentence Transformers RAG |
| 4 | `04_feast_rag_ingest.py` | kfto-sft-feast-rag | Ingest Wikipedia data into Feast + Milvus |
| 5 | `05_sft_feast_rag.py` | kfto-sft-feast-rag | SFT training with Feast RAG retriever |
| 6 | `06_deepspeed_finetune.py` | ray-finetune-llm-deepspeed | DeepSpeed ZeRO-2 LLM fine-tuning |
| 7a | `07_dreambooth.py` | kfto-dreambooth | DreamBooth Stable Diffusion training |
| 7b | `07_dreambooth_serve.py` | stable-diffusion-dreambooth | FastAPI image generation server |
| 8 | `08_hpo_optuna.py` | hpo-raytune | Hyperparameter optimization with Optuna |

## Architecture

### What Changed from Kubernetes

| Kubernetes | Single Machine |
|------------|---------------|
| PyTorchJob / TrainJob CRDs | `docker compose run training python scripts/...` |
| RayCluster + Ray Train | Direct Python / DeepSpeed via Accelerate |
| Kueue resource quotas | Removed (single machine) |
| ConfigMaps | Volume-mounted config files |
| PVCs | Docker named volumes / bind mounts |
| KServe | FastAPI serving container |
| Ray Tune | Optuna |
| FSDP (multi-GPU) | Single-GPU with gradient checkpointing + accumulation |
| `flash_attention_2` | `sdpa` (native in PyTorch 2.9+, no ARM64 flash-attn wheels) |

### Docker Compose Services

**Infrastructure** (`docker-compose.yml`):
- `minio` — S3 object storage for models/datasets (ports 9000, 9090)
- `milvus` — Vector DB for RAG Feast online store (port 19530)
- `minio-setup` — One-shot bucket creation

**GPU** (`docker-compose.gpu.yml`):
- `training` — All training scripts (NGC ARM64 + CUDA 13.0 base)
- `serving` — DreamBooth FastAPI inference server (port 8000)

### Directory Structure

```
single-machine-workflow/
├── docker-compose.yml              # Infrastructure services
├── docker-compose.gpu.yml          # GPU training/serving overlay
├── docker/
│   ├── Dockerfile.training         # NGC ARM64 base + ML deps
│   └── Dockerfile.serving          # FastAPI serving image
├── configs/
│   ├── feast/                      # Feast feature store configs
│   ├── deepspeed/                  # DeepSpeed ZeRO-2 configs
│   └── training/                   # Training YAML configs
├── scripts/
│   ├── common/                     # Shared utilities
│   └── *.py                        # Training/serving scripts
├── tests/                          # Shell-based smoke tests
├── data/                           # Runtime data (gitignored)
└── models/                         # Model checkpoints (gitignored)
```

## Makefile Targets

```bash
# Build & Infrastructure
make build              # Build Docker images
make up                 # Start infrastructure (MinIO, Milvus)
make down               # Stop everything
make clean              # Remove data volumes
make shell              # Open bash in training container

# Run Examples
make run-feast          # Feast feature store demo
make run-sft            # SFT fine-tuning
make run-rag-hf         # HuggingFace RAG
make run-rag-st         # Sentence Transformers RAG
make run-feast-rag-ingest  # Ingest data into Feast+Milvus
make run-sft-feast-rag  # SFT + Feast RAG training
make run-deepspeed      # DeepSpeed fine-tuning
make run-dreambooth     # DreamBooth training
make run-hpo            # HPO with Optuna
make create-dataset     # Create GSM8K dataset

# Tests
make test               # Run all smoke tests
make test-<name>        # Run individual test (feast, sft, rag, etc.)
```

All `run-*` targets accept `ARGS=` for passing script arguments:

```bash
make run-sft ARGS="--model-name-or-path ibm-granite/granite-3.0-1b-a400m-base --num-epochs 1"
make run-deepspeed ARGS="--model-name meta-llama/Meta-Llama-3.1-8B --lora"
```

## Running the Blog Example (Feast RAG Pipeline)

The Feast RAG pipeline (examples 4 + 5) demonstrates the full workflow from the blog post:

```bash
# 1. Start infrastructure
make up

# 2. Ingest Wikipedia data into Feast + Milvus
make run-feast-rag-ingest

# 3. Train RAG model using Feast-backed retrieval
make run-sft-feast-rag
```

## Test Mode

All scripts support `--test-mode` for quick validation (limited samples, epochs, and steps):

```bash
python scripts/02_sft_llm.py --test-mode       # 1 epoch, 5 steps
python scripts/06_deepspeed_finetune.py --test-mode  # 20 samples, 5 steps
python scripts/07_dreambooth.py --test-mode     # 10 steps
python scripts/08_hpo_optuna.py --test-mode     # 2 trials, 3 epochs
```

## DGX Spark Notes

- The NGC base image (`nvcr.io/nvidia/pytorch:25.09-py3`) provides native ARM64 + CUDA 13.0 + GB10 (sm_121a) support
- `TRITON_PTXAS_PATH` is set automatically in all scripts and containers
- All scripts use `attn_implementation: sdpa` (not `flash_attention_2`) since flash-attn lacks ARM64 wheels
- DeepSpeed ZeRO-2 (not ZeRO-3) is used: sufficient for single-GPU memory optimization without multi-GPU communication overhead
- The PyTorch CUDA capability warning for GB10 can be safely ignored
