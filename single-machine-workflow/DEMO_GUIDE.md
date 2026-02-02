# ML Demo Suite: A Guided Tour of Production ML Techniques

This guide walks you through 9 hands-on demos that cover the core techniques used to build, train, and serve production ML systems. Each demo runs on a single machine and produces validated output with metrics, charts, and comparison tables so you can see exactly what happened and why it matters.

The demos are self-contained Python scripts that originally ran as distributed Kubernetes workloads (PyTorchJob, RayCluster, KServe). They've been adapted to run locally with Docker Compose infrastructure, making them accessible for learning, experimentation, and demo purposes without a cluster.

---

## Table of Contents

1. [What You'll Learn](#what-youll-learn)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Running the Demos](#running-the-demos)
5. [Demo Walkthrough](#demo-walkthrough)
   - [Demo 01: Feature Store (Feast)](#demo-01-feature-store-feast)
   - [Demo 02: SFT Fine-Tuning with LoRA](#demo-02-sft-fine-tuning-with-lora)
   - [Demo 03a: RAG with HuggingFace DPR + FAISS](#demo-03a-rag-with-huggingface-dpr--faiss)
   - [Demo 03b: RAG with Sentence Transformers](#demo-03b-rag-with-sentence-transformers)
   - [Demo 04: Vector DB Ingestion (Feast + Milvus)](#demo-04-vector-db-ingestion-feast--milvus)
   - [Demo 05: Combined Feast + RAG + SFT Training](#demo-05-combined-feast--rag--sft-training)
   - [Demo 06: DeepSpeed ZeRO-2 Fine-Tuning](#demo-06-deepspeed-zero-2-fine-tuning)
   - [Demo 07: DreamBooth (Stable Diffusion)](#demo-07-dreambooth-stable-diffusion)
   - [Demo 08: Hyperparameter Optimization (Optuna)](#demo-08-hyperparameter-optimization-optuna)
6. [How the Demos Connect](#how-the-demos-connect)
7. [Understanding the Output](#understanding-the-output)
8. [Running the Full Suite](#running-the-full-suite)
9. [Customizing the Demos](#customizing-the-demos)
10. [Troubleshooting](#troubleshooting)

---

## What You'll Learn

By running these demos, you'll see working examples of the techniques that production ML teams use every day:

| Technique | Why It Matters | Demos |
|-----------|---------------|-------|
| **Feature Stores** | Prevent training/serving skew by centralizing feature definitions | 01, 04, 05 |
| **LoRA / PEFT** | Fine-tune billion-parameter models on a single GPU by training <1% of weights | 02, 06 |
| **Retrieval-Augmented Generation** | Ground LLM answers in real documents to reduce hallucination | 03a, 03b, 04, 05 |
| **Vector Databases** | Store and search embeddings at scale for production RAG | 04, 05 |
| **DeepSpeed ZeRO** | Partition optimizer state to fit larger models in GPU memory | 06 |
| **DreamBooth** | Teach a diffusion model a new visual concept from a handful of images | 07 |
| **Hyperparameter Optimization** | Systematically find better hyperparameters than manual guessing | 08 |
| **ONNX Export** | Convert models to a portable format for cross-platform serving | 08 |

Each demo produces a **validation report** (PASS/FAIL/WARN checks), **metrics.json** for programmatic analysis, and often **ASCII charts** and **baseline comparison tables** so you can see the impact of each technique.

---

## Prerequisites

- **GPU**: NVIDIA GPU with Docker support (tested on DGX Spark GB10, works on any CUDA GPU)
- **Docker**: Docker Engine with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
- **Docker Compose v2**: Comes bundled with modern Docker Desktop and Docker Engine
- **Disk space**: ~15 GB for model downloads and checkpoints
- **HuggingFace account**: Free account at [huggingface.co](https://huggingface.co) (needed for some model downloads)

If you're running directly on the host (not in Docker), you also need:
- Python 3.12 with a CUDA-enabled PyTorch installation
- The Python packages listed in `docker/Dockerfile.training`

---

## Setup

### Option A: Docker (Recommended)

```bash
cd single-machine-workflow/

# 1. Create your environment file
cp .env.example .env

# 2. Add your HuggingFace token (get one at https://huggingface.co/settings/tokens)
#    This is needed for gated models like Llama. Open .env and set:
#    HF_TOKEN=hf_your_token_here

# 3. Build the training container image
make build

# 4. Start infrastructure services (MinIO object storage, Milvus vector DB, etcd)
make up

# 5. Verify services are healthy
docker compose ps
#    You should see minio, etcd, and milvus all "healthy"
```

### Option B: Native (without Docker)

If you have a working CUDA Python environment:

```bash
cd single-machine-workflow/

# Start just the infrastructure
docker compose -f docker-compose.yml up -d

# Run demos directly
python scripts/01_feast_feature_store.py --test-mode
```

The `run_all_demos.sh` script uses this approach — it runs Python directly and manages infrastructure automatically.

---

## Running the Demos

### Quick test (all 9 demos, ~15 minutes with GPU)

```bash
bash scripts/run_all_demos.sh
```

This runs every demo in `--test-mode` (minimal data, few training steps), prints a roadmap, validates each demo's output, and finishes with a combined metrics report and key takeaways summary.

### Individual demos via Make

```bash
# In Docker (uses the training container with all dependencies)
make run-feast ARGS="--test-mode"
make run-sft ARGS="--test-mode"
make run-deepspeed ARGS="--test-mode"

# Full training (longer, better results)
make run-sft                    # ~30 min, 10 epochs on GSM8K
make run-dreambooth             # ~20 min, 300 training steps
make run-hpo                    # ~5 min, 10 Optuna trials
```

### Individual demos directly

```bash
python scripts/01_feast_feature_store.py --test-mode
python scripts/02_sft_llm.py --test-mode
python scripts/08_hpo_optuna.py --n-trials 20
```

Every script accepts `--help` for a full list of options.

### Run a subset

```bash
# Run only specific demos by ID
bash scripts/run_all_demos.sh --only 01,02,08
```

---

## Demo Walkthrough

### Demo 01: Feature Store (Feast)

**Script:** `scripts/01_feast_feature_store.py`

**What it does:**
Creates a Feast feature store with driver performance data (conversion rate, acceleration rate, daily trips), retrieves historical features with point-in-time correctness, and converts them to a JSONL training dataset suitable for LLM fine-tuning.

**Why this matters:**
In production ML, training data and serving data often come from different pipelines, and subtle differences between them cause **training/serving skew** — the model performs well in training but poorly in production. Feature stores solve this by providing a single source of truth for feature definitions. Feast is an open-source feature store that integrates with both offline (batch) and online (real-time) data sources.

**What to watch for in the output:**
- The feature schema showing typed columns (Float32, Int32) with non-null counts
- Point-in-time correctness: each driver's features are joined at the correct historical timestamp, not the latest values
- The generated JSONL format with instruction/input/output fields ready for LLM training
- All 5 validation checks passing (valid JSON, correct row count, file sizes)

**Key concepts:** Feature stores, entity joins, point-in-time correctness, offline feature retrieval, training data generation

```bash
python scripts/01_feast_feature_store.py --test-mode
```

---

### Demo 02: SFT Fine-Tuning with LoRA

**Script:** `scripts/02_sft_llm.py`

**What it does:**
Fine-tunes an LLM (IBM Granite 1B by default) on the GSM8K math dataset using Supervised Fine-Tuning (SFT) with LoRA adapters. First runs the base model on sample math questions to establish a baseline, trains the model, then runs the same questions through the fine-tuned model to compare answers.

**Why this matters:**
Pre-trained LLMs are general-purpose, but most applications need specialized behavior. Full fine-tuning of a billion-parameter model requires enormous GPU memory (14+ bytes per parameter for weights, gradients, and optimizer states). **LoRA** (Low-Rank Adaptation) solves this by freezing the original weights and training small adapter matrices — typically <1% of total parameters. This demo trains only 2.7M parameters out of 1.3B total (0.2%), making fine-tuning practical on a single GPU.

**What to watch for in the output:**
- `trainable params: 2,752,512 || all params: 1,337,377,792 || trainable%: 0.2058` — LoRA trains a tiny fraction
- The training loss curve (should trend downward over full training; may not in test mode's 5 steps)
- Base vs fine-tuned answer comparison table — the model's responses before and after training
- In test mode: the "Training loss decreased" check shows WARN instead of FAIL, because 5 steps isn't enough for convergence — this is expected and the demo still passes

**Key concepts:** SFT, LoRA, PEFT, adapter weights, TRL SFTTrainer, gradient accumulation

```bash
# Test mode (5 steps, ~2 min)
python scripts/02_sft_llm.py --test-mode

# Full training (10 epochs, ~30 min)
python scripts/02_sft_llm.py

# With a different model
python scripts/02_sft_llm.py --model-name-or-path meta-llama/Meta-Llama-3.1-8B --test-mode
```

---

### Demo 03a: RAG with HuggingFace DPR + FAISS

**Script:** `scripts/03_rag_huggingface.py`

**What it does:**
Builds a Retrieval-Augmented Generation pipeline using Facebook's Dense Passage Retrieval (DPR) encoder to create embeddings, FAISS for vector similarity search, and the RAG-Sequence model for answer generation. Loads a cat-facts dataset, encodes it into vectors, retrieves relevant passages for a query, and generates an answer grounded in the retrieved documents.

**Why this matters:**
LLMs can hallucinate — confidently generating plausible but incorrect answers. **RAG** mitigates this by retrieving relevant documents from a knowledge base before generating an answer, grounding the response in factual content. This is the foundation of most enterprise LLM applications (chatbots, search, Q&A systems). The DPR + FAISS approach is the original RAG architecture from Facebook Research.

**What to watch for in the output:**
- The FAISS index creation with embedding dimensions (768-dim vectors)
- The generated answer referencing actual content from the cat-facts dataset
- The summary table showing dataset size, embedding dimensions, and the answer
- All validation checks passing (non-empty dataset, valid embeddings, non-empty answer)

**Key concepts:** Dense Passage Retrieval, FAISS indexing, vector similarity search, RAG-Sequence generation, document grounding

```bash
python scripts/03_rag_huggingface.py --test-mode

# With a custom query
python scripts/03_rag_huggingface.py --query "how long do cats sleep"
```

---

### Demo 03b: RAG with Sentence Transformers

**Script:** `scripts/03_rag_sentence_transformers.py`

**What it does:**
Builds a simpler RAG pipeline using Sentence Transformers for semantic search and a causal LM (IBM Granite 2B Instruct) for generation. Loads the same cat-facts dataset, encodes it using a compact embedding model, retrieves the top-k most similar chunks, then generates answers both without context (baseline) and with retrieved context (RAG) to demonstrate the difference.

**Why this matters:**
While Demo 03a uses the full DPR + RAG-Sequence architecture from the research paper, real-world applications often use a simpler two-stage approach: a lightweight embedding model for retrieval and any LLM for generation. **Sentence Transformers** provides high-quality sentence embeddings in a much simpler API. This demo shows the same RAG concept with a more practical, production-friendly architecture, and directly compares answers with and without retrieval.

**What to watch for in the output:**
- Similarity scores for each retrieved chunk (0-1 range, higher = more relevant)
- The bar chart of retrieval similarity scores
- **The answer comparison table**: "No Context" vs "With RAG" — this is the key demonstration. The RAG answer should be more specific and grounded in the retrieved facts
- The baseline comparison metrics (answer length, context chunks)

**Key concepts:** Sentence embeddings, semantic search, top-k retrieval, context injection, baseline comparison

```bash
python scripts/03_rag_sentence_transformers.py --test-mode

# With a custom query and more context
python scripts/03_rag_sentence_transformers.py --query "tell me about cat mummies" --top-k 10
```

---

### Demo 04: Vector DB Ingestion (Feast + Milvus)

**Script:** `scripts/04_feast_rag_ingest.py`

**What it does:**
Takes the RAG concept from Demos 03a/03b and makes it production-ready. Loads Wikipedia passages, chunks them into manageable pieces, generates DPR embeddings for each chunk, stores everything in a Parquet file (offline store), sets up a Feast feature repository with vector-indexed features, and writes the embeddings to a Milvus vector database (online store) for real-time similarity search.

**Why this matters:**
In-memory FAISS indices (Demo 03a) work for demos but don't scale. Production RAG systems need a **vector database** that can handle millions of embeddings, support concurrent queries, persist data across restarts, and provide consistent search performance. **Milvus** is a purpose-built vector database, and **Feast** provides a unified interface for managing both the offline feature data (Parquet) and the online vector store (Milvus), with automatic schema management and versioning.

**What to watch for in the output:**
- The pipeline stages table showing passages → chunks → embeddings → Parquet → Feast → Milvus
- Embedding dimensions (768) matching the DPR encoder output
- Chunk count (how text was split for embedding)
- Parquet file size (the offline store)
- Milvus write status (OK means data is in the vector DB and searchable)

**Key concepts:** Text chunking, DPR embeddings, Parquet offline store, Milvus vector database, Feast feature repository, online/offline store separation

```bash
# Requires infrastructure (MinIO + Milvus) to be running
python scripts/04_feast_rag_ingest.py --test-mode

# With more data
python scripts/04_feast_rag_ingest.py --dataset-pct 5
```

---

### Demo 05: Combined Feast + RAG + SFT Training

**Script:** `scripts/05_sft_feast_rag.py`

**What it does:**
Ties together everything from the previous demos into a single training pipeline. Builds a RAG model (DPR question encoder + BART generator) backed by a Feast retriever that queries the Milvus vector DB for relevant passages. Trains this model on question-answer pairs using SFT with the Seq2Seq trainer. After training, runs inference on sample questions to demonstrate the trained model answering with RAG-retrieved context.

**Why this matters:**
This is the capstone demo for the RAG track. In production, you don't just retrieve documents — you train models that learn to use retrieved context effectively. This demo shows the complete pipeline: **features from Feast (Demo 01) + embeddings in Milvus (Demo 04) + RAG retrieval (Demos 03) + SFT training (Demo 02)**. The Feast-backed retriever means the same feature infrastructure serves both training and inference, preventing the training/serving skew that plagues ad-hoc RAG systems.

**What to watch for in the output:**
- The test forward pass succeeding (model + retriever connected correctly)
- Training loss decreasing (the model is learning to use retrieved context)
- The inference results table showing questions and the model's RAG-augmented answers
- The loss curve chart

**Key concepts:** End-to-end RAG training, Feast-backed retrieval, Seq2Seq training, RAG model architecture (question encoder + retriever + generator)

```bash
# Requires Demo 04 to have been run first (data must be in Milvus)
python scripts/05_sft_feast_rag.py --test-mode
```

---

### Demo 06: DeepSpeed ZeRO-2 Fine-Tuning

**Script:** `scripts/06_deepspeed_finetune.py`

**What it does:**
Fine-tunes the same Granite 1B model as Demo 02, but uses DeepSpeed ZeRO Stage 2 for memory-optimized training. ZeRO-2 partitions optimizer states and gradients across the training process, dramatically reducing per-GPU memory usage. The demo measures actual peak GPU memory and compares it against a naive estimate (what training would require without ZeRO), showing the memory savings.

**Why this matters:**
Training LLMs is fundamentally a memory problem. A 1B parameter model needs ~14 bytes per parameter for full training (2B for bf16 weights + 8B for Adam optimizer states + 4B for gradients = 14B × 1B params = 14 GB). **DeepSpeed ZeRO-2** partitions the optimizer states and gradients so they don't all need to be in GPU memory simultaneously, typically reducing peak memory by 50-75%. This is what makes training larger models possible on limited hardware. On multi-GPU systems, ZeRO-2 partitions across GPUs; on a single GPU, it optimizes memory layout and offloads to CPU.

**What to watch for in the output:**
- The **baseline comparison table**: "Naive vs ZeRO-2 Memory" showing peak memory reduction (typically ~75%)
- The bar chart comparing naive estimate vs actual DeepSpeed memory usage
- Training loss over steps and evaluation perplexity
- Epoch timing (how long each training epoch takes)

**Key concepts:** DeepSpeed ZeRO Stage 2, optimizer state partitioning, CPU offloading, gradient checkpointing, memory efficiency, Accelerate integration

```bash
python scripts/06_deepspeed_finetune.py --test-mode

# Without LoRA (full fine-tuning, more memory)
python scripts/06_deepspeed_finetune.py --no-lora --test-mode

# With a different DeepSpeed config
python scripts/06_deepspeed_finetune.py --ds-config configs/deepspeed/ds_config_zero2_offload.json
```

---

### Demo 07: DreamBooth (Stable Diffusion)

**Script:** `scripts/07_dreambooth.py`

**What it does:**
Fine-tunes a Stable Diffusion model using the DreamBooth technique to learn a new visual concept (a corgi dog) from just 5 example images. Trains only the UNet (the denoising network) while freezing the text encoder and VAE. After training, loads the fine-tuned pipeline and generates a sample image of the learned concept.

**Why this matters:**
**DreamBooth** is a technique for personalizing text-to-image models with very few examples (3-5 images). It works by associating a unique token (like "ccorgi") with the visual concept, so prompts like "a photo of ccorgi dog on the beach" generate images of that specific dog. This has practical applications in product photography, personalized content, and creative tools. The technique demonstrates that large diffusion models can be adapted to new concepts without retraining from scratch.

**What to watch for in the output:**
- The training progress bar with loss values at each step
- The loss curve chart (should trend downward over 300 steps; may fluctuate in test mode's 10 steps)
- **The generated image** saved at `models/dreambooth/generated_sample.png` — open it to see the model's attempt at generating the learned concept
- The baseline comparison showing early vs late training loss
- In test mode: loss and baseline checks show WARN (10 steps isn't enough for the model to learn the concept well, but the pipeline runs correctly)

**Key concepts:** DreamBooth, Stable Diffusion, UNet fine-tuning, noise prediction, DDPM scheduler, concept learning, image generation

```bash
python scripts/07_dreambooth.py --test-mode

# Full training (300 steps, ~20 min)
python scripts/07_dreambooth.py

# View the generated image
open models/dreambooth/generated_sample.png    # macOS
xdg-open models/dreambooth/generated_sample.png # Linux
```

---

### Demo 08: Hyperparameter Optimization (Optuna)

**Script:** `scripts/08_hpo_optuna.py`

**What it does:**
Demonstrates automatic hyperparameter optimization using Optuna. Trains a small neural network on a sine-wave regression task (learn to approximate sin(x)) with Optuna searching over hidden layer size and learning rate. First trains with default hyperparameters to establish a baseline loss, then runs multiple Optuna trials that explore the search space. The best model is exported to ONNX format and validated with ONNX Runtime for cross-platform inference.

**Why this matters:**
Choosing hyperparameters manually (learning rate, model size, batch size, etc.) is time-consuming and often suboptimal. **Hyperparameter optimization** automates this search, systematically exploring combinations and pruning unpromising trials early. Optuna is a modern HPO framework that uses efficient algorithms (Tree-structured Parzen Estimator) to find good hyperparameters faster than grid search or random search. The sine-wave task makes loss values interpretable — you can see that Optuna's best trial genuinely fits the data better than naive defaults.

**What to watch for in the output:**
- The baseline loss with default hyperparameters (hidden=10, lr=0.01)
- The **trial summary table** showing each trial's hidden size, learning rate, loss, and state
- The **hyperparameter comparison table**: default vs Optuna's best choices
- The **baseline comparison**: default loss vs Optuna's best loss — Optuna should find a lower loss
- ONNX export and validation (the model runs in ONNX Runtime, proving it's portable)

**Key concepts:** Hyperparameter optimization, Optuna trials, search space definition, trial pruning, ONNX export, cross-platform inference

```bash
python scripts/08_hpo_optuna.py --test-mode

# More thorough search
python scripts/08_hpo_optuna.py --n-trials 50 --n-epochs 20
```

---

## How the Demos Connect

The demos are designed to build on each other. Here's how they fit into a production ML workflow:

```
┌─────────────────────────────────────────────────────────────┐
│                      FOUNDATIONS                            │
│                                                             │
│  [01] Feature Store ──────────────────── Feature management │
│  [02] SFT + LoRA ────────────────────── Basic fine-tuning   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                 RETRIEVAL-AUGMENTED GENERATION               │
│                                                             │
│  [03a] HuggingFace RAG (DPR + FAISS) ── Research approach  │
│  [03b] Sentence Transformers RAG ─────── Production approach│
│  [04]  Feast + Milvus Ingestion ──────── Scalable storage   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    ADVANCED TRAINING                         │
│                                                             │
│  [05] Feast + RAG + SFT combined ─────── Full RAG pipeline  │
│  [06] DeepSpeed ZeRO-2 ──────────────── Memory efficiency   │
│  [07] DreamBooth ─────────────────────── Image fine-tuning   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                       CAPSTONE                              │
│                                                             │
│  [08] HPO with Optuna ────────────────── Optimize any demo  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Dependency chain for the RAG track:**
```
01 (features) ─┐
               ├──→ 04 (ingest into Milvus) ──→ 05 (train with RAG)
03a/03b (RAG) ─┘
```

**Independent demos** (can run in any order): 01, 02, 03a, 03b, 06, 07, 08

**Demos that need infrastructure** (Milvus must be running): 04, 05

---

## Understanding the Output

Every demo produces structured output designed to help you understand what happened:

### Intro Banner

Each demo starts with a colored banner explaining what it does, why it matters, and what to watch for:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ Demo 02: SFT Fine-Tuning with LoRA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ WHAT:  Fine-tune an LLM on math problems using SFT + LoRA
┃ WHY:   LoRA lets you fine-tune billion-param models on a single GPU
┃ WATCH: Loss decreasing, base vs fine-tuned answers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Validation Report

Each demo ends with a validation report. Checks can be PASS, FAIL, or WARN:

```
============================================================
Validation Report: SFT Fine-Tuning
============================================================
  [PASS] Model checkpoint dir exists and non-empty
  [PASS] adapter_config.json exists
  [PASS] adapter_model.safetensors >= 1KB
  [PASS] Final loss 0.7656 in [0, 15]
  [WARN] Training loss decreased [test-mode: only 5 steps, loss may not decrease]
  [PASS] adapter_config.json is valid JSON

5/6 checks passed, 1 warnings
============================================================
```

- **PASS**: The check succeeded
- **WARN**: The check failed, but this is expected in test mode (too few training steps for convergence). The demo still passes
- **FAIL**: Something genuinely went wrong

### Metrics JSON

Every demo saves a `metrics.json` file with structured data:

```bash
# View a demo's metrics
cat models/sft_output/metrics.json | python -m json.tool

# View all metrics at once
python scripts/report_all_demos.py
```

### Narrative Connections

Demos print a `>> Next:` message at the end suggesting which demo to run next:

```
>> Next: Scale this training with DeepSpeed (Demo 06)
```

### Combined Report

When running the full suite, the final report includes:
- A metrics table summarizing all 9 demos (timing, loss curves, baseline comparisons)
- A KEY TAKEAWAYS section with one-line summaries of what each demo demonstrated

---

## Running the Full Suite

```bash
bash scripts/run_all_demos.sh
```

This script:

1. **Pre-flight checks**: Verifies Python, CUDA, Docker, and Python dependencies
2. **Infrastructure**: Automatically starts MinIO, etcd, and Milvus if not already running
3. **Suite roadmap**: Prints the demo categories (Foundations, RAG, Advanced Training, Capstone)
4. **Runs all 9 demos** in order with `--test-mode`, logging output to `logs/`
5. **Combined report**: Shows pass/fail status for each demo
6. **Metrics summary**: Aggregated table from all `metrics.json` files
7. **Key takeaways**: Narrative synthesis of what was demonstrated

Expected result: **9 passed, 0 failed, 0 skipped**

To run only specific demos:

```bash
bash scripts/run_all_demos.sh --only 01,02,08
```

---

## Customizing the Demos

### Using a different model

Most training demos accept `--model-name-or-path` or `--model-name`:

```bash
# SFT with Llama (requires HF_TOKEN for gated access)
python scripts/02_sft_llm.py --model-name-or-path meta-llama/Meta-Llama-3.1-8B --test-mode

# DeepSpeed with a different model
python scripts/06_deepspeed_finetune.py --model-name microsoft/phi-2 --test-mode
```

### Adjusting training parameters

```bash
# More epochs, larger batch
python scripts/02_sft_llm.py --num-epochs 5 --batch-size 8 --learning-rate 1e-4

# Different LoRA rank
python scripts/02_sft_llm.py --lora-r 32 --lora-alpha 64

# More HPO trials
python scripts/08_hpo_optuna.py --n-trials 50 --n-epochs 30
```

### Using the DeepSpeed CPU offload config

For larger models that don't fit in GPU memory even with ZeRO-2:

```bash
python scripts/06_deepspeed_finetune.py \
  --ds-config configs/deepspeed/ds_config_zero2_offload.json \
  --model-name meta-llama/Meta-Llama-3.1-8B
```

### Serving DreamBooth models

After training a DreamBooth model, you can serve it via FastAPI:

```bash
make build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up serving

# Generate images via API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a photo of ccorgi dog on the beach"}'
```

---

## Troubleshooting

### "Infrastructure not running" / Milvus errors

Demos 04 and 05 need Milvus. Start it with:

```bash
docker compose -f docker-compose.yml up -d
# Wait ~30 seconds for Milvus to be healthy
curl http://localhost:9091/healthz
```

### "No module named 'torch'" or "CUDA not available"

You're using the wrong Python environment. If running natively on DGX Spark:

```bash
export PYTHON=/home/robbie/training-test/venv/bin/python
$PYTHON -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

Or use the Docker container which has everything pre-installed:

```bash
make shell
# Inside the container:
python scripts/01_feast_feature_store.py --test-mode
```

### PyTorch CUDA capability warning

```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
```

This warning is harmless and can be ignored. It's a known PyTorch bug — the GB10 works correctly. The demos suppress this warning inside scripts, but it may appear during pre-flight checks.

### Out of GPU memory

- Use `--test-mode` to reduce data sizes and training steps
- Use DeepSpeed with CPU offload: `--ds-config configs/deepspeed/ds_config_zero2_offload.json`
- Reduce batch size: `--batch-size 1`
- Enable gradient checkpointing: `--gradient-checkpointing`
- Use QLoRA (4-bit quantization): `--load-in-4bit`

### Demo 05 fails with "Feast repo not found"

Demo 05 depends on Demo 04 having run first to create the Feast repository and ingest data into Milvus:

```bash
python scripts/04_feast_rag_ingest.py --test-mode
python scripts/05_sft_feast_rag.py --test-mode
```

### Logs

All demo output is logged to `logs/`:

```bash
ls logs/
# 01_feast_feature_store.log
# 02_sft_llm.log
# ...

# View a specific log
cat logs/02_sft_llm.log
```
