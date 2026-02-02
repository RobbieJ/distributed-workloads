#!/usr/bin/env python3
"""Ingest data into Feast + Milvus for RAG.

Ported from examples/kfto-sft-feast-rag/sft_feast_rag_model.ipynb (ingest portion).
Loads Wikipedia DPR dataset, chunks text, generates embeddings, creates Parquet
offline store, applies Feast feature definitions, and writes to Milvus online store.

Requires: docker compose up (Milvus must be running)

Usage:
    python scripts/04_feast_rag_ingest.py [--test-mode]
    python scripts/04_feast_rag_ingest.py --dataset-pct 5
"""

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# GPU setup
sys.path.insert(0, ".")
from scripts.common.gpu_utils import setup_cuda_env

setup_cuda_env()

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer


def generate_synthetic_passages(n=20):
    """Generate synthetic Wikipedia-like passages for testing."""
    topics = [
        ("Python (programming)", "Python is a high-level programming language. It was created by Guido van Rossum and released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming."),
        ("Machine Learning", "Machine learning is a subset of artificial intelligence that focuses on algorithms that improve through experience. Common approaches include supervised learning, unsupervised learning, and reinforcement learning."),
        ("Neural Networks", "Artificial neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes called neurons organized in layers that process information."),
        ("Natural Language Processing", "Natural language processing is a field of AI focused on enabling computers to understand and generate human language. Key tasks include text classification, named entity recognition, and machine translation."),
        ("Deep Learning", "Deep learning uses multiple layers of neural networks to progressively extract features from raw input. It has achieved breakthrough results in computer vision, speech recognition, and NLP."),
        ("Transformers", "The Transformer architecture was introduced in the paper Attention Is All You Need. It relies on self-attention mechanisms and has become the foundation for modern language models."),
        ("BERT", "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model developed by Google. It uses masked language modeling and next sentence prediction."),
        ("GPT", "GPT (Generative Pre-trained Transformer) is a family of language models developed by OpenAI. It uses autoregressive pre-training on large text corpora."),
        ("Computer Vision", "Computer vision is a field of AI that trains computers to interpret visual information. Applications include image classification, object detection, and semantic segmentation."),
        ("Reinforcement Learning", "Reinforcement learning trains agents to make decisions by interacting with an environment. The agent learns to maximize cumulative reward through trial and error."),
    ]
    passages = []
    for i in range(n):
        title, text = topics[i % len(topics)]
        passages.append({"id": str(i), "title": f"{title} ({i})", "text": text})
    return passages


def load_and_chunk_dataset(dataset_pct, max_chars=380, test_mode=False):
    """Load Wikipedia DPR dataset and chunk it."""
    from scripts.common.data_utils import chunk_text

    if test_mode:
        print("Using synthetic passages for test mode...")
        raw_passages = generate_synthetic_passages(20)
        print(f"Generated {len(raw_passages)} synthetic passages")
    else:
        print(f"Loading Wikipedia DPR dataset ({dataset_pct}% of train split)...")
        dataset = load_dataset(
            "facebook/wiki_dpr",
            "psgs_w100.nq.exact",
            split=f"train[:{dataset_pct}%]",
            with_index=False,
        )
        print(f"Loaded {len(dataset)} passages")
        raw_passages = dataset

    # Chunk the dataset
    print(f"Chunking to max {max_chars} chars per chunk...")
    all_chunks = []
    all_ids = []
    all_titles = []

    for i, example in enumerate(raw_passages):
        chunks = chunk_text(example["text"], max_chars)
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{example['id']}_{j}")
            all_titles.append(example["title"])

    print(f"Created {len(all_chunks)} chunks from {len(raw_passages)} passages")
    return all_chunks, all_ids, all_titles


def generate_embeddings(sentences, batch_size=16):
    """Generate DPR embeddings for all chunks."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/dpr-ctx_encoder-single-nq-base"

    print(f"Loading DPR encoder: {model_name} on {device}")
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
    model = DPRContextEncoder.from_pretrained(model_name).to(device)

    print(f"Generating embeddings for {len(sentences)} chunks...")
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_texts = sentences[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings = model(**inputs).pooler_output
            all_embeddings.append(embeddings.float().cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    del model
    torch.cuda.empty_cache()

    return embeddings


def create_parquet(sentences, embeddings, output_path, batch_size=256):
    """Create Parquet file for Feast offline store."""
    print(f"Creating Parquet file: {output_path}")

    first_batch_df = pd.DataFrame(
        {
            "passage_id": list(range(min(batch_size, len(sentences)))),
            "passage_text": sentences[:batch_size],
            "embedding": pd.Series(
                [e.tolist() for e in embeddings[:batch_size]], dtype=object
            ),
            "event_timestamp": [datetime.now(timezone.utc)] * min(batch_size, len(sentences)),
        }
    )

    pqwriter = pq.ParquetWriter(
        output_path, pa.Table.from_pandas(first_batch_df).schema
    )
    pqwriter.write_table(pa.Table.from_pandas(first_batch_df))

    for i in range(batch_size, len(sentences), batch_size):
        batch_end = min(i + batch_size, len(sentences))
        batch_df = pd.DataFrame(
            {
                "passage_id": list(range(i, batch_end)),
                "passage_text": sentences[i:batch_end],
                "embedding": pd.Series(
                    [e.tolist() for e in embeddings[i:batch_end]]
                ),
                "event_timestamp": [datetime.now(timezone.utc)] * (batch_end - i),
            }
        )
        pqwriter.write_table(pa.Table.from_pandas(batch_df))

    pqwriter.close()
    print(f"Parquet file saved: {output_path}")


def setup_feast_repo(feast_repo_dir, parquet_path):
    """Set up Feast repository with config and feature definitions."""
    feast_repo = Path(feast_repo_dir)
    feast_repo.mkdir(parents=True, exist_ok=True)
    data_dir = feast_repo / "data"
    data_dir.mkdir(exist_ok=True)

    # Copy Parquet file to data/
    target = data_dir / "wiki_dpr.parquet"
    if Path(parquet_path).resolve() != target.resolve():
        shutil.copy2(parquet_path, target)

    # Generate feature store config with correct Milvus host/port
    milvus_host_raw = os.environ.get("MILVUS_HOST", "milvus")
    # pymilvus requires URI scheme prefix
    milvus_host = milvus_host_raw if "://" in milvus_host_raw else f"http://{milvus_host_raw}"
    milvus_port = os.environ.get("MILVUS_PORT", "19530")
    yaml_content = f"""project: ragproject
provider: local
registry: data/registry.db
online_store:
  type: milvus
  host: {milvus_host}
  port: {milvus_port}
  vector_enabled: true
  embedding_dim: 768
  index_type: FLAT
  metric_type: COSINE
offline_store:
  type: file
entity_key_serialization_version: 3
auth:
  type: no_auth
"""
    (feast_repo / "feature_store.yaml").write_text(yaml_content)

    # Copy feature definitions
    repo_py_src = Path("configs/feast/ragproject_repo.py")
    if repo_py_src.exists():
        shutil.copy2(repo_py_src, feast_repo / "ragproject_repo.py")
    (feast_repo / "__init__.py").write_text("")

    return feast_repo


def apply_and_write(feast_repo_dir, batch_size=10000):
    """Apply Feast definitions and write data to Milvus online store."""
    import subprocess

    from feast import FeatureStore

    # Apply
    print("Applying Feast feature definitions...")
    feast_bin = shutil.which("feast") or os.path.join(
        os.path.dirname(sys.executable), "feast"
    )
    subprocess.run([feast_bin, "apply"], cwd=feast_repo_dir, check=True)

    # Write to online store
    print("Writing to Milvus online store...")
    store = FeatureStore(repo_path=feast_repo_dir)
    parquet_file = pq.ParquetFile(os.path.join(feast_repo_dir, "data", "wiki_dpr.parquet"))

    for batch_num, batch in enumerate(
        parquet_file.iter_batches(batch_size=batch_size), 1
    ):
        batch_df = batch.to_pandas()
        try:
            print(f"Writing batch {batch_num} ({len(batch_df)} rows)...")
            store.write_to_online_store(
                feature_view_name="wiki_passages", df=batch_df
            )
            print(f"Batch {batch_num} written successfully.")
        except Exception as e:
            print(f"Warning: Skipping batch {batch_num}: {e}")

    print("All data written to online store.")


def main():
    parser = argparse.ArgumentParser(description="Feast RAG Data Ingestion")
    parser.add_argument(
        "--dataset-pct",
        type=int,
        default=1,
        help="Percentage of Wikipedia DPR dataset to use",
    )
    parser.add_argument(
        "--feast-repo-dir",
        type=str,
        default="./data/feast_rag_repo",
        help="Feast repository directory",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (minimal data)",
    )
    args = parser.parse_args()

    if args.test_mode:
        args.dataset_pct = 1

    from scripts.common.demo_utils import print_intro, suppress_known_warnings

    suppress_known_warnings()
    print_intro(
        "Demo 04: Feast RAG Ingestion",
        "Ingest RAG embeddings into Feast + Milvus vector DB",
        "Production RAG needs a scalable vector store, not just in-memory FAISS",
        "Embedding dims, chunk count, Milvus collection stats",
    )

    print("=" * 60)
    print("Feast RAG Data Ingestion")
    print("=" * 60)

    # Load and chunk
    sentences, ids, titles = load_and_chunk_dataset(
        args.dataset_pct, test_mode=args.test_mode
    )
    num_passages = len(set(i.rsplit("_", 1)[0] for i in ids))

    # Generate embeddings
    embeddings = generate_embeddings(sentences)

    # Create Parquet
    parquet_path = os.path.join(args.feast_repo_dir, "data", "wiki_dpr.parquet")
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    create_parquet(sentences, embeddings, parquet_path)

    # Setup Feast repo
    feast_repo = setup_feast_repo(args.feast_repo_dir, parquet_path)

    # Apply and write to Milvus
    milvus_ok = True
    try:
        apply_and_write(str(feast_repo))
    except Exception as e:
        print(f"Milvus write skipped (not running?): {e}")
        milvus_ok = False

    # ── Metrics, Charts & Validation ──────────────────────────────────────
    from scripts.common.demo_utils import MetricsCollector, OutputValidator, plot_table

    mc = MetricsCollector("04_feast_rag_ingest")
    mc.set_metadata(
        dataset_pct=args.dataset_pct,
        num_passages=num_passages,
        num_chunks=len(sentences),
        embedding_shape=list(embeddings.shape),
        parquet_path=parquet_path,
        feast_repo_dir=args.feast_repo_dir,
        milvus_write=milvus_ok,
    )
    metrics_path = os.path.join(args.feast_repo_dir, "metrics.json")
    mc.save(metrics_path)

    # Pipeline stages table
    parquet_size = os.path.getsize(parquet_path) if os.path.isfile(parquet_path) else 0
    plot_table(
        ["Stage", "Count / Size", "Details"],
        [
            ["Passages loaded", str(num_passages), f"{args.dataset_pct}% of dataset"],
            ["Chunks created", str(len(sentences)), f"from {num_passages} passages"],
            ["Embeddings", f"{embeddings.shape[0]} x {embeddings.shape[1]}", "DPR encoder"],
            ["Parquet file", f"{parquet_size:,} bytes", os.path.basename(parquet_path)],
            ["Feast repo", "applied", str(feast_repo)],
            ["Milvus write", "OK" if milvus_ok else "SKIPPED", "online store"],
        ],
        title="Feast RAG Ingestion Pipeline",
    )

    # Validation
    v = OutputValidator("Feast RAG Data Ingestion")
    v.check(len(sentences) > 0, f"Chunks created ({len(sentences)} chunks)")
    v.check(
        embeddings.shape[0] == len(sentences),
        f"Embedding count matches chunk count ({embeddings.shape[0]} == {len(sentences)})",
    )
    v.check(
        embeddings.shape[1] == 768,
        f"Embedding dimension is 768 (got {embeddings.shape[1]})",
    )
    v.check_file_min_size(parquet_path, 1024, "Parquet file >= 1KB")
    v.check_file_exists(
        os.path.join(args.feast_repo_dir, "feature_store.yaml"),
        "feature_store.yaml exists",
    )
    v.check_dir_exists(args.feast_repo_dir, "Feast repo directory exists and non-empty")
    v.check_file_exists(metrics_path, "metrics.json saved")
    v.print_report()

    print("\nFeast RAG Data Ingestion completed successfully!")
    print("\n>> Next: Combine Feast features + RAG retrieval + SFT (Demo 05)")


if __name__ == "__main__":
    main()
