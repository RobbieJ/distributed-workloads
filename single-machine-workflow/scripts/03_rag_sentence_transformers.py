#!/usr/bin/env python3
"""Sentence Transformers RAG example.

Ported from examples/rag-llm/sentence_transformers.ipynb.
Uses sentence-transformers for semantic search and a causal LM for generation.

Usage:
    python scripts/03_rag_sentence_transformers.py [--test-mode]
    python scripts/03_rag_sentence_transformers.py --query "tell me about cat mummies"
"""

import argparse
import sys
import urllib.request

# GPU setup
sys.path.insert(0, ".")
from scripts.common.gpu_utils import setup_cuda_env

setup_cuda_env()

import torch
import transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search


def load_dataset_from_url(url):
    """Load dataset from a URL, one line per chunk."""
    dataset = []
    for line in urllib.request.urlopen(url):
        dataset.append(line.decode("utf-8").strip())
    print(f"Loaded {len(dataset)} entries")
    return dataset


def retrieve_context(query, dataset, embedder_model, top_k=5):
    """Retrieve top-k relevant chunks using semantic search."""
    embedder = SentenceTransformer(embedder_model)

    query_embedding = embedder.encode(query, convert_to_tensor=True)
    dataset_embedding = embedder.encode(dataset, convert_to_tensor=True)

    results = semantic_search(
        query_embeddings=query_embedding,
        corpus_embeddings=dataset_embedding,
        top_k=top_k,
    )

    print("Retrieved knowledge:")
    for corpus in results[0]:
        print(f"  - (similarity: {corpus['score']:.2f}) {dataset[corpus['corpus_id']]}")

    return results[0], dataset


def generate_answer(query, retrieved_chunks, dataset, generator_model):
    """Generate answer using retrieved context."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(generator_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = 0 if torch.cuda.is_available() else -1
    pipeline = transformers.pipeline(
        "text-generation",
        model=generator_model,
        tokenizer=tokenizer,
        device=device,
    )

    # Build system prompt with retrieved context
    context = "".join(
        [f" - {dataset[corpus['corpus_id']]}\n" for corpus in retrieved_chunks]
    )
    instruction_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. "
        "Don't make up any new information:\n"
        f"{context}"
    )

    messages = [
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": query},
    ]

    outputs = pipeline(messages, max_new_tokens=1024)

    # Extract assistant response
    response = ""
    for turn in outputs:
        for item in turn["generated_text"]:
            if item["role"] == "assistant":
                response = item["content"]

    # Clean up
    del pipeline
    torch.cuda.empty_cache()

    return response


def main():
    parser = argparse.ArgumentParser(description="Sentence Transformers RAG Demo")
    parser.add_argument(
        "--embedder-model",
        type=str,
        default="ibm-granite/granite-embedding-30m-english",
        help="Sentence transformer model for embeddings",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default="ibm-granite/granite-3.2-2b-instruct",
        help="Generator model for answer generation",
    )
    parser.add_argument(
        "--dataset-url",
        type=str,
        default="https://huggingface.co/ngxson/demo_simple_rag_py/raw/main/cat-facts.txt",
        help="URL for dataset (one entry per line)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="tell me about cat mummies",
        help="Query to ask",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/rag_st_output",
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (quick validation)",
    )
    args = parser.parse_args()

    from scripts.common.demo_utils import print_intro, suppress_known_warnings

    suppress_known_warnings()
    print_intro(
        "Demo 03b: Sentence Transformers RAG",
        "Build a RAG pipeline with Sentence Transformers",
        "Sentence Transformers offers simpler semantic search with high quality",
        "Similarity scores, how context changes the answer",
    )

    print("=" * 60)
    print("Sentence Transformers RAG Demo")
    print("=" * 60)

    # Load dataset
    print("\n--- Loading dataset ---")
    dataset = load_dataset_from_url(args.dataset_url)

    # Retrieve context
    print(f"\n--- Retrieving context for: {args.query} ---")
    retrieved_chunks, dataset = retrieve_context(
        args.query, dataset, args.embedder_model, top_k=args.top_k
    )

    # Baseline: generate answer without any retrieved context
    print(f"\n--- Baseline: generating answer without context ---")
    no_context_answer = generate_answer(args.query, [], dataset, args.generator_model)
    print(f"No-context answer: {no_context_answer}")

    # Generate answer with RAG context
    print(f"\n--- Generating answer with RAG context ---")
    answer = generate_answer(args.query, retrieved_chunks, dataset, args.generator_model)
    print(f"\nQuery: {args.query}")
    print(f"Answer: {answer}")

    # ── Metrics, Charts & Validation ──────────────────────────────────────
    import os

    from scripts.common.demo_utils import (
        BaselineComparator,
        MetricsCollector,
        OutputValidator,
        plot_bar_chart,
        plot_table,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    mc = MetricsCollector("03_rag_sentence_transformers")
    mc.set_metadata(
        embedder_model=args.embedder_model,
        generator_model=args.generator_model,
        dataset_url=args.dataset_url,
        dataset_entries=len(dataset),
        top_k=args.top_k,
        query=args.query,
        answer=answer,
    )
    chunk_scores = []
    for chunk in retrieved_chunks:
        chunk_scores.append({
            "corpus_id": chunk["corpus_id"],
            "score": round(chunk["score"], 4),
            "text": dataset[chunk["corpus_id"]][:100],
        })
    mc.set_metadata(retrieved_chunks=chunk_scores)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    mc.save(metrics_path)

    # Bar chart of similarity scores
    if retrieved_chunks:
        labels = [f"Chunk {c['corpus_id']}" for c in retrieved_chunks]
        scores = [c["score"] for c in retrieved_chunks]
        plot_bar_chart(labels, scores, title="Retrieval Similarity Scores")

    # Table of retrieved chunks
    table_rows = []
    for chunk in retrieved_chunks:
        text = dataset[chunk["corpus_id"]]
        table_rows.append([
            chunk["corpus_id"],
            f"{chunk['score']:.4f}",
            text[:60] + ("..." if len(text) > 60 else ""),
        ])
    plot_table(
        ["Chunk ID", "Score", "Text (truncated)"],
        table_rows,
        title="Retrieved Context Chunks",
    )

    # Baseline comparison: no context vs RAG
    bc = BaselineComparator("RAG: No Context vs With Context")
    bc.add_metric(
        "Answer Length", len(no_context_answer), len(answer),
        unit=" chars", lower_is_better=False,
    )
    bc.add_metric("Context Chunks", 0, len(retrieved_chunks), lower_is_better=False)
    bc.render()

    # Show answer comparison table
    trunc = lambda s, n=80: s[:n] + ("..." if len(s) > n else "")
    plot_table(
        ["Source", "Answer (truncated)"],
        [
            ["No Context", trunc(no_context_answer)],
            ["With RAG", trunc(answer)],
        ],
        title="Answer Comparison",
    )
    mc.set_metadata(
        no_context_answer=no_context_answer,
        baseline_comparison=bc.to_dict(),
    )

    # Validation
    v = OutputValidator("Sentence Transformers RAG")
    v.check(len(dataset) > 0, f"Dataset loaded ({len(dataset)} entries)")
    v.check(
        len(retrieved_chunks) == args.top_k,
        f"Retrieved {len(retrieved_chunks)} chunks (expected {args.top_k})",
    )
    scores = [c["score"] for c in retrieved_chunks]
    v.check(
        all(0.0 <= s <= 1.0 for s in scores),
        f"All similarity scores in [0, 1] (range: {min(scores):.4f}-{max(scores):.4f})",
    )
    v.check(
        isinstance(answer, str) and len(answer.strip()) > 0,
        "Answer is non-empty string",
    )
    v.check_file_exists(metrics_path, "metrics.json saved")
    bc.add_validations(v)
    v.print_report()

    print("\nSentence Transformers RAG Demo completed successfully!")
    print("\n>> Next: Store these embeddings in a production vector DB (Demo 04)")


if __name__ == "__main__":
    main()
