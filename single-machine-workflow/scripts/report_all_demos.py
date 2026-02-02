#!/usr/bin/env python3
"""Aggregate all metrics.json files and print a combined summary."""

import glob
import json
import os
import sys

sys.path.insert(0, ".")
from scripts.common.demo_utils import plot_table

METRICS_LOCATIONS = [
    ("01 Feast Feature Store", "data/feast_output/metrics.json"),
    ("02 SFT Fine-Tuning", "models/sft_output/metrics.json"),
    ("03a RAG HuggingFace", "data/rag_hf_output/metrics.json"),
    ("03b RAG Sentence Trans.", "data/rag_st_output/metrics.json"),
    ("04 Feast RAG Ingest", "data/feast_rag_repo/metrics.json"),
    ("05 SFT Feast RAG", "models/feast_rag_output/metrics.json"),
    ("06 DeepSpeed", "models/deepspeed_output/metrics.json"),
    ("07 DreamBooth", "models/dreambooth/metrics.json"),
    ("08 HPO Optuna", "models/hpo/metrics.json"),
]


def load_metrics():
    results = []
    for label, path in METRICS_LOCATIONS:
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            results.append((label, path, data))
    return results


def summarize_training(data):
    """Extract a one-line training summary from metrics data."""
    parts = []
    if "steps" in data and data["steps"]:
        first_loss = data["steps"][0].get("loss")
        last_loss = data["steps"][-1].get("loss")
        n = len(data["steps"])
        if first_loss is not None and last_loss is not None:
            parts.append(f"{n} steps, loss {first_loss:.3f} -> {last_loss:.3f}")
        else:
            parts.append(f"{n} steps")
    if "epochs" in data and data["epochs"]:
        last = data["epochs"][-1]
        if "eval_loss" in last:
            parts.append(f"eval_loss={last['eval_loss']:.3f}")
        if "perplexity" in last:
            parts.append(f"ppl={last['perplexity']:.1f}")
    if "trials" in data and data["trials"]:
        completed = [t for t in data["trials"] if t.get("value") is not None]
        if completed:
            best = min(completed, key=lambda t: t["value"])
            parts.append(f"{len(completed)} trials, best={best['value']:.4f}")
    meta = data.get("metadata", {})
    if "num_chunks" in meta:
        parts.append(f"{meta['num_chunks']} chunks")
    if "embedding_shape" in meta:
        shape = meta["embedding_shape"]
        parts.append(f"emb {shape[0]}x{shape[1]}")
    if "row_count" in meta:
        parts.append(f"{meta['row_count']} rows")
    if "answer" in meta:
        ans = meta["answer"]
        parts.append(f'answer="{ans[:40]}{"..." if len(ans) > 40 else ""}"')
    if "dataset_entries" in meta and not parts:
        parts.append(f"{meta['dataset_entries']} entries")
    return "; ".join(parts) if parts else "completed"


def summarize_baseline(data):
    """Extract a one-line baseline comparison summary from metrics data."""
    meta = data.get("metadata", {})
    bc = meta.get("baseline_comparison")
    if not bc or not bc.get("metrics"):
        return ""
    parts = []
    for m in bc["metrics"]:
        name = m.get("name", "")
        baseline = m.get("baseline")
        technique = m.get("technique")
        change_pct = m.get("change_pct")
        if change_pct is not None and baseline is not None and technique is not None:
            try:
                b_str = f"{float(baseline):.2f}" if isinstance(baseline, (int, float)) else str(baseline)
                t_str = f"{float(technique):.2f}" if isinstance(technique, (int, float)) else str(technique)
                parts.append(f"{name}: {b_str}->{t_str} ({change_pct:+.0f}%)")
            except (TypeError, ValueError):
                pass
    return "; ".join(parts) if parts else ""


def print_narrative_synthesis(results):
    """Print KEY TAKEAWAYS section summarizing what was demonstrated."""
    if not results:
        return

    BOLD = "\033[1m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    print(f"\n{CYAN}{'━' * 64}{RESET}")
    print(f"{CYAN}┃{RESET} {BOLD}KEY TAKEAWAYS{RESET}")
    print(f"{CYAN}{'━' * 64}{RESET}")

    for label, path, data in results:
        meta = data.get("metadata", {})
        bullet = None

        if "01" in label:
            row_count = meta.get("row_count", "?")
            bullet = f"Feast feature store registered and served {row_count} training examples with point-in-time correctness"
        elif "02" in label:
            steps = data.get("steps", [])
            if steps:
                first_loss = steps[0].get("loss", "?")
                last_loss = steps[-1].get("loss", "?")
                bullet = f"SFT + LoRA fine-tuning: loss {first_loss:.3f} -> {last_loss:.3f} over {len(steps)} steps"
        elif "03a" in label:
            answer = meta.get("answer", "")
            if answer:
                bullet = f"DPR + FAISS RAG retrieved context and generated: \"{answer[:50]}...\""
        elif "03b" in label:
            answer = meta.get("answer", "")
            if answer:
                bullet = f"Sentence Transformers RAG grounded answer in retrieved context"
        elif "04" in label:
            num_chunks = meta.get("num_chunks", "?")
            shape = meta.get("embedding_shape", [])
            dim = shape[1] if len(shape) > 1 else "?"
            bullet = f"Ingested {num_chunks} chunks with {dim}-dim embeddings into Feast + Milvus"
        elif "05" in label:
            steps = data.get("steps", [])
            if steps:
                bullet = f"Combined Feast features + RAG retrieval + SFT training over {len(steps)} steps"
        elif "06" in label:
            bc = meta.get("baseline_comparison", {})
            metrics = bc.get("metrics", [])
            for m in metrics:
                if "Memory" in m.get("name", ""):
                    pct = m.get("change_pct")
                    if pct is not None:
                        bullet = f"DeepSpeed ZeRO-2 reduced peak memory by {abs(pct):.0f}% vs naive estimate"
            if not bullet:
                steps = data.get("steps", [])
                if steps:
                    bullet = f"DeepSpeed ZeRO-2 training completed over {len(steps)} steps"
        elif "07" in label:
            steps = data.get("steps", [])
            if steps:
                bullet = f"DreamBooth fine-tuned Stable Diffusion on custom concept ({len(steps)} steps)"
        elif "08" in label:
            trials = data.get("trials", [])
            completed = [t for t in trials if t.get("value") is not None]
            if completed:
                best = min(completed, key=lambda t: t["value"])
                bullet = f"Optuna found best loss {best['value']:.4f} across {len(completed)} trials (sine-wave regression)"

        if bullet:
            print(f"{CYAN}┃{RESET}  - {label}: {bullet}")

    print(f"{CYAN}{'━' * 64}{RESET}")
    print()


def main():
    results = load_metrics()

    if not results:
        print("No metrics.json files found.")
        return

    has_baseline = any(
        d.get("metadata", {}).get("baseline_comparison") for _, _, d in results
    )

    rows = []
    for label, path, data in results:
        elapsed = data.get("elapsed_seconds", "?")
        summary = summarize_training(data)
        row = [label, f"{elapsed}s", summary]
        if has_baseline:
            row.append(summarize_baseline(data))
        rows.append(row)

    headers = ["Demo", "Time", "Summary"]
    if has_baseline:
        headers.append("vs Baseline")

    plot_table(
        headers,
        rows,
        title="All Demo Metrics",
    )

    print(f"Found {len(results)}/{len(METRICS_LOCATIONS)} metrics files.\n")

    # ── Narrative Synthesis ─────────────────────────────────────────────
    print_narrative_synthesis(results)


if __name__ == "__main__":
    main()
