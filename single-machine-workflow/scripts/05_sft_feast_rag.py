#!/usr/bin/env python3
"""SFT + Feast RAG training.

Ported from examples/kfto-sft-feast-rag/sft_feast_rag_model.ipynb (training portion).
Fine-tunes a RAG model (RagSequenceForGeneration) using data from Feast + Milvus.
Single-GPU, no FSDP.

Requires:
    - Milvus running (docker compose up)
    - Data ingested (python scripts/04_feast_rag_ingest.py)

Usage:
    python scripts/05_sft_feast_rag.py [--test-mode]
    python scripts/05_sft_feast_rag.py --feast-repo-dir ./data/feast_rag_repo --num-epochs 3
"""

import argparse
import os
import sys
from pathlib import Path

# GPU setup
sys.path.insert(0, ".")
from scripts.common.gpu_utils import setup_cuda_env

setup_cuda_env()

import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    GenerationConfig,
    RagConfig,
    RagSequenceForGeneration,
    RagTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)


def build_rag_model(
    generator_model_name,
    question_encoder_model_name,
    feast_repo_path,
    feature_view,
    device,
):
    """Build the RAG model with Feast-backed retriever."""
    from scripts.common.feast_rag_retriever import FeastIndex, FeastRAGRetriever

    # Question encoder
    qe_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_model_name)
    qe_model = DPRQuestionEncoder.from_pretrained(
        question_encoder_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Generator
    gen_tokenizer = AutoTokenizer.from_pretrained(
        generator_model_name, trust_remote_code=True, use_fast=True
    )
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(
        generator_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    # RAG config
    qe_config = {
        "model_type": "dpr",
        "hidden_size": 768,
        "vocab_size": qe_tokenizer.vocab_size,
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "projection_dim": 0,
        "torch_dtype": "bfloat16",
    }

    rag_config = RagConfig(
        question_encoder=qe_config,
        generator=gen_model.config.to_dict(),
        index_name="custom",
        index={"index_name": "feast_dummy_index", "custom_type": "FeastIndex"},
        n_docs=10,
    )

    features_to_retrieve = [
        "wiki_passages:passage_text",
        "wiki_passages:embedding",
        "wiki_passages:passage_id",
    ]

    # Retriever
    feast_index = FeastIndex()
    rag_retriever = FeastRAGRetriever(
        question_encoder_tokenizer=qe_tokenizer,
        generator_tokenizer=gen_tokenizer,
        question_encoder=qe_model,
        generator_model=gen_model,
        feast_repo_path=feast_repo_path,
        feature_view=feature_view,
        features=features_to_retrieve,
        search_type="vector",
        config=rag_config,
        index=feast_index,
    )

    # RAG model
    model = RagSequenceForGeneration(
        question_encoder=qe_model,
        config=rag_config,
        generator=gen_model,
        retriever=rag_retriever,
    )
    model.generation_config = GenerationConfig(
        max_length=128, num_beams=1, do_sample=False, length_penalty=1.0
    )

    model = model.to(device)
    return model, qe_tokenizer, gen_tokenizer, rag_retriever


def prepare_dataset(qe_tokenizer, gen_tokenizer, cache_dir, test_mode=False):
    """Prepare or load cached NQ dataset for RAG training."""
    cache_path = Path(cache_dir)

    if (cache_path / "train").exists() and (cache_path / "test").exists():
        print(f"Loading cached dataset from {cache_dir}")
        loaded = load_from_disk(cache_dir)
        return loaded["train"], loaded["test"]

    print("Preparing dataset (this will be cached for future runs)...")
    from datasets import Dataset, load_dataset

    # Use a simpler approach: generate synthetic Q&A pairs for training
    # In production, use the Natural Questions dataset
    num_samples = 50 if test_mode else 500
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the boiling point of water?",
        "Who painted the Mona Lisa?",
        "What is the largest planet?",
    ] * (num_samples // 5)

    answers = [
        "Paris",
        "William Shakespeare",
        "100 degrees Celsius",
        "Leonardo da Vinci",
        "Jupiter",
    ] * (num_samples // 5)

    def preprocess(question, answer):
        q_tokens = qe_tokenizer(
            question, truncation=True, max_length=32, padding="max_length"
        )
        a_tokens = gen_tokenizer(
            text_target=answer, truncation=True, max_length=32, padding="max_length"
        )
        return {
            "input_ids": q_tokens["input_ids"],
            "attention_mask": q_tokens["attention_mask"],
            "labels": a_tokens["input_ids"],
        }

    records = [preprocess(q, a) for q, a in zip(questions, answers)]

    split_idx = int(len(records) * 0.9)
    train_data = Dataset.from_list(records[:split_idx])
    test_data = Dataset.from_list(records[split_idx:])

    dataset_dict = DatasetDict({"train": train_data, "test": test_data})
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(cache_dir)

    return train_data, test_data


def main():
    parser = argparse.ArgumentParser(description="SFT + Feast RAG Training")
    parser.add_argument(
        "--generator-model",
        type=str,
        default="facebook/bart-large",
        help="Generator model (must be Seq2Seq like BART or T5)",
    )
    parser.add_argument(
        "--question-encoder",
        type=str,
        default="facebook/dpr-question_encoder-single-nq-base",
        help="Question encoder model",
    )
    parser.add_argument(
        "--feast-repo-dir",
        type=str,
        default="./data/feast_rag_repo",
        help="Feast repository directory",
    )
    parser.add_argument(
        "--dataset-cache",
        type=str,
        default="./data/rag_dataset_cache",
        help="Cache directory for preprocessed dataset",
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=4e-6)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/feast_rag_output",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument("--test-mode", action="store_true", help="Quick test run")
    args = parser.parse_args()

    if args.test_mode:
        args.num_epochs = 1

    from scripts.common.demo_utils import print_intro, suppress_known_warnings

    suppress_known_warnings()
    print_intro(
        "Demo 05: SFT + Feast RAG",
        "Combine Feast features + RAG retrieval + SFT training",
        "Ties together features (01), RAG (03/04), and fine-tuning (02) into one pipeline",
        "How retrieved context shapes training data",
    )

    print("=" * 60)
    print("SFT + Feast RAG Training")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(42)

    # Load feature view definition
    sys.path.insert(0, "configs/feast")
    from ragproject_repo import wiki_passage_feature_view

    # Build RAG model
    print("\n--- Building RAG model ---")
    model, qe_tokenizer, gen_tokenizer, retriever = build_rag_model(
        args.generator_model,
        args.question_encoder,
        args.feast_repo_dir,
        wiki_passage_feature_view,
        device,
    )

    # Prepare dataset
    print("\n--- Preparing dataset ---")
    train_dataset, test_dataset = prepare_dataset(
        qe_tokenizer, gen_tokenizer, args.dataset_cache, test_mode=args.test_mode
    )
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        warmup_steps=200 if not args.test_mode else 5,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=True,
        eval_strategy="epoch" if not args.test_mode else "no",
        save_strategy="epoch" if not args.test_mode else "no",
        save_total_limit=1,
        logging_steps=1,
        report_to=[],
        remove_unused_columns=False,
        predict_with_generate=True,
        max_steps=5 if args.test_mode else -1,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if not args.test_mode else None,
        tokenizer=gen_tokenizer,
        data_collator=default_data_collator,
    )

    # Test forward pass
    print("\n--- Test forward pass ---")
    test_query = "What is the capital of France?"
    test_answer = "Paris"
    q_tok = qe_tokenizer(test_query, return_tensors="pt")
    a_tok = gen_tokenizer(text_target=test_answer, return_tensors="pt")
    test_input = {
        "input_ids": q_tok["input_ids"].to(device),
        "attention_mask": q_tok["attention_mask"].to(device),
        "labels": a_tok["input_ids"].to(device),
    }
    with torch.no_grad():
        model(**test_input)
    print("Forward pass successful")

    # Train
    print("\n--- Starting training ---")
    trainer.train()
    print("Training complete!")

    # Save
    if not args.test_mode:
        save_path = os.path.join(args.output_dir, "final")
        trainer.save_model(save_path)
        print(f"Model saved to {save_path}")

    # ── Inference demo: run sample questions through RAG pipeline ────────
    print("\n--- Inference: running sample questions through fine-tuned RAG ---")
    sample_questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet?",
    ]
    inference_answers = []
    try:
        model.eval()
        rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        for q in sample_questions:
            input_dict = rag_tokenizer.prepare_seq2seq_batch(
                q, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_dict["input_ids"],
                    max_length=64,
                    num_beams=1,
                )
            answer = gen_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            inference_answers.append(answer)
            print(f"  Q: {q}")
            print(f"  A: {answer}")
    except Exception as e:
        print(f"  Inference skipped: {e}")
        inference_answers = ["(skipped)"] * len(sample_questions)

    # ── Metrics, Charts & Validation ──────────────────────────────────────
    from scripts.common.demo_utils import (
        MetricsCollector,
        OutputValidator,
        plot_loss_curve,
        plot_multi_loss,
        plot_table,
    )

    mc = MetricsCollector("05_sft_feast_rag")
    mc.set_metadata(
        generator_model=args.generator_model,
        question_encoder=args.question_encoder,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_samples=len(train_dataset),
        test_samples=len(test_dataset),
    )

    # Extract metrics from trainer log history
    log_history = trainer.state.log_history
    train_steps, train_losses = [], []
    eval_epochs, eval_losses = [], []
    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
            mc.add_step(step=entry["step"], loss=entry["loss"])
        if "eval_loss" in entry and "epoch" in entry:
            eval_epochs.append(entry["epoch"])
            eval_losses.append(entry["eval_loss"])
            mc.add_epoch(epoch=entry["epoch"], eval_loss=entry["eval_loss"])

    if inference_answers and inference_answers[0] != "(skipped)":
        mc.set_metadata(
            inference_questions=sample_questions,
            inference_answers=inference_answers,
        )
        trunc = lambda s, n=60: s[:n] + ("..." if len(s) > n else "")
        plot_table(
            ["Question", "Answer"],
            [[q, trunc(a)] for q, a in zip(sample_questions, inference_answers)],
            title="RAG Inference Results",
        )

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    mc.save(metrics_path)

    # Charts
    if train_steps and train_losses:
        plot_loss_curve(train_steps, train_losses, title="Feast RAG Training Loss")
    if eval_epochs and eval_losses:
        plot_multi_loss(
            {"eval_loss": (eval_epochs, eval_losses)},
            title="Feast RAG Eval Loss",
            xlabel="Epoch",
        )

    # Validation
    v = OutputValidator("SFT + Feast RAG Training")
    v.check_dir_exists(args.output_dir, "Output directory exists and non-empty")
    if train_losses:
        v.check_metric_range(
            train_losses[-1], 0.0, 500.0,
            f"Final loss {train_losses[-1]:.4f} in [0, 500]",
        )
    if len(train_losses) >= 2:
        v.check_loss_decreased(
            train_losses[0], train_losses[-1], "Training loss decreased"
        )
    if not args.test_mode:
        final_path = os.path.join(args.output_dir, "final")
        v.check_dir_exists(final_path, "Final model saved")
    v.check_file_exists(metrics_path, "metrics.json saved")
    v.print_report()

    print("\nSFT + Feast RAG Training completed successfully!")


if __name__ == "__main__":
    main()
