#!/usr/bin/env python3
"""Supervised Fine-Tuning (SFT) for LLMs.

Ported from examples/kfto-sft-llm/sft.ipynb.
Uses TRL SFTTrainer with PEFT/LoRA for parameter-efficient fine-tuning.
Runs on a single GPU (no FSDP/distributed).

Usage:
    python scripts/02_sft_llm.py [--test-mode]
    python scripts/02_sft_llm.py --model-name-or-path ibm-granite/granite-3.0-1b-a400m-base --num-epochs 3
"""

import argparse
import random
import sys

# GPU setup
sys.path.insert(0, ".")
from scripts.common.gpu_utils import setup_cuda_env

setup_cuda_env()

from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


def build_parameters(args):
    """Build the YAML-like parameter dict from CLI args."""
    return {
        # Model
        "model_name_or_path": args.model_name_or_path,
        "model_revision": "main",
        "torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "use_liger_kernel": False,
        # PEFT / LoRA
        "use_peft": args.use_peft,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "lora_modules_to_save": [],
        # QLoRA
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": False,
        # Dataset
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_train_split": "train",
        "dataset_test_split": "test",
        "dataset_text_field": "text",
        "dataset_kwargs": {
            "add_special_tokens": False,
            "append_concat_token": False,
        },
        # SFT
        "max_length": args.max_seq_length,
        "packing": False,
        # Training
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "auto_find_batch_size": False,
        "eval_strategy": "epoch",
        "bf16": True,
        "tf32": False,
        "learning_rate": args.learning_rate,
        "warmup_steps": 10,
        "lr_scheduler_type": "inverse_sqrt",
        "optim": "adamw_torch_fused",
        "max_grad_norm": 1.0,
        "seed": 42,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # No FSDP on single GPU
        # Checkpointing
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "resume_from_checkpoint": False,
        # Logging
        "log_level": "warning",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "report_to": ["tensorboard"],
        "output_dir": args.output_dir,
    }


def train(parameters):
    """Main training function."""
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_dict(parameters)

    set_seed(training_args.seed)

    # Model kwargs
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=not training_args.gradient_checkpointing,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        right_pad_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
        if right_pad_id is not None:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    train_dataset = load_dataset(
        path=script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    test_dataset = None
    if training_args.eval_strategy != "no":
        test_dataset = load_dataset(
            path=script_args.dataset_name,
            name=script_args.dataset_config,
            split=script_args.dataset_test_split,
        )

    # Template datasets for GSM8K (question/answer -> chat format)
    def template_dataset(sample):
        messages = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
        if tokenizer.chat_template:
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
        # Fallback for models without a chat template
        return {
            "text": f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        }

    # Check if dataset has 'question'/'answer' columns (GSM8K format)
    if "question" in train_dataset.column_names:
        train_dataset = train_dataset.map(
            template_dataset, remove_columns=["question", "answer"]
        )
        if test_dataset is not None:
            test_dataset = test_dataset.map(
                template_dataset, remove_columns=["question", "answer"]
            )

    # Log samples
    for index in random.sample(range(len(train_dataset)), min(2, len(train_dataset))):
        print(f"Sample {index}: {train_dataset[index]['text'][:200]}...")

    # Train
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
        processing_class=tokenizer,
    )

    if trainer.accelerator.is_main_process and hasattr(
        trainer.model, "print_trainable_parameters"
    ):
        trainer.model.print_trainable_parameters()

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)

    print(f"Training completed, model checkpoint saved to {training_args.output_dir}")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="SFT Fine-Tuning")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="ibm-granite/granite-3.0-1b-a400m-base",
        help="Model to fine-tune",
    )
    parser.add_argument("--num-epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument("--use-peft", action="store_true", default=True, help="Use PEFT/LoRA")
    parser.add_argument("--no-peft", dest="use_peft", action="store_false", help="Disable PEFT/LoRA")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use QLoRA 4-bit")
    parser.add_argument(
        "--dataset-name", type=str, default="gsm8k", help="Dataset name"
    )
    parser.add_argument(
        "--dataset-config", type=str, default="main", help="Dataset config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/sft_output",
        help="Output directory for model checkpoint",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (1 epoch, small batch, 10 samples)",
    )
    args = parser.parse_args()

    if args.test_mode:
        args.num_epochs = 1
        args.batch_size = 1
        args.max_seq_length = 256

    from scripts.common.demo_utils import print_intro, suppress_known_warnings

    suppress_known_warnings()
    print_intro(
        "Demo 02: SFT Fine-Tuning with LoRA",
        "Fine-tune an LLM on math problems using SFT + LoRA",
        "LoRA lets you fine-tune billion-param models on a single GPU by training only low-rank adapters",
        "Loss decreasing, base vs fine-tuned answers",
    )

    print("=" * 60)
    print("SFT Fine-Tuning")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Epochs: {args.num_epochs}, Batch: {args.batch_size}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    parameters = build_parameters(args)

    # ── Baseline: run base model inference on sample prompts ──────────────
    import torch
    import transformers

    baseline_prompts = [
        "What is 2 + 3?",
        "If a train travels 60 miles in 1 hour, how far does it go in 3 hours?",
        "A store sells apples for $2 each. How much do 5 apples cost?",
    ]
    base_answers = []
    print("\n--- Baseline: base model inference ---")
    try:
        bl_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if bl_tokenizer.pad_token is None:
            bl_tokenizer.pad_token = bl_tokenizer.eos_token
        bl_device = 0 if torch.cuda.is_available() else -1
        bl_pipe = transformers.pipeline(
            "text-generation",
            model=args.model_name_or_path,
            tokenizer=bl_tokenizer,
            device=bl_device,
            torch_dtype=torch.bfloat16,
        )
        for prompt in baseline_prompts:
            out = bl_pipe(prompt, max_new_tokens=50, do_sample=False)
            answer = out[0]["generated_text"][len(prompt):].strip()
            base_answers.append(answer)
            print(f"  Q: {prompt}")
            print(f"  A: {answer[:100]}")
        del bl_pipe
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Baseline inference skipped: {e}")
        base_answers = ["(skipped)"] * len(baseline_prompts)

    if args.test_mode:
        parameters["max_steps"] = 5
        parameters["eval_strategy"] = "no"
        parameters["save_strategy"] = "no"
        parameters["report_to"] = []

    trainer = train(parameters)

    # ── Metrics, Charts & Validation ──────────────────────────────────────
    import os

    from scripts.common.demo_utils import (
        BaselineComparator,
        MetricsCollector,
        OutputValidator,
        plot_loss_curve,
        plot_multi_loss,
        plot_table,
    )

    mc = MetricsCollector("02_sft_llm")
    mc.set_metadata(
        model=args.model_name_or_path,
        epochs=args.num_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
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

    output_dir = args.output_dir
    metrics_path = os.path.join(output_dir, "metrics.json")
    mc.save(metrics_path)

    # Charts
    if train_steps and train_losses:
        plot_loss_curve(train_steps, train_losses, title="SFT Training Loss")
    if eval_epochs and eval_losses:
        plot_multi_loss(
            {"eval_loss": (eval_epochs, eval_losses)},
            title="SFT Eval Loss",
            xlabel="Epoch",
        )

    # Validation
    v = OutputValidator("SFT Fine-Tuning", test_mode=args.test_mode)
    v.check_dir_exists(output_dir, "Model checkpoint dir exists and non-empty")
    adapter_config_path = os.path.join(output_dir, "adapter_config.json")
    adapter_weights_path = os.path.join(output_dir, "adapter_model.safetensors")
    v.check_file_exists(adapter_config_path, "adapter_config.json exists")
    v.check_file_min_size(
        adapter_weights_path, 1024, "adapter_model.safetensors >= 1KB"
    )
    if train_losses:
        v.check_metric_range(
            train_losses[-1], 0.0, 15.0,
            f"Final loss {train_losses[-1]:.4f} in [0, 15]",
        )
    if len(train_losses) >= 2:
        v.check_or_warn(
            float(train_losses[-1]) < float(train_losses[0]),
            "Training loss decreased",
            test_mode_note="only 5 steps, loss may not decrease",
        )
    # Validate adapter_config.json is valid JSON
    if os.path.isfile(adapter_config_path):
        import json

        def _check_adapter_json():
            with open(adapter_config_path) as f:
                json.load(f)
        v.check_callable(_check_adapter_json, "adapter_config.json is valid JSON")
    v.check_file_exists(metrics_path, "metrics.json saved")

    # ── Fine-tuned model inference and baseline comparison ────────────────
    finetuned_answers = []
    bc = None
    if base_answers and base_answers[0] != "(skipped)":
        print("\n--- Fine-tuned model inference ---")
        try:
            from peft import PeftModel

            ft_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
            if ft_tokenizer.pad_token is None:
                ft_tokenizer.pad_token = ft_tokenizer.eos_token
            ft_device = 0 if torch.cuda.is_available() else -1
            ft_pipe = transformers.pipeline(
                "text-generation",
                model=output_dir,
                tokenizer=ft_tokenizer,
                device=ft_device,
                torch_dtype=torch.bfloat16,
            )
            for prompt in baseline_prompts:
                out = ft_pipe(prompt, max_new_tokens=50, do_sample=False)
                answer = out[0]["generated_text"][len(prompt):].strip()
                finetuned_answers.append(answer)
                print(f"  Q: {prompt}")
                print(f"  A: {answer[:100]}")
            del ft_pipe
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Fine-tuned inference skipped: {e}")
            finetuned_answers = ["(skipped)"] * len(baseline_prompts)

    if finetuned_answers and finetuned_answers[0] != "(skipped)":
        # Answer comparison table
        trunc = lambda s, n=60: s[:n] + ("..." if len(s) > n else "")
        answer_rows = []
        for i, prompt in enumerate(baseline_prompts):
            answer_rows.append([
                trunc(prompt, 40),
                trunc(base_answers[i]),
                trunc(finetuned_answers[i]),
            ])
        plot_table(
            ["Question", "Base Model", "Fine-Tuned"],
            answer_rows,
            title="Base vs Fine-Tuned Answers",
        )

        # Metrics comparison
        avg_base_len = sum(len(a) for a in base_answers) / len(base_answers)
        avg_ft_len = sum(len(a) for a in finetuned_answers) / len(finetuned_answers)

        bc = BaselineComparator("SFT: Base Model vs Fine-Tuned")
        bc.add_metric(
            "Avg Answer Length", avg_base_len, avg_ft_len,
            unit=" chars", lower_is_better=False,
        )
        if train_losses:
            bc.add_metric(
                "Training Loss (first)", train_losses[0], train_losses[-1],
                lower_is_better=True,
            )
        bc.render()
        mc.set_metadata(
            baseline_prompts=baseline_prompts,
            base_answers=base_answers,
            finetuned_answers=finetuned_answers,
            baseline_comparison=bc.to_dict(),
        )
        mc.save(metrics_path)
        # In test mode, fine-tuned answers may be identical due to minimal training
        if args.test_mode:
            v.check_or_warn(
                finetuned_answers != base_answers,
                "Fine-tuned answers differ from baseline",
                test_mode_note="only 5 steps, answers may be identical",
            )
        else:
            bc.add_validations(v)

    v.print_report()

    print("\nSFT Fine-Tuning completed successfully!")
    print("\n>> Next: Scale this training with DeepSpeed (Demo 06)")


if __name__ == "__main__":
    main()
