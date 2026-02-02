#!/usr/bin/env python3
"""DeepSpeed LLM fine-tuning.

Ported from examples/ray-finetune-llm-deepspeed/ray_finetune_llm_deepspeed.py.
Removes all Ray dependencies. Uses Accelerate + DeepSpeed ZeRO-2 for
memory-efficient single-GPU training with LoRA support.

Usage:
    python scripts/06_deepspeed_finetune.py [--test-mode]
    python scripts/06_deepspeed_finetune.py --model-name meta-llama/Meta-Llama-3.1-8B --lora
"""

import argparse
import functools
import json
import math
import os
import sys
import time

# GPU setup
sys.path.insert(0, ".")
from scripts.common.gpu_utils import setup_cuda_env

setup_cuda_env()

import torch
import torch.nn as nn
import tqdm
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

OPTIM_BETAS = (0.9, 0.999)
OPTIM_EPS = 1e-8
NUM_WARMUP_STEPS = 10
OPTIM_WEIGHT_DECAY = 0.0


def collate_fn(batch, tokenizer, block_size, device):
    """Tokenize and collate a batch of messages."""
    texts = []
    for item in batch:
        messages = item["messages"]
        if isinstance(messages, str):
            texts.append(messages)
        elif tokenizer.chat_template:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    add_special_tokens=False,
                )
            )
        else:
            # Fallback for models without a chat template
            parts = []
            for msg in messages:
                parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
            texts.append("\n".join(parts))

    out_batch = tokenizer(
        texts,
        padding="max_length",
        max_length=block_size,
        truncation=True,
        return_tensors="pt",
    )
    out_batch["labels"] = out_batch["input_ids"].clone()
    out_batch = {k: v.to(device) for k, v in out_batch.items()}
    return out_batch


def get_number_of_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, eval_dataloader, accelerator, as_test=False):
    """Evaluate model and return perplexity + loss."""
    model.eval()
    losses = []

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather(loss[None]))
        if as_test:
            break

    losses = torch.stack(losses)
    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss


def training_function(args, ds_config_path):
    """Main training loop with DeepSpeed + Accelerate. Returns MetricsCollector."""
    from scripts.common.demo_utils import MetricsCollector

    mc = MetricsCollector("06_deepspeed_finetune")
    mc.set_metadata(model=args.model_name, lora=args.lora)
    # Ensure distributed env vars are set for single-GPU DeepSpeed
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    # DeepSpeed plugin
    ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config_path)
    ds_plugin.hf_ds_config.config["train_micro_batch_size_per_gpu"] = (
        args.batch_size_per_device
    )

    accelerator = Accelerator(
        deepspeed_plugin=ds_plugin,
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mx,
    )

    set_seed(42)

    # Load dataset
    if args.train_path and os.path.isfile(args.train_path):
        train_ds = load_dataset("json", data_files=args.train_path, split="train")
    else:
        train_ds = load_dataset("gsm8k", "main", split="train")
        train_ds = train_ds.map(
            lambda x: {
                "messages": [
                    {"role": "user", "content": x["question"]},
                    {"role": "assistant", "content": x["answer"]},
                ]
            },
            remove_columns=["question", "answer"],
        )

    if args.test_path and os.path.isfile(args.test_path):
        valid_ds = load_dataset("json", data_files=args.test_path, split="train")
    else:
        valid_ds = load_dataset("gsm8k", "main", split="test")
        valid_ds = valid_ds.map(
            lambda x: {
                "messages": [
                    {"role": "user", "content": x["question"]},
                    {"role": "assistant", "content": x["answer"]},
                ]
            },
            remove_columns=["question", "answer"],
        )

    if args.as_test:
        train_ds = train_ds.select(range(min(20, len(train_ds))))
        valid_ds = valid_ds.select(range(min(10, len(valid_ds))))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load config if exists
    chat_template = None
    special_tokens = None
    if args.dataset_config and os.path.isfile(args.dataset_config):
        with open(args.dataset_config) as f:
            dataset_config = json.load(f)
            chat_template = dataset_config.get("chat_template")
            special_tokens = dataset_config.get("special_tokens")

    if special_tokens:
        tokenizer.add_tokens(special_tokens, special_tokens=True)
    if chat_template:
        tokenizer.chat_template = chat_template

    collate_partial = functools.partial(
        collate_fn,
        tokenizer=tokenizer,
        block_size=args.ctx_len,
        device=accelerator.device,
    )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=args.batch_size_per_device,
        shuffle=True,
        collate_fn=collate_partial,
    )
    eval_dataloader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size_per_device,
        shuffle=False,
        collate_fn=collate_partial,
    )

    # Model
    print(f"Loading model: {args.model_name}")
    s = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="sdpa",
    )
    print(f"Model loaded in {time.time() - s:.1f}s")

    total_param_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()

    model.resize_token_embeddings(len(tokenizer))

    # LoRA
    if args.lora:
        lora_config_dict = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "task_type": "CAUSAL_LM",
            "bias": "none",
        }
        if args.lora_config and os.path.isfile(args.lora_config):
            with open(args.lora_config) as f:
                lora_config_dict = json.load(f)

        lora_config = LoraConfig(**lora_config_dict)
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        print(f"LoRA applied. Trainable params: {get_number_of_params(model):,}")

    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()

    # Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(
        model.parameters(),
        lr=args.lr,
        betas=OPTIM_BETAS,
        weight_decay=OPTIM_WEIGHT_DECAY,
        eps=OPTIM_EPS,
    )

    # Scheduler
    num_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum)
    total_training_steps = num_steps_per_epoch * args.num_epochs

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer,
            warmup_num_steps=NUM_WARMUP_STEPS,
            total_num_steps=total_training_steps,
        )

    # Prepare
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # Training loop
    print("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        loss_sum = torch.tensor(0.0).to(accelerator.device)
        s_epoch = time.time()

        for step, batch in tqdm.tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                loss_val = loss.item()
                loss_sum += loss_val
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            mc.add_step(step=epoch * len(train_dataloader) + step, loss=loss_val)

            if accelerator.is_main_process and step % 10 == 0:
                print(f"[epoch {epoch} step {step}] loss: {loss.item():.4f}")

            if args.as_test and step >= 5:
                break

        e_epoch = time.time()
        print(f"Epoch {epoch} train time: {e_epoch - s_epoch:.1f}s")

        # Evaluate
        print("Running evaluation...")
        perplex, eloss = evaluate(
            model, eval_dataloader, accelerator, as_test=args.as_test
        )
        print(f"Eval loss: {eloss:.4f}, Perplexity: {perplex:.2f}")
        mc.add_epoch(
            epoch=epoch,
            eval_loss=eloss,
            perplexity=perplex,
            train_time=e_epoch - s_epoch,
        )

        # Save checkpoint
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            tokenizer.save_pretrained(checkpoint_dir)

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                checkpoint_dir,
                is_main_process=True,
                save_function=accelerator.save,
                safe_serialization=True,
                state_dict=accelerator.get_state_dict(model),
            )
            print(f"Checkpoint saved to {checkpoint_dir}")

        if args.as_test:
            break

    print("Training complete!")
    if torch.cuda.is_available():
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        mc.set_metadata(
            peak_memory_bytes=peak_memory_bytes,
            total_param_count=total_param_count,
        )
    return mc


def main():
    parser = argparse.ArgumentParser(description="DeepSpeed LLM Fine-tuning")
    parser.add_argument(
        "--model-name",
        type=str,
        default="ibm-granite/granite-3.0-1b-a400m-base",
        help="Model to fine-tune",
    )
    parser.add_argument("--train-path", type=str, default=None, help="Training JSONL")
    parser.add_argument("--test-path", type=str, default=None, help="Test JSONL")
    parser.add_argument(
        "--dataset-config", type=str, default=None, help="Dataset config JSON"
    )
    parser.add_argument("--mx", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument(
        "--ds-config",
        type=str,
        default="./configs/deepspeed/ds_config_zero2.json",
        help="DeepSpeed config",
    )
    parser.add_argument("--lora", action="store_true", default=True, help="Enable LoRA")
    parser.add_argument("--no-lora", dest="lora", action="store_false")
    parser.add_argument("--lora-config", type=str, default=None, help="LoRA config JSON")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--batch-size-per-device", type=int, default=4)
    parser.add_argument("--eval-batch-size-per-device", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./models/deepspeed_output")
    parser.add_argument("--no-grad-ckpt", action="store_true")
    parser.add_argument("--as-test", action="store_true", help="Quick test run")
    parser.add_argument("--test-mode", action="store_true", help="Alias for --as-test")
    args = parser.parse_args()

    if args.test_mode:
        args.as_test = True

    from scripts.common.demo_utils import print_intro, suppress_known_warnings

    suppress_known_warnings()
    print_intro(
        "Demo 06: DeepSpeed ZeRO-2 Fine-Tuning",
        "Fine-tune with DeepSpeed ZeRO-2 memory optimization",
        "ZeRO-2 partitions optimizer states across GPUs, enabling larger models",
        "Memory savings vs naive training, loss convergence",
    )

    print("=" * 60)
    print("DeepSpeed LLM Fine-Tuning")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"LoRA: {args.lora}")
    print(f"DeepSpeed config: {args.ds_config}")

    mc = training_function(args, args.ds_config)

    # ── Charts & Validation ───────────────────────────────────────────────
    from scripts.common.demo_utils import (
        BaselineComparator,
        OutputValidator,
        plot_comparison_bar,
        plot_loss_curve,
        plot_multi_loss,
    )

    metrics_path = os.path.join(args.output_dir, "metrics.json")

    # Baseline comparison: naive memory estimate vs DeepSpeed actual
    bc = None
    param_count = mc.metadata.get("total_param_count")
    peak_mem = mc.metadata.get("peak_memory_bytes")
    if param_count and peak_mem:
        # Naive estimate: params * 2 (bf16 weights) + params * 4 * 2 (Adam states) + params * 4 (grads) = params * 14
        naive_bytes = param_count * 14
        naive_gb = naive_bytes / (1024 ** 3)
        actual_gb = peak_mem / (1024 ** 3)
        trainable_params = mc.metadata.get("trainable_params", param_count)

        bc = BaselineComparator("DeepSpeed: Naive vs ZeRO-2 Memory")
        bc.add_metric("Peak Memory", naive_gb, actual_gb, unit=" GB", lower_is_better=True)
        bc.add_metric("Total Params", param_count, param_count, lower_is_better=False)
        bc.render()
        plot_comparison_bar(
            ["Peak Memory (GB)"],
            [naive_gb],
            [actual_gb],
            title="Naive Estimate vs DeepSpeed ZeRO-2",
        )
        mc.set_metadata(baseline_comparison=bc.to_dict())

    mc.save(metrics_path)

    # Charts
    if mc.steps:
        steps = [s["step"] for s in mc.steps]
        losses = [s["loss"] for s in mc.steps]
        plot_loss_curve(steps, losses, title="DeepSpeed Training Loss")

    if mc.epochs:
        epoch_nums = [e["epoch"] for e in mc.epochs]
        eval_losses = [e["eval_loss"] for e in mc.epochs]
        perplexities = [e["perplexity"] for e in mc.epochs]
        plot_multi_loss(
            {
                "eval_loss": (epoch_nums, eval_losses),
                "perplexity": (epoch_nums, perplexities),
            },
            title="DeepSpeed Eval Metrics",
            xlabel="Epoch",
        )

    # Validation
    v = OutputValidator("DeepSpeed Fine-Tuning", test_mode=args.as_test)
    v.check_dir_exists(args.output_dir, "Output directory exists")
    checkpoint_dir = os.path.join(args.output_dir, "epoch_0")
    v.check_dir_exists(checkpoint_dir, "epoch_0/ checkpoint exists and non-empty")
    config_path = os.path.join(checkpoint_dir, "config.json")
    v.check_or_warn(
        os.path.isfile(config_path),
        "config.json in checkpoint",
        test_mode_note="LoRA checkpoints may use adapter_config.json instead",
    )

    if mc.steps:
        final_loss = mc.steps[-1]["loss"]
        v.check_metric_range(
            final_loss, 0.0, 20.0, f"Final loss {final_loss:.4f} in [0, 20]"
        )
    if mc.epochs:
        last_eval = mc.epochs[-1]["eval_loss"]
        last_ppl = mc.epochs[-1]["perplexity"]
        v.check_metric_range(
            last_eval, 0.0, 20.0, f"Eval loss {last_eval:.4f} in [0, 20]"
        )
        v.check_metric_range(
            last_ppl, 1.0, 1e6, f"Perplexity {last_ppl:.2f} in [1, 1e6]"
        )
    v.check_file_exists(metrics_path, "metrics.json saved")
    if bc is not None:
        bc.add_validations(v)
    v.print_report()

    print("\nDeepSpeed Fine-Tuning completed successfully!")


if __name__ == "__main__":
    main()
