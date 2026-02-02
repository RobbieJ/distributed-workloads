#!/usr/bin/env python3
"""Create GSM8K training and test datasets in JSONL format.

Ported from examples/ray-finetune-llm-deepspeed/create_dataset.py.
Generates chat-formatted JSONL files suitable for SFT and DeepSpeed training.

Usage:
    python scripts/create_dataset.py [--output-dir ./data]
    python scripts/create_dataset.py --format tokens  # token-based format
    python scripts/create_dataset.py --format chat     # custom chat template
    python scripts/create_dataset.py --format hf       # HuggingFace chat template (default)
"""

import argparse
import json
import os

from datasets import load_dataset


def gsm8k_hf_chat_template(output_dir):
    """Default: HuggingFace chat template format."""
    dataset = load_dataset("gsm8k", "main")
    dataset_splits = {"train": dataset["train"], "test": dataset["test"]}

    os.makedirs(output_dir, exist_ok=True)

    for key, ds in dataset_splits.items():
        path = os.path.join(output_dir, f"{key}.jsonl")
        with open(path, "w") as f:
            for item in ds:
                record = {
                    "messages": [
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["answer"]},
                    ]
                }
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(ds)} examples to {path}")


def gsm8k_qa_tokens_template(output_dir):
    """Token-based format with special delimiters."""
    dataset = load_dataset("gsm8k", "main")
    dataset_splits = {"train": dataset["train"], "test": dataset["test"]}

    os.makedirs(output_dir, exist_ok=True)

    config = {
        "chat_template": "{{ messages }}",
        "special_tokens": ["<START_Q>", "<END_Q>", "<START_A>", "<END_A>"],
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(json.dumps(config))

    for key, ds in dataset_splits.items():
        path = os.path.join(output_dir, f"{key}.jsonl")
        with open(path, "w") as f:
            for item in ds:
                record = {
                    "messages": (
                        f"<START_Q>{item['question']}<END_Q>"
                        f"<START_A>{item['answer']}<END_A>"
                    )
                }
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(ds)} examples to {path}")


def gsm8k_qa_chat_template(output_dir):
    """Custom chat template with role-based formatting."""
    dataset = load_dataset("gsm8k", "main")
    dataset_splits = {"train": dataset["train"], "test": dataset["test"]}

    os.makedirs(output_dir, exist_ok=True)

    config = {
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '\n\nQuestion: ' + message['content'] +  eos_token }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAnswer: '  + message['content'] +  eos_token  }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '\n\nAnswer: ' }}"
            "{% endif %}"
        )
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(json.dumps(config))

    for key, ds in dataset_splits.items():
        path = os.path.join(output_dir, f"{key}.jsonl")
        with open(path, "w") as f:
            for item in ds:
                record = {
                    "messages": [
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["answer"]},
                    ]
                }
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(ds)} examples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Create GSM8K dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["hf", "tokens", "chat"],
        default="hf",
        help="Dataset format: hf (default), tokens, or chat",
    )
    args = parser.parse_args()

    print(f"Creating GSM8K dataset ({args.format} format) in {args.output_dir}")

    if args.format == "tokens":
        gsm8k_qa_tokens_template(args.output_dir)
    elif args.format == "chat":
        gsm8k_qa_chat_template(args.output_dir)
    else:
        gsm8k_hf_chat_template(args.output_dir)

    print("Dataset creation complete!")


if __name__ == "__main__":
    main()
