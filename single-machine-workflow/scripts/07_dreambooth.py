#!/usr/bin/env python3
"""DreamBooth training for Stable Diffusion.

Ported from examples/kfto-dreambooth/dreambooth.ipynb.
Removes Kubernetes/Ray dependencies. Uses Accelerate for single-GPU training.
Fine-tunes UNet while freezing text encoder and VAE.

Usage:
    python scripts/07_dreambooth.py [--test-mode]
    python scripts/07_dreambooth.py --concept-name sccorgi --concept-type dog --max-steps 300
"""

import argparse
import math
import os
import sys

# GPU setup
sys.path.insert(0, ".")
from scripts.common.gpu_utils import setup_cuda_env

setup_cuda_env()

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


class DreamBoothDataset(Dataset):
    """Dataset for DreamBooth fine-tuning."""

    def __init__(self, dataset, instance_prompt, tokenizer, size=512):
        self.dataset = dataset
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        return {
            "instance_images": self.transforms(image),
            "instance_prompt_ids": self.tokenizer(
                self.instance_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids,
        }


def collate_fn(examples, tokenizer):
    """Collate function for DreamBooth DataLoader."""
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    input_ids = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids
    return {"input_ids": input_ids, "pixel_values": pixel_values}


def train(args):
    """Main DreamBooth training function. Returns MetricsCollector."""
    from scripts.common.demo_utils import MetricsCollector

    mc = MetricsCollector("07_dreambooth")
    mc.set_metadata(
        concept_name=args.concept_name,
        concept_type=args.concept_type,
        model_id=args.model_id,
        max_train_steps=args.max_train_steps,
    )

    instance_prompt = f"a photo of {args.concept_name} {args.concept_type}"
    print(f"Instance prompt: {instance_prompt}")

    # Load tokenizer and dataset
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    dataset = load_dataset(args.dataset_id, split=args.dataset_split)
    train_dataset = DreamBoothDataset(dataset, instance_prompt, tokenizer)

    # Load model components
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    # Accelerator setup
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    set_seed(args.seed)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Optimizer
    if args.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(unet.parameters(), lr=args.learning_rate)

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, tokenizer),
    )

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # Training loop
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215

                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            mc.add_step(step=global_step, loss=logs["loss"])

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Save pipeline
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving pipeline to {args.output_dir}...")
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            steps_offset=1,
        )
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
            feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(args.output_dir)
        print(f"Pipeline saved to {args.output_dir}")

        # Upload to MinIO if requested
        if args.upload:
            from scripts.common.storage_utils import upload_directory

            upload_directory(args.output_dir, "models", "dreambooth")

    return mc


def main():
    parser = argparse.ArgumentParser(description="DreamBooth Training")
    parser.add_argument("--concept-name", type=str, default="ccorgi")
    parser.add_argument("--concept-type", type=str, default="dog")
    parser.add_argument(
        "--dataset-id", type=str, default="diffusers/dog-example"
    )
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument(
        "--model-id", type=str, default="CompVis/stable-diffusion-v1-4"
    )
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--max-train-steps", type=int, default=300)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--use-8bit-adam", action="store_true", default=True)
    parser.add_argument("--no-8bit-adam", dest="use_8bit_adam", action="store_false")
    parser.add_argument("--seed", type=int, default=3434554)
    parser.add_argument("--output-dir", type=str, default="./models/dreambooth")
    parser.add_argument("--upload", action="store_true", help="Upload to MinIO")
    parser.add_argument("--test-mode", action="store_true", help="Quick test (10 steps)")
    args = parser.parse_args()

    if args.test_mode:
        args.max_train_steps = 10

    from scripts.common.demo_utils import print_intro, suppress_known_warnings

    suppress_known_warnings()
    print_intro(
        "Demo 07: DreamBooth Stable Diffusion",
        "Fine-tune Stable Diffusion with DreamBooth",
        "DreamBooth teaches a diffusion model a new concept from just a few images",
        "Loss decreasing, generated image of learned concept",
    )

    print("=" * 60)
    print("DreamBooth Training")
    print("=" * 60)
    print(f"Concept: {args.concept_name} ({args.concept_type})")
    print(f"Model: {args.model_id}")
    print(f"Max steps: {args.max_train_steps}")

    mc = train(args)

    # ── Generate sample image ────────────────────────────────────────────
    generated_image_path = os.path.join(args.output_dir, "generated_sample.png")
    try:
        print("\n--- Generating sample image ---")
        gen_pipe = StableDiffusionPipeline.from_pretrained(
            args.output_dir, torch_dtype=torch.float16
        )
        gen_pipe = gen_pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        inference_steps = 5 if args.test_mode else 15
        prompt = f"a photo of {args.concept_name} {args.concept_type}"
        image = gen_pipe(prompt, num_inference_steps=inference_steps).images[0]
        image.save(generated_image_path)
        print(f"Generated image saved to {generated_image_path}")
        del gen_pipe
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Image generation skipped: {e}")

    # ── Charts & Validation ───────────────────────────────────────────────
    import json

    from scripts.common.demo_utils import (
        BaselineComparator,
        OutputValidator,
        plot_comparison_bar,
        plot_loss_curve,
    )

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    mc.save(metrics_path)

    # Chart
    if mc.steps:
        steps = [s["step"] for s in mc.steps]
        losses = [s["loss"] for s in mc.steps]
        plot_loss_curve(steps, losses, title="DreamBooth Training Loss")

    # Baseline comparison: first 3 steps vs last 3 steps
    bc = None
    if mc.steps and len(mc.steps) > 5:
        avg_first3 = sum(s["loss"] for s in mc.steps[:3]) / 3
        avg_last3 = sum(s["loss"] for s in mc.steps[-3:]) / 3
        bc = BaselineComparator("DreamBooth: Before vs After Training")
        bc.add_metric("Avg Loss (3 steps)", avg_first3, avg_last3, lower_is_better=True)
        bc.add_metric("Total Steps", len(mc.steps), len(mc.steps), lower_is_better=False)
        bc.render()
        plot_comparison_bar(
            ["Avg Loss"],
            [avg_first3],
            [avg_last3],
            title="DreamBooth: Early vs Late Loss",
        )
        mc.set_metadata(baseline_comparison=bc.to_dict())

    # Validation
    v = OutputValidator("DreamBooth Training", test_mode=args.test_mode)
    v.check_dir_exists(args.output_dir, "Pipeline directory exists and non-empty")
    model_index = os.path.join(args.output_dir, "model_index.json")
    v.check_file_exists(model_index, "model_index.json exists")
    for subdir in ["unet", "vae", "tokenizer"]:
        v.check_dir_exists(
            os.path.join(args.output_dir, subdir), f"{subdir}/ subdir exists"
        )
    if mc.steps:
        final_loss = mc.steps[-1]["loss"]
        v.check_metric_range(
            final_loss, 0.0, 5.0, f"Final loss {final_loss:.4f} in [0, 5]"
        )
        if len(mc.steps) > 5:
            avg_first = sum(s["loss"] for s in mc.steps[:3]) / 3
            avg_last = sum(s["loss"] for s in mc.steps[-3:]) / 3
            v.check_or_warn(
                avg_last < avg_first,
                f"Loss trend decreased (avg first 3: {avg_first:.4f} > avg last 3: {avg_last:.4f})",
                test_mode_note="only 10 steps, loss may not converge",
            )
    if os.path.isfile(model_index):
        def _check_model_index():
            with open(model_index) as f:
                json.load(f)
        v.check_callable(_check_model_index, "model_index.json is valid JSON")
    v.check_file_exists(metrics_path, "metrics.json saved")
    if os.path.isfile(generated_image_path):
        v.check_file_min_size(generated_image_path, 100, "Generated image saved")
    if bc is not None:
        if args.test_mode:
            # In test mode, baseline comparison may fail due to insufficient training
            for m in bc.metrics:
                try:
                    b_num, t_num = float(m["baseline"]), float(m["technique"])
                    if m["lower_is_better"]:
                        v.check_or_warn(
                            t_num <= b_num,
                            f"{m['name']}: technique <= baseline",
                            test_mode_note="only 10 steps",
                        )
                except (TypeError, ValueError):
                    pass
        else:
            bc.add_validations(v)
    v.print_report()

    print("\nDreamBooth Training completed successfully!")


if __name__ == "__main__":
    main()
