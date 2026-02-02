#!/usr/bin/env python3
"""FastAPI serving for DreamBooth Stable Diffusion.

Replaces KServe with a simple FastAPI server for text-to-image generation.
Loads a fine-tuned Stable Diffusion pipeline and serves it via HTTP.

Usage:
    uvicorn scripts.07_dreambooth_serve:app --host 0.0.0.0 --port 8000
    # Or run directly:
    python scripts/07_dreambooth_serve.py --model-path ./models/dreambooth --port 8000
"""

import argparse
import base64
import io
import os
import sys

sys.path.insert(0, ".")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="DreamBooth Stable Diffusion Server")

# Global pipeline reference
pipeline = None


class GenerateRequest(BaseModel):
    prompt: str
    num_images: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 50


class GenerateResponse(BaseModel):
    images: list[str]  # Base64-encoded PNG images


def load_pipeline(model_path):
    """Load the Stable Diffusion pipeline."""
    global pipeline
    from diffusers import StableDiffusionPipeline

    print(f"Loading pipeline from {model_path}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    print("Pipeline loaded and ready.")


@app.on_event("startup")
async def startup():
    model_path = os.environ.get("MODEL_PATH", "./models/dreambooth")
    if os.path.exists(model_path):
        load_pipeline(model_path)
    else:
        print(f"Warning: Model path {model_path} not found. Load manually via /load endpoint.")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/load")
async def load_model(model_path: str):
    """Load or reload the model from a path."""
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Path not found: {model_path}")
    load_pipeline(model_path)
    return {"status": "loaded", "model_path": model_path}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate images from a text prompt."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    images_b64 = []
    for _ in range(request.num_images):
        result = pipeline(
            request.prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
        )
        image = result.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        images_b64.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

    return GenerateResponse(images=images_b64)


def main():
    parser = argparse.ArgumentParser(description="DreamBooth Serving")
    parser.add_argument("--model-path", type=str, default="./models/dreambooth")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = args.model_path
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
