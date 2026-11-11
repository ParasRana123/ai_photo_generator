"""
FastAPI server exposing Phase 1 Training and Phase 2 Inference endpoints.
Bridges Express backend and React frontend with Python AI engines.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.utils.device import get_device_info
from src.utils.inference_config import InferenceConfig
from src.preprocessing.dataset_builder import DatasetBuilder
from src.inference.generator import SDXLInferenceEngine
from src.inference.style_presets import list_available_styles
from src.inference.metadata_writer import PNGMetadataWriter
from .state import TrainingJobManager


app = FastAPI(
    title="AI Photo Generator - Python AI Service",
    version="0.3.0",
    description="REST API wrapping SDXL LoRA training (Phase 1) and inference (Phase 2)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

job_manager = TrainingJobManager()
MOCK_MODE_DEFAULT = os.getenv("AI_MOCK_MODE", "true").lower() == "true"


class TrainRequest(BaseModel):
    input_dir: str = "data/raw"
    output_dir: str = "data/processed"
    trigger_word: str = "sks person"
    steps: int = 10
    rank: int = 16
    dry_run: Optional[bool] = None


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    style_preset: Optional[str] = "photorealistic"
    trigger_word: str = "sks person"
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_weights_path: str = "checkpoints/person_sdxl_lora.safetensors"
    lora_scale: float = 0.85
    steps: int = 30
    guidance_scale: float = 7.0
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    num_images: int = 1
    sampler: str = "dpm++_2m_karras"
    output_dir: str = "outputs/generated"
    dry_run: Optional[bool] = None


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Returns Python AI service health and compute device info."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    cached_models = []
    if hf_cache.exists():
        for d in hf_cache.iterdir():
            if d.is_dir() and d.name.startswith("models--"):
                cached_models.append(d.name.replace("models--", "").replace("--", "/"))

    return {
        "status": "ok",
        "service": "python-ai-engine",
        "device": get_device_info(),
        "mock_mode_default": MOCK_MODE_DEFAULT,
        "cached_models": cached_models
    }


@app.post("/preprocess")
def run_preprocessing(input_dir: str = "data/raw", output_dir: str = "data/processed", trigger_word: str = "sks person") -> Dict[str, Any]:
    """Preprocesses raw photos into cropped/resized dataset with trigger captions."""
    builder = DatasetBuilder(
        input_dir=input_dir,
        output_dir=output_dir,
        trigger_word=trigger_word
    )
    summary = builder.build()
    return {
        "status": "completed",
        "processed_count": summary["processed_count"],
        "validation": summary["validation"]
    }


@app.post("/train")
def start_training(req: TrainRequest) -> Dict[str, Any]:
    """Triggers asynchronous SDXL LoRA training."""
    if job_manager.is_running():
        raise HTTPException(status_code=409, detail="Training job is already running.")

    use_dry_run = req.dry_run if req.dry_run is not None else MOCK_MODE_DEFAULT
    status = job_manager.start_training(
        input_dir=req.input_dir,
        output_dir=req.output_dir,
        trigger_word=req.trigger_word,
        steps=req.steps,
        rank=req.rank,
        dry_run=use_dry_run
    )
    return status


@app.get("/train/status")
def get_training_status() -> Dict[str, Any]:
    """Returns live progress, loss, and completion status for active training job."""
    return job_manager.get_status()


@app.post("/generate")
def generate_images(req: GenerateRequest) -> Dict[str, Any]:
    """Generates images from a prompt using trained LoRA weights."""
    use_dry_run = req.dry_run if req.dry_run is not None else MOCK_MODE_DEFAULT

    try:
        cfg = InferenceConfig(
            base_model=req.base_model,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or InferenceConfig().negative_prompt,
            style_preset=req.style_preset,
            trigger_word=req.trigger_word,
            lora_weights_path=req.lora_weights_path,
            lora_scale=req.lora_scale,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            seed=req.seed,
            sampler=req.sampler,
            output_dir=req.output_dir
        )
        cfg.validate()

        engine = SDXLInferenceEngine(config=cfg, mock=use_dry_run)
        result = engine.generate(
            prompt=req.prompt,
            style_preset=req.style_preset,
            num_images=req.num_images,
            seed=req.seed
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    items = []
    for path_str in result.saved_paths:
        p = Path(path_str)
        items.append({
            "filename": p.name,
            "url": f"/outputs/generated/{p.name}",
            "prompt": result.prompt,
            "seed": result.seed,
            "style_preset": req.style_preset,
            "duration_seconds": result.duration_seconds,
            "is_mock": use_dry_run
        })

    return {
        "status": "completed",
        "is_mock": use_dry_run,
        "images": items,
        "metadata": result.metadata
    }


@app.get("/styles")
def get_styles() -> Dict[str, Any]:
    """Lists available photographic style presets."""
    return {"styles": list_available_styles()}


@app.get("/gallery")
def get_gallery(output_dir: str = "outputs/generated") -> Dict[str, Any]:
    """Lists generated images with embedded PNG metadata."""
    out_path = Path(output_dir)
    if not out_path.exists():
        return {"images": []}

    pngs = sorted(out_path.glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
    gallery_items = []
    for f in pngs:
        meta = {}
        try:
            meta = PNGMetadataWriter.read_metadata(f)
        except Exception:
            pass
        gallery_items.append({
            "filename": f.name,
            "url": f"/outputs/generated/{f.name}",
            "prompt": meta.get("prompt", ""),
            "style_preset": meta.get("style_preset", ""),
            "seed": meta.get("seed", ""),
            "created_at": meta.get("timestamp", f.stat().st_mtime)
        })

    return {"images": gallery_items}

