"""Unit tests for SDXL pipeline loader and mock pipeline."""

import torch
from src.inference.pipeline_loader import SDXLPipelineLoader, MockSDXLInferencePipeline
from src.utils.inference_config import InferenceConfig


def test_mock_pipeline_generation():
    pipe = MockSDXLInferencePipeline()
    pipe.load_lora_weights("checkpoints/fake_lora.safetensors")
    pipe.set_adapters(["default"], adapter_weights=[0.75])

    out = pipe(
        prompt="portrait of sks person",
        width=512,
        height=512,
        num_inference_steps=5,
        num_images_per_prompt=2
    )

    assert len(out.images) == 2
    assert out.images[0].size == (512, 512)
    assert pipe.lora_scale == 0.75


def test_pipeline_loader_initialization():
    cfg = InferenceConfig(sampler="euler_a", mixed_precision="fp32")
    loader = SDXLPipelineLoader(cfg)
    assert loader.dtype == torch.float32

    # Load mock pipeline
    pipe = loader.load_pipeline(mock=True)
    assert isinstance(pipe, MockSDXLInferencePipeline)
    assert pipe.scheduler_name == "euler_a"

