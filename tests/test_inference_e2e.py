"""End-to-end integration test for Phase 2: SDXL LoRA Inference Pipeline."""

from pathlib import Path
from src.inference.generator import SDXLInferenceEngine
from src.inference.metadata_writer import PNGMetadataWriter
from src.utils.inference_config import InferenceConfig


def test_full_phase2_inference_pipeline_e2e(tmp_path):
    output_dir = tmp_path / "generated"

    # Step 1 & 2: Configure base model & trained LoRA
    cfg = InferenceConfig(
        output_dir=str(output_dir),
        lora_weights_path="checkpoints/person_sdxl_lora.safetensors",
        lora_scale=0.85,
        trigger_word="sks person",
        sampler="dpm++_2m_karras",
        width=512,
        height=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        seed=777
    )

    # Initialize Engine (Loads model & LoRA)
    engine = SDXLInferenceEngine(config=cfg, mock=True)

    # Step 3: Accept text prompt + style preset
    user_prompt = "sitting by the fireplace with a warm smile"
    style = "cinematic"

    # Step 4: Generate image
    result = engine.generate(
        prompt=user_prompt,
        style_preset=style,
        seed=777
    )

    # Step 5: Save result and verify
    assert len(result.images) == 1
    assert len(result.saved_paths) == 1

    saved_image_path = Path(result.saved_paths[0])
    assert saved_image_path.exists()
    assert saved_image_path.suffix == ".png"

    # Verify PNG embedded metadata
    metadata = PNGMetadataWriter.read_metadata(saved_image_path)
    assert "parameters" in metadata
    assert "sks person" in metadata["prompt"]
    assert "cinematic" in metadata["prompt"].lower()
    assert metadata["seed"] == "777"
    assert metadata["sampler"] == "dpm++_2m_karras"

