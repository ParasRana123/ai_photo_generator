"""Unit tests for inference generator and PNG metadata persistence."""

from PIL import Image
from src.inference.generator import SDXLInferenceEngine
from src.inference.metadata_writer import PNGMetadataWriter
from src.utils.inference_config import InferenceConfig


def test_png_metadata_roundtrip(tmp_path):
    img = Image.new("RGB", (256, 256), color=(50, 100, 150))
    metadata = {
        "prompt": "a portrait of sks person",
        "negative_prompt": "blurry",
        "seed": 999,
        "sampler": "dpm++_2m_karras",
        "guidance_scale": 7.5,
        "num_inference_steps": 35,
        "lora_scale": 0.8
    }

    out_file = tmp_path / "metadata_test.png"
    PNGMetadataWriter.save_image_with_metadata(img, out_file, metadata)

    assert out_file.exists()
    extracted = PNGMetadataWriter.read_metadata(out_file)
    assert extracted["seed"] == "999"
    assert extracted["sampler"] == "dpm++_2m_karras"
    assert "parameters" in extracted


def test_engine_single_and_batch_generation(tmp_path):
    cfg = InferenceConfig(
        output_dir=str(tmp_path),
        prompt="sks person at a cafe",
        seed=123
    )
    engine = SDXLInferenceEngine(config=cfg, mock=True)

    # 1. Single generation
    res = engine.generate()
    assert len(res.images) == 1
    assert len(res.saved_paths) == 1
    assert res.seed == 123

    # 2. Batch generation
    batch_prompts = [
        "sks person in a library",
        "sks person on a mountain"
    ]
    batch_res = engine.generate_batch(batch_prompts, base_seed=500)
    assert len(batch_res) == 2
    assert batch_res[0].seed == 500
    assert batch_res[1].seed == 501

