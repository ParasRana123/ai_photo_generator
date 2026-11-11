"""Integration tests for Fast Instant FaceID + LCM-LoRA pipeline and FastAPI routes."""

import pytest
from PIL import Image
from src.training.face_encoder import FaceIdentityEncoder
from src.utils.inference_config import InferenceConfig
from src.inference.generator import SDXLInferenceEngine
from src.inference.metadata_writer import PNGMetadataWriter
from src.api.server import generate_images, GenerateRequest


def test_fast_faceid_config_validation():
    cfg = InferenceConfig.from_yaml("configs/fast_faceid_config.yaml")
    assert cfg.engine_mode == "instant_faceid"
    assert cfg.sampler == "lcm"
    assert cfg.use_lcm is True
    assert cfg.num_inference_steps == 6
    assert cfg.guidance_scale == 1.5

    bad_mode = InferenceConfig(engine_mode="invalid_mode")
    with pytest.raises(ValueError, match="Invalid engine_mode"):
        bad_mode.validate()


def test_instant_faceid_end_to_end_mock(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i in range(3):
        Image.new("RGB", (512, 512), color=(120, 80, 60)).save(raw_dir / f"p_{i}.png")

    face_id_file = tmp_path / "person_face_id.safetensors"
    encoder = FaceIdentityEncoder()
    encoder.enroll_directory(input_dir=raw_dir, output_path=face_id_file, mock=True)

    out_dir = tmp_path / "generated"
    cfg = InferenceConfig(
        engine_mode="instant_faceid",
        base_model="dreamlike-art/dreamlike-diffusion-1.0",
        face_id_path=str(face_id_file),
        ip_adapter_scale=0.75,
        use_lcm=True,
        sampler="lcm",
        width=512,
        height=512,
        num_inference_steps=6,
        guidance_scale=1.5,
        seed=555,
        output_dir=str(out_dir)
    )
    engine = SDXLInferenceEngine(config=cfg, mock=True)
    result = engine.generate(prompt="portrait of a man in a sharp tuxedo")

    assert len(result.saved_paths) == 1
    meta = PNGMetadataWriter.read_metadata(result.saved_paths[0])
    assert meta["engine_mode"] == "instant_faceid"
    assert meta["use_lcm"] == "True"
    assert meta["seed"] == "555"
    assert "sks person" not in meta["prompt"]


def test_fastapi_generate_instant_faceid(tmp_path):
    req = GenerateRequest(
        prompt="man smiling in sunlit park",
        engine_mode="instant_faceid",
        base_model="dreamlike-art/dreamlike-diffusion-1.0",
        use_lcm=True,
        sampler="lcm",
        steps=6,
        guidance_scale=1.5,
        width=512,
        height=512,
        output_dir=str(tmp_path),
        dry_run=True
    )
    resp = generate_images(req)
    assert resp["status"] == "completed"
    assert resp["is_mock"] is True
    assert resp["metadata"]["engine_mode"] == "instant_faceid"

