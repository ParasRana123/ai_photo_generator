"""Unit and integration tests for Free Cloud ZeroGPU engine and reference selector."""

from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image

from src.utils.inference_config import InferenceConfig
from src.training.reference_selector import ReferencePhotoSelector
from src.inference.cloud_client import CloudZeroGPUClient
from src.inference.generator import SDXLInferenceEngine
from src.inference.metadata_writer import PNGMetadataWriter
from src.api.server import generate_images, GenerateRequest


def test_cloud_zerogpu_config_validation():
    cfg = InferenceConfig.from_yaml("configs/cloud_zerogpu_config.yaml")
    assert cfg.engine_mode == "cloud_zerogpu"
    assert cfg.cloud_provider == "auto"
    assert cfg.id_weight == 1.0

    bad_provider = InferenceConfig(engine_mode="cloud_zerogpu", cloud_provider="unknown_cloud")
    with pytest.raises(ValueError, match="Invalid cloud_provider"):
        bad_provider.validate()


def test_reference_photo_selector_profile_ranking(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i in range(5):
        Image.new("RGB", (600, 600), color=(i * 40, 100, 120)).save(raw_dir / f"user_{i}.jpg")

    profile_path = tmp_path / "cloud_face_profile.json"
    selector = ReferencePhotoSelector(use_face_detection=False)
    progress = []
    profile = selector.build_profile(
        input_dir=raw_dir,
        output_path=profile_path,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert profile_path.exists()
    assert profile["total_photos_analyzed"] == 5
    assert len(profile["top_photos"]) == 4
    assert len(progress) == 5

    loaded = ReferencePhotoSelector.load_profile(profile_path)
    assert loaded is not None
    assert loaded["primary_photo"] == profile["primary_photo"]


def test_cloud_client_auto_failover(tmp_path):
    ref_img = tmp_path / "face.jpg"
    Image.new("RGB", (512, 512), color=(200, 150, 120)).save(ref_img)

    client = CloudZeroGPUClient()
    fake_img = Image.new("RGB", (768, 1024), color=(50, 80, 120))

    with patch.object(client, "_run_pulid_flux", side_effect=RuntimeError("ZeroGPU quota busy")):
        with patch.object(
            client,
            "_run_instantid",
            return_value=(fake_img, {"cloud_space": "InstantX/InstantID", "cloud_model": "SDXL + InstantID"})
        ):
            img, meta = client.generate(
                reference_photos=[str(ref_img)],
                prompt="portrait in a suit",
                provider="auto",
                seed=99,
            )

    assert img.size == (768, 1024)
    assert meta["provider_used"] == "instantid"
    assert meta["cloud_space"] == "InstantX/InstantID"


def test_cloud_zerogpu_engine_e2e(tmp_path):
    ref_img = tmp_path / "ref.jpg"
    Image.new("RGB", (512, 512), color=(180, 140, 110)).save(ref_img)
    out_dir = tmp_path / "out"

    cfg = InferenceConfig(
        engine_mode="cloud_zerogpu",
        cloud_provider="pulid_flux",
        reference_image_path=str(ref_img),
        width=896,
        height=1152,
        num_inference_steps=20,
        guidance_scale=4.0,
        id_weight=1.0,
        seed=101,
        output_dir=str(out_dir),
    )

    engine = SDXLInferenceEngine(config=cfg, mock=True)
    result = engine.generate(prompt="young man in a sharp tuxedo")

    assert len(result.saved_paths) == 1
    meta = PNGMetadataWriter.read_metadata(result.saved_paths[0])
    assert meta["engine_mode"] == "cloud_zerogpu"
    assert meta["cloud_provider"] == "pulid_flux"
    assert meta["seed"] == "101"


def test_fastapi_generate_cloud_zerogpu(tmp_path):
    req = GenerateRequest(
        prompt="man in Tokyo neon lights",
        engine_mode="cloud_zerogpu",
        cloud_provider="auto",
        output_dir=str(tmp_path),
        dry_run=True,
    )
    resp = generate_images(req)
    assert resp["status"] == "completed"
    assert resp["is_mock"] is True
    assert resp["images"][0]["cloud_space"] == "mock/pulid_flux"
