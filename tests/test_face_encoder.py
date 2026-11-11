"""Unit tests for sub-second InsightFace Identity Encoder and SafeTensors persistence."""

import pytest
import torch
from PIL import Image
from src.training.face_encoder import FaceIdentityEncoder


def test_face_encoder_enroll_and_load(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i in range(4):
        Image.new("RGB", (512, 512), color=(i * 50, 100, 150)).save(raw_dir / f"img_{i}.png")

    out_path = tmp_path / "face_id.safetensors"
    encoder = FaceIdentityEncoder()
    progress_calls = []

    summary = encoder.enroll_directory(
        input_dir=raw_dir,
        output_path=out_path,
        progress_callback=lambda done, total, det: progress_calls.append((done, total, det)),
        mock=True
    )

    assert out_path.exists()
    assert summary["faces_detected"] == 4
    assert summary["images_scanned"] == 4
    assert summary["embedding_dim"] == 512
    assert len(progress_calls) == 4

    emb = FaceIdentityEncoder.load_face_embedding(out_path)
    assert emb.shape == (1, 1, 512)
    # Verify unit normalization
    norm = torch.norm(emb).item()
    assert abs(norm - 1.0) < 1e-4


def test_prepare_ip_adapter_embeds_cfg_shapes():
    face_emb = torch.randn(1, 1, 512)

    # With CFG > 1.0 -> negative + positive concatenated
    cfg_embeds = FaceIdentityEncoder.prepare_ip_adapter_embeds(
        face_emb, do_classifier_free_guidance=True
    )
    assert len(cfg_embeds) == 1
    assert cfg_embeds[0].shape == (2, 1, 512)
    assert torch.allclose(cfg_embeds[0][0], torch.zeros(1, 512))

    # Without CFG (<= 1.0) -> positive only
    no_cfg_embeds = FaceIdentityEncoder.prepare_ip_adapter_embeds(
        face_emb, do_classifier_free_guidance=False
    )
    assert no_cfg_embeds[0].shape == (1, 1, 512)


def test_face_encoder_empty_dir_raises(tmp_path):
    encoder = FaceIdentityEncoder()
    with pytest.raises(ValueError, match="No photos found"):
        encoder.enroll_directory(input_dir=tmp_path, output_path=tmp_path / "out.safetensors", mock=True)

