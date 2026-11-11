"""Integration tests for the Phase 3 Python FastAPI AI microservice."""

from pathlib import Path
from src.api.server import (
    health_check,
    get_styles,
    get_training_status,
    generate_images,
    get_gallery,
    GenerateRequest,
)


def test_python_api_health_and_styles():
    health = health_check()
    assert health["status"] == "ok"
    assert health["service"] == "python-ai-engine"

    styles_resp = get_styles()
    assert "styles" in styles_resp
    assert len(styles_resp["styles"]) >= 7


def test_python_api_training_status_initial():
    status = get_training_status()
    assert "status" in status
    assert "progress_percent" in status


def test_python_api_generate_and_gallery(tmp_path):
    req = GenerateRequest(
        prompt="sks person reading a book in a sunlit garden",
        style_preset="photorealistic",
        steps=15,
        seed=321,
        output_dir=str(tmp_path),
        dry_run=True,
    )
    gen_resp = generate_images(req)
    assert gen_resp["status"] == "completed"
    assert len(gen_resp["images"]) == 1
    assert gen_resp["images"][0]["seed"] == 321

    gallery_resp = get_gallery(output_dir=str(tmp_path))
    assert len(gallery_resp["images"]) == 1
    assert gallery_resp["images"][0]["filename"] == gen_resp["images"][0]["filename"]

