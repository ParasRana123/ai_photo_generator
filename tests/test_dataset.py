"""Unit tests for SDXL dataset loading and tensor formatting."""

import pytest
import torch
from PIL import Image
from src.training.dataset import SDXLDataset
from src.training.latent_cache import LatentCache, CachedLatentDataset


def test_sdxl_dataset_item_structure(tmp_path):
    # Create 2 test processed images and caption files
    for i in range(2):
        img = Image.new("RGB", (1024, 1024), color=(i * 100, 50, 50))
        img_path = tmp_path / f"image_{i:03d}.png"
        img.save(img_path)
        txt_path = tmp_path / f"image_{i:03d}.txt"
        with open(txt_path, "w") as f:
            f.write(f"a photo of sks person, variant {i}")

    dataset = SDXLDataset(data_dir=tmp_path, trigger_word="sks person")
    assert len(dataset) == 2

    item = dataset[0]
    assert "pixel_values" in item
    assert "input_ids_1" in item
    assert "input_ids_2" in item
    assert "original_size" in item

    # Verify normalization range [-1, 1]
    assert item["pixel_values"].shape == (3, 1024, 1024)
    assert item["pixel_values"].min() >= -1.01
    assert item["pixel_values"].max() <= 1.01


def test_latent_cache_save_and_load(tmp_path):
    cache_dir = tmp_path / "cache"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    img = Image.new("RGB", (1024, 1024), color=(120, 120, 120))
    img.save(data_dir / "img_001.png")

    dataset = SDXLDataset(data_dir=data_dir)
    cache = LatentCache(cache_dir=cache_dir)
    cached_paths = cache.cache_dataset(dataset)
    assert len(cached_paths) == 1

    cached_ds = CachedLatentDataset(cache_dir=cache_dir)
    assert len(cached_ds) == 1
    sample = cached_ds[0]
    assert sample["latents"].shape == (4, 128, 128)
    assert sample["prompt_embeds"].shape == (77, 2048)
    assert sample["pooled_prompt_embeds"].shape == (1280,)

