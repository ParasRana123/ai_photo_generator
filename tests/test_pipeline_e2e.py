"""End-to-end integration test for Phase 1 AI training pipeline."""

import pytest
from pathlib import Path
from src.preprocessing.generate_sample_dataset import generate_sample_portraits
from src.preprocessing.dataset_builder import DatasetBuilder
from src.training.dataset import SDXLDataset
from src.training.latent_cache import LatentCache, CachedLatentDataset
from src.training.model_loader import MockSDXLUNet
from src.training.lora_network import LoRANetworkManager
from src.training.trainer import SDXLLoRATrainer
from src.training.saver import LoRASaver
from src.validation.evaluator import LoRAEvaluator
from src.utils.config import TrainingConfig


def test_full_phase1_pipeline_integration(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    cache_dir = tmp_path / "cache"
    checkpoints_dir = tmp_path / "checkpoints"
    validation_dir = tmp_path / "validation"

    # Step 1: Generate sample portraits and preprocess
    samples = generate_sample_portraits(raw_dir, count=12)
    assert len(samples) == 12

    builder = DatasetBuilder(
        input_dir=raw_dir,
        output_dir=processed_dir,
        trigger_word="sks person",
        use_face_detection=False
    )
    summary = builder.build()
    assert summary["processed_count"] == 12

    # Step 2: Latent Caching
    dataset = SDXLDataset(data_dir=processed_dir)
    cache = LatentCache(cache_dir=cache_dir)
    cache.cache_dataset(dataset)
    cached_ds = CachedLatentDataset(cache_dir=cache_dir)
    assert len(cached_ds) == 12

    # Step 3: LoRA Model Injection
    unet = MockSDXLUNet()
    mgr = LoRANetworkManager(rank=8, alpha=16, target_modules=["cross_attn"])
    lora_unet = mgr.inject_lora_unet(unet)

    # Step 4: Trainer Loop
    cfg = TrainingConfig(max_train_steps=3, gradient_accumulation_steps=1)
    trainer = SDXLLoRATrainer(
        lora_model=lora_unet,
        noise_scheduler=None,
        dataset=cached_ds,
        config=cfg
    )
    res = trainer.train()
    assert res["total_steps"] == 3

    # Step 5: Save SafeTensors & Evaluate Prompts
    saver = LoRASaver(
        output_dir=checkpoints_dir,
        output_name="phase1_test_lora",
        trigger_word="sks person"
    )
    safetensors_path = saver.save(lora_unet, training_config=cfg)
    assert safetensors_path.exists()

    evaluator = LoRAEvaluator(
        lora_weights_path=safetensors_path,
        output_dir=validation_dir
    )
    eval_report = evaluator.evaluate(dry_run=True)
    assert eval_report["total_prompts_tested"] == 5
    assert len(list(validation_dir.glob("*.png"))) == 5

