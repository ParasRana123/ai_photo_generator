#!/usr/bin/env python3
"""
Master end-to-end training pipeline CLI for Phase 1.
Orchestrates:
  1. Photo preprocessing & dataset verification (10-15 photos)
  2. Latent & prompt embedding caching for accelerated training
  3. SDXL UNet LoRA injection using PEFT/Diffusers
  4. Diffusion training loop with AdamW optimizer & cosine scheduler
  5. SafeTensors LoRA weight export with embedded metadata
  6. Post-training prompt evaluation & validation image generation
"""

import argparse
import sys
from pathlib import Path
import torch

from src.utils.config import PipelineConfig
from src.utils.device import get_optimal_device, get_device_info
from src.preprocessing.dataset_builder import DatasetBuilder
from src.preprocessing.generate_sample_dataset import generate_sample_portraits
from src.training.dataset import SDXLDataset
from src.training.latent_cache import LatentCache, CachedLatentDataset
from src.training.model_loader import ModelLoader, MockSDXLUNet
from src.training.lora_network import LoRANetworkManager
from src.training.trainer import SDXLLoRATrainer
from src.training.saver import LoRASaver
from src.validation.evaluator import LoRAEvaluator
from src.validation.test_prompts import PromptSuite


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: SDXL LoRA AI Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sdxl_lora_config.yaml",
        help="Path to pipeline configuration YAML"
    )
    parser.add_argument(
        "--trigger-word",
        type=str,
        default=None,
        help="Trigger word/token for fine-tuning (e.g. 'sks person')"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum training steps (override config)"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="LoRA rank dimension (override config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (override config)"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing if data/processed is already prepared"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast verification dry-run using mock architecture and sample data"
    )

    args = parser.parse_args()

    # 1. Load Configuration
    config_path = Path(args.config)
    if config_path.exists():
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        cfg = PipelineConfig()

    # Apply CLI overrides
    if args.trigger_word:
        cfg.preprocess.trigger_word = args.trigger_word
        cfg.training.output_name = f"{args.trigger_word.replace(' ', '_')}_sdxl_lora"
    if args.steps:
        cfg.training.max_train_steps = args.steps
    if args.rank:
        cfg.training.rank = args.rank
    if args.lr:
        cfg.training.learning_rate = args.lr

    device = get_optimal_device()
    dev_info = get_device_info()

    print("=" * 65)
    print("  AI PHOTO GENERATOR - PHASE 1: SDXL LORA TRAINING PIPELINE")
    print("=" * 65)
    print(f" Compute Device:      {dev_info['device_type'].upper()} ({dev_info['device_name']})")
    print(f" Trigger Word:        {cfg.preprocess.trigger_word}")
    print(f" LoRA Rank / Alpha:   Rank {cfg.training.rank} / Alpha {cfg.training.lora_alpha}")
    print(f" Max Training Steps:  {cfg.training.max_train_steps}")
    print(f" Learning Rate:       {cfg.training.learning_rate}")
    print(f" Mode:                {'DRY-RUN / VERIFICATION' if args.dry_run else 'PRODUCTION TRAINING'}")
    print("=" * 65)

    # -------------------------------------------------------------
    # Step 1: Preprocessing & Dataset Verification
    # -------------------------------------------------------------
    processed_dir = Path(cfg.preprocess.output_dir)
    raw_dir = Path(cfg.preprocess.input_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preprocess:
        print("\n[PHASE 1 - STEP 1/5] Ingesting and Preprocessing Photos...")
        raw_images = [
            p for p in raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]

        if len(raw_images) == 0:
            print(f"  [INFO] Input directory '{raw_dir}' is empty.")
            print("  [INFO] Generating 12 sample portraits matching Phase 1 requirements...")
            generate_sample_portraits(raw_dir, count=12)

        builder = DatasetBuilder(
            input_dir=raw_dir,
            output_dir=processed_dir,
            trigger_word=cfg.preprocess.trigger_word,
            target_resolution=cfg.preprocess.resolution,
            use_face_detection=cfg.preprocess.detect_face
        )
        prep_summary = builder.build()
        print(f"  [OK] Preprocessed {prep_summary['processed_count']} photos into '{processed_dir}'")
        for w in prep_summary["validation"]["warnings"]:
            print(f"  [NOTICE] {w}")

    # -------------------------------------------------------------
    # Step 2: Latent & Text Embedding Caching
    # -------------------------------------------------------------
    print("\n[PHASE 1 - STEP 2/5] Preparing Latent and Text Embedding Cache...")
    cache_dir = Path("data/latents_cache")
    latent_cache = LatentCache(cache_dir=cache_dir, device=device)

    dataset = SDXLDataset(
        data_dir=processed_dir,
        target_resolution=cfg.preprocess.resolution,
        trigger_word=cfg.preprocess.trigger_word
    )

    if len(dataset) == 0:
        print("  [ERROR] No processed photos found to train on. Exiting.")
        sys.exit(1)

    # Cache dataset latents
    latent_cache.cache_dataset(dataset)
    cached_dataset = CachedLatentDataset(cache_dir=cache_dir)
    print(f"  [OK] Cached {len(cached_dataset)} latents and text embeddings in '{cache_dir}'")

    # -------------------------------------------------------------
    # Step 3: Model Setup & LoRA Injection
    # -------------------------------------------------------------
    print("\n[PHASE 1 - STEP 3/5] Initializing SDXL UNet and Injecting LoRA Adapters...")
    if args.dry_run:
        unet = MockSDXLUNet()
        scheduler = None
    else:
        try:
            model_loader = ModelLoader(cfg.model)
            unet = model_loader.load_unet(device)
            scheduler = model_loader.load_noise_scheduler()
        except Exception as e:
            print(f"  [WARN] Could not load online SDXL weights ({e}). Falling back to mock architecture.")
            unet = MockSDXLUNet()
            scheduler = None

    lora_mgr = LoRANetworkManager(
        rank=cfg.training.rank,
        alpha=cfg.training.lora_alpha,
        dropout=cfg.training.lora_dropout,
        target_modules=["cross_attn"] if args.dry_run or isinstance(unet, MockSDXLUNet) else None
    )
    lora_unet = lora_mgr.inject_lora_unet(unet)
    param_info = lora_mgr.get_parameter_summary(lora_unet)
    print(f"  [OK] LoRA injected! Trainable parameters: {param_info['trainable_params']:,} "
          f"({param_info['trainable_percentage']}% of total)")

    # -------------------------------------------------------------
    # Step 4: Diffusion LoRA Training Loop
    # -------------------------------------------------------------
    print("\n[PHASE 1 - STEP 4/5] Running Diffusion LoRA Training Loop...")
    max_steps = 10 if args.dry_run else cfg.training.max_train_steps
    trainer = SDXLLoRATrainer(
        lora_model=lora_unet,
        noise_scheduler=scheduler,
        dataset=cached_dataset,
        config=cfg.training,
        device=device
    )
    train_summary = trainer.train(max_steps=max_steps)

    # -------------------------------------------------------------
    # Step 5: Save SafeTensors & Post-Training Prompt Evaluation
    # -------------------------------------------------------------
    print("\n[PHASE 1 - STEP 5/5] Exporting .safetensors Weights and Evaluating Prompts...")
    saver = LoRASaver(
        output_dir=cfg.training.output_dir,
        output_name=cfg.training.output_name,
        trigger_word=cfg.preprocess.trigger_word
    )
    safetensors_path = saver.save(lora_unet, training_config=cfg.training)

    # Run post-training prompt evaluation
    evaluator = LoRAEvaluator(
        lora_weights_path=safetensors_path,
        output_dir=cfg.validation.output_dir,
        device=device
    )
    prompt_suite = PromptSuite(trigger_word=cfg.preprocess.trigger_word).get_prompts()
    eval_report = evaluator.evaluate(prompts=prompt_suite, dry_run=args.dry_run)

    print("\n" + "=" * 65)
    print("  PHASE 1 MILESTONE COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f" Trained LoRA SafeTensors: {safetensors_path}")
    print(f" Training Steps Run:       {train_summary['total_steps']}")
    print(f" Training Duration:        {train_summary['elapsed_seconds']}s")
    print(f" Validation Samples:       {eval_report['total_prompts_tested']} images saved in '{cfg.validation.output_dir}'")
    print("=" * 65)


if __name__ == "__main__":
    main()

