#!/usr/bin/env python3
"""
CLI entry point for Phase 1 photo preprocessing.
Ingests user photos, validates count (10-15 recommended), performs face-aware cropping,
normalizes resolutions to SDXL standards, and creates paired trigger word captions.
"""

import argparse
import sys
from pathlib import Path
from src.utils.config import PipelineConfig, PreprocessConfig
from src.preprocessing.dataset_builder import DatasetBuilder
from src.preprocessing.generate_sample_dataset import generate_sample_portraits


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Photo Preprocessing & Dataset Builder for SDXL LoRA Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocess_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing 10-15 raw input photos"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed photos and captions"
    )
    parser.add_argument(
        "--trigger-word",
        type=str,
        default=None,
        help="Unique identifier/trigger word (e.g. 'sks person')"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Target base resolution (default: 1024)"
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate 12 synthetic sample photos in input-dir for immediate pipeline testing"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        try:
            cfg = PipelineConfig.from_yaml(config_path).preprocess
        except Exception:
            # Fallback to direct PreprocessConfig if standalone yaml
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
                cfg = PreprocessConfig(**d)
    else:
        cfg = PreprocessConfig()

    # CLI overrides
    if args.input_dir:
        cfg.input_dir = args.input_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.trigger_word:
        cfg.trigger_word = args.trigger_word
    if args.resolution:
        cfg.resolution = args.resolution

    cfg.validate()

    print("=" * 60)
    print("  AI PHOTO GENERATOR - PHASE 1: PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f" Input Directory:    {cfg.input_dir}")
    print(f" Output Directory:   {cfg.output_dir}")
    print(f" Trigger Word:       {cfg.trigger_word}")
    print(f" Target Resolution:  {cfg.resolution}x{cfg.resolution}")

    # Generate samples if requested or if input directory is empty
    input_path = Path(cfg.input_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    existing_images = [
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    if args.generate_samples or len(existing_images) == 0:
        print(f"\n[INFO] Generating 12 sample photos in '{cfg.input_dir}' for pipeline testing...")
        samples = generate_sample_portraits(output_dir=cfg.input_dir, count=12)
        print(f"[OK] Generated {len(samples)} sample portraits.")

    # Execute dataset building
    builder = DatasetBuilder(
        input_dir=cfg.input_dir,
        output_dir=cfg.output_dir,
        trigger_word=cfg.trigger_word,
        target_resolution=cfg.resolution,
        min_photos=cfg.min_photos,
        max_photos=cfg.max_photos,
        use_face_detection=cfg.detect_face
    )

    print("\n[INFO] Starting image processing and captioning...")
    summary = builder.build()

    print("\n" + "=" * 60)
    print("  PREPROCESSING SUMMARY")
    print("=" * 60)
    validation = summary["validation"]
    print(f" Total Raw Images Found:   {validation['total_images']}")
    print(f" Processed Images Saved:   {summary['processed_count']}")
    print(f" Meets Phase 1 (10-15):    {'YES' if validation['meets_phase1_target'] else 'NO'}")
    print(f" Metadata Manifest:        {summary['metadata_jsonl']}")

    if validation["warnings"]:
        print("\n[WARNINGS / RECOMMENDATIONS]:")
        for w in validation["warnings"]:
            print(f"  * {w}")

    print("\n[OK] Phase 1 Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
