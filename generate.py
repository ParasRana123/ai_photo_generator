#!/usr/bin/env python3
"""
CLI entry point for Phase 2: SDXL LoRA Inference & Photo Generation.
Implements the core AI inference loop:
  1. Load SDXL base model
  2. Load trained LoRA (.safetensors)
  3. Accept text prompt (CLI argument or interactive prompt shell)
  4. Generate high-resolution image
  5. Save result with embedded PNG metadata
"""

import argparse
import sys
from pathlib import Path

from src.utils.inference_config import InferenceConfig, VALID_SAMPLERS
from src.inference.generator import SDXLInferenceEngine
from src.inference.style_presets import list_available_styles


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: SDXL LoRA Personal Photo Generator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt describing the desired photo (e.g. 'sks person drinking coffee in Paris')"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt tags to suppress unwanted artifacts"
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        default=None,
        help="Path to trained .safetensors LoRA file (default: checkpoints/person_sdxl_lora.safetensors)"
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=None,
        help="LoRA weight strength multiplier (0.0 to 1.5, default: 0.85)"
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Stylistic preset (e.g. 'photorealistic', 'cinematic', 'studio_portrait', 'cyberpunk')"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Denoising inference steps (default: 30)"
    )
    parser.add_argument(
        "--cfg", "--guidance-scale",
        dest="guidance_scale",
        type=float,
        default=None,
        help="Classifier-Free Guidance scale (default: 7.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random generation seed for reproducibility"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=VALID_SAMPLERS,
        help=f"Sampling scheduler algorithm (default: dpm++_2m_karras)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated photos (default: outputs/generated)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yaml",
        help="Path to inference YAML configuration"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal prompt shell"
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="Print list of available photographic style presets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast dry-run using mock pipeline for offline verification"
    )

    return parser.parse_args()


def display_styles():
    print("=" * 65)
    print("  AVAILABLE PHOTOGRAPHIC STYLE PRESETS")
    print("=" * 65)
    for s in list_available_styles():
        print(f"  * {s['name']:<18} : {s['display_name']} - {s['description']}")
    print("=" * 65)


def run_interactive_shell(engine: SDXLInferenceEngine):
    print("\n" + "=" * 65)
    print("  INTERACTIVE SDXL LORA PHOTO STUDIO")
    print("  Type your prompt and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 65)

    while True:
        try:
            prompt_input = input("\nEnter prompt > ").strip()
            if prompt_input.lower() in ("exit", "quit", "q"):
                print("Exiting interactive photo studio. Happy generating!")
                break
            if not prompt_input:
                continue

            print("[INFO] Synthesizing image...")
            result = engine.generate(prompt=prompt_input)
            print(f"[OK] Generated in {result.duration_seconds}s!")
            for p in result.saved_paths:
                print(f"  Saved to: {p}")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break


def main():
    args = parse_args()

    if args.list_styles:
        display_styles()
        sys.exit(0)

    # 1. Load Configuration
    config_path = Path(args.config)
    if config_path.exists():
        cfg = InferenceConfig.from_yaml(config_path)
    else:
        cfg = InferenceConfig()

    # Apply CLI overrides
    if args.prompt:
        cfg.prompt = args.prompt
    if args.negative_prompt:
        cfg.negative_prompt = args.negative_prompt
    if args.lora_weights:
        cfg.lora_weights_path = args.lora_weights
    if args.lora_scale is not None:
        cfg.lora_scale = args.lora_scale
    if args.style:
        cfg.style_preset = args.style
    if args.steps:
        cfg.num_inference_steps = args.steps
    if args.guidance_scale:
        cfg.guidance_scale = args.guidance_scale
    if args.seed is not None:
        cfg.seed = args.seed
    if args.sampler:
        cfg.sampler = args.sampler
    if args.output_dir:
        cfg.output_dir = args.output_dir

    cfg.validate()

    print("=" * 65)
    print("  AI PHOTO GENERATOR - PHASE 2: INFERENCE PIPELINE")
    print("=" * 65)
    print(f" Base Model:          {cfg.base_model}")
    print(f" LoRA Weights:        {cfg.lora_weights_path} (Scale: {cfg.lora_scale})")
    print(f" Sampler / Steps:     {cfg.sampler} / {cfg.num_inference_steps} steps")
    print(f" Guidance Scale (CFG):{cfg.guidance_scale}")
    print(f" Style Preset:        {cfg.style_preset or 'None'}")
    print(f" Output Directory:    {cfg.output_dir}")
    print(f" Mode:                {'DRY-RUN / VERIFICATION' if args.dry_run else 'PRODUCTION DIFFUSION'}")
    print("=" * 65)

    # 2. Initialize Engine
    engine = SDXLInferenceEngine(config=cfg, mock=args.dry_run)

    # 3. Interactive Mode or Single/Batch Run
    if args.interactive:
        run_interactive_shell(engine)
    else:
        print(f"\n[PHASE 2] Synthesizing image(s) for prompt:\n  \"{cfg.prompt}\"\n")
        result = engine.generate(
            num_images=args.num_images,
            seed=cfg.seed
        )

        print("=" * 65)
        print("  GENERATION COMPLETE")
        print("=" * 65)
        print(f" Final Enhanced Prompt: {result.prompt}")
        print(f" Seed:                  {result.seed}")
        print(f" Time Taken:            {result.duration_seconds}s")
        print(f" Saved Image(s):")
        for p in result.saved_paths:
            print(f"  * {p}")
        print("=" * 65)


if __name__ == "__main__":
    main()

