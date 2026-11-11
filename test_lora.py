#!/usr/bin/env python3
"""
CLI entry point for Phase 1 LoRA prompt testing and validation.
Loads trained .safetensors LoRA onto base SDXL, executes the test prompt suite,
and saves the resulting photos with metadata.
"""

import argparse
from pathlib import Path
from src.validation.evaluator import LoRAEvaluator
from src.validation.test_prompts import PromptSuite, TestPrompt


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Test Trained SDXL LoRA with Multiple Prompts"
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        default="checkpoints/person_sdxl_lora.safetensors",
        help="Path to trained .safetensors LoRA weight file"
    )
    parser.add_argument(
        "--trigger-word",
        type=str,
        default="sks person",
        help="Trigger word/token for the trained subject"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validation",
        help="Directory to save generated test images"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom test prompt to evaluate (if omitted, runs full benchmark suite)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in offline dry-run mode for pipeline verification"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  AI PHOTO GENERATOR - PHASE 1: LORA PROMPT TEST SUITE")
    print("=" * 60)
    print(f" LoRA Weights:       {args.lora_weights}")
    print(f" Trigger Word:       {args.trigger_word}")
    print(f" Output Directory:   {args.output_dir}")
    print(f" Execution Mode:     {'Dry-Run / Verification' if args.dry_run else 'Full Diffusion Inference'}")

    evaluator = LoRAEvaluator(
        lora_weights_path=args.lora_weights,
        output_dir=args.output_dir
    )

    prompts = None
    if args.prompt:
        prompts = [
            TestPrompt(
                name="custom_user_prompt",
                category="User Custom",
                prompt=args.prompt,
                negative_prompt=PromptSuite.DEFAULT_NEGATIVE,
                seed=42
            )
        ]

    report = evaluator.evaluate(prompts=prompts, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print(f" Total Samples Generated: {report['total_prompts_tested']}")
    print(f" Report Manifest:         {Path(args.output_dir) / 'evaluation_report.json'}")
    print("\n[OK] Phase 1 LoRA prompt testing completed successfully.")


if __name__ == "__main__":
    main()
