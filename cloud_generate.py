#!/usr/bin/env python3
"""
Free Cloud ZeroGPU Personal AI Photo Generator CLI (fal.ai Free Alternative).
1. Scans uploaded photos in data/raw and ranks the clearest face portraits (~1-5s)
2. Dispatches prompt + top reference photo(s) to free Hugging Face ZeroGPU Spaces
   (PuLID-FLUX -> InstantID -> PuLID-SDXL automatic failover)
3. Saves high-resolution generated photos with embedded PNG metadata
"""

import argparse
from pathlib import Path
from src.training.reference_selector import ReferencePhotoSelector
from src.utils.inference_config import InferenceConfig, VALID_CLOUD_PROVIDERS
from src.inference.generator import SDXLInferenceEngine


def main():
    parser = argparse.ArgumentParser(
        description="Free Cloud ZeroGPU Personal Photo Generator (fal.ai Alternative)"
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Rank and select optimal face photos from --input-dir"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Folder containing your 10-15 photos (default: data/raw)"
    )
    parser.add_argument(
        "--profile-path",
        type=str,
        default="checkpoints/cloud_face_profile.json",
        help="Path to save ranked cloud face profile JSON"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt (e.g. 'portrait of a man in a leather jacket in Tokyo at night')"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=VALID_CLOUD_PROVIDERS,
        help="Cloud ZeroGPU provider: auto, pulid_flux, instantid, or pulid_sdxl"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="photorealistic",
        help="Style preset (photorealistic, cinematic, studio_portrait, cyberpunk, etc.)"
    )
    parser.add_argument(
        "--id-weight",
        type=float,
        default=1.0,
        help="Face identity preservation strength (0.2 to 1.5, default: 1.0)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Inference steps (default: 20 for PuLID-FLUX)"
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=4.0,
        help="Guidance scale (default: 4.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional free Hugging Face token for 4x daily ZeroGPU quota"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run offline mock verification without contacting HuggingFace Spaces"
    )

    args = parser.parse_args()

    if not args.enroll and not args.prompt:
        parser.print_help()
        return

    profile_file = Path(args.profile_path)
    if args.enroll or (args.prompt and not profile_file.exists()):
        print("=" * 60)
        print("  SMART PHOTO RANKING & CLOUD PROFILE BUILDER")
        print("=" * 60)
        selector = ReferencePhotoSelector()
        profile = selector.build_profile(
            input_dir=args.input_dir,
            output_path=profile_file,
            mock=args.dry_run
        )
        print(
            f"[OK] Ranked {profile['total_photos_analyzed']} photos "
            f"({profile['faces_detected']} faces) in {profile['duration_seconds']}s"
        )
        print(f"  Primary Face Reference: {profile['primary_photo']}")

    if args.prompt:
        print("=" * 60)
        print(f"  FREE CLOUD ZEROGPU GENERATION ({args.provider.upper()})")
        print("=" * 60)
        cfg = InferenceConfig(
            engine_mode="cloud_zerogpu",
            cloud_provider=args.provider,
            cloud_profile_path=str(profile_file),
            id_weight=args.id_weight,
            hf_token=args.hf_token,
            style_preset=args.style,
            auto_inject_trigger=False,
            width=896,
            height=1152,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed,
        )
        cfg.validate()

        engine = SDXLInferenceEngine(config=cfg, mock=args.dry_run)
        result = engine.generate(prompt=args.prompt, style_preset=args.style, seed=args.seed)
        print(
            f"[OK] Generated via '{result.metadata.get('cloud_provider')}' "
            f"({result.metadata.get('cloud_space')}) in {result.duration_seconds}s"
        )
        for p in result.saved_paths:
            print(f"  Saved: {p}")


if __name__ == "__main__":
    main()
