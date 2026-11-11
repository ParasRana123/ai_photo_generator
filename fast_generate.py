#!/usr/bin/env python3
"""
Fast Zero-Shot Personal AI Photo Generator CLI (Instant FaceID + LCM-LoRA).
Enrolls a person's face from 10-15 photos in seconds via InsightFace ArcFace,
then synthesizes stylized portraits in 4-8 LCM steps using IP-Adapter-FaceID.
"""

import argparse
from pathlib import Path
from src.training.face_encoder import FaceIdentityEncoder
from src.utils.inference_config import InferenceConfig
from src.inference.generator import SDXLInferenceEngine


def main():
    parser = argparse.ArgumentParser(
        description="Fast Open-Source FaceID + LCM Photo Generator (~2s Enrollment / 6-Step Inference)"
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Run fast InsightFace enrollment on photos in --input-dir"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Folder containing 10-15 photos of the person (default: data/raw)"
    )
    parser.add_argument(
        "--face-id-path",
        type=str,
        default="checkpoints/person_face_id.safetensors",
        help="Output/input path for the .safetensors FaceID embedding"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt to generate image (e.g. 'person in a sharp tuxedo in Paris')"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="dreamlike-art/dreamlike-diffusion-1.0",
        help="Base SD 1.5 Diffusers model ID"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="photorealistic",
        help="Style preset (photorealistic, cinematic, studio_portrait, cyberpunk, etc.)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="LCM denoising steps (default: 6)"
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=1.5,
        help="Classifier-Free Guidance scale (1.0 - 2.0 recommended for LCM)"
    )
    parser.add_argument(
        "--ip-scale",
        type=float,
        default=0.7,
        help="IP-Adapter FaceID identity strength (0.0 - 1.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast dry-run verification without downloading/running diffusion weights"
    )

    args = parser.parse_args()

    if not args.enroll and not args.prompt:
        parser.print_help()
        return

    face_id_file = Path(args.face_id_path)

    # Step 1: Instant Face Enrollment (~seconds)
    if args.enroll or (args.prompt and not face_id_file.exists()):
        print("=" * 60)
        print("  FAST FACE ENROLLMENT (InsightFace ArcFace)")
        print("=" * 60)
        encoder = FaceIdentityEncoder()
        summary = encoder.enroll_directory(
            input_dir=args.input_dir,
            output_path=face_id_file,
            mock=args.dry_run
        )
        print(
            f"[OK] Enrolled {summary['faces_detected']}/{summary['images_scanned']} faces "
            f"in {summary['duration_seconds']}s -> {summary['safetensors_path']}"
        )

    # Step 2: Fast LCM + IP-Adapter-FaceID Generation
    if args.prompt:
        print("=" * 60)
        print("  FAST LCM + IP-ADAPTER-FACEID GENERATION")
        print("=" * 60)
        cfg = InferenceConfig(
            engine_mode="instant_faceid",
            base_model=args.base_model,
            face_id_path=str(face_id_file),
            ip_adapter_scale=args.ip_scale,
            use_lcm=True,
            sampler="lcm",
            style_preset=args.style,
            auto_inject_trigger=False,
            width=512,
            height=512,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            seed=args.seed,
            mixed_precision="fp32"
        )
        cfg.validate()

        engine = SDXLInferenceEngine(config=cfg, mock=args.dry_run)
        result = engine.generate(prompt=args.prompt, style_preset=args.style, seed=args.seed)
        print(f"[OK] Generated in {result.duration_seconds}s (Seed: {result.seed})")
        for path in result.saved_paths:
            print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

