#!/usr/bin/env python3
"""
Standalone Text-to-Video Generation CLI.
Generates MP4 video clips, animated GIF previews, and poster thumbnails from text prompts.
Supports camera motion trajectories, temporal styling, and multi-provider cloud routing.
"""

import argparse
import sys
import time
from pathlib import Path

from src.utils.video_config import (
    VideoConfig,
    VALID_VIDEO_PROVIDERS,
    VALID_CAMERA_MOTIONS,
    VALID_VIDEO_FORMATS
)
from src.video.engine import TextToVideoEngine
from src.video.prompt_enhancer import VIDEO_PRESETS


def main():
    parser = argparse.ArgumentParser(
        description="PersonaAI Text-to-Video Synthesis CLI"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the desired video scene and motion"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="zoom_in",
        choices=VALID_CAMERA_MOTIONS,
        help="Camera motion trajectory (default: zoom_in)"
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=5.0,
        help="Motion intensity from 1.0 (subtle) to 10.0 (dynamic, default: 5.0)"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="cinematic_movie",
        choices=list(VIDEO_PRESETS.keys()),
        help="Video cinematic style preset"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=VALID_VIDEO_PROVIDERS,
        help="Video rendering provider (default: auto)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Video duration in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frame rate (default: 16)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Frame width (default: 512)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Frame height (default: 512)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic random seed"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key for fal.ai or Google Gemini/Veo"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/videos",
        help="Destination directory for generated video outputs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run offline mock generation without contacting remote cloud spaces"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("  PERSONAAI - TEXT-TO-VIDEO GENERATOR")
    print("=" * 65)
    print(f"  Prompt      : {args.prompt}")
    print(f"  Camera      : {args.camera.upper()} (intensity {args.intensity})")
    print(f"  Style       : {args.style}")
    print(f"  Provider    : {args.provider}")
    print(f"  Duration    : {args.duration}s @ {args.fps} FPS ({int(args.duration * args.fps)} frames)")
    print(f"  Resolution  : {args.width}x{args.height}")
    print(f"  Mock Mode   : {args.dry_run}")
    print("=" * 65)

    cfg = VideoConfig(
        prompt=args.prompt,
        camera_motion=args.camera,
        motion_intensity=args.intensity,
        provider=args.provider,
        duration_seconds=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        api_key=args.api_key,
        output_dir=args.out_dir,
    )
    cfg.validate()

    engine = TextToVideoEngine(config=cfg, mock=args.dry_run)
    result = engine.generate(
        prompt=args.prompt,
        camera_motion=args.camera,
        motion_intensity=args.intensity,
        duration_seconds=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        provider=args.provider,
        seed=args.seed,
        style_preset=args.style,
    )

    print("\n[OK] Video generated successfully!")
    print(f"  • Video File : {result.video_path}")
    print(f"  • Thumbnail  : {result.thumbnail_path}")
    if result.gif_path:
        print(f"  • GIF Preview: {result.gif_path}")
    print(f"  • Generation Time: {result.generation_time_seconds}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
