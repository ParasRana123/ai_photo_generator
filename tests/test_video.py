"""
Unit and integration tests for the Text-to-Video generation subsystem.
Covers configuration, prompt enhancement, camera motion transforms,
multi-format video encoding, mock inference, and FastAPI endpoints.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.utils.video_config import VideoConfig
from src.video import (
    VideoPromptEnhancer,
    CameraMotionController,
    VideoEncoder,
    CloudZeroGPUVideoClient,
    VideoAPIProvider,
    TextToVideoEngine,
    VideoGenerationResult,
)
from src.api.server import (
    health_check,
    get_video_motions,
    get_video_styles,
    generate_video,
    get_video_gallery,
    VideoGenerateRequest,
)


def test_video_config_defaults_and_validation():
    """Validates VideoConfig defaults and boundary constraint checks."""
    cfg = VideoConfig(prompt="A cinematic drone shot over a misty lake")
    assert cfg.prompt == "A cinematic drone shot over a misty lake"
    assert cfg.fps == 16
    assert cfg.duration_seconds == 3.0
    assert cfg.camera_motion == "static"
    assert cfg.provider == "auto"

    cfg.validate()

    # Empty prompt must raise ValueError
    with pytest.raises(ValueError):
        invalid_cfg = VideoConfig(prompt="")
        invalid_cfg.validate()

    # Invalid FPS must raise ValueError
    with pytest.raises(ValueError):
        invalid_fps_cfg = VideoConfig(prompt="test", fps=999)
        invalid_fps_cfg.validate()


def test_video_prompt_enhancer_and_presets():
    """Tests prompt expansion, style presets, and camera motion cue injection."""
    enhancer = VideoPromptEnhancer()

    # Check available presets
    presets = enhancer.list_presets()
    assert len(presets) >= 5
    preset_names = [p["name"] for p in presets]
    assert "cinematic_movie" in preset_names
    assert "slow_motion" in preset_names
    assert "drone_aerial" in preset_names

    # Test cinematic enhancement
    pos_prompt, neg_prompt = enhancer.build_video_prompts(
        prompt="A runner sprinting on an athletic track",
        style_name="cinematic_movie",
        camera_instruction="slow cinematic push-in dolly forward",
    )
    assert "35mm" in pos_prompt.lower() or "cinematic" in pos_prompt.lower()
    assert "push" in pos_prompt.lower() or "dolly" in pos_prompt.lower()
    assert "flicker" in neg_prompt.lower() or "jitter" in neg_prompt.lower()


def test_camera_motion_transforms():
    """Tests synthetic camera motion transformations and prompt descriptors."""
    controller = CameraMotionController()
    motions = controller.list_motions()
    assert len(motions) >= 6

    # Test compute_frame_transforms for Zoom In
    transforms = controller.compute_frame_transforms(
        motion_type="zoom_in",
        num_frames=16,
        intensity=5.0,
    )
    assert len(transforms) == 16
    assert "zoom" in transforms[0]
    assert "dx" in transforms[0]
    assert "dy" in transforms[0]

    # Test prompt descriptor
    desc = controller.get_prompt_descriptor("zoom_in", 5.0)
    assert "push-in" in desc.lower() or "zoom" in desc.lower()


def test_video_encoder_mp4_gif_thumbnail(tmp_path):
    """Tests MP4 writing, GIF rendering, poster thumbnail generation, and sidecar metadata."""
    # Generate 16 colorful synthetic test frames
    frames = []
    for i in range(16):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[:, :, 0] = int((i / 16) * 255)  # Red gradient over time
        frame[:, :, 1] = 120
        frame[:, :, 2] = 200
        frames.append(frame)

    mp4_path = tmp_path / "unit_test_video.mp4"
    thumb_path = tmp_path / "unit_test_video_thumb.jpg"
    gif_path = tmp_path / "unit_test_video.gif"

    # 1. Save MP4
    saved_mp4 = VideoEncoder.save_mp4(frames, mp4_path, fps=16)
    assert Path(saved_mp4).exists()
    assert Path(saved_mp4).stat().st_size > 0

    # 2. Save Thumbnail
    saved_thumb = VideoEncoder.save_thumbnail(frames, thumb_path)
    assert Path(saved_thumb).exists()
    assert Path(saved_thumb).stat().st_size > 0

    # 3. Save GIF
    saved_gif = VideoEncoder.save_gif(frames, gif_path, fps=16)
    assert Path(saved_gif).exists()
    assert Path(saved_gif).stat().st_size > 0

    # 4. Sidecar metadata
    metadata = {
        "prompt": "Color transition test",
        "camera_motion": "none",
        "duration_seconds": 1.0,
        "fps": 16,
    }
    sidecar_path = VideoEncoder.write_metadata_sidecar(saved_mp4, metadata)
    assert Path(sidecar_path).exists()

    loaded_meta = VideoEncoder.read_metadata_sidecar(saved_mp4)
    assert loaded_meta["prompt"] == "Color transition test"


def test_cloud_zerogpu_video_client_mock():
    """Tests Free Cloud ZeroGPU mock fallback frame generator."""
    frames = CloudZeroGPUVideoClient.generate_mock_video_frames(
        prompt="A tranquil waterfall in a lush tropical jungle",
        width=128,
        height=128,
        num_frames=12,
        camera_motion="pan_right",
    )
    assert len(frames) == 12
    assert isinstance(frames[0], Image.Image)
    assert frames[0].size == (128, 128)


def test_commercial_video_client():
    """Tests commercial API provider interface and parameter validation."""
    provider = VideoAPIProvider()
    assert provider is not None
    # Missing API key must raise informative ValueError
    with pytest.raises(ValueError) as excinfo:
        provider.generate_fal_video(prompt="Test prompt", api_key="")
    assert "fal.ai API key is required" in str(excinfo.value)


def test_text_to_video_engine_mock_generation(tmp_path):
    """Tests end-to-end TextToVideoEngine pipeline generation in mock mode."""
    cfg = VideoConfig(
        prompt="A cybernetic lion roaring on a neon rooftop",
        camera_motion="zoom_in",
        motion_intensity=5.0,
        duration_seconds=1.0,
        fps=12,
        width=128,
        height=128,
        output_dir=str(tmp_path),
        mock=True,
    )

    engine = TextToVideoEngine(config=cfg, mock=True)
    result: VideoGenerationResult = engine.generate(
        prompt="A cybernetic lion roaring on a neon rooftop",
        camera_motion="zoom_in",
        duration_seconds=1.0,
        fps=12,
        seed=42,
    )

    assert Path(result.video_path).exists()
    assert Path(result.thumbnail_path).exists()
    assert Path(result.gif_path).exists()
    assert result.num_frames == 12
    assert result.fps == 12
    assert result.seed == 42
    assert result.generation_time_seconds >= 0.0


def test_fastapi_video_endpoints(tmp_path):
    """Tests FastAPI video endpoints directly for motions, styles, generation, and gallery."""
    # 1. Health check includes video engine info
    health = health_check()
    assert health["video_engine_ready"] is True
    assert "video_motions" in health
    assert "video_styles" in health

    # 2. Get motions
    motions_resp = get_video_motions()
    assert "motions" in motions_resp
    assert len(motions_resp["motions"]) >= 5

    # 3. Get styles
    styles_resp = get_video_styles()
    assert "styles" in styles_resp
    assert len(styles_resp["styles"]) >= 5

    # 4. Generate video via endpoint
    req = VideoGenerateRequest(
        prompt="A fast motorcycle speeding down an empty desert highway",
        camera_motion="pan_left",
        motion_intensity=5.0,
        duration_seconds=1.0,
        fps=12,
        width=256,
        height=256,
        style_preset="cinematic_movie",
        export_gif=True,
        output_dir=str(tmp_path),
        seed=101,
        dry_run=True,
    )
    gen_resp = generate_video(req)
    assert gen_resp["status"] == "completed"
    assert req.prompt in gen_resp["video"]["prompt"]
    assert gen_resp["video"]["camera_motion"] == "pan_left"
    assert gen_resp["video"]["seed"] == 101

    # 5. Gallery endpoint listing
    gallery_resp = get_video_gallery(output_dir=str(tmp_path))
    assert len(gallery_resp["videos"]) >= 1
    assert gallery_resp["videos"][0]["filename"] == gen_resp["video"]["filename"]
