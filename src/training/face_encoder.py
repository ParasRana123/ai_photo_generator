"""
Sub-second Zero-Shot Face Identity Encoder using InsightFace (ArcFace / buffalo_l).
Extracts and averages 512-dimensional facial embeddings across 10-15 reference photos
in seconds on CPU/GPU and serializes them into .safetensors format for IP-Adapter-FaceID.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import time
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from safetensors.torch import save_file, load_file


class FaceIdentityEncoder:
    """Encodes multiple reference photos into a single averaged ArcFace identity tensor."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    _shared_app = None

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple = (320, 320),
        use_gpu: Optional[bool] = None
    ):
        self.model_name = model_name
        self.det_size = det_size
        self.use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu

    def _get_app(self):
        """Lazily initializes and caches InsightFace FaceAnalysis model."""
        if FaceIdentityEncoder._shared_app is not None:
            return FaceIdentityEncoder._shared_app

        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if self.use_gpu else -1

        app = FaceAnalysis(
            name=self.model_name,
            allowed_modules=["detection", "recognition"],
            providers=providers
        )
        app.prepare(ctx_id=ctx_id, det_size=self.det_size)
        FaceIdentityEncoder._shared_app = app
        return app

    @staticmethod
    def _load_bgr_image(image_path: Union[str, Path], max_dim: int = 640) -> np.ndarray:
        """
        Loads an image via PIL (handling EXIF orientation & Unicode paths),
        downscales oversized photos for ultra-fast detection, and converts to BGR.
        """
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
        rgb = np.array(img)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def extract_embedding(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Extracts the 512-dim L2-normalized ArcFace embedding for the largest face in an image.
        Returns None if no face is detected.
        """
        app = self._get_app()
        bgr = self._load_bgr_image(image_path)
        faces = app.get(bgr)
        if not faces:
            return None

        # Pick largest face by bounding box area
        primary_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        emb = np.asarray(primary_face.normed_embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return (emb / norm) if norm > 0 else emb

    def enroll_directory(
        self,
        input_dir: Union[str, Path] = "data/raw",
        output_path: Union[str, Path] = "checkpoints/person_face_id.safetensors",
        trigger_word: str = "sks person",
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        mock: bool = False
    ) -> Dict[str, Any]:
        """
        Scans a folder of user photos, extracts facial embeddings, computes a unified
        L2-normalized identity vector, and saves it as a .safetensors checkpoint.
        """
        start_time = time.time()
        in_dir = Path(input_dir)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not in_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {in_dir}")

        image_files = sorted([
            p for p in in_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ])

        if not image_files:
            raise ValueError(f"No photos found in '{in_dir}'. Upload 10-15 photos first.")

        # Prefer user-uploaded photos over synthetic placeholders if both exist
        non_sample = [p for p in image_files if not p.name.startswith("sample_photo_")]
        scan_files = non_sample if non_sample else image_files

        embeddings: List[np.ndarray] = []

        if mock:
            rng = np.random.default_rng(42)
            for idx, _ in enumerate(scan_files):
                vec = rng.normal(size=(512,)).astype(np.float32)
                vec /= np.linalg.norm(vec)
                embeddings.append(vec)
                if progress_callback:
                    progress_callback(idx + 1, len(scan_files), len(embeddings))
        else:
            # Warm up model before timing per-photo progress
            self._get_app()
            for idx, img_path in enumerate(scan_files):
                emb = self.extract_embedding(img_path)
                if emb is not None:
                    embeddings.append(emb)
                if progress_callback:
                    progress_callback(idx + 1, len(scan_files), len(embeddings))

        if not embeddings:
            raise RuntimeError(
                f"No human faces could be detected across {len(scan_files)} photo(s) in '{in_dir}'. "
                "Please upload clear front/side photos with visible faces."
            )

        stacked = np.stack(embeddings, axis=0)  # (N, 512)
        mean_emb = np.mean(stacked, axis=0)
        mean_norm = np.linalg.norm(mean_emb)
        if mean_norm > 0:
            mean_emb = mean_emb / mean_norm

        # Shape expected by Diffusers IP-Adapter-FaceID: (1, 1, 512)
        faceid_tensor = torch.from_numpy(mean_emb).float().unsqueeze(0).unsqueeze(0)
        individual_tensor = torch.from_numpy(stacked).float()

        duration = round(time.time() - start_time, 3)

        metadata = {
            "format": "pt",
            "encoder": f"insightface_{self.model_name}",
            "faces_detected": str(len(embeddings)),
            "images_scanned": str(len(scan_files)),
            "embedding_dim": "512",
            "trigger_word": trigger_word,
            "duration_seconds": str(duration),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "is_mock": str(mock).lower()
        }

        save_file(
            {
                "faceid_embeds": faceid_tensor,
                "individual_embeds": individual_tensor
            },
            str(out_file),
            metadata=metadata
        )

        return {
            "safetensors_path": str(out_file),
            "faces_detected": len(embeddings),
            "images_scanned": len(scan_files),
            "embedding_dim": 512,
            "duration_seconds": duration,
            "is_mock": mock
        }

    @staticmethod
    def load_face_embedding(path: Union[str, Path]) -> torch.Tensor:
        """Loads saved faceid_embeds tensor of shape (1, 1, 512) from .safetensors."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"FaceID embedding file not found: {path}")
        tensors = load_file(str(p))
        if "faceid_embeds" not in tensors:
            raise KeyError(f"'faceid_embeds' tensor missing in {path}")
        emb = tensors["faceid_embeds"].float()
        if emb.ndim == 1:
            emb = emb.unsqueeze(0).unsqueeze(0)
        elif emb.ndim == 2:
            emb = emb.unsqueeze(0)
        return emb

    @staticmethod
    def prepare_ip_adapter_embeds(
        face_embed: torch.Tensor,
        do_classifier_free_guidance: bool = True
    ) -> List[torch.Tensor]:
        """
        Formats the (1, 1, 512) face embedding for Diffusers StableDiffusionPipeline
        ip_adapter_image_embeds argument.
        """
        if do_classifier_free_guidance:
            neg_embed = torch.zeros_like(face_embed)
            return [torch.cat([neg_embed, face_embed], dim=0)]
        return [face_embed]
