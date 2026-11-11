"""
Training job state tracker and background runner for Python AI API.
Executes Phase 1 preprocessing, latent caching, and LoRA training asynchronously
while exposing live step progress, loss metrics, and completion status.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import time

from src.utils.config import PipelineConfig
from src.utils.device import get_optimal_device
from src.preprocessing.dataset_builder import DatasetBuilder
from src.preprocessing.generate_sample_dataset import generate_sample_portraits
from src.training.dataset import SDXLDataset
from src.training.latent_cache import LatentCache, CachedLatentDataset
from src.training.model_loader import ModelLoader, MockSDXLUNet
from src.training.lora_network import LoRANetworkManager
from src.training.trainer import SDXLLoRATrainer
from src.training.saver import LoRASaver
from src.utils.logger import TrainingLogger


@dataclass
class TrainingJobState:
    status: str = "idle"  # idle, preprocessing, caching, training, saving, completed, failed
    progress_percent: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    current_loss: float = 0.0
    message: str = "Ready to train"
    lora_path: Optional[str] = None
    trigger_word: str = "sks person"
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiveProgressLogger(TrainingLogger):
    """Subclass of TrainingLogger that updates shared TrainingJobState on every step."""

    def __init__(self, state_ref: TrainingJobState, lock: threading.Lock):
        super().__init__()
        self.state_ref = state_ref
        self.lock = lock

    def log_step(self, step: int, total_steps: int, loss: float, lr: float) -> str:
        msg = super().log_step(step, total_steps, loss, lr)
        with self.lock:
            self.state_ref.current_step = step
            self.state_ref.total_steps = total_steps
            self.state_ref.current_loss = round(loss, 4)
            # Map training loop to 25% -> 95% progress range
            ratio = (step / total_steps) if total_steps > 0 else 1.0
            self.state_ref.progress_percent = round(25.0 + (ratio * 70.0), 1)
            self.state_ref.message = f"Training LoRA step {step}/{total_steps} (loss: {loss:.4f})"
        return msg


class TrainingJobManager:
    """Thread-safe manager coordinating background SDXL LoRA training."""

    def __init__(self):
        self.state = TrainingJobState()
        self.lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return self.state.to_dict()

    def is_running(self) -> bool:
        with self.lock:
            return self.state.status in ("preprocessing", "caching", "training", "saving")

    def start_training(
        self,
        input_dir: str = "data/raw",
        output_dir: str = "data/processed",
        trigger_word: str = "sks person",
        steps: int = 10,
        rank: int = 16,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        if self.is_running():
            raise RuntimeError("A training job is already in progress.")

        with self.lock:
            self.state = TrainingJobState(
                status="preprocessing",
                progress_percent=5.0,
                current_step=0,
                total_steps=steps,
                trigger_word=trigger_word,
                message="Starting photo preprocessing...",
                started_at=time.time()
            )

        self._thread = threading.Thread(
            target=self._run_worker,
            args=(input_dir, output_dir, trigger_word, steps, rank, dry_run),
            daemon=True
        )
        self._thread.start()
        return self.get_status()

    def _run_worker(
        self,
        input_dir: str,
        output_dir: str,
        trigger_word: str,
        steps: int,
        rank: int,
        dry_run: bool
    ) -> None:
        try:
            device = get_optimal_device()
            cfg = PipelineConfig()
            cfg.preprocess.input_dir = input_dir
            cfg.preprocess.output_dir = output_dir
            cfg.preprocess.trigger_word = trigger_word
            cfg.training.max_train_steps = steps
            cfg.training.rank = rank

            raw_path = Path(input_dir)
            raw_path.mkdir(parents=True, exist_ok=True)
            raw_images = [
                p for p in raw_path.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
            if len(raw_images) == 0:
                generate_sample_portraits(raw_path, count=12)

            # Stage 1: Preprocessing
            builder = DatasetBuilder(
                input_dir=raw_path,
                output_dir=output_dir,
                trigger_word=trigger_word,
                target_resolution=cfg.preprocess.resolution,
                use_face_detection=False if dry_run else True
            )
            builder.build()

            # Stage 2: Latent Caching
            with self.lock:
                self.state.status = "caching"
                self.state.progress_percent = 15.0
                self.state.message = "Caching VAE latents & text embeddings..."

            cache_dir = Path("data/latents_cache")
            if dry_run:
                dataset = SDXLDataset(data_dir=output_dir, trigger_word=trigger_word)
                cache = LatentCache(cache_dir=cache_dir, device=device)
                cache.cache_dataset(dataset)
                cached_ds = CachedLatentDataset(cache_dir=cache_dir)

                # Stage 3: LoRA Setup & Training
                with self.lock:
                    self.state.status = "training"
                    self.state.progress_percent = 25.0
                    self.state.message = "Injecting LoRA rank adapters (Dry-Run)..."

                unet = MockSDXLUNet()
                scheduler = None
            else:
                if device.type == "cpu":
                    cfg.model.variant = None
                loader = ModelLoader(cfg.model)
                tok1, tok2 = loader.load_tokenizers()
                dataset = SDXLDataset(
                    data_dir=output_dir,
                    tokenizer_1=tok1,
                    tokenizer_2=tok2,
                    trigger_word=trigger_word
                )
                vae = loader.load_vae(device)
                te1, te2 = loader.load_text_encoders(device)
                cache = LatentCache(cache_dir=cache_dir, device=device, dtype=loader.dtype)
                cache.cache_dataset(dataset, vae=vae, text_encoder_1=te1, text_encoder_2=te2)
                cached_ds = CachedLatentDataset(cache_dir=cache_dir)
                del vae, te1, te2

                # Stage 3: LoRA Setup & Training
                with self.lock:
                    self.state.status = "training"
                    self.state.progress_percent = 25.0
                    self.state.message = "Loading SDXL UNet and injecting LoRA..."

                unet = loader.load_unet(device)
                scheduler = loader.load_noise_scheduler()

            mgr = LoRANetworkManager(
                rank=rank,
                alpha=rank * 2,
                target_modules=["cross_attn"] if isinstance(unet, MockSDXLUNet) else None
            )
            lora_unet = mgr.inject_lora_unet(unet)

            live_logger = LiveProgressLogger(self.state, self.lock)
            cfg.training.gradient_accumulation_steps = 1
            trainer = SDXLLoRATrainer(
                lora_model=lora_unet,
                noise_scheduler=scheduler,
                dataset=cached_ds,
                config=cfg.training,
                device=device,
                logger=live_logger
            )
            trainer.train(max_steps=steps)

            # Stage 4: Export .safetensors
            with self.lock:
                self.state.status = "saving"
                self.state.progress_percent = 95.0
                self.state.message = "Exporting trained LoRA .safetensors..."

            saver = LoRASaver(
                output_dir=cfg.training.output_dir,
                output_name="person_sdxl_lora",
                trigger_word=trigger_word
            )
            saved_path = saver.save(lora_unet, training_config=cfg.training)

            with self.lock:
                self.state.status = "completed"
                self.state.progress_percent = 100.0
                self.state.lora_path = str(saved_path)
                self.state.message = "Training completed! Model ready for inference."
                self.state.completed_at = time.time()

        except Exception as exc:
            with self.lock:
                self.state.status = "failed"
                self.state.error = str(exc)
                self.state.message = f"Training failed: {exc}"
                self.state.completed_at = time.time()

