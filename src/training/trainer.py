"""
Core SDXL LoRA training engine.
Orchestrates the forward diffusion process, noise addition, UNet cross-attention prediction,
loss computation, gradient accumulation, and learning rate scheduling.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler

from src.utils.config import TrainingConfig
from src.utils.logger import TrainingLogger
from src.utils.device import get_optimal_device, clear_memory


class SDXLLoRATrainer:
    """Trains LoRA adapter weights on SDXL UNet representations."""

    def __init__(
        self,
        lora_model: nn.Module,
        noise_scheduler: Optional[Any],
        dataset: Any,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        logger: Optional[TrainingLogger] = None
    ):
        self.config = config or TrainingConfig()
        self.device = device or get_optimal_device()
        self.model = lora_model.to(self.device)
        self.noise_scheduler = noise_scheduler
        self.dataset = dataset
        self.logger = logger or TrainingLogger()

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Configures the AdamW optimizer with trainable LoRA parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon
        )
        return optimizer

    def _build_lr_scheduler(self) -> Any:
        """Constructs the learning rate scheduler with warm-up."""
        scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=self.config.max_train_steps
        )
        return scheduler

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Executes a single forward-backward optimization step."""
        latents = batch["latents"].to(self.device)
        prompt_embeds = batch["prompt_embeds"].to(self.device)
        pooled_embeds = batch["pooled_prompt_embeds"].to(self.device)

        # 1. Sample Gaussian noise matching latent dimensions
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # 2. Sample random timesteps uniformly
        num_timesteps = 1000
        if self.noise_scheduler is not None and hasattr(self.noise_scheduler, "config"):
            num_timesteps = getattr(self.noise_scheduler.config, "num_train_timesteps", 1000)

        timesteps = torch.randint(
            0, num_timesteps, (bsz,), device=self.device, dtype=torch.long
        )

        # 3. Add noise to latents according to the diffusion schedule
        if self.noise_scheduler is not None and hasattr(self.noise_scheduler, "add_noise"):
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        else:
            # Fallback for mock scheduler
            noisy_latents = latents + 0.1 * noise

        # 4. Prepare SDXL time and size conditioning vectors
        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": torch.cat([
                batch["original_size"].to(self.device),
                batch["crop_coords_top_left"].to(self.device),
                batch["target_size"].to(self.device)
            ], dim=-1)
        }

        # 5. Predict the noise residual with UNet + LoRA
        model_pred = self.model(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=added_cond_kwargs
        )

        pred_sample = model_pred.sample if hasattr(model_pred, "sample") else model_pred[0]

        # Target prediction: epsilon (noise)
        target = noise
        loss = F.mse_loss(pred_sample.float(), target.float(), reduction="mean")

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        return float(loss.item())

    def train(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Runs the complete LoRA training loop."""
        total_steps = max_steps or self.config.max_train_steps
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True
        )

        self.model.train()
        self.optimizer.zero_grad()

        step = 0
        running_loss = 0.0

        print(f"[INFO] Beginning LoRA training on {self.device.type.upper()} for {total_steps} steps...")

        while step < total_steps:
            for batch in loader:
                loss_val = self.train_step(batch)
                running_loss += loss_val

                # Gradient accumulation boundary
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                step += 1
                current_lr = self.lr_scheduler.get_last_lr()[0]
                status_msg = self.logger.log_step(step, total_steps, loss_val, current_lr)

                if step % 20 == 0 or step == total_steps:
                    print(f"  {status_msg}")

                if step >= total_steps:
                    break

        clear_memory()
        summary = self.logger.get_summary()
        print(f"\n[OK] Training completed in {summary['elapsed_seconds']}s with avg loss: {summary['average_loss']}")
        return summary
