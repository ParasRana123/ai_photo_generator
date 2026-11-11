"""
Training logger and progress reporter.
Tracks loss values, computes moving averages, estimates remaining time,
and logs training diagnostics.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import time
import json


class TrainingLogger:
    """Tracks training metrics and outputs human-readable progress updates."""

    def __init__(self, log_file: Optional[Union[str, Path]] = None, window_size: int = 20):
        self.log_file = Path(log_file) if log_file else None
        self.window_size = window_size
        self.loss_history: List[float] = []
        self.step_times: List[float] = []
        self.start_time = time.time()
        self.last_step_time = time.time()

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("")  # Clear or initialize

    def log_step(self, step: int, total_steps: int, loss: float, lr: float) -> str:
        """Records a step, calculates moving average loss, and returns formatted status."""
        current_time = time.time()
        step_duration = current_time - self.last_step_time
        self.last_step_time = current_time

        self.loss_history.append(loss)
        self.step_times.append(step_duration)

        recent_losses = self.loss_history[-self.window_size:]
        avg_loss = sum(recent_losses) / len(recent_losses)

        recent_durations = self.step_times[-self.window_size:]
        avg_step_duration = sum(recent_durations) / len(recent_durations)

        remaining_steps = total_steps - step
        eta_seconds = remaining_steps * avg_step_duration
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        msg = (
            f"[Step {step:04d}/{total_steps:04d}] "
            f"Loss: {loss:.4f} (Avg: {avg_loss:.4f}) | "
            f"LR: {lr:.2e} | "
            f"Speed: {avg_step_duration:.2f}s/step | "
            f"ETA: {eta_str}"
        )

        if self.log_file:
            entry = {
                "step": step,
                "loss": round(loss, 6),
                "avg_loss": round(avg_loss, 6),
                "learning_rate": lr,
                "timestamp": current_time
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        return msg

    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregate training summary metrics."""
        elapsed = time.time() - self.start_time
        total_steps = len(self.loss_history)
        avg_loss = (sum(self.loss_history) / total_steps) if total_steps > 0 else 0.0

        return {
            "total_steps": total_steps,
            "elapsed_seconds": round(elapsed, 2),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
            "average_loss": round(avg_loss, 6),
            "min_loss": min(self.loss_history) if self.loss_history else 0.0
        }

