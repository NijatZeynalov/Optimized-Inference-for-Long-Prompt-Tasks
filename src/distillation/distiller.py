import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel
from ..utils.logger import get_logger
from ..utils.metrics import MetricsTracker

logger = get_logger(__name__)


class KVCacheDistiller:
    """
    Implements distillation for KV cache optimization.
    """

    def __init__(
            self,
            teacher_model: PreTrainedModel,
            student_model: PreTrainedModel,
            temperature: float = 2.0,
            alpha: float = 0.5
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.metrics = MetricsTracker()

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

    def train_step(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Single distillation training step.
        """
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Get student outputs
        student_outputs = self.student(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Calculate losses
        distillation_loss = self._compute_distillation_loss(
            teacher_outputs.logits,
            student_outputs.logits
        )

        kv_loss = self._compute_kv_loss(
            teacher_outputs.hidden_states,
            student_outputs.hidden_states
        )

        # Combined loss
        total_loss = (
                self.alpha * distillation_loss +
                (1 - self.alpha) * kv_loss
        )

        return {
            "total_loss": total_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "kv_loss": kv_loss.item()
        }

    def _compute_distillation_loss(
            self,
            teacher_logits: torch.Tensor,
            student_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        """
        # Apply temperature scaling
        teacher_probs = (teacher_logits / self.temperature).softmax(dim=-1)
        student_log_probs = (student_logits / self.temperature).log_softmax(dim=-1)

        return self.kl_loss(student_log_probs, teacher_probs)

    def _compute_kv_loss(
            self,
            teacher_states: Tuple[torch.Tensor, ...],
            student_states: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Compute loss for KV cache alignment.
        """
        total_loss = 0.0

        # Compare hidden states at key layers
        for t_state, s_state in zip(teacher_states, student_states):
            state_loss = self.mse_loss(t_state, s_state)
            total_loss += state_loss

        return total_loss / len(teacher_states)

    def save_student(self, path: str) -> None:
        """Save distilled student model."""
        self.student.save_pretrained(path)
        logger.info(f"Saved distilled model to {path}")

    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics."""
        return self.metrics.get_metrics()