import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InferenceOptimizer:
    """
    Optimizes model inference for long prompts.
    """

    def __init__(
            self,
            model: PreTrainedModel,
            batch_size: int = 1,
            max_length: int = 2048,
            dtype: torch.dtype = torch.float16
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.dtype = dtype

        self._optimize_model()

    def _optimize_model(self) -> None:
        """Apply inference optimizations to model."""
        self.model.eval()  # Set to evaluation mode

        # Enable memory efficient attention if available
        if hasattr(self.model.config, "use_memory_efficient_attention"):
            self.model.config.use_memory_efficient_attention = True

        # Apply optimizations
        self._optimize_attention()
        self._optimize_memory()
        self._apply_fusion()

    def _optimize_attention(self) -> None:
        """Optimize attention computation."""
        for layer in self.model.layers:
            if hasattr(layer, "self_attn"):
                # Enable flash attention if available
                if hasattr(layer.self_attn, "enable_flash_attention"):
                    layer.self_attn.enable_flash_attention = True

                # Set optimal attention settings
                layer.self_attn._use_sdpa = True
                if hasattr(layer.self_attn, "head_dim"):
                    layer.self_attn.head_dim = min(
                        layer.self_attn.head_dim,
                        128  # Optimal for most hardware
                    )

    def _optimize_memory(self) -> None:
        """Apply memory optimizations."""
        # Convert to optimal dtype
        self.model.to(self.dtype)

        # Enable gradient checkpointing if needed
        if self.max_length > 1024:
            self.model.gradient_checkpointing_enable()

        # Enable memory efficient forward pass
        for module in self.model.modules():
            if hasattr(module, "config"):
                module.config.use_cache = True

    def _apply_fusion(self) -> None:
        """Apply operator fusion optimizations."""
        for module in self.model.modules():
            # Fuse LayerNorm if possible
            if isinstance(module, nn.LayerNorm):
                module.forward = torch.jit.script(module.forward)

            # Fuse attention operations
            if hasattr(module, "self_attn"):
                self._fuse_attention(module.self_attn)

    def _fuse_attention(self, attention_module: nn.Module) -> None:
        """Fuse attention operations for faster inference."""
        if not hasattr(attention_module, "q_proj"):
            return

        # Fuse QKV projections
        qkv_weight = torch.cat([
            attention_module.q_proj.weight,
            attention_module.k_proj.weight,
            attention_module.v_proj.weight
        ], dim=0)

        qkv_bias = torch.cat([
            attention_module.q_proj.bias,
            attention_module.k_proj.bias,
            attention_module.v_proj.bias
        ], dim=0)

        # Create fused projection
        attention_module.qkv_proj = nn.Linear(
            attention_module.q_proj.in_features,
            3 * attention_module.q_proj.out_features,
            bias=True
        )

        attention_module.qkv_proj.weight.data = qkv_weight
        attention_module.qkv_proj.bias.data = qkv_bias

    def optimize_forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Optimized forward pass.
        """
        try:
            # Apply input optimizations
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            # Split long sequences if needed
            if input_ids.shape[1] > self.max_length:
                return self._process_long_sequence(input_ids, attention_mask)

            # Regular optimized forward
            with torch.cuda.amp.autocast(dtype=self.dtype):
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )

            return outputs

        except Exception as e:
            logger.error(f"Error in optimized forward pass: {str(e)}")
            raise

    def _process_long_sequence(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> Dict:
        """
        Process sequences longer than max_length.
        """
        # Split into chunks
        chunks = []
        for i in range(0, input_ids.shape[1], self.max_length):
            chunk_ids = input_ids[:, i:i + self.max_length]
            chunk_mask = attention_mask[:, i:i + self.max_length]

            # Process chunk
            with torch.cuda.amp.autocast(dtype=self.dtype):
                chunk_output = self.model(
                    chunk_ids,
                    attention_mask=chunk_mask,
                    use_cache=True
                )
            chunks.append(chunk_output)

        # Combine chunk outputs
        return self._combine_chunks(chunks)

    def _combine_chunks(self, chunks: List[Dict]) -> Dict:
        """Combine outputs from multiple chunks."""
        combined = {}

        # Combine logits
        combined["logits"] = torch.cat(
            [chunk["logits"] for chunk in chunks],
            dim=1
        )

        # Combine hidden states if present
        if "hidden_states" in chunks[0]:
            combined["hidden_states"] = tuple(
                torch.cat([chunk["hidden_states"][i] for chunk in chunks], dim=1)
                for i in range(len(chunks[0]["hidden_states"]))
            )

        return combined