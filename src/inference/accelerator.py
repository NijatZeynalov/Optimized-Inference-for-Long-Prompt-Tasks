import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel
from ..cache import SingleInputKVCache, AcrossKVCache
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InferenceAccelerator:
    """
    Accelerates inference for long prompts using optimized KV caching.
    """

    def __init__(
            self,
            model: PreTrainedModel,
            use_across_kv: bool = True,
            combine_layers: int = 2
    ):
        self.model = model
        self.use_across_kv = use_across_kv

        # Initialize caches
        self.single_kv_cache = SingleInputKVCache(
            num_layers=model.config.num_hidden_layers,
            hidden_size=model.config.hidden_size,
            num_heads=model.config.num_attention_heads
        )

        if use_across_kv:
            self.across_kv_cache = AcrossKVCache(
                num_layers=model.config.num_hidden_layers,
                hidden_size=model.config.hidden_size,
                num_heads=model.config.num_attention_heads,
                combine_layers=combine_layers
            )

    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            max_length: int = 512,
            **kwargs
    ) -> torch.Tensor:
        """
        Optimized text generation.
        """
        try:
            batch_size = input_ids.shape[0]
            device = input_ids.device

            # Initialize generation
            generated_tokens = input_ids
            past_key_values = None

            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass with caching
                outputs = self._forward_with_cache(
                    generated_tokens,
                    attention_mask,
                    past_key_values
                )

                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                generated_tokens = torch.cat(
                    [generated_tokens, next_token.unsqueeze(-1)],
                    dim=-1
                )

                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=device)
                    ], dim=-1)

                past_key_values = outputs.past_key_values

            return generated_tokens

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

    def _forward_with_cache(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Tuple]
    ) -> Dict:
        """
        Forward pass with cache optimization.
        """
        # Try using cached states first
        if past_key_values is not None:
            cached_outputs = self._forward_with_kv_cache(
                input_ids,
                attention_mask,
                past_key_values
            )
            if cached_outputs is not None:
                return cached_outputs

        # Regular forward pass with cache updates
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True
        )

        # Update caches
        self._update_caches(outputs, input_ids)

        return outputs

    def _forward_with_kv_cache(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            past_key_values: Tuple
    ) -> Optional[Dict]:
        """
        Attempt forward pass using cached KV states.
        """
        shape = input_ids.shape

        # Get cached states
        kv_states = self.single_kv_cache.get_kv_states(
            layer_idx=0,  # Check first layer
            input_shape=shape
        )

        if kv_states is not None:
            if self.use_across_kv:
                kv_states = self._combine_with_across_cache(
                    kv_states,
                    input_ids
                )

            # Use cached states
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=(kv_states,) + past_key_values[1:],
                use_cache=True
            )

            return outputs

        return None

    def _combine_with_across_cache(
            self,
            kv_states: Tuple[torch.Tensor, torch.Tensor],
            input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine SingleKV cache with AcrossKV cache.
        """
        if not self.use_across_kv:
            return kv_states

        across_states = self.across_kv_cache.get_kv_states(
            layer_idx=0,
            position_ids=input_ids
        )

        if across_states is not None:
            # Combine states with weighted average
            key_states = 0.7 * kv_states[0] + 0.3 * across_states[0]
            value_states = 0.7 * kv_states[1] + 0.3 * across_states[1]
            return (key_states, value_states)

        return kv_states

    def _update_caches(
            self,
            outputs: Dict,
            input_ids: torch.Tensor
    ) -> None:
        """
        Update both caches with new states.
        """
        if not hasattr(outputs, "past_key_values"):
            return

        for idx, (key_states, value_states) in enumerate(outputs.past_key_values):
            self.single_kv_cache.update(
                idx,
                key_states,
                value_states,
                input_ids
            )

            if self.use_across_kv:
                self.across_kv_cache.update(
                    idx,
                    key_states,
                    value_states,
                    input_ids
                )

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.single_kv_cache.clear()
        if self.use_across_kv:
            self.across_kv_cache.clear()