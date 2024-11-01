import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SingleInputKVCache:
    """
    Implements SingleInputKV cache optimization for long prompts.
    Reuses KV cache from earlier layers to skip computation in later layers.
    """

    def __init__(
            self,
            num_layers: int,
            hidden_size: int,
            num_heads: int,
            max_length: int = 2048
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_length = max_length

        # Initialize caches
        self.key_cache = {}
        self.value_cache = {}
        self.reuse_map = self._init_reuse_map()

    def _init_reuse_map(self) -> Dict[int, int]:
        """Initialize layer reuse mapping."""
        reuse_map = {}
        # Lower layers compute new KV, higher layers reuse
        reuse_threshold = self.num_layers // 2
        for layer_idx in range(self.num_layers):
            if layer_idx >= reuse_threshold:
                reuse_map[layer_idx] = layer_idx - reuse_threshold
        return reuse_map

    def update(
            self,
            layer_idx: int,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            input_ids: torch.Tensor
    ) -> None:
        """Update cache with new KV states."""
        if layer_idx not in self.reuse_map:
            cache_key = (layer_idx, tuple(input_ids.shape))
            self.key_cache[cache_key] = key_states.detach()
            self.value_cache[cache_key] = value_states.detach()

    def get_kv_states(
            self,
            layer_idx: int,
            input_shape: Tuple[int, ...]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached or reused KV states."""
        cache_key = (layer_idx, input_shape)

        # Check if layer should reuse cache
        if layer_idx in self.reuse_map:
            reuse_layer = self.reuse_map[layer_idx]
            reuse_key = (reuse_layer, input_shape)

            if reuse_key in self.key_cache:
                return (
                    self.key_cache[reuse_key],
                    self.value_cache[reuse_key]
                )

        # Return cached states if available
        if cache_key in self.key_cache:
            return (
                self.key_cache[cache_key],
                self.value_cache[cache_key]
            )

        return None

    def clear(self) -> None:
        """Clear all caches."""
        self.key_cache.clear()
        self.value_cache.clear()


class LayerOutputCache:
    """Caches layer outputs for reuse."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}

    def update(
            self,
            layer_idx: int,
            output: torch.Tensor,
            input_ids: torch.Tensor
    ) -> None:
        """Update cache with layer output."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        cache_key = (layer_idx, tuple(input_ids.shape))
        self.cache[cache_key] = output.detach()

    def get(
            self,
            layer_idx: int,
            input_shape: Tuple[int, ...]
    ) -> Optional[torch.Tensor]:
        """Get cached layer output."""
        cache_key = (layer_idx, input_shape)
        return self.cache.get(cache_key)

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()