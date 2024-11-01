import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AcrossKVCache:
    """
    Implements AcrossKV cache optimization.
    Combines KV caches across neighboring layers to reduce memory usage.
    """

    def __init__(
            self,
            num_layers: int,
            hidden_size: int,
            num_heads: int,
            combine_layers: int = 2
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.combine_layers = combine_layers

        # Initialize combined cache
        self.combined_cache = {}
        self.layer_groups = self._init_layer_groups()

    def _init_layer_groups(self) -> Dict[int, List[int]]:
        """Initialize layer grouping for cache combination."""
        groups = {}
        for i in range(0, self.num_layers, self.combine_layers):
            group_layers = list(range(i, min(i + self.combine_layers, self.num_layers)))
            for layer in group_layers:
                groups[layer] = group_layers
        return groups

    def update(
            self,
            layer_idx: int,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            position_ids: torch.Tensor
    ) -> None:
        """Update combined cache with new KV states."""
        group = self.layer_groups[layer_idx]
        group_key = (min(group), tuple(position_ids.shape))

        if group_key not in self.combined_cache:
            self.combined_cache[group_key] = {
                'keys': [],
                'values': [],
                'position_ids': position_ids
            }

        cache_entry = self.combined_cache[group_key]
        relative_idx = group.index(layer_idx)

        # Update or append states
        while len(cache_entry['keys']) <= relative_idx:
            cache_entry['keys'].append(None)
            cache_entry['values'].append(None)

        cache_entry['keys'][relative_idx] = key_states.detach()
        cache_entry['values'][relative_idx] = value_states.detach()

    def get_kv_states(
            self,
            layer_idx: int,
            position_ids: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get combined KV states for layer."""
        group = self.layer_groups[layer_idx]
        group_key = (min(group), tuple(position_ids.shape))

        if group_key not in self.combined_cache:
            return None

        cache_entry = self.combined_cache[group_key]
        relative_idx = group.index(layer_idx)

        if (len(cache_entry['keys']) <= relative_idx or
                cache_entry['keys'][relative_idx] is None):
            return None

        # Combine with neighboring layers
        combined_key = self._combine_states(
            cache_entry['keys'],
            relative_idx
        )
        combined_value = self._combine_states(
            cache_entry['values'],
            relative_idx
        )

        return combined_key, combined_value

    def _combine_states(
            self,
            states: List[torch.Tensor],
            center_idx: int
    ) -> torch.Tensor:
        """Combine states from neighboring layers."""
        weights = self._get_combination_weights(len(states), center_idx)
        combined = None

        for i, state in enumerate(states):
            if state is not None:
                weighted = state * weights[i]
                if combined is None:
                    combined = weighted
                else:
                    combined += weighted

        return combined

    def _get_combination_weights(
            self,
            num_states: int,
            center_idx: int
    ) -> torch.Tensor:
        """Get weights for combining states."""
        weights = torch.zeros(num_states)

        # Gaussian weighting centered on target layer
        for i in range(num_states):
            distance = abs(i - center_idx)
            weights[i] = torch.exp(torch.tensor(-distance / 2.0))

        # Normalize weights
        weights = weights / weights.sum()
        return weights.to(next(iter(self.combined_cache.values()))['keys'][0].device)

    def clear(self) -> None:
        """Clear combined cache."""
        self.combined_cache.clear()