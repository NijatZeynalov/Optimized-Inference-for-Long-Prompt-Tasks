import torch
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
from .logger import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """
    Track and compute optimization metrics.
    """

    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_stats = {}

    def update(
            self,
            metrics: Dict[str, float],
            prefix: Optional[str] = None
    ) -> None:
        """
        Update metrics.
        """
        for name, value in metrics.items():
            key = f"{prefix}/{name}" if prefix else name
            self.metrics[key].append(value)
            self.current_stats[key] = value

    def compute_inference_metrics(
            self,
            batch_size: int,
            sequence_length: int,
            start_time: float,
            end_time: float,
            memory_start: int,
            memory_end: int
    ) -> Dict[str, float]:
        """
        Compute inference performance metrics.
        """
        duration = end_time - start_time
        memory_used = memory_end - memory_start

        metrics = {
            "tokens_per_second": (batch_size * sequence_length) / duration,
            "latency": duration * 1000,  # ms
            "memory_mb": memory_used / (1024 * 1024),
            "memory_per_token": memory_used / (batch_size * sequence_length)
        }

        self.update(metrics, "inference")
        return metrics

    def compute_cache_metrics(
            self,
            hits: int,
            misses: int,
            cache_size: int
    ) -> Dict[str, float]:
        """
        Compute cache performance metrics.
        """
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0

        metrics = {
            "hit_rate": hit_rate,
            "miss_rate": 1 - hit_rate,
            "cache_size_mb": cache_size / (1024 * 1024)
        }

        self.update(metrics, "cache")
        return metrics

    def get_average_metrics(
            self,
            window_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get averaged metrics.
        """
        averages = {}

        for key, values in self.metrics.items():
            if window_size:
                values = values[-window_size:]
            averages[key] = float(np.mean(values))

        return averages

    def reset(self) -> None:
        """Reset metrics."""
        self.metrics.clear()
        self.current_stats.clear()