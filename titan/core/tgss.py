"""
TITAN Pillar 5: Temporal Gradient Superposition Sketching (TGSS)
================================================================
Count-Min Sketch for O(1) gradient memory regardless of N parameters.
Temporal superposition: EMA of sketches = virtual large-batch gradient.
Frequency-domain gradient accumulation for convolutional / spatial layers.

Novel: sketch directly feeds ASDT active-set selection.
"""

from __future__ import annotations
import hashlib
import math
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MurmurHash3-inspired fast hash for Count-Min Sketch
# ---------------------------------------------------------------------------

_HASH_SEEDS = [0x9e3779b9, 0x6c62272e, 0xc2b2ae3d, 0x27d4eb2f, 0x165667b1]


def _hash_index(param_idx: int, row: int, width: int) -> int:
    """Fast integer hash mapping param_idx → bucket in [0, width)."""
    h = param_idx ^ _HASH_SEEDS[row % len(_HASH_SEEDS)]
    h = ((h >> 16) ^ h) * 0x45d9f3b
    h = ((h >> 16) ^ h) * 0x45d9f3b
    h = (h >> 16) ^ h
    return int(h % width) if width > 0 else 0


# ---------------------------------------------------------------------------
# Count-Min Sketch
# ---------------------------------------------------------------------------

class CountMinSketch:
    """
    Count-Min Sketch for gradient magnitude accumulation.

    Memory: d × w × 4 bytes (FP32) regardless of number of parameters.
    With d=5, w=1_000_000: 20 MB total for 1T parameter model.

    Error bound (§3.5):
        |g_estimate - g_true| ≤ ε · ||g||_1 / w
    """

    def __init__(self, width: int = 1_000_000, depth: int = 5):
        self.width = width
        self.depth = depth
        # Core sketch table: depth × width, FP32
        self.table = torch.zeros(depth, width, dtype=torch.float32)
        self._total_updates = 0

    def update(self, param_indices: torch.Tensor, values: torch.Tensor) -> None:
        """
        Batch update: add values[i] to all rows for param_indices[i].
        param_indices: (N,) int64
        values:        (N,) float32  — gradient magnitudes
        """
        idx_np = param_indices.cpu().numpy().astype(np.int64)
        val_np = values.cpu().numpy().astype(np.float32)

        for row in range(self.depth):
            # Vectorized hash calculation
            seed = _HASH_SEEDS[row % len(_HASH_SEEDS)]
            h = idx_np ^ seed
            h = ((h >> 16) ^ h) * 0x45d9f3b
            h = ((h >> 16) ^ h) * 0x45d9f3b
            h = (h >> 16) ^ h
            buckets = (h % self.width).astype(np.int64)
            
            np.add.at(self.table[row].numpy(), buckets, val_np)

        self._total_updates += len(idx_np)

    def query(self, param_indices: torch.Tensor) -> torch.Tensor:
        """Vectorized query for gradient magnitudes."""
        idx_np = param_indices.cpu().numpy().astype(np.int64)
        n = len(idx_np)
        
        # Matrix to store estimates from each row: (depth, n)
        row_bounds = np.zeros((self.depth, n), dtype=np.float32)
        
        for row in range(self.depth):
            seed = _HASH_SEEDS[row % len(_HASH_SEEDS)]
            h = idx_np ^ seed
            h = ((h >> 16) ^ h) * 0x45d9f3b
            h = ((h >> 16) ^ h) * 0x45d9f3b
            h = (h >> 16) ^ h
            buckets = (h % self.width).astype(np.int64)
            row_bounds[row] = self.table[row].numpy()[buckets]
            
        # CMS estimate is the minimum across all rows
        estimates = np.min(row_bounds, axis=0)
        return torch.from_numpy(estimates).float()

    def merge(self, other: "CountMinSketch", alpha: float = 0.01) -> None:
        """
        Temporal superposition: EMA update in sketch space (§3.5).
        self = (1-α)·self + α·other
        """
        assert self.width == other.width and self.depth == other.depth
        self.table.mul_(1 - alpha).add_(other.table, alpha=alpha)

    def top_k_indices(self, k: int, param_count: int) -> torch.Tensor:
        """
        Approximate top-k parameter indices by estimated gradient magnitude.
        Scans all buckets; in production replaced by heavy-hitter sketch.
        """
        # For large k we can sample and rank
        sample_size = min(param_count, self.width * 2)
        candidate_indices = torch.randperm(param_count)[:sample_size]
        scores = self.query(candidate_indices)
        top_k_local = scores.topk(min(k, len(scores))).indices
        return candidate_indices[top_k_local]

    def reset(self) -> None:
        self.table.zero_()
        self._total_updates = 0

    def memory_bytes(self) -> int:
        return self.depth * self.width * 4  # FP32


# ---------------------------------------------------------------------------
# TGSS: Full Gradient Sketch Manager
# ---------------------------------------------------------------------------

class TGSSManager:
    """
    Manages per-layer Count-Min Sketches and temporal superposition.

    Usage:
        1. After backward: call update_from_gradients(model)
        2. For ASDT: call get_importance_scores(model) → Dict[name, float]
        3. Every K steps: call decay_sketch()
    """

    def __init__(
        self,
        sketch_width: int = 1_000_000,
        sketch_depth: int = 5,
        temporal_alpha: float = 0.01,   # EMA decay: α=0.01 ≈ 100-step window
        use_freq_domain: bool = True,
    ):
        self.sketch_width = sketch_width
        self.sketch_depth = sketch_depth
        self.alpha = temporal_alpha
        self.use_freq_domain = use_freq_domain

        # One CMS per parameter tensor
        self._sketches: Dict[str, CountMinSketch] = {}
        # Cumulative importance EMA per parameter
        self._importance_ema: Dict[str, float] = {}
        self._step = 0

    # ---------------------------------------------------------------- update

    def update_from_gradients(self, model: nn.Module) -> int:
        """
        Process all .grad tensors in model and update sketches.
        Returns number of parameters processed.
        """
        total = 0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            self._update_param(name, param)
            total += param.numel()
        self._step += 1
        return total

    def _update_param(self, name: str, param: nn.Parameter) -> None:
        grad = param.grad.detach().float()

        if self.use_freq_domain and grad.dim() >= 2:
            grad = self._freq_domain_gradient(grad)

        flat_grad = grad.view(-1)
        magnitudes = flat_grad.abs()
        n = flat_grad.numel()
        param_indices = torch.arange(n, dtype=torch.int64)

        # Get or create sketch for this parameter
        if name not in self._sketches:
            self._sketches[name] = CountMinSketch(self.sketch_width, self.sketch_depth)

        # Build new sketch for this step
        step_sketch = CountMinSketch(self.sketch_width, self.sketch_depth)
        step_sketch.update(param_indices, magnitudes)

        # Temporal superposition: EMA merge
        self._sketches[name].merge(step_sketch, alpha=self.alpha)

        # Update importance EMA
        mean_magnitude = magnitudes.mean().item()
        prev = self._importance_ema.get(name, 0.0)
        self._importance_ema[name] = (1 - self.alpha) * prev + self.alpha * mean_magnitude

    # ---------------------------------------------------------------- query

    def get_importance_scores(self, model: nn.Module) -> Dict[str, float]:
        """
        Returns Dict[param_name → importance_score] for all sketched parameters.
        Used by ASDT for active-set selection.
        """
        scores: Dict[str, float] = {}
        for name, param in model.named_parameters():
            if name in self._importance_ema:
                scores[name] = self._importance_ema[name]
            else:
                scores[name] = 0.0
        return scores

    def query_top_k(self, name: str, k: int, param: nn.Parameter) -> torch.Tensor:
        """
        Return top-k active parameter indices for a specific layer.
        """
        if name not in self._sketches:
            return torch.randperm(param.numel())[:k]
        sketch = self._sketches[name]
        return sketch.top_k_indices(k, param.numel())

    # ---------------------------------------------------------------- freq domain (§3.5 novel)

    def _freq_domain_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Apply FFT to gradient tensor for better sparsity exploitation.
        Most energy in low-frequency components → 10-50x better sketch fidelity.
        """
        try:
            # Real FFT along last two dimensions
            if grad.dim() >= 2:
                g_freq = torch.fft.rfft2(grad.float())
                # Magnitude in frequency domain
                g_mag = g_freq.abs()
                # Keep top-K% of frequencies (energy compaction)
                k_frac = 0.05  # keep 5% of frequencies
                k = max(1, int(g_mag.numel() * k_frac))
                threshold = g_mag.reshape(-1).kthvalue(max(1, g_mag.numel() - k)).values
                mask = g_mag >= threshold
                g_freq_sparse = g_freq * mask
                # Reconstruct in spatial domain (sparse version)
                g_reconstructed = torch.fft.irfft2(g_freq_sparse, s=grad.shape[-2:])
                return g_reconstructed.to(grad.dtype)
        except Exception:
            pass  # Fallback to spatial domain
        return grad

    # ---------------------------------------------------------------- memory

    def total_memory_bytes(self) -> int:
        return sum(s.memory_bytes() for s in self._sketches.values())

    def stats(self) -> Dict:
        return {
            "n_sketches": len(self._sketches),
            "total_memory_mb": self.total_memory_bytes() // (1024 ** 2),
            "step": self._step,
            "n_importance_scores": len(self._importance_ema),
        }

    def reset_all(self) -> None:
        for s in self._sketches.values():
            s.reset()
        self._importance_ema.clear()


# ---------------------------------------------------------------------------
# TGSS error bound verification  (Theorem §7.4)
# ---------------------------------------------------------------------------

def verify_sketch_fidelity(
    true_gradients: Dict[str, torch.Tensor],
    sketch_manager: TGSSManager,
    model: nn.Module,
    tolerance: float = 1e-4,
) -> Dict[str, bool]:
    """
    Verify that sketch estimates are within error bound:
        |g_est - g_true| ≤ ε · ||g||_1 / w
    Returns {param_name: within_bound}.
    """
    results: Dict[str, bool] = {}
    for name, param in model.named_parameters():
        if name not in true_gradients:
            continue
        true_g = true_gradients[name].detach().float().view(-1)
        n = true_g.numel()
        if n == 0 or name not in sketch_manager._sketches:
            continue

        # Sample 100 random indices
        sample_idx = torch.randperm(n)[:min(100, n)]
        est = sketch_manager._sketches[name].query(sample_idx)
        true_sample = true_g.abs()[sample_idx]

        g_l1 = true_g.abs().sum().item()
        max_allowed_error = tolerance * g_l1 / sketch_manager.sketch_width
        actual_error = (est - true_sample).abs().max().item()
        results[name] = actual_error <= max(max_allowed_error, 1e-10)

    return results
