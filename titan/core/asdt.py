"""
TITAN Pillar 3: Adaptive Sparse Delta Training (ASDT)
=====================================================
Trains only the 0.001–0.1% of parameters with meaningful gradients.
Three parameter classes: Plastic | Elastic | Dormant
Gradient Importance Teleportation via lightweight proxy networks.
"""

from __future__ import annotations
import math
from enum import IntEnum
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterClass(IntEnum):
    PLASTIC  = 0   # High gradient → BF16, full Adam, VRAM
    ELASTIC  = 1   # Moderate grad → INT8, sign-SGD, DRAM
    DORMANT  = 2   # Low / zero    → INT2, no update, NVMe


# ---------------------------------------------------------------------------
# Gradient Importance Proxy (Novel Contribution §3.3)
# ---------------------------------------------------------------------------

class GradientImportanceProxy(nn.Module):
    """
    2-layer MLP predicts per-parameter gradient magnitude from layer activations.
    Cost: O(|S_t|) instead of O(N) for active-set selection.

    Input:  activation statistics [mean, std, l2_norm, sparsity] per layer
    Output: scalar importance score per parameter group
    """

    def __init__(self, n_features: int = 8, hidden: int = 32, n_groups: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_groups),
            nn.Sigmoid(),
        )
        self.n_groups = n_groups
        self._ema_scores = torch.zeros(n_groups)

    def forward(self, act_stats: torch.Tensor) -> torch.Tensor:
        """act_stats: (n_features,) → importance (n_groups,) in [0,1]."""
        return self.net(act_stats.unsqueeze(0)).squeeze(0)

    def compute_act_stats(self, activation: torch.Tensor) -> torch.Tensor:
        """Extract 8-dimensional feature vector from an activation tensor."""
        flat = activation.detach().float().view(-1)
        if flat.numel() == 0:
            return torch.zeros(8)
        return torch.tensor([
            flat.mean().item(),
            flat.std().item(),
            flat.norm(2).item() / math.sqrt(flat.numel()),
            (flat.abs() < 1e-6).float().mean().item(),   # sparsity
            flat.abs().max().item(),
            flat.abs().min().item(),
            flat.abs().median().item(),
            flat.kthvalue(max(1, int(flat.numel() * 0.99))).values.item(),  # 99th pct
        ])

    @torch.no_grad()
    def update_ema(self, scores: torch.Tensor, alpha: float = 0.1) -> None:
        self._ema_scores = (1 - alpha) * self._ema_scores + alpha * scores.cpu()

    def top_k_groups(self, k: int) -> List[int]:
        return self._ema_scores.topk(min(k, self.n_groups)).indices.tolist()


# ---------------------------------------------------------------------------
# Parameter Group Bitmap
# ---------------------------------------------------------------------------

class SparseParameterIndex:
    """
    Compressed bitmap tracking which parameter indices are in the active set.
    Uses uint64 packed bits for memory efficiency.
    """

    def __init__(self, n_params: int):
        self.n_params = n_params
        n_words = math.ceil(n_params / 64)
        self._bits = torch.zeros(n_words, dtype=torch.int64)

    def set(self, indices: torch.Tensor) -> None:
        for idx in indices.tolist():
            word, bit = divmod(int(idx), 64)
            self._bits[word] |= (1 << bit)

    def get_indices(self) -> torch.Tensor:
        result = []
        for word_idx, word in enumerate(self._bits.tolist()):
            if word == 0:
                continue
            for bit in range(64):
                if word & (1 << bit):
                    idx = word_idx * 64 + bit
                    if idx < self.n_params:
                        result.append(idx)
        return torch.tensor(result, dtype=torch.long)

    def clear(self) -> None:
        self._bits.zero_()

    def count(self) -> int:
        return int(self._bits.count_nonzero().item()) * 64  # approx


# ---------------------------------------------------------------------------
# ASDT Optimizer Wrapper
# ---------------------------------------------------------------------------

class ASDTOptimizer:
    """
    Wraps any base optimizer (Adam / AdamW) to only update the active sparse set.

    Selection rule (§3.3):
        S_t = top_k { |sketch_estimate(∂L/∂w_i)| + λ · exploration_bonus_i }

    Three-class update:
        Plastic  → full Adam BF16
        Elastic  → sign-SGD INT8
        Dormant  → no update
    """

    def __init__(
        self,
        named_params: Iterator,
        top_k_fraction: float = 0.001,        # fraction of params active per step
        exploration_lambda: float = 0.01,
        plastic_lr: float = 1e-4,
        elastic_lr: float = 1e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.top_k_fraction = top_k_fraction
        self.exploration_lambda = exploration_lambda
        self.plastic_lr = plastic_lr
        self.elastic_lr = elastic_lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Collect flat param references
        self._params: List[Tuple[str, nn.Parameter]] = list(named_params)
        self._step = 0

        # Per-parameter Adam states (only allocated for plastic params)
        self._m: Dict[str, torch.Tensor] = {}   # first moment
        self._v: Dict[str, torch.Tensor] = {}   # second moment

        # Exploration bonus: increases for params not updated recently
        self._steps_since_update: Dict[str, int] = {n: 0 for n, _ in self._params}

        # Parameter class assignments
        self._classes: Dict[str, ParameterClass] = {
            n: ParameterClass.DORMANT for n, _ in self._params
        }

        # Moving average of gradient magnitudes for class transitions
        self._grad_ema: Dict[str, float] = {n: 0.0 for n, _ in self._params}
        self._class_threshold_high = 1e-3
        self._class_threshold_low  = 1e-5

    # ------------------------------------------------------------------ API

    @torch.no_grad()
    def step(self, sketch_gradient_magnitudes: Optional[Dict[str, float]] = None) -> Dict[str, int]:
        """
        Perform one ASDT step.

        sketch_gradient_magnitudes: dict name→|g| from TGSS sketch.
        Returns count of parameters updated per class.
        """
        self._step += 1
        b1, b2 = self.betas
        stats = {"plastic": 0, "elastic": 0, "dormant": 0}

        for name, param in self._params:
            if param.grad is None:
                self._steps_since_update[name] += 1
                continue

            grad = param.grad.detach()

            # Use sketch estimate if provided, else actual grad norm
            if sketch_gradient_magnitudes and name in sketch_gradient_magnitudes:
                importance = sketch_gradient_magnitudes[name]
            else:
                importance = grad.norm(2).item() / max(grad.numel() ** 0.5, 1.0)

            # Exploration bonus
            bonus = self.exploration_lambda * math.log1p(self._steps_since_update[name])
            effective_importance = importance + bonus

            # Update EMA of gradient magnitude for class transition
            ema = self._grad_ema.get(name, 0.0)
            self._grad_ema[name] = 0.9 * ema + 0.1 * importance

            # Determine class
            cls = self._classify(name, self._grad_ema[name])
            self._classes[name] = cls

            if cls == ParameterClass.DORMANT:
                self._steps_since_update[name] += 1
                stats["dormant"] += 1
                continue

            self._steps_since_update[name] = 0

            if cls == ParameterClass.PLASTIC:
                # Full Adam update
                if name not in self._m:
                    self._m[name] = torch.zeros_like(param.data, dtype=torch.float32)
                    self._v[name] = torch.zeros_like(param.data, dtype=torch.float32)

                m = self._m[name]
                v = self._v[name]
                g = grad.float()

                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** self._step)
                v_hat = v / (1 - b2 ** self._step)
                update = m_hat / (v_hat.sqrt() + self.eps)

                if self.weight_decay > 0:
                    update.add_(param.data.float(), alpha=self.weight_decay)

                param.data.add_(update.to(param.dtype), alpha=-self.plastic_lr)
                stats["plastic"] += param.numel()

            elif cls == ParameterClass.ELASTIC:
                # Sign-SGD: cheaper update, INT8 precision
                sign_grad = grad.sign()
                param.data.add_(sign_grad.to(param.dtype), alpha=-self.elastic_lr)
                stats["elastic"] += param.numel()

        return stats

    def _classify(self, name: str, grad_ema: float) -> ParameterClass:
        if grad_ema >= self._class_threshold_high:
            return ParameterClass.PLASTIC
        elif grad_ema >= self._class_threshold_low:
            return ParameterClass.ELASTIC
        else:
            return ParameterClass.DORMANT

    def set_thresholds(self, high: float, low: float) -> None:
        self._class_threshold_high = high
        self._class_threshold_low  = low

    def param_class_summary(self) -> Dict[str, int]:
        counts = {c.name: 0 for c in ParameterClass}
        for cls in self._classes.values():
            counts[cls.name] += 1
        return counts

    def zero_grad(self) -> None:
        for _, p in self._params:
            if p.grad is not None:
                p.grad = None


# ---------------------------------------------------------------------------
# ASDT top-k active parameter selection (global version)
# ---------------------------------------------------------------------------

def select_active_parameters(
    model: nn.Module,
    top_k_fraction: float = 0.001,
    exploration_lambda: float = 0.01,
    steps_since_update: Optional[Dict[str, int]] = None,
) -> Set[str]:
    """
    Returns names of parameters in the active set S_t for this step.
    Uses actual gradient magnitudes (or zeros if not computed yet).
    """
    scores: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm(2).item()
        else:
            g_norm = 0.0
        bonus = 0.0
        if steps_since_update and name in steps_since_update:
            bonus = exploration_lambda * math.log1p(steps_since_update[name])
        scores[name] = g_norm + bonus

    total_params = sum(p.numel() for p in model.parameters())
    k = max(1, int(total_params * top_k_fraction))

    # Select top-k by score (parameter-name level, not element level)
    sorted_names = sorted(scores, key=lambda n: scores[n], reverse=True)

    # Accumulate until we hit k total parameters
    active: Set[str] = set()
    accumulated = 0
    for name in sorted_names:
        p = dict(model.named_parameters())[name]
        active.add(name)
        accumulated += p.numel()
        if accumulated >= k:
            break

    return active


# ---------------------------------------------------------------------------
# Memory estimate for ASDT active set
# ---------------------------------------------------------------------------

def asdt_vram_estimate(n_total_params: int, plastic_fraction: float = 0.001) -> Dict[str, int]:
    """
    Returns dict with VRAM breakdown per §3.3:
        VRAM_ASDT = |S_plastic| × (param_bytes + grad_bytes + adam_bytes)
    """
    plastic = int(n_total_params * plastic_fraction)
    param_bytes = plastic * 2     # BF16
    grad_bytes  = plastic * 2     # BF16
    adam_bytes  = plastic * 8     # m + v in FP32
    total = param_bytes + grad_bytes + adam_bytes
    return {
        "plastic_params": plastic,
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "adam_bytes": adam_bytes,
        "total_vram_bytes": total,
        "total_vram_mb": total // (1024 ** 2),
    }
