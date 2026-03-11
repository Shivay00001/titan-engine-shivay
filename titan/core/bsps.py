"""
TITAN Pillar 6: Biologically-Inspired Synaptic Plasticity Scheduling (BSPS)
=============================================================================
4-phase parameter lifecycle: GROWTH → ELASTIC → SLEEPING → FROZEN
Novel: Task Relevance Reawakening (TRR) for continual learning.
Manages automatic migration between VRAM / DRAM / NVMe tiers.
"""

from __future__ import annotations
import math
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Phase(IntEnum):
    GROWTH   = 0   # High grad → full Adam BF16, VRAM
    ELASTIC  = 1   # LTP/LTD   → sign-SGD INT8, DRAM
    SLEEPING = 2   # Consolidating → EMA decay, NVMe INT4
    FROZEN   = 3   # Long-term memory → no update, NVMe INT2


PHASE_NAMES = {Phase.GROWTH: "GROWTH", Phase.ELASTIC: "ELASTIC",
               Phase.SLEEPING: "SLEEPING", Phase.FROZEN: "FROZEN"}


# ---------------------------------------------------------------------------
# Task Relevance Reawakening Probe  (Novel Contribution §3.6)
# ---------------------------------------------------------------------------

class TaskRelevanceProbe(nn.Module):
    """
    2-layer MLP predicts which frozen parameter groups are relevant
    to the current training task, enabling reawakening from FROZEN → ELASTIC.

    Input:  [layer_embedding (embed_dim), task_embedding (task_dim)]
    Output: relevance score in [0, 1]
    """

    def __init__(self, embed_dim: int = 64, task_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.task_dim = task_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim + task_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        # Layer embeddings: one per parameter group
        self._layer_embeddings: Dict[str, nn.Parameter] = {}
        self._task_embedding: Optional[torch.Tensor] = None
        self._reawaken_threshold = 0.5

    def register_layer(self, name: str, param: nn.Parameter) -> None:
        """Register a parameter group with a learnable embedding."""
        emb = nn.Parameter(torch.randn(self.embed_dim) * 0.01)
        self._layer_embeddings[name] = emb

    def set_task_embedding(self, task_examples: torch.Tensor) -> None:
        """
        Compute task embedding from first ~100 training examples.
        task_examples: (N, feature_dim) tensor.
        """
        with torch.no_grad():
            # Simple mean pooling + PCA-like projection to task_dim
            mean_embed = task_examples.float().mean(0)
            if mean_embed.numel() >= self.task_dim:
                self._task_embedding = mean_embed[:self.task_dim]
            else:
                self._task_embedding = F.pad(mean_embed, (0, self.task_dim - mean_embed.numel()))

    @torch.no_grad()
    def score_layer(self, name: str) -> float:
        """
        Returns relevance score for layer `name` to current task.
        Relevance(w_i, task) = Probe_MLP(layer_embedding_i, task_embedding)
        """
        if self._task_embedding is None:
            return 0.0
        if name not in self._layer_embeddings:
            return 0.0
        layer_emb = self._layer_embeddings[name].detach()
        task_emb  = self._task_embedding
        combined  = torch.cat([layer_emb, task_emb.to(layer_emb.device)], dim=-1)
        return float(self.net(combined.unsqueeze(0)).item())

    def should_reawaken(self, name: str) -> bool:
        return self.score_layer(name) >= self._reawaken_threshold


# ---------------------------------------------------------------------------
# Per-Parameter Phase State
# ---------------------------------------------------------------------------

class ParameterPhaseState:
    __slots__ = [
        "phase", "grad_ema", "steps_in_phase",
        "steps_without_update", "last_grad_norm",
    ]

    def __init__(self):
        self.phase: Phase = Phase.FROZEN
        self.grad_ema: float = 0.0
        self.steps_in_phase: int = 0
        self.steps_without_update: int = 0
        self.last_grad_norm: float = 0.0


# ---------------------------------------------------------------------------
# BSPS Phase Manager
# ---------------------------------------------------------------------------

class BSPSManager:
    """
    Manages the 4-phase lifecycle for all parameters in a model.

    Phase transition conditions (§3.6):
        GROWTH  → ELASTIC:  moving_avg(|g|) < τ_high  for M1 steps
        ELASTIC → SLEEPING: moving_avg(|g|) < τ_low   for M2 steps
        SLEEPING→ FROZEN:   no activation for K_freeze steps
        FROZEN  → GROWTH:   task relevance score > threshold (TRR)

    VRAM budget = |GROWTH| × 12 bytes + rest on DRAM/NVMe
    """

    def __init__(
        self,
        tau_high: float    = 1e-3,
        tau_low: float     = 1e-5,
        m1_steps: int      = 50,
        m2_steps: int      = 200,
        k_freeze: int      = 1000,
        ema_decay: float   = 0.95,
        task_probe: Optional[TaskRelevanceProbe] = None,
    ):
        self.tau_high = tau_high
        self.tau_low  = tau_low
        self.m1 = m1_steps
        self.m2 = m2_steps
        self.k_freeze = k_freeze
        self.ema_decay = ema_decay
        self.task_probe = task_probe

        self._states: Dict[str, ParameterPhaseState] = {}
        self._step = 0

        # Hooks for tier migration callbacks
        self._on_phase_change: List[Callable] = []

    # ---------------------------------------------------------------- registration

    def register_model(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            state = ParameterPhaseState()
            state.phase = Phase.FROZEN  # start frozen, warm up naturally
            self._states[name] = state
            if self.task_probe:
                self.task_probe.register_layer(name, param)

    # ---------------------------------------------------------------- step update

    def step(self, model: nn.Module) -> Dict[str, int]:
        """
        Update phase states based on current gradients.
        Returns phase counts: {phase_name: count}.
        """
        self._step += 1
        for name, param in model.named_parameters():
            if name not in self._states:
                state = ParameterPhaseState()
                self._states[name] = state
            state = self._states[name]

            # Compute gradient norm for this parameter
            if param.grad is not None:
                g_norm = param.grad.detach().float().norm(2).item()
                g_norm /= max(param.numel() ** 0.5, 1.0)  # normalize by sqrt(n)
            else:
                g_norm = 0.0

            state.last_grad_norm = g_norm
            state.grad_ema = self.ema_decay * state.grad_ema + (1 - self.ema_decay) * g_norm

            # Update step counters
            if g_norm > 0:
                state.steps_without_update = 0
            else:
                state.steps_without_update += 1
            state.steps_in_phase += 1

            # Phase transitions
            old_phase = state.phase
            new_phase = self._compute_transition(name, state)
            if new_phase != old_phase:
                state.phase = new_phase
                state.steps_in_phase = 0
                self._notify_phase_change(name, old_phase, new_phase, param)

        return self._phase_counts()

    def _compute_transition(self, name: str, state: ParameterPhaseState) -> Phase:
        g = state.grad_ema

        if state.phase == Phase.GROWTH:
            if g < self.tau_high and state.steps_in_phase >= self.m1:
                return Phase.ELASTIC
            return Phase.GROWTH

        elif state.phase == Phase.ELASTIC:
            if g < self.tau_low and state.steps_in_phase >= self.m2:
                return Phase.SLEEPING
            if g >= self.tau_high:
                return Phase.GROWTH  # re-activate
            return Phase.ELASTIC

        elif state.phase == Phase.SLEEPING:
            if state.steps_without_update >= self.k_freeze:
                return Phase.FROZEN
            if g >= self.tau_low:
                return Phase.ELASTIC  # re-activate
            return Phase.SLEEPING

        elif state.phase == Phase.FROZEN:
            # Check task relevance reawakening
            if self.task_probe and self.task_probe.should_reawaken(name):
                return Phase.ELASTIC
            # Also reawaken if gradient appears (e.g. due to new training signal)
            if state.last_grad_norm > self.tau_low:
                return Phase.ELASTIC
            return Phase.FROZEN

        return state.phase

    # ---------------------------------------------------------------- queries

    def get_phase(self, name: str) -> Phase:
        if name in self._states:
            return self._states[name].phase
        return Phase.FROZEN

    def growth_params(self) -> Set[str]:
        return {n for n, s in self._states.items() if s.phase == Phase.GROWTH}

    def elastic_params(self) -> Set[str]:
        return {n for n, s in self._states.items() if s.phase == Phase.ELASTIC}

    def frozen_params(self) -> Set[str]:
        return {n for n, s in self._states.items() if s.phase == Phase.FROZEN}

    def should_update(self, name: str) -> bool:
        phase = self.get_phase(name)
        return phase in (Phase.GROWTH, Phase.ELASTIC)

    def update_rule(self, name: str) -> str:
        phase = self.get_phase(name)
        return {Phase.GROWTH: "adam_bf16", Phase.ELASTIC: "sign_sgd_int8",
                Phase.SLEEPING: "ema_decay", Phase.FROZEN: "none"}[phase]

    def vram_estimate_bytes(self, param_sizes: Dict[str, int]) -> int:
        """
        Estimate VRAM usage: GROWTH params × 12 bytes (param+grad+adam).
        """
        total = 0
        for name, state in self._states.items():
            if state.phase == Phase.GROWTH:
                n = param_sizes.get(name, 0)
                total += n * 12  # 2 (BF16 param) + 2 (grad) + 8 (adam m+v FP32)
        return total

    # ---------------------------------------------------------------- callbacks

    def on_phase_change(self, callback: Callable) -> None:
        """Register callback(name, old_phase, new_phase, param) for tier migration."""
        self._on_phase_change.append(callback)

    def _notify_phase_change(
        self, name: str, old: Phase, new: Phase, param: nn.Parameter
    ) -> None:
        for cb in self._on_phase_change:
            try:
                cb(name, old, new, param)
            except Exception as e:
                pass  # non-fatal

    def _phase_counts(self) -> Dict[str, int]:
        counts = {p.name: 0 for p in Phase}
        for state in self._states.values():
            counts[state.phase.name] += 1
        return counts

    # ---------------------------------------------------------------- sleeping EMA decay

    def apply_sleeping_decay(self, model: nn.Module, decay: float = 0.9999) -> int:
        """
        For SLEEPING parameters: w ← w × decay (slow consolidation).
        Does not require gradient; applied directly to weights.
        """
        count = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if self.get_phase(name) == Phase.SLEEPING:
                    param.data.mul_(decay)
                    count += 1
        return count

    def report(self) -> str:
        counts = self._phase_counts()
        total = sum(counts.values())
        lines = [f"BSPS Phase Report (step {self._step}):"]
        for phase_name, cnt in counts.items():
            pct = 100 * cnt / max(total, 1)
            lines.append(f"  {phase_name:10s}: {cnt:6d} ({pct:.1f}%)")
        return "\n".join(lines)
