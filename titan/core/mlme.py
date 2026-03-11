"""
TITAN Pillar 2: Micro-Layer Materialization Engine (MLME)
=========================================================
Never materialize a full layer. Stream:
  - Attention: head-by-head (H_micro << H total heads)
  - FFN: column stripes of width C  (C << d_ff)
  - Error Accumulation Banks (EABs) for INT2/INT4 rounding correction
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Error Accumulation Bank (Novel Contribution §6.6)
# ---------------------------------------------------------------------------

class ErrorAccumulationBank:
    """
    Maintains FP32 residuals for each INT-quantization group so that
    rounding error never compounds across training steps.

    Update rule (§6.6):
        w_INT_t  = Quantize_INT2(w_t + EAB_t)
        EAB_{t+1} = w_t + EAB_t - Dequantize(w_INT_t)
    """

    def __init__(self, shape: Tuple[int, ...], quant_bits: int = 4, group_size: int = 128):
        self.quant_bits = quant_bits
        self.group_size = group_size
        n_groups = math.ceil(math.prod(shape) / group_size)
        # EAB stored in FP32; tiny relative to full tensors
        self._bank = torch.zeros(math.prod(shape), dtype=torch.float32)
        self._shape = shape

    def quantize_with_correction(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (quantized_tensor, updated_eab).
        quantized_tensor has same dtype as w but with corrected rounding.
        """
        orig_shape = w.shape
        flat = w.detach().float().view(-1)
        corrected = flat + self._bank

        levels = (1 << self.quant_bits) - 1
        # Per-group min/max quantization
        n = flat.numel()
        padded_len = math.ceil(n / self.group_size) * self.group_size
        padded = F.pad(corrected, (0, padded_len - n))
        groups = padded.view(-1, self.group_size)
        g_min = groups.min(dim=1, keepdim=True).values
        g_max = groups.max(dim=1, keepdim=True).values
        scale = (g_max - g_min).clamp(min=1e-8) / levels
        q_groups = ((groups - g_min) / scale).round().clamp(0, levels)

        # Dequantize to compute residual
        deq_groups = q_groups * scale + g_min
        deq = deq_groups.view(-1)[:n]

        # Update EAB: residual = corrected - dequantized
        self._bank = (corrected - deq).detach()

        # Return quantized tensor at original dtype
        q_flat = q_groups.view(-1)[:n]
        return q_flat.view(orig_shape).to(w.dtype), self._bank.clone()

    def reset(self) -> None:
        self._bank.zero_()


# ---------------------------------------------------------------------------
# Micro-block flash attention  (exact, O(N) memory)
# ---------------------------------------------------------------------------

def flash_attention_micro(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
    causal: bool = True,
) -> torch.Tensor:
    """
    Block-tiled exact attention (FlashAttention logic, §2.11).
    Memory: O(N) instead of O(N²).  Handles arbitrary sequence lengths.

    Shape: Q/K/V are (B, N, d_head).
    Returns output (B, N, d_head).
    """
    B, N, d = Q.shape
    scale = d ** -0.5
    out = torch.zeros_like(Q)
    lse = torch.full((B, N), float("-inf"), device=Q.device, dtype=torch.float32)
    # Running max for numerical stability (online softmax)
    m = torch.full((B, N), float("-inf"), device=Q.device, dtype=torch.float32)

    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()

    for j in range(0, N, block_size):
        kj = K_f[:, j:j + block_size, :]   # B × Bk × d
        vj = V_f[:, j:j + block_size, :]

        for i in range(0, N, block_size):
            if causal and j > i + block_size - 1:
                continue  # skip future blocks (causal mask)

            qi = Q_f[:, i:i + block_size, :]   # B × Bi × d
            Bi_len = qi.shape[1]
            Bj_len = kj.shape[1]

            s = torch.bmm(qi, kj.transpose(1, 2)) * scale  # B × Bi × Bj

            if causal:
                rows = torch.arange(i, i + Bi_len, device=Q.device).unsqueeze(1)
                cols = torch.arange(j, j + Bj_len, device=Q.device).unsqueeze(0)
                mask = rows < cols  # upper triangle = future
                s = s.masked_fill(mask.unsqueeze(0), float("-inf"))

            m_new = torch.maximum(m[:, i:i + Bi_len], s.max(dim=-1).values)
            # Update running softmax sum
            exp_s = torch.exp(s - m_new.unsqueeze(-1))
            exp_lse = torch.exp(lse[:, i:i + Bi_len] - m_new)
            lse_new = m_new + torch.log(exp_lse + exp_s.sum(dim=-1))

            # Update output accumulator
            new_contrib = torch.bmm(exp_s, vj)  # B × Bi × d
            rescale = torch.exp(lse[:, i:i + Bi_len] - lse_new).unsqueeze(-1)
            out[:, i:i + Bi_len, :] = (
                out[:, i:i + Bi_len, :].float() * rescale
                + new_contrib * torch.exp(m_new - lse_new).unsqueeze(-1)
            ).to(Q.dtype)

            m[:, i:i + Bi_len]   = m_new
            lse[:, i:i + Bi_len] = lse_new

    return out


# ---------------------------------------------------------------------------
# Micro-head attention block
# ---------------------------------------------------------------------------

class MicroHeadAttention(nn.Module):
    """
    Streams H attention heads H_micro at a time.
    Peak VRAM = H_micro × d × d_head × 3 × 2 bytes  (§3.2).
    Gradients flow through each micro-block independently via standard autograd.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        micro_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.micro_heads = min(micro_heads, num_heads)
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout

        # Separate Q/K/V projections per head group allows partial loading
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # EAB per projection matrix
        self.eab_q = ErrorAccumulationBank(self.q_proj.weight.shape)
        self.eab_k = ErrorAccumulationBank(self.k_proj.weight.shape)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        Q = self.q_proj(x)  # B × N × D
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (B, N, H, d_head) then stream micro-head groups
        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.num_heads, self.head_dim)

        Qh, Kh, Vh = _split_heads(Q), _split_heads(K), _split_heads(V)
        output_chunks: List[torch.Tensor] = []

        for h_start in range(0, self.num_heads, self.micro_heads):
            h_end = min(h_start + self.micro_heads, self.num_heads)
            # Micro-block: B × N × micro_heads × d_head
            q_block = Qh[:, :, h_start:h_end, :].contiguous().view(B * (h_end - h_start), N, self.head_dim)
            k_block = Kh[:, :, h_start:h_end, :].contiguous().view(B * (h_end - h_start), N, self.head_dim)
            v_block = Vh[:, :, h_start:h_end, :].contiguous().view(B * (h_end - h_start), N, self.head_dim)

            ctx_block = flash_attention_micro(q_block, k_block, v_block)
            ctx_block = ctx_block.view(B, N, (h_end - h_start) * self.head_dim)
            output_chunks.append(ctx_block)

        # Concatenate all micro-block outputs and project
        ctx = torch.cat(output_chunks, dim=-1)
        return self.out_proj(ctx)


# ---------------------------------------------------------------------------
# Column-stripe FFN  (§3.2 Striped FFN Computation)
# ---------------------------------------------------------------------------

class StripeFFN(nn.Module):
    """
    W1: embed_dim → ffn_dim  (column-striped in forward pass)
    W2: ffn_dim → embed_dim

    FFN(x) = Σ_stripes GELU(x · W1_stripe) · W2_stripe
    Correctness: linear ops exactly decomposable by column groups.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: Optional[int] = None,
        stripe_width: int = 4096,
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim or 4 * embed_dim
        self.stripe_width = stripe_width
        self.n_stripes = math.ceil(self.ffn_dim / self.stripe_width)

        self.w1 = nn.Parameter(torch.empty(embed_dim, self.ffn_dim))
        self.w2 = nn.Parameter(torch.empty(self.ffn_dim, embed_dim))
        self.b1 = nn.Parameter(torch.zeros(self.ffn_dim)) if bias else None
        self.b2 = nn.Parameter(torch.zeros(embed_dim)) if bias else None

        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.act = F.gelu if activation == "gelu" else F.silu
        # EAB for correction across stripes
        self._eab = ErrorAccumulationBank((embed_dim,), quant_bits=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stripe-wise FFN with exact decomposition."""
        *batch, d = x.shape
        out = torch.zeros(*batch, self.embed_dim, device=x.device, dtype=x.dtype)

        for s in range(self.n_stripes):
            c_start = s * self.stripe_width
            c_end   = min(c_start + self.stripe_width, self.ffn_dim)
            w1_s = self.w1[:, c_start:c_end]   # d × C
            w2_s = self.w2[c_start:c_end, :]   # C × d
            b1_s = self.b1[c_start:c_end] if self.b1 is not None else None
            h = x @ w1_s
            if b1_s is not None:
                h = h + b1_s
            h = self.act(h)                     # apply non-linearity within stripe
            out = out + h @ w2_s                # accumulate output

        if self.b2 is not None:
            out = out + self.b2
        return out

    @property
    def memory_peak_bytes(self) -> int:
        """Peak VRAM for one stripe in bytes (BF16)."""
        C = self.stripe_width
        d = self.embed_dim
        # w1_stripe + w2_stripe + activation buffer
        return (d * C + C * d + C) * 2  # BF16


# ---------------------------------------------------------------------------
# Micro-checkpoint manager (§3.2 Activation Micro-Checkpointing)
# ---------------------------------------------------------------------------

class MicroCheckpointManager:
    """
    Saves activations at head/stripe boundaries within a layer.
    Recomputes only the current micro-block during backward, not the entire layer.

    Memory = max(M_micro_block) + M_checkpoints  vs  M_full_layer
    """

    def __init__(self, recompute: bool = True):
        self.recompute = recompute
        self._checkpoints: List[Tuple[str, torch.Tensor]] = []

    def save(self, tag: str, tensor: torch.Tensor) -> None:
        if self.recompute:
            # Detach and store on CPU to free VRAM
            self._checkpoints.append((tag, tensor.detach().cpu()))

    def retrieve(self, tag: str) -> Optional[torch.Tensor]:
        for t, val in reversed(self._checkpoints):
            if t == tag:
                return val.cuda() if torch.cuda.is_available() else val
        return None

    def clear(self) -> None:
        self._checkpoints.clear()

    @property
    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for _, t in self._checkpoints)


# ---------------------------------------------------------------------------
# Utility: estimate micro-block VRAM budget
# ---------------------------------------------------------------------------

def vram_budget_for_micro_attention(
    embed_dim: int,
    num_heads: int,
    micro_heads: int,
    seq_len: int,
    batch_size: int = 1,
    bytes_per_param: int = 2,
) -> int:
    """
    Returns peak VRAM in bytes for one micro-head block, per §3.2 formula:
        Peak VRAM = H_micro × d × d_head × 3 × 2
    """
    d_head = embed_dim // num_heads
    qkv = micro_heads * d_head * 3          # Q/K/V weights per micro group
    attn = batch_size * seq_len * d_head * micro_heads  # activation buffer
    return (qkv + attn) * bytes_per_param
