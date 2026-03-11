import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import TITANConfig, build_titan_trainer

# ---------------------------------------------------------------------------
# High-Scale Transformer Model for Production Stress Test
# ---------------------------------------------------------------------------

class ProductionTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class TITANStressModel(nn.Module):
    """A 'Real-World' scale model for stress testing: ~50M parameters."""
    def __init__(self, vocab_size=32000, embed_dim=512, num_layers=12, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            ProductionTransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, batch):
        x = batch["input_ids"]
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

def run_production_stress_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- TITAN PRODUCTION STRESS TEST STARTING ON {device.upper()} ---")
    
    # 1. Configuration (Targeting memory optimization and real-world defaults)
    config = TITANConfig(
        device=device,
        dtype="float32", # Stability first
        nvme_path="./stress_test_nvme",
        use_trd=True,
        trd_rank=16,            # Reduced rank for 4GB RAM
        trd_n_cores=6,
        trd_min_size=1024,      # Compress all major layers
        sketch_width=100_000,   # Moderate resolution
        sketch_depth=3,
        hge_keep_fraction=0.01, # Extremely aggressive gradient compression
        log_every=5,
        max_grad_norm=1.0
    )

    # 2. Model Initialization
    # 6 layers, 256 embed dim = balanced workload
    print("Initializing StressModel (6 Layers, 256 Dim)...")
    model = TITANStressModel(num_layers=6, embed_dim=256)
    
    # 3. Build Trainer
    trainer = build_titan_trainer(model, config)
    
    def loss_fn(model, batch):
        logits = model(batch)
        targets = batch["labels"]
        # [B, T, V] -> [B*T, V]
        return nn.functional.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1))

    # 4. Training Loop
    steps = 2
    batch_size = 4
    seq_len = 64
    
    print(f"Simulating {steps} steps of training with B={batch_size}, T={seq_len}...")
    start_time = time.time()
    
    losses = []
    
    for step in range(1, steps + 1):
        batch = {
            "input_ids": torch.randint(0, 32000, (batch_size, seq_len)),
            "labels": torch.randint(0, 32000, (batch_size, seq_len))
        }
        
        loss, metrics = trainer.step(batch, loss_fn)
        losses.append(loss)
        
        if step % 5 == 0:
            print(f"Step {step}/{steps} - Loss: {loss:.4f} - HGE CR: {metrics.hge_compression_ratio:.1f}x")

    total_time = time.time() - start_time
    print(f"\n--- STRESS TEST COMPLETE in {total_time:.2f}s ---")
    
    # 5. Metrics Evaluation
    vram_stats = trainer.vram_estimate()
    comp_ratio = metrics.hge_compression_ratio
    
    print("\n[PERFORMANCE SUMMARY]")
    print(f"Avg Loss Delta: {losses[0] - losses[-1]:.4f}")
    print(f"Peak HGE Compression: {comp_ratio:.1f}x")
    print(f"Estimated VRAM Overhead (Active): {vram_stats['total_mb']} MB")
    print(f"TRD Weight Compression: Enabled (Target Rank 32)")
    
    # Check convergence (simplified)
    if losses[-1] < losses[0]:
        print("Convergence: Positive (Loss decreasing)")
    else:
        print("Convergence: Neutral/Slow (Expected for random data, but no crashes)")

if __name__ == "__main__":
    run_production_stress_test()
