import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from titan import TITANConfig, build_titan_trainer
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # MLME StripeFFN equivalent projection expansion
        # Using standard linear for now, which TITAN converts to TRD.
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class DummyModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=8)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, batch):
        x = batch["input_ids"]
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        logits = self.head(h)
        return logits

def main():
    parser = argparse.ArgumentParser(description="TITAN Production Training Example")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    print(f"=== Starting TITAN Training on {args.device.upper()} ===")
    
    # 1. Configuration
    config = TITANConfig(
        device=args.device,
        dtype=args.dtype,
        nvme_path="./titan_nvme_store",
        use_trd=True,
        trd_rank=16,
        trd_n_cores=4,
        sketch_width=100_000,
        sketch_depth=3,
        use_hge_for_update=False
    )
    
    # 2. Build Model
    model = DummyModel()
    
    # 3. Setup Trainer
    trainer = build_titan_trainer(model, config)
    
    # 4. Define Loss
    def loss_fn(model, batch):
        logits = model(batch) # [B, T, V]
        targets = batch["labels"] # [B, T]
        return nn.functional.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1))
    
    # 5. Training Loop
    batch_size, seq_len = 4, 32
    print("\nStarting Training Loop...")
    for step in range(args.steps):
        # Generate synthetic data
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len))
        }
        
        loss, metrics = trainer.step(batch, loss_fn)
        
        if step == args.steps - 1:
            print("\nFinal State VRAM Breakdown:")
            vram_stats = trainer.vram_estimate()
            for k, v in vram_stats.items():
                print(f"  {k}: {v}")
                
            print("\nBSPS Phase Breakdown:")
            print(trainer.phase_report())
            
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
