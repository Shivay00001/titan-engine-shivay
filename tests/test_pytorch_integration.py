import os
import sys
import tempfile
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to sys.path to allow imports like `from core...`
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import TITANConfig, build_titan_trainer

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, batch):
        x = batch["x"]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def simple_loss_fn(model, batch):
    logits = model(batch)
    targets = batch["y"]
    return nn.functional.cross_entropy(logits, targets)

def test_titan_pytorch_integration():
    print("Initializing TITAN integration test...")
    
    # We use a temporary directory for the NVMe store
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Setup configuration
        config = TITANConfig(
            nvme_path=tmp_dir,
            device="cpu", # Force CPU for tests unless GPU available
            dtype="float32",
            max_grad_norm=1.0,
            use_trd=True,
            trd_rank=4,
            trd_n_cores=2,
            trd_min_size=10, # Force TRD to hit our tiny layers
            sketch_width=1000, # Reduce memory for test
            sketch_depth=3,
        )

        model = TinyModel()
        
        # Test input data
        batch_size = 8
        batch = {
            "x": torch.randn(batch_size, 32),
            "y": torch.randint(0, 10, (batch_size,))
        }

        print("Building TITAN Trainer...")
        trainer = build_titan_trainer(model, config)

        print("Executing 1 step of TITAN...")
        loss, metrics = trainer.step(batch, simple_loss_fn)
        
        assert loss > 0, f"Loss should be positive, got {loss}"
        assert metrics.step == 1, "Metrics step should be 1"
        assert 'total_bytes' in trainer.vram_estimate(), "VRAM estimate missing total_bytes"
        
        print("Integration test passed successfully!")
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    test_titan_pytorch_integration()
