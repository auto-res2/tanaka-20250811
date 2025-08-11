import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import dataloader from preprocess module
from preprocess import get_toy_dataloaders


def fake_quantise(x: torch.Tensor, bits: int, symmetric: bool = True):
    """Uniform quantisation with STE backward (for ≤8 bits)."""
    qmin = -(2 ** (bits - 1)) if symmetric else 0
    qmax = (2 ** (bits - 1)) - 1 if symmetric else (2 ** bits) - 1
    scale = x.detach().abs().max() / qmax + 1e-8
    
    def _quantise(y):
        y_div = y / scale
        y_clamped = y_div.clamp(qmin, qmax).round()
        return y_clamped * scale
    
    # STE: forward uses quantised value, backward uses identity
    return _quantise(x) + (x - _quantise(x)).detach()


class HierarchicalCodebookLinear(nn.Module):
    """Linear layer with 2-bit main indices and 1-bit residue (total 3 effective bits)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        # Store indices instead of full weights
        self.main_index = nn.Parameter(torch.randint(-2, 3, (out_features, in_features), dtype=torch.int8))
        self.residue_index = nn.Parameter(torch.randint(-1, 2, (out_features, in_features), dtype=torch.int8))
        self.scale = nn.Parameter(torch.full((1,), 1e-1))

    def forward(self, x):
        w_int = self.main_index.to(torch.float32) + self.residue_index.to(torch.float32) * 0.25
        w = w_int * self.scale
        return F.linear(x, w)


class RA_LayerNorm(nn.Module):
    """Range-Aware Residual LayerNorm – tiny variant."""

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.ln(x + self.beta)


class TinyTransformerBlock(nn.Module):
    """A tiny transformer block incorporating our quantisation modules."""

    def __init__(self, d_model=64, n_heads=2, mlp_ratio=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ra_ln1 = RA_LayerNorm(d_model)
        self.mlp = nn.Sequential(
            HierarchicalCodebookLinear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            HierarchicalCodebookLinear(d_model * mlp_ratio, d_model),
        )
        self.ra_ln2 = RA_LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ra_ln1(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ra_ln2(x)
        return x


class TinyGPT(nn.Module):
    """A GPT-like toy model (~100K parameters) for quick experiments."""

    def __init__(self, vocab_size=256, seq_len=32, layers=2, d_model=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model))
        self.blocks = nn.ModuleList([TinyTransformerBlock(d_model) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        # idx shape: [batch, seq_len]
        x = self.emb(idx) + self.pos[:idx.size(1)]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def train_one_epoch(model, loader, opt, device, max_steps=100, bits_act=8):
    model.train()
    total_loss = 0.0
    n = 0
    start = time.time()
    
    for step, (x, y) in enumerate(loader):
        if step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if bits_act < 8:
            logits = fake_quantise(logits, bits_act)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n += 1
        print(f"Training step {step}: Loss = {loss.item():.4f}")
    
    elapsed = time.time() - start
    tok_s = (n * x.numel()) / elapsed
    return total_loss / n, tok_s


def train_model(device="cpu", quick=True):
    print("Starting training...")
    loader = get_toy_dataloaders(batch_size=16)
    torch.manual_seed(0)
    model = TinyGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    steps = 30 if quick else 200
    loss_curve = []

    for step in range(steps):
        # Quick curriculum: use full precision activations at first, then drop to 4-bit quantisation
        bits_a = 8 if step < steps // 2 else 4
        loss, tok_s = train_one_epoch(model, loader, optimizer, device, max_steps=1, bits_act=bits_a)
        loss_curve.append(loss)
    
    # Save the trained model
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "tinygpt.pt")
    print("Training completed. Model saved to models/tinygpt.pt")
    
    # Save training loss plot in high-quality PDF
    image_dir = Path(".research/iteration1/images")
    image_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")
    plt.figure(figsize=(4, 3))
    plt.plot(loss_curve, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(image_dir / "training_loss.pdf", bbox_inches="tight")
    plt.close()
    print(f"Training loss plot saved to {image_dir / 'training_loss.pdf'}")
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train TinyGPT toy model")
    parser.add_argument("--full", action="store_true", help="Run full training (longer run)")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(device=device, quick=(not args.full))
