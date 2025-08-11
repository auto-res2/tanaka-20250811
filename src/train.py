import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Ensure required directories exist
MODEL_DIR = os.path.join(os.getcwd(), 'models')
IMAGE_DIR = os.path.join(os.getcwd(), '.research', 'iteration1', 'images')
DATA_DIR = os.path.join(os.getcwd(), 'data')
for d in [MODEL_DIR, IMAGE_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------------
# Model and Helper Definitions
# -----------------------------

class HierarchicalCodebookLinear(nn.Module):
    """Linear layer with 2-bit main + 1-bit residue (3 effective bits)."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.main_index = nn.Parameter(torch.randint(-2, 3, (out_f, in_f), dtype=torch.int8))
        self.residue_index = nn.Parameter(torch.randint(-1, 2, (out_f, in_f), dtype=torch.int8))
        self.scale = nn.Parameter(torch.full((1,), 1e-1))

    def forward(self, x):
        w_int = self.main_index.to(torch.float32) + self.residue_index.to(torch.float32) * 0.25
        w = w_int * self.scale
        return F.linear(x, w)

class RA_LayerNorm(nn.Module):
    """Range-Aware Residual LayerNorm (RA-RLN) – tiny variant."""

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.ln(x + self.beta)

class TinyTransformerBlock(nn.Module):
    """A 2-head miniature transformer block."""

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
    """GPT-like toy model – suitable for quick training tests."""

    def __init__(self, vocab_size=256, seq_len=32, layers=2, d_model=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(seq_len, d_model))
        self.blocks = nn.ModuleList([TinyTransformerBlock(d_model) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        x = self.emb(idx) + self.pos[:idx.size(1)]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def fake_quantise(x: torch.Tensor, bits: int, symmetric: bool = True):
    """Uniform quantisation with STE backward (for ≤8 bits)."""
    qmin = -(2 ** (bits - 1)) if symmetric else 0
    qmax = (2 ** (bits - 1)) - 1 if symmetric else 2 ** bits - 1
    scale = x.detach().abs().max() / qmax + 1e-8

    def _quantise(y):
        y_div = y / scale
        y_clamped = y_div.clamp(qmin, qmax).round()
        return y_clamped * scale

    return _quantise(x) + (x - _quantise(x)).detach()


def make_toy_dataloader(seq_len=32, vocab=256, num_tokens=10000, batch_size=16):
    """Generate toy data loader from synthetic data."""
    torch.manual_seed(0)
    data = torch.randint(0, vocab, (num_tokens,))
    x = data.unfold(0, seq_len, seq_len)[:-1]
    y = data.unfold(0, seq_len, seq_len)[1:]
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_one_epoch(model, loader, opt, device, max_steps=100, bits_act=8):
    model.train()
    total_loss, n = 0.0, 0
    start = time.time()
    for step, (x, y) in enumerate(loader):
        if step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if bits_act < 8:
            logits = fake_quantise(logits, bits_act)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


def train_model(quick=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    # Use toy data loader; alternatively load from preprocessed data
    loader = make_toy_dataloader(batch_size=16)
    model = TinyGPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    steps = 30 if quick else 200
    loss_curve = []
    for step in range(steps):
        # For curriculum: use full precision activations first, then drop bits later
        bits_a = 8 if step < steps // 2 else 4
        loss = train_one_epoch(model, loader, opt, device, max_steps=1, bits_act=bits_a)
        loss_curve.append(loss)
        print(f"Step {step+1}/{steps} - Loss: {loss:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot training loss curve
    sns.set(style='whitegrid')
    plt.figure(figsize=(4,3))
    plt.plot(loss_curve, label='training loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(IMAGE_DIR, 'training_loss.pdf')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Training loss plot saved to {plot_path}")
    
    return model

if __name__ == '__main__':
    train_model(quick=True)
