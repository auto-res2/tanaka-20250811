import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Ensure required directories exist
MODEL_DIR = os.path.join(os.getcwd(), 'models')
IMAGE_DIR = os.path.join(os.getcwd(), '.research', 'iteration1', 'images')
DATA_DIR = os.path.join(os.getcwd(), 'data')
for d in [MODEL_DIR, IMAGE_DIR, DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# Re-define TinyGPT and helper functions (must match training definitions)

class HierarchicalCodebookLinear(nn.Module):
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
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.ln(x + self.beta)

class TinyTransformerBlock(nn.Module):
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


def make_toy_dataloader(seq_len=32, vocab=256, num_tokens=10000, batch_size=16):
    torch.manual_seed(0)
    data = torch.randint(0, vocab, (num_tokens,))
    x = data.unfold(0, seq_len, seq_len)[:-1]
    y = data.unfold(0, seq_len, seq_len)[1:]
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def eval_perplexity(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            n += 1
    ppl = np.exp(total_loss / n)
    return ppl


def evaluate_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on device: {device}")
    # Load preprocessed data if available, otherwise generate toy data
    loader = make_toy_dataloader(batch_size=16)
    model = TinyGPT().to(device)
    model_path = os.path.join(MODEL_DIR, 'model.pt')
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Please run train.py first.")
        return
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    ppl = eval_perplexity(model, loader, device)
    print(f"Evaluation completed. Perplexity: {ppl:.2f}")

    # Create a simple bar plot for perplexity
    sns.set(style='whitegrid')
    plt.figure(figsize=(3,2.5))
    plt.bar(['Perplexity'], [ppl], color='skyblue')
    plt.ylabel('Perplexity')
    plt.tight_layout()
    plot_path = os.path.join(IMAGE_DIR, 'evaluation.pdf')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Evaluation plot saved to {plot_path}")

if __name__ == '__main__':
    evaluate_model()
