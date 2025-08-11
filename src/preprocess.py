import torch
from torch.utils.data import DataLoader, TensorDataset


def get_toy_dataloaders(seq_len=32, vocab=256, num_tokens=10000, batch_size=32):
    """Generates a toy dataloader with two data patterns for robustness tests."""
    rng = torch.Generator().manual_seed(0)
    data_a = torch.randint(0, vocab, (num_tokens,), generator=rng)
    data_b = (torch.arange(num_tokens) % vocab).to(torch.long)
    data = torch.stack([data_a, data_b]).flatten()[:num_tokens]
    x = data.unfold(0, seq_len, seq_len)[:-1]
    y = data.unfold(0, seq_len, seq_len)[1:]
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader


if __name__ == "__main__":
    # Quick test to verify dataloader works
    loader = get_toy_dataloaders()
    batch = next(iter(loader))
    print(f"Batch shapes: {[t.shape for t in batch]}")
