import os
import torch
from torch.utils.data import TensorDataset

DATA_DIR = os.path.join(os.getcwd(), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def preprocess_data(seq_len=32, vocab=256, num_tokens=10000):
    """Generates synthetic data and saves preprocessed tensors."""
    torch.manual_seed(0)
    # Generate a simple sequence of random tokens
    data = torch.randint(0, vocab, (num_tokens,))
    # Create overlapping sequences
    xs = data.unfold(0, seq_len, seq_len)[:-1]
    ys = data.unfold(0, seq_len, seq_len)[1:]
    dataset = TensorDataset(xs, ys)
    data_path = os.path.join(DATA_DIR, 'preprocessed_data.pt')
    torch.save({'xs': xs, 'ys': ys}, data_path)
    print(f"Preprocessed data saved to {data_path}")
    return dataset


if __name__ == '__main__':
    preprocess_data()
