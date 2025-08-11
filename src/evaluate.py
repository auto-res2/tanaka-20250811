import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Import dataloader from preprocess module and model from train module
from preprocess import get_toy_dataloaders
from train import TinyGPT


def eval_perplexity(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            n += 1
    ppl = np.exp(total_loss / n)
    return ppl


def evaluate_model(device="cpu"):
    print("Starting evaluation...")
    loader = get_toy_dataloaders(batch_size=16)
    model = TinyGPT().to(device)
    model_path = Path("models/tinygpt.pt")
    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print("Model loaded from models/tinygpt.pt")
    else:
        print("Model not found. Please run the training script first.")
        return
    
    ppl = eval_perplexity(model, loader, device)
    print(f"Model Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TinyGPT toy model")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_model(device=device)
