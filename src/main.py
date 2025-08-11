import argparse

# Import training and evaluation modules
from train import train_model
from evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="GNTQ-EFD++ Toy Experiment Pipeline")
    parser.add_argument("--full", action="store_true", help="Run full experiments (longer run)")
    args = parser.parse_args()

    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n=== TRAINING PHASE ===")
    model = train_model(device=device, quick=(not args.full))

    print("\n=== EVALUATION PHASE ===")
    evaluate_model(device=device)

    print("\nExperiment pipeline completed successfully.")


if __name__ == "__main__":
    main()
