import argparse
import os

# Import our modules
import preprocess
import train
import evaluate


def main(quick=True):
    print('Starting the full experimental pipeline...')
    # Step 1: Preprocess data
    print('Running preprocessing...')
    preprocess.preprocess_data()
    
    # Step 2: Train model
    print('Starting training...')
    model = train.train_model(quick=quick)
    
    # Step 3: Evaluate model
    print('Starting evaluation...')
    evaluate.evaluate_model()
    
    print('Experiment completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full GNTQ-EFD++ toy experiment')
    parser.add_argument('--quick', action='store_true', help='Run the quick test version (default)')
    parser.add_argument('--full', action='store_true', help='Run the full experiment (longer)')
    args = parser.parse_args()

    # If full flag is provided, override quick
    quick = False if args.full else True

    main(quick=quick)
