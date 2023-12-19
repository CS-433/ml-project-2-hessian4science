# Import necessary libraries
from torch.utils.data import DataLoader
import torch
import argparse
from dataset import process_dataset

# Import custom modules
from utils import model_selection, learn_models

# Main function
if __name__ == "__main__":
    """
    Main function to run the script. It parses command-line arguments, loads the dataset, splits it into training and validation sets,
    creates data loaders, sets the loss function, and performs cross-validation to find the best hyperparameters and optimizer.
    """
    # Create argument parser
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument("--batch_size", default=32, type=int, help="The batch size for training.")
    parser.add_argument("--epochs", default=2, type=int, help="The number of epochs to train for.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the training and validation curves.")
    parser.add_argument("--lr", default="0.001,0.001", help="The list of learning rates for the optimizers.")
    parser.add_argument("--optimizer", default="SCRN,SCRN_Momentum",
                        help="The list of optimizers to use for training.")
    parser.add_argument("--save", action="store_true", help="Whether to save the trained model.")
    parser.add_argument("--save_path", default="./results/",
                        help="The directory where the trained model should be saved.")
    parser.add_argument("--criterion", default="cross_entropy", help="The loss function to use for training.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print detailed training progress.")
    parser.add_argument("--scheduler", action="store_true", help="Whether to use a learning rate scheduler.")
    parser.add_argument("--model_selection", action="store_true", help="Whether to perform model_selection.")

    # Parse the arguments
    args = parser.parse_args()

    # Convert string arguments to appropriate data types
    lrs = [float(lr) for lr in args.lr.split(",")]
    print(f"Learning rates: {lrs}")
    optimizers_ = args.optimizer.split(",")
    print(f"Optimizers: {optimizers_}")

    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    print(f"Scheduler: {args.scheduler}")
    

    datasets = 


    # Create data loaders for the training, validation, and test sets
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.model_selection:
        # Perform model selection to find the best hyperparameters
        model_selection(lrs, optimizers_,
                        dataloader, val_dataloader,
                        test_dataloader, device=device,
                        args=args, verbose=args.verbose)

    else:
        # Train the models
        learn_models(lrs, optimizers_,
                     dataloader, val_dataloader,
                     test_dataloader, device=device,
                     args=args, verbose=args.verbose)