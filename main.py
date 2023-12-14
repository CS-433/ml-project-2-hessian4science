# Import necessary libraries
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import argparse

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
    parser.add_argument("--dataset", default="MNIST", help="The dataset to use for training and testing.")
    parser.add_argument("--hidden", default=128, type=int, help="The number of hidden units in the model.")
    parser.add_argument("--num_layers", default="3", help="The list of numbers of layers in the model.")
    parser.add_argument("--conv_number", default="3", help="The list of numbers of convolutional layers in the model.")
    parser.add_argument("--batch_size", default=128, type=int, help="The batch size for training.")
    parser.add_argument("--epochs", default=20, type=int, help="The number of epochs to train for.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the training and validation curves.")
    parser.add_argument("--lr", default="0.001", help="The list of learning rates for the optimizers.")
    parser.add_argument("--optimizer", default="SGD",
                        help="The list of optimizers to use for training.")
    parser.add_argument("--activation", default="relu", help="The activation function to use in the model.")
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
    num_layers = [int(layer) for layer in args.num_layers.split(",")]
    print(f"Number of layers: {num_layers}")
    conv_numbers = [int(conv) for conv in args.conv_number.split(",")]
    print(f"Number of convolutional layers: {conv_numbers}")
    # Check if CUDA is available and set the device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    print(f"Scheduler: {args.scheduler}")

    # Load the appropriate dataset based on the argument
    if args.dataset == "MNIST":
        transformer_mnist = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,))
        ])
        dataset = MNIST("data", train=True, download=True, transform=transformer_mnist)
        test_dataset = MNIST("data", train=False, download=True, transform=transformer_mnist)
        input_shape = [1, 28, 28]
        n_class = 10
    elif args.dataset == "CIFAR10":
        transformer_cifar = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CIFAR10("data", train=True, download=True, transform=transformer_cifar)
        test_dataset = CIFAR10("data", train=False, download=True, transform=transformer_cifar)
        input_shape = [3, 32, 32]
        n_class = 10
    elif args.dataset == "CIFAR100":
        transformer_cifar = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CIFAR100("data", train=True, download=True, transform=transformer_cifar)
        test_dataset = CIFAR100("data", train=False, download=True, transform=transformer_cifar)
        input_shape = [3, 32, 32]
        n_class = 100
    else:
        raise NotImplementedError

    # Split the dataset into training and validation sets
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders for the training, validation, and test sets
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.model_selection:
        # Perform model selection to find the best hyperparameters
        model_selection(lrs, optimizers_, num_layers,
                        conv_numbers,
                        dataloader, val_dataloader,
                        test_dataloader, input_shape,
                        n_class, device=device,
                        args=args, verbose=args.verbose)

    else:
        # Train the models
        learn_models(lrs, optimizers_, num_layers,
                     conv_numbers,
                     dataloader, val_dataloader,
                     test_dataloader, input_shape,
                     n_class, device=device,
                     args=args, verbose=args.verbose)
