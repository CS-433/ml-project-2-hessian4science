from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch import optim
import torch
import os
import argparse

from models import NN
from utils import learn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MNIST")
    parser.add_argument("--hidden", default=128, type=int)
    parser.add_argument("--max_iters", default=1000, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--conv_number", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_path", default="results")
    parser.add_argument("--criterion", default="cross_entropy")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

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

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.criterion == "cross_entropy":
        criterion = torch.nn.functional.cross_entropy
    else:
        raise NotImplementedError

    hidden_layers = [args.hidden] * args.num_layers
    hidden_layers.append(n_class)

    model = NN(input_shape, hidden_layers, activation=args.activation, conv_number=args.conv_number)
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_acc_history, train_loss_history, val_acc_history, val_loss_history, lr_history = learn(model, dataloader,
                                                                                                 test_dataloader,
                                                                                                 optimizer,
                                                                                                 epochs=args.epochs,
                                                                                                 device=device,
                                                                                                 plot=args.plot)
    if args.save:
        torch.save(model.state_dict(), os.path.join(args.save_path,
                                                    f"{args.dataset}_{args.optimizer}_{args.activation}_{args.criterion}_{args.gamma}_{args.lr}_{args.weight_decay}_{args.hidden}_{args.num_layers}_{args.conv_number}.pth"))
