import gc
import os

import numpy as np
import torch
from torch import nn, autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

import optimizers
from models import NN


@torch.no_grad()
def validate(model, device, val_loader, criterion):
    """
    Validate a model
    :param model: model to validate
    :param device: device to use
    :param val_loader: dataloader for validation set
    :param criterion: loss function
    :return: validation loss, validation accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item() * len(data)
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    return test_loss, correct / len(val_loader.dataset)


def train_epoch(model, optimizer, criterion, train_loader, epoch, device, verbose=True, scheduler=None):
    """
    Train a model for one epoch
    :param model: model to train
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param criterion: loss function
    :param train_loader: dataloader for training set
    :param epoch: current epoch
    :param device: device to use
    :param verbose: boolean to print or not
    :return: train loss history, train accuracy history, learning rate history
    """
    model.train()
    # loss_history = []
    # accuracy_history = []
    # lr_history = []
    total_loss = 0.0
    total_accuracy = 0.0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # with autocast(device_type=device):
        #     output = model(data)
        #     loss = criterion(output, target)
        # scaler.scale(loss).backward()
        # optimizer.set_f(model, data, target, criterion)
        # scaler.step(optimizer)
        # scaler.update()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.set_f(model, data, target, criterion)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        accuracy_float = (output.argmax(dim=1) == target).float().mean().item()

        loss_float = loss.item()
        total_loss += loss_float * len(data)
        total_accuracy += accuracy_float * len(data)
        total += len(data)
        # loss_history.append(loss_float)
        #
        # accuracy_history.append(accuracy_float)
        # if scheduler is not None:
        #     lr_history.append(scheduler.get_last_lr()[0])
        # else:
        #     lr_history.append(optimizer.param_groups[0]['lr'])
        if verbose and batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
            if scheduler is None:
                lr = optimizer.param_groups[0]['lr']
            else:
                lr = scheduler.get_last_lr()[0]
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "
                f"batch_loss={loss_float:0.2e} "
                f"batch_acc={accuracy_float:0.3f} "
                f"lr={lr:0.3e} "
            )
    if scheduler is None:
        lr = optimizer.param_groups[0]['lr']
    else:
        lr = scheduler.get_last_lr()[0]
    return total_loss / total, total_accuracy / total, lr


def learn(model, train_loader, val_loader, optimizer, criterion, epochs=10, device="cpu", verbose=True, with_scheduler=True):
    """
    Train a model
    :param model: model to train
    :param train_loader: dataloader for training set
    :param val_loader: dataloader for validation set
    :param optimizer: optimizer
    :param criterion: loss function
    :param epochs: number of epochs
    :param device: device to use
    :param verbose: boolean to print or not
    :param with_scheduler: boolean to use a scheduler or not
    :return: train accuracy history, train loss history, validation accuracy history, validation loss history, learning rate history
    """
    model = model.to(device=device)
    scheduler = None
    if with_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(len(train_loader.dataset) * epochs) // train_loader.batch_size,
        )

    # ===== Train Model =====
    lr_history = []
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    # scaler = GradScaler()
    pbar = tqdm(total=epochs, unit="epochs")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, lr = train_epoch(
            model, optimizer, criterion, train_loader, epoch, device, verbose=verbose, scheduler=scheduler
        )

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        lr_history.append(lr)

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        pbar.update(1)
        pbar.set_description_str(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        torch.cuda.empty_cache()
        gc.collect()
        if verbose:
            print(torch.cuda.memory_summary(device=device))
    pbar.close()

    return train_acc_history, train_loss_history, val_acc_history, val_loss_history, lr_history


def cross_validation(lrs, optimizers_, num_layers, conv_numbers,
                     dataloader, val_dataloader, test_dataloader, input_shape, n_class, device='cpu', args=None,
                     verbose=True):
    """
        Cross validation for the best hyperparameters
        :param lrs: list of learning rates
        :param optimizers_: list of optimizers
        :param num_layers: list of number of layers
        :param conv_numbers: list of number of convolutional layers
        :param dataloader: dataloader for training set
        :param val_dataloader: dataloader for validation set
        :param test_dataloader: dataloader for test set
        :param n_class: number of classes
        :param input_shape: shape of the input
        :param device: device to use
        :param args: arguments
        :param verbose: boolean to print or not
        :return: best hyperparameters
        """
    best_acc = 0
    best_lr = 0
    best_opt = None
    best_num_layers = 0
    best_conv_number = 0
    save_path = os.path.join(args.save_path, f"{args.dataset}/")
    os.makedirs(save_path, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    for num_layer in num_layers:
        for conv_number in conv_numbers:
            for lr in lrs:
                print(f"lr: {lr}, num_layer: {num_layer}, conv_number: {conv_number}")
                train_acc_history_list = []
                train_loss_history_list = []
                val_acc_history_list = []
                val_loss_history_list = []
                lr_history_list = []
                test_acc_history = []
                test_loss_history = []
                for opt in optimizers_:
                    print(f"Optimizer: {opt}")
                    hidden_layers = [args.hidden] * num_layer
                    hidden_layers.append(n_class)
                    model = NN(input_shape, hidden_layers, activation=args.activation, conv_number=conv_number)
                    optimizer = getattr(optimizers, opt)(model.parameters(), lr=lr)
                    (train_acc_history,
                     train_loss_history, val_acc_history,
                     val_loss_history,
                     lr_history) = learn(model,
                                         dataloader,
                                         val_dataloader,
                                         optimizer,
                                         criterion,
                                         epochs=args.epochs,
                                         device=device,
                                         verbose=verbose,
                                         with_scheduler=args.scheduler)
                    train_acc_history_list.append(train_acc_history)
                    train_loss_history_list.append(train_loss_history)
                    val_acc_history_list.append(val_acc_history)
                    val_loss_history_list.append(val_loss_history)
                    lr_history_list.append(lr_history)
                    test_loss, test_acc = validate(model, device, test_dataloader, criterion)
                    test_acc_history.append(test_acc)
                    test_loss_history.append(test_loss)

                    if val_acc_history[-1] > best_acc:
                        best_acc = val_acc_history[-1]
                        best_lr = lr
                        best_opt = opt
                        best_num_layers = num_layer
                        best_conv_number = conv_number
                    if args.save:
                        torch.save(model.state_dict(),
                                   os.path.join(save_path, f"{lr}_{num_layer}_{conv_number}_{opt}_{args.scheduler}.pt"))
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                    if verbose:
                        print(f"Test Acc for {opt}: {test_acc:0.2f}%")
                        print(torch.cuda.memory_summary(device=device))

                if args.plot:
                    n_train = len(train_acc_history_list[0])
                    t_train = args.epochs * np.arange(n_train) / n_train
                    t_val = np.arange(1, args.epochs + 1)

                    for i, opt in enumerate(optimizers_):
                        plt.plot(t_train, train_acc_history_list[i], label=opt)
                    plt.legend()
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.savefig(os.path.join(save_path, f"training_curves_{lr}_{num_layer}_{conv_number}_{args.scheduler}.png"))
                    plt.close()

                    for i, opt in enumerate(optimizers_):
                        plt.plot(t_train, train_loss_history_list[i], label=opt)
                    plt.legend()
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.savefig(os.path.join(save_path, f"loss_curves_{lr}_{num_layer}_{conv_number}_{args.scheduler}.png"))
                    plt.close()

                    for i, opt in enumerate(optimizers_):
                        plt.plot(t_train, lr_history_list[i], label=opt)
                    plt.legend()
                    plt.xlabel("Epoch")
                    plt.ylabel("Learning Rate")
                    plt.savefig(os.path.join(save_path, f"lr_curves_{lr}_{num_layer}_{conv_number}_{args.scheduler}.png"))
                    plt.close()

                    for i, opt in enumerate(optimizers_):
                        plt.plot(t_val, val_acc_history_list[i], label=opt)
                    plt.legend()
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.savefig(os.path.join(save_path, f"val_acc_curves_{lr}_{num_layer}_{conv_number}_{args.scheduler}.png"))
                    plt.close()

                    for i, opt in enumerate(optimizers_):
                        plt.plot(t_val, val_loss_history_list[i], label=opt)
                    plt.legend()
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.savefig(os.path.join(save_path, f"val_loss_curves_{lr}_{num_layer}_{conv_number}_{args.scheduler}.png"))
                    plt.close()

    return best_lr, best_opt, best_num_layers, best_conv_number
