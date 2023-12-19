import copy
import gc
import os
import csv

import numpy as np
import torch
from torch import nn
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
        if optimizer.name == 'SCRN' or optimizer.name == 'SCRN_Momentum':
            ind = len(data) // 2
            data_for_grad = data[:ind]
            target_for_grad = target[:ind]
            data_for_hessian = data[ind:]
            target_for_hessian = target[ind:]

            optimizer.zero_grad()
            optimizer.set_f(model, data_for_hessian, target_for_hessian, criterion)
            output = model(data_for_grad)
            loss = criterion(output, target_for_grad)
            loss.backward()
            optimizer.step()
            accuracy_float = (output.argmax(dim=1) == target_for_grad).float().mean().item()

            loss_float = loss.item()
            total_loss += loss_float * len(data_for_grad)
            total_accuracy += accuracy_float * len(data_for_grad)
            total += len(data_for_grad)
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            accuracy_float = (output.argmax(dim=1) == target).float().mean().item()

            loss_float = loss.item()
            total_loss += loss_float * len(data)
            total_accuracy += accuracy_float * len(data)
            total += len(data)

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
        if scheduler is not None:
            scheduler.step()
    if scheduler is None:
        lr = optimizer.param_groups[0]['lr']
    else:
        lr = scheduler.get_last_lr()[0]
    return total_loss / total, total_accuracy / total, lr
    # return loss_history, accuracy_history, lr_history


def learn(model, train_loader, val_loader, optimizer, criterion, epochs=10, device="cpu", verbose=True,
          with_scheduler=True):
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

    print("Initial validation")
    initial_train_loss, initial_train_acc = validate(model, device, train_loader, criterion)
    print(f"Initial train loss: {initial_train_loss:.4f}, Initial train acc: {initial_train_acc * 100:.2f}%")
    initial_valid_loss, initial_valid_acc = validate(model, device, val_loader, criterion)
    print(f"Initial valid loss: {initial_valid_loss:.4f}, Initial valid acc: {initial_valid_acc * 100:.2f}%")
    train_loss_history = [initial_train_loss]
    train_acc_history = [initial_train_acc]
    val_loss_history = [initial_valid_loss]
    val_acc_history = [initial_valid_acc]
    # scaler = GradScaler()
    pbar = tqdm(total=epochs, unit="epochs")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, lr = train_epoch(
            model, optimizer, criterion, train_loader, epoch, device, verbose=verbose, scheduler=scheduler
        )

        # train_loss_history.extend(train_loss)
        train_loss_history.append(train_loss)
        # train_acc_history.extend(train_acc)
        train_acc_history.append(train_acc)
        # lr_history.extend(lr)
        lr_history.append(lr)
        train_loss_avg = np.mean(train_loss)
        train_acc_avg = np.mean(train_acc) * 100

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        pbar.update(1)
        pbar.set_description_str(
            f"Epoch {epoch}/{epochs}, Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

    pbar.close()

    return train_acc_history, train_loss_history, val_acc_history, val_loss_history, lr_history


def model_selection(lrs, optimizers_, num_layers, conv_numbers,
                    dataloader, val_dataloader, test_dataloader, input_shape, n_class, device='cpu', args=None,
                    verbose=True):
    """
        Model selection for the best hyperparameters
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
        """

    save_path = os.path.join(args.save_path, f"{args.dataset}/")
    os.makedirs(save_path, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    for num_layer in num_layers:
        for conv_number in conv_numbers:
            print(f"num_layer: {num_layer}, conv_number: {conv_number}")
            train_acc_history_list = []
            train_loss_history_list = []
            val_acc_history_list = []
            val_loss_history_list = []
            lr_history_list = []

            for opt in optimizers_:
                best_loss = np.inf
                best_lr = 0
                best_train_acc_history = []
                best_train_loss_history = []
                best_val_acc_history = []
                best_val_loss_history = []
                best_lr_history = []
                best_test_acc = 0
                best_test_loss = 0

                hidden_layers = [args.hidden] * num_layer
                hidden_layers.append(n_class)
                base_model = NN(input_shape, hidden_layers, activation=args.activation, conv_number=conv_number)
                for lr in lrs:
                    print(f"Optimizer: {opt}")
                    model = copy.deepcopy(base_model)
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
                    test_loss, test_acc = validate(model, device, test_dataloader, criterion)
                    if args.save:
                        torch.save(model.state_dict(),
                                   os.path.join(save_path, f"{lr}_{num_layer}_{conv_number}_{opt}_{args.scheduler}.pt"))

                    if val_loss_history[-1] < best_loss:
                        best_loss = val_acc_history[-1]
                        best_lr = lr
                        best_train_acc_history = train_acc_history
                        best_train_loss_history = train_loss_history
                        best_val_acc_history = val_acc_history
                        best_val_loss_history = val_loss_history
                        best_lr_history = lr_history
                        best_test_acc = test_acc
                        best_test_loss = test_loss
                    model.zero_grad()
                    del model
                    del optimizer
                    torch.cuda.empty_cache()
                    gc.collect()
                    if verbose:
                        print(f"The test accuracy: {test_acc}, the test loss: {test_loss}")
                        print(f"memory_allocated: {torch.cuda.memory_allocated(device=device) / 1024 ** 3} GB")
                        print(f"memory_reserved: {torch.cuda.memory_reserved(device=device) / 1024 ** 3} GB")
                print(
                    f"Best learning rate: {best_lr}, Best test accuracy: {best_test_acc}, Best test loss: {best_test_loss}")
                train_acc_history_list.append(best_train_acc_history)
                train_loss_history_list.append(best_train_loss_history)
                val_acc_history_list.append(best_val_acc_history)
                val_loss_history_list.append(best_val_loss_history)
                lr_history_list.append(best_lr_history)

            if args.plot:
                n_train = len(train_acc_history_list[0])
                t_train = args.epochs * np.arange(n_train) / n_train
                t_val = np.arange(1, args.epochs + 1)
                min_loss = min([min(loss) for loss in val_loss_history_list]+[min(loss) for loss in train_loss_history_list])
                max_loss = max([max(loss) for loss in val_loss_history_list]+[max(loss) for loss in train_loss_history_list])

                for i, opt in enumerate(optimizers_):
                    plt.plot(t_train, train_acc_history_list[i], label=opt)
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.savefig(
                    os.path.join(save_path, f"training_curves_{num_layer}_{conv_number}_{args.scheduler}.png"))
                plt.close()

                for i, opt in enumerate(optimizers_):
                    plt.plot(t_train, train_loss_history_list[i], label=opt)
                plt.legend()
                plt.ylim(min_loss, max_loss)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(
                    os.path.join(save_path, f"loss_curves_{num_layer}_{conv_number}_{args.scheduler}.png"))
                plt.close()

                for i, opt in enumerate(optimizers_):
                    plt.plot(t_train, lr_history_list[i], label=opt)
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Learning Rate")
                plt.savefig(
                    os.path.join(save_path, f"lr_curves_{num_layer}_{conv_number}_{args.scheduler}.png"))
                plt.close()

                for i, opt in enumerate(optimizers_):
                    plt.plot(t_val, val_acc_history_list[i], label=opt)
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.savefig(
                    os.path.join(save_path, f"val_acc_curves_{num_layer}_{conv_number}_{args.scheduler}.png"))
                plt.close()

                for i, opt in enumerate(optimizers_):
                    plt.plot(t_val, val_loss_history_list[i], label=opt)
                plt.legend()
                plt.ylim(min_loss, max_loss)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(
                    os.path.join(save_path, f"val_loss_curves_{num_layer}_{conv_number}_{args.scheduler}.png"))
                plt.close()


def learn_models(lrs, optimizers_, num_layers, conv_numbers,
                 dataloader, val_dataloader, test_dataloader, input_shape, n_class, device='cpu', args=None,
                 verbose=True, num_iter=5):
    """
    Learn models
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
        """

    save_path = os.path.join(args.save_path, f"{args.dataset}/")
    os.makedirs(save_path, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    lr_dict = {opt: lr for opt, lr in zip(optimizers_, lrs)}
    for num_layer in num_layers:
        for conv_number in conv_numbers:
            print(f"num_layer: {num_layer}, conv_number: {conv_number}")
            for i in range(num_iter):
                print(f"num_iter: {i}")
                hidden_layers = [args.hidden] * num_layer
                hidden_layers.append(n_class)
                base_model = NN(input_shape, hidden_layers, activation=args.activation, conv_number=conv_number)
                train_acc_history_list = []
                train_loss_history_list = []
                val_acc_history_list = []
                val_loss_history_list = []
                lr_history_list = []

                for opt in optimizers_:
                    print(f"Optimizer: {opt}")
                    model = copy.deepcopy(base_model)
                    optimizer = getattr(optimizers, opt)(model.parameters(), lr=lr_dict[opt])
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
                    test_loss, test_acc = validate(model, device, test_dataloader, criterion)

                    train_acc_history_list.append(train_acc_history)
                    train_loss_history_list.append(train_loss_history)
                    val_acc_history_list.append(val_acc_history)
                    val_loss_history_list.append(val_loss_history)
                    lr_history_list.append(lr_history)
                    if args.save:
                        file_name = f"history_lin:{num_layer}_conv:{conv_number}_iter:{i}_opt:{opt}_sch:{args.scheduler}.csv"

                        with open(file_name, mode='w', newline='') as file:
                            writer = csv.writer(file)

                            writer.writerow(["train_acc"," train_loss", "val_acc", "val_loss"])
                            for row in zip(train_acc_history, train_loss_history, val_acc_history, val_loss_history):
                                writer.writerow(row)
    
                        torch.save(model.state_dict(),
                                   os.path.join(save_path, f"lin:{num_layer}_conv:{conv_number}_iter:{i}_opt:{opt}_sch:{args.scheduler}.pt"))

                    model.zero_grad()
                    del model
                    del optimizer
                    torch.cuda.empty_cache()
                    gc.collect()
                    if verbose:
                        print(f"The test accuracy: {test_acc}, the test loss: {test_loss}")
                        print(f"memory_allocated: {torch.cuda.memory_allocated(device=device) / 1024 ** 3} GB")
                        print(f"memory_reserved: {torch.cuda.memory_reserved(device=device) / 1024 ** 3} GB")

                    if args.plot:
                        n_train = len(train_acc_history_list[0])
                        t_train = args.epochs * np.arange(n_train) / n_train
                        t_val = np.arange(0, args.epochs + 1)

                        for i, opt in enumerate(optimizers_):
                            plt.plot(t_train, train_acc_history_list[i], label=opt)
                        plt.legend()
                        plt.xlabel("Epoch")
                        plt.ylabel("Accuracy")
                        plt.savefig(
                            os.path.join(save_path, f"training_curves_{num_layer}_{conv_number}_iter:{i}_{args.scheduler}.png"))
                        plt.close()

                        for i, opt in enumerate(optimizers_):
                            plt.plot(t_train, train_loss_history_list[i], label=opt)
                        plt.legend()
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.savefig(
                            os.path.join(save_path, f"loss_curves_{num_layer}_{conv_number}_iter:{i}_{args.scheduler}.png"))
                        plt.close()

                        for i, opt in enumerate(optimizers_):
                            plt.plot(t_train, lr_history_list[i], label=opt)
                        plt.legend()
                        plt.xlabel("Epoch")
                        plt.ylabel("Learning Rate")
                        plt.savefig(
                            os.path.join(save_path, f"lr_curves_{num_layer}_{conv_number}_iter:{i}_{args.scheduler}.png"))
                        plt.close()

                        for i, opt in enumerate(optimizers_):
                            plt.plot(t_val, val_acc_history_list[i], label=opt)
                        plt.legend()
                        plt.xlabel("Epoch")
                        plt.ylabel("Accuracy")
                        plt.savefig(
                            os.path.join(save_path, f"val_acc_curves_{num_layer}_{conv_number}_iter:{i}_{args.scheduler}.png"))
                        plt.close()

                        for i, opt in enumerate(optimizers_):
                            plt.plot(t_val, val_loss_history_list[i], label=opt)
                        plt.legend()
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.savefig(
                            os.path.join(save_path, f"val_loss_curves_{num_layer}_{conv_number}_iter:{i}_{args.scheduler}.png"))
                        plt.close()
