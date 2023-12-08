from functools import partial

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


@torch.no_grad()
def validate(model, device, val_loader, criterion, verbose=True):
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
    if verbose:
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss,
                correct,
                len(val_loader.dataset),
                100.0 * correct / len(val_loader.dataset),
            )
        )
    return test_loss, correct / len(val_loader.dataset)


def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device, verbose=True):
    model.train()
    loss_history = []
    accuracy_history = []
    lr_history = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        accuracy_float = (output.argmax(dim=1) == target).float().mean().item()

        loss_float = loss.item()
        loss_history.append(loss_float)

        accuracy_history.append(accuracy_float)
        lr_history.append(scheduler.get_last_lr()[0])
        if verbose and batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "
                f"batch_loss={loss_float:0.2e} "
                f"batch_acc={accuracy_float:0.3f} "
                f"lr={scheduler.get_last_lr()[0]:0.3e} "
            )

    return loss_history, accuracy_history, lr_history


def learn(model, train_loader, val_loader, optimizer, epochs=10, device="cpu", plot=True, verbose=True):
    model = model.to(device=device)
    criterion = torch.nn.functional.cross_entropy
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
    pbar = tqdm(total=epochs, unit="epochs")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, lrs = train_epoch(
            model, optimizer, scheduler, criterion, train_loader, epoch, device, verbose=verbose
        )
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        lr_history.extend(lrs)

        val_loss, val_acc = validate(model, device, val_loader, criterion, verbose=verbose)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        pbar.update(1)
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train_acc = sum(train_acc) / len(train_acc)
        pbar.set_description_str(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%")
    pbar.close()

    # ===== Plot training curves =====
    if plot:
        n_train = len(train_acc_history)
        t_train = epochs * np.arange(n_train) / n_train
        t_val = np.arange(1, epochs + 1)

        plt.figure(figsize=(6.4 * 3, 4.8))
        plt.subplot(1, 3, 1)
        plt.plot(t_train, train_acc_history, label="Train")
        plt.plot(t_val, val_acc_history, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.subplot(1, 3, 2)
        plt.plot(t_train, train_loss_history, label="Train")
        plt.plot(t_val, val_loss_history, label="Val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 3, 3)
        plt.plot(t_train, lr_history)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")

        plt.savefig('training_curves.png')
        plt.close()

    return train_acc_history, train_loss_history, val_acc_history, val_loss_history, lr_history
