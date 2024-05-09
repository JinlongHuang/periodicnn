import os
import wandb
import json
from time import time
from itertools import chain

import torch
import torch.optim as optim

import models
from dataloader import load_data


def train(model, data, loss_func, device, optimizer):
    model.train()
    total_loss = 0
    for X, Y in data:
        X, Y = X.to(device), Y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_func(pred, Y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data)
    return avg_loss


def validate_crypto(model, data, device):
    model.eval()
    pnl = []
    preds = []
    targets = []
    with torch.no_grad():
        for X, Y in data:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            pnl.append(_val_pnl(pred, Y))
            preds.append(list(pred.cpu().squeeze().numpy()))
            targets.append(list(Y.cpu().squeeze().numpy()))
    pnl = list(chain.from_iterable(pnl))
    preds = list(chain.from_iterable(preds))
    targets = list(chain.from_iterable(targets))

    return pnl, preds, targets


def validate(model, data, loss_func, device, checkpoint):
    model.eval()
    total_loss = 0
    preds = []
    with torch.no_grad():
        for X, Y in data:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = loss_func(pred, Y)
            total_loss += loss.item()

            preds.append(list(pred.cpu().squeeze().numpy()))
    avg_loss = total_loss / len(data)
    checkpoint["val_preds"] = preds  # only save the last ts_index preds
    return avg_loss


def test(model, data, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, Y in data:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            predictions.append(pred.cpu())
    return predictions


def _neg_pnl_loss(pred, y):
    """pred, y: (batch_size, out_seq_len)"""
    if pred.shape != y.shape:
        raise ValueError("pred and y must have the same shape")
    return -torch.mean(torch.sum(pred * y, dim=1))


def _val_pnl(pred, y):
    """pred, y: (batch_size, out_seq_len)"""
    if pred.shape != y.shape:
        raise ValueError("pred and y must have the same shape")
    return list(torch.sum(pred * y, dim=1).numpy())


def run_crypto(i: int, use_wandb: bool):
    if use_wandb:
        wandb.init(project='BNN', entity='edgesky',
                   name=f'real-data-seed-{i}', reinit=True)

    with open("config.json") as f:
        config = json.load(f)
        in_seq_len = config["data"]["in_seq_len"]
        out_seq_len = config["data"]["out_seq_len"]
        assets = config["data"]["ts_for_preprocess"]

        num_epochs = config["train"]["num_epochs"]
        learning_rate = config["train"]["learning_rate"]
        saved_epoch = config["train"]["saved_epoch"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.AdaFunc(in_seq_len, out_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    starting_epoch = 0

    # Load saved checkpoints into model and optimizer
    if saved_epoch >= 0:
        checkpoint_file = f"checkpoints/OneLinear_epoch_{saved_epoch}.pt"
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            starting_epoch = checkpoint["epoch"] + 1
            sum_val_ret = -sum(checkpoint["val_loss"])
            print(
                f"Epoch {checkpoint['epoch']:>5}, "
                f"Train Loss: {checkpoint['train_loss']:>10.5f},  "
                f"Total Val PnL: {sum_val_ret:>10.5f}"
            )

    train_data, val_data, test_data = load_data()

    train_start_time = time()
    for epoch in range(starting_epoch, num_epochs):
        for entry in os.scandir("data/"):
            # Select assets to train
            if not entry.is_dir():
                continue
            asset = entry.name[:-4]
            if asset not in assets:
                continue
            if len(train_data[asset]) == 0 or len(val_data[asset]) == 0:
                continue

            train_loss = train(model, train_data[asset], device, optimizer)
            val_pnl, val_preds, val_targets = validate_crypto(
                    model, val_data[asset], device)
            total_val_pnl = sum(val_pnl)

        # Log metrics and save checkpoint
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:>5},      "
                f"Train Loss: {train_loss:>10.5f},  "
                f"Total Val PnL: {total_val_pnl:>10.5f}"
            )
            if use_wandb:
                wandb.log({
                           'Train Loss': train_loss,
                           'Total Val PnL': total_val_pnl
                           })

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_pnl": val_pnl,
                "val_preds": val_preds,
                "val_targets": val_targets,
            }
            torch.save(checkpoint, f"checkpoints/_epoch_{epoch}.pt")

    if use_wandb:
        wandb.finish()
    print(f"Training finished in {time() - train_start_time:.2f} seconds")


def run_monash():
    with open("config.json") as f:
        config = json.load(f)
        in_seq_len = config["data"]["in_seq_len"]
        out_seq_len = config["data"]["out_seq_len"]
        ts_name_list = config["data"]["ts_for_preprocess"]

        model_name = config["train"]["model_name"]
        num_epochs = config["train"]["num_epochs"]
        learning_rate = config["train"]["learning_rate"]
        loss_func_str = config["train"]["loss_func"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if loss_func_str == "pnl":
        loss_func = _neg_pnl_loss
    elif loss_func_str == "mse":
        loss_func = torch.nn.MSELoss()

    if model_name == 'OneLinear':
        model = models.OneLinear(in_seq_len, out_seq_len).to(device)
    elif model_name == 'AdaFunc':
        model = models.AdaFunc(in_seq_len, out_seq_len).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    starting_epoch = 0

    train_data, val_data, test_data, zero_shot_data = load_data()
    train_start_time = time()
    for epoch in range(starting_epoch, num_epochs):
        checkpoint = {"epoch": epoch}
        epoch_start_time = time()
        train_loss = 0
        val_loss = 0
        for ts_name in ts_name_list:
            if len(train_data[ts_name]) == 0 or len(val_data[ts_name]) == 0:
                continue

            for ts_index in range(len(train_data[ts_name])):
                train_loss += train(
                        model, train_data[ts_name][ts_index], loss_func,
                        device, optimizer)
                val_loss += validate(
                        model, val_data[ts_name][ts_index], loss_func,
                        device, checkpoint)

        train_loss /= len(ts_name_list)
        train_loss /= len(train_data[ts_name])
        val_loss /= len(ts_name_list)
        val_loss /= len(val_data[ts_name])
        # if epoch % 10 == 0:
        print(
            f"Epoch {epoch:>5},      "
            f"Train Loss: {train_loss:>10.5f},  "
            f"Val Loss: {val_loss:>10.5f}"
        )
        print(f"Epoch {epoch:>5} finished in "
              f"{time() - epoch_start_time:.2f} seconds")

        checkpoint.update({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        torch.save(checkpoint, f"checkpoints/{model_name}_epoch_{epoch}.pt")

    print(f"Training finished in {time() - train_start_time:.2f} seconds")
