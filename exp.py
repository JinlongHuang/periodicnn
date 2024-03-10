import os
import json
from time import time
from itertools import chain

import torch
import torch.optim as optim

from models import OneLinear
from dataloader import load_data


def train(model, data, device, optimizer):
    model.train()
    total_loss = 0
    for x, y in data:
        x, y = x.to(device), y.to(device)

        # Forward pass
        pred = model(x)
        loss = _neg_pnl_loss(pred, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data)
    return avg_loss


def validate(model, data, device):
    model.eval()
    loss = []
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss.append(_val_loss(pred, y))
            preds.append(list(pred.cpu().squeeze().numpy()))
            targets.append(list(y.cpu().squeeze().numpy()))
    loss = list(chain.from_iterable(loss))
    preds = list(chain.from_iterable(preds))
    targets = list(chain.from_iterable(targets))

    return loss, preds, targets


def test(model, data, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            predictions.append(pred.cpu())
    return predictions


def _neg_pnl_loss(pred, y):
    """pred, y: (batch_size, out_seq_len)"""
    if pred.shape != y.shape:
        raise ValueError("pred and y must have the same shape")
    return -torch.mean(torch.sum(pred * y, dim=1))


def _val_loss(pred, y):
    """pred, y: (batch_size, out_seq_len)"""
    if pred.shape != y.shape:
        raise ValueError("pred and y must have the same shape")
    return list(-torch.sum(pred * y, dim=1).numpy())


def run():
    with open("config.json") as f:
        config = json.load(f)
        in_seq_len = config['data']['in_seq_len']
        out_seq_len = config['data']['out_seq_len']
        assets = config['data']['assets_for_preprocess']

        num_epochs = config['train']['num_epochs']
        learning_rate = config['train']['learning_rate']
        saved_epoch = config['train']['saved_epoch']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneLinear(in_seq_len, out_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    starting_epoch = 0

    # Load saved checkpoints into model and optimizer
    if saved_epoch >= 0:
        checkpoint_file = f'checkpoints/OneLinear_epoch_{saved_epoch}.pt'
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_epoch = checkpoint['epoch'] + 1
            sum_val_ret = - sum(checkpoint['val_loss'])
            print(f"Epoch {checkpoint['epoch']:>5}, "
                  f"Train Loss: {checkpoint['train_loss']:>10.5f},  "
                  f"Total Val PnL: {sum_val_ret:>10.5f}")

    train_data, val_data, test_data = load_data()

    train_start_time = time()
    for epoch in range(starting_epoch, num_epochs):
        for entry in os.scandir('data/'):
            # Select assets to train
            if not entry.is_dir():
                continue
            asset = entry.name[:-4]
            if asset not in assets:
                continue
            if len(train_data[asset]) == 0 or len(val_data[asset]) == 0:
                continue

            train_loss = train(model, train_data[asset], device, optimizer)
            val_loss, val_preds, val_targets = validate(
                    model, val_data[asset], device)
            sum_val_ret = - sum(val_loss)

            print(f"Epoch {epoch:>5},      "
                  f"Asset: {asset:>6},     "
                  f"Train Loss: {train_loss:>10.5f},  "
                  f"Total Val PnL: {sum_val_ret:>10.5f}")

        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_preds': val_preds,
                'val_targets': val_targets
            }
            torch.save(checkpoint, f"checkpoints/OneLinear_epoch_{epoch}.pt")

    print(f"Training finished in {time() - train_start_time:.2f} seconds")
