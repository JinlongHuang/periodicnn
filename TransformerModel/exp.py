import os
import json
from time import time
from itertools import chain

import torch
import torch.optim as optim

from models import OneLinear
from play import TransformerModel
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
    print(avg_loss)
    return avg_loss


def validate(model, data, device):
    model.eval()
    pnl = []
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pnl.append(_val_pnl(pred, y))
            preds.append(list(pred.cpu().squeeze().numpy()))
            targets.append(list(y.cpu().squeeze().numpy()))
    pnl = list(chain.from_iterable(pnl))
    preds = list(chain.from_iterable(preds))
    targets = list(chain.from_iterable(targets))

    return pnl, preds, targets


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


def _val_pnl(pred, y):
    """pred, y: (batch_size, out_seq_len)"""
    if pred.shape != y.shape:
        raise ValueError("pred and y must have the same shape")
    return list((torch.sum(pred * y, dim=1).cpu().numpy()))


def run():
    with open("config.json") as f:
        config = json.load(f)
        in_seq_len = config['data']['in_seq_len']
        out_seq_len = config['data']['out_seq_len']
        batch_size = config['data']['batch_size']
        assets = config['data']['ts_for_preprocess']

        num_epochs = config['train']['num_epochs']
        learning_rate = config['train']['learning_rate']
        saved_epoch = config['train']['saved_epoch']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(in_seq_len, out_seq_len, batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    starting_epoch = 0
    print(in_seq_len)
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
    #print("train_data x in run is {}".format(train_data['ETH'][0][0].size()))
    #print("train_data x in run is {}".format(type(train_data['ETH'][0][0])))
    #print("train_data y in run is {}".format(train_data['ETH'][0][1].size()))
    #print("train_data x in run is {}".format(train_data['ETH'][1][0].size()))
    #print("train_data x in run is {}".format(train_data['ETH'][2][0].size()))
    #print("train_data x in run is {}".format(len(train_data['ETH'])))
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
            print(train_loss)
            val_pnl, val_preds, val_targets = validate(
                    model, val_data[asset], device)
            total_val_pnl = sum(val_pnl)

            print(f"Epoch {epoch:>5},      "
                  f"Asset: {asset:>6},     "
                  f"Train Loss: {train_loss:>10.5f},  "
                  f"Total Val PnL: {total_val_pnl:>10.5f}")

        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_pnl': val_pnl,
                'val_preds': val_preds,
                'val_targets': val_targets
            }
            torch.save(checkpoint, f"checkpoints/OneLinear_epoch_{epoch}.pt")

    print(f"Training finished in {time() - train_start_time:.2f} seconds")
