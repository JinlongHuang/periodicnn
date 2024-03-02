import os
import json

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
        loss = _neg_return_loss(pred, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data)
    return avg_loss


def validate(model, data, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = _neg_return_loss(pred, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(data)
    return avg_loss


def test(model, data, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x in data:
            x = x.to(device)
            pred = model(x)
            predictions.append(pred.cpu())
    return predictions


def _neg_return_loss(pred, y):
    """pred, y: (batch_size, out_seq_len)"""
    if pred.shape != y.shape:
        raise ValueError("pred and y must have the same shape")
    return -torch.mean(torch.sum(pred * y, dim=1))


def run():
    with open("config.json") as f:
        config = json.load(f)
        in_seq_len = config['data']['in_seq_len']
        out_seq_len = config['data']['out_seq_len']

        num_epochs = config['train']['num_epochs']
        learning_rate = config['train']['learning_rate']
        early_stopping_patience = config['train']['early_stopping_patience']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneLinear(in_seq_len, out_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    starting_epoch = 0

    # load saved checkpoints into model and optimizer
    saved_epoch = 10  # change this number to the epoch you want to load
    checkpoint_file = f'checkpoints/OneLinear_epoch_{saved_epoch}.pt'
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print(f"Epoch {checkpoint['epoch']:>3}, "
              f"Asset: {checkpoint['asset']:>6},     "
              f"Train Loss: {checkpoint['train_loss']:>8.3f},  "
              f"Val Loss: {checkpoint['val_loss']:>8.3f}")

    train_data, val_data, test_data = load_data()

    min_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(starting_epoch, num_epochs):
        for entry in os.scandir('data/'):
            if not entry.is_dir():
                continue
            asset = entry.name[:-4]
            if len(train_data[asset]) == 0 or len(val_data[asset]) == 0:
                continue
            train_loss = train(model, train_data[asset], device, optimizer)
            val_loss = validate(model, val_data[asset], device)

            print(f"Epoch {epoch:>3}, "
                  f"Asset: {asset:>6},     "
                  f"Train Loss: {train_loss:>8.3f},  "
                  f"Val Loss: {val_loss:>8.3f}")

        if epoch % 10 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'asset': asset,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f"checkpoints/OneLinear_epoch_{epoch}.pt")

        # Early stopping check
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
