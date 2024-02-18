import torch
import torch.nn as nn
import torch.optim as optim

from models import TwoLinear


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def test(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            predictions.append(pred.cpu())
    return predictions


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoLinear().to(device)
    loss_fn = nn.MSELoss()  # Example loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Placeholder for your DataLoader
    train_dataloader = None
    val_dataloader = None

    # Early stopping parameters
    early_stopping_patience = 10
    min_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):  # Example epoch count
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device)
        val_loss = validate(model, val_dataloader, loss_fn, device)

        print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        # Early stopping check
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
