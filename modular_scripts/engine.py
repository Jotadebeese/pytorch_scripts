"""
Contains functions for training and testing a PyTorch model.
"""

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# Defining train_step
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
"""Trains a PyTorch model for a single epoch.

Turns a target PyTorch model to training mode and runs it
through all the required training steps (forward pass,
loss calculation, optimizer step).

Args:
    model: A PyTorch model to be trained.
    dataLoader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on.

Returns:
    A tuple of training loss and training accuracy metrics,
    in the form (train_loss, train_accuracy).
    Example:

    (0.1123, 0.8755)
"""
# Put model in train mode
model.train()

# Setup train loss and train accuracy values
train_loss, train_acc = 0, 0

# Loop through data loader data batches
for batch, (X, y) in enumerate(dataloader):
    # Send data to target device
    X, y = X.to(device), y.to(device)

    # 1. Forward pass
    y_pred = model(X)

    # 2. Calculate and accumulate loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Calculate and accumulate accuracy metric across all batches
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_pred)

# Adjust metrics to get average loss and accuracy per batch
train_loss = train_loss / len(dataloader)
train_acc = train_acc / len(dataloader)

return train_loss, train_acc