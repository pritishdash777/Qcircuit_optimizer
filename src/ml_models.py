# src/ml_models.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ---------------------------
# Neural Network
# ---------------------------

class CircuitOptimizerNN(nn.Module):
    """
    Predicts how many redundant gates
    exist in a circuit.
    """

    def __init__(self, input_size=6):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------
# Training Function
# ---------------------------

def train_model(X, y, epochs=100):

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = CircuitOptimizerNN()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(epochs):

        predictions = model(X)

        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | Loss = {loss.item():.4f}"
            )

    return model


# ---------------------------
# Prediction Function
# ---------------------------

def predict_redundancy(model, features):

    x = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        prediction = model(x)

    return float(prediction.item())