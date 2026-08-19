import torch
import torch.nn as nn
import torch.optim as optim


class CircuitAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder:
        # 50 gate values -> 32 -> 16 -> 8 compressed values
        self.encoder = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 8)
        )

        # Decoder:
        # 8 compressed values -> reconstruct 50 gate values
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),

            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, 50)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        return reconstructed


def train_autoencoder(
    model,
    dataset,
    epochs=100
):

    # Convert NumPy dataset into PyTorch tensor
    X = torch.tensor(
        dataset,
        dtype=torch.float32
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()

    for epoch in range(epochs):

        # Forward pass
        output = model(X)

        # Compare reconstructed circuit with original circuit
        loss = criterion(
            output,
            X
        )

        # Reset previous gradients
        optimizer.zero_grad()

        # Calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Show training progress
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | Loss = {loss.item():.4f}"
            )

    return model