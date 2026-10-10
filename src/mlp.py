import torch
from torch import nn, optim
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], classes: int, dropout: float):
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1].")

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))

        for i, _ in enumerate(hidden_dims[:-1]):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        self.layers.append(nn.Linear(hidden_dims[-1], classes))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        out = X
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
            out = self.dropout(out)
        out = self.layers[-1](out)
        return out

    def fit(self, X, Y, epochs: int, lr: float):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        losses = []

        super().train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self(X)
            loss = loss_fn(logits, Y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        return losses
    def predict(self, X):
        super().eval()
        with torch.no_grad():
            return self(X).argmax(dim=1)


# TODO
def evaluate_random_mlp(self, X, Y):
    """
    Data is split into train/validation/test, hyper parameter tuning of MLP,
    best parameters are returned.
    """
    # hyper parameters: 
    # 1. hidden dimensions, number of layers
    # 2. learning rate
    # 3. dropout rate
    # 4. epoch number
    raise NotImplementedError
