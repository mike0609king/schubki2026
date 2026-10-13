import torch
from torch import nn, optim
from torch.nn import functional as F

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

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

class MLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, input_dim: int, hidden_dims: list[int], classes: int, dropout: float = 0.0, lr: float = 1e-3, epochs: int = 50, random_seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.classes = classes
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        torch.manual_seed(self.random_seed)

        self.model_ = MLP(input_dim=self.input_dim, hidden_dims=list(self.hidden_dims), classes=self.classes, dropout=self.dropout)

        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.long)

        self.losses_ = self.model_.fit(X_tensor, y_tensor, epochs=self.epochs, lr=self.lr)
        self.classes_ = np.arange(self.classes)
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")

        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(X, dtype=torch.float32)

        predictions = self.model_.predict(X_tensor)
        return predictions.cpu().numpy()

def evaluate_random_mlp(X, Y, random_seed: int = 42):
    """
    Data is split into train/validation/test, hyper parameter tuning of MLP,
    best parameters are returned.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=random_seed, stratify=Y)

    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(input_dim=X_train.shape[1], classes=len(np.unique(Y_train)), random_state=42))
    ])

    param_grid = {
        "hidden_dims": [(32,), (64,), (64, 32), (128, 64)],
        "dropout": [0.0, 0.2, 0.4],
        "lr": [1e-2, 1e-3, 1e-4],
        "epochs": [30, 50],
    }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="accuracy", cv=5, refit=True, n_jobs=1, verbose=2)

    grid_search.fit(X_train, Y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)

    Y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_pred)
    
    print("Test accuracy:", test_accuracy)

    return grid_search.best_params_
