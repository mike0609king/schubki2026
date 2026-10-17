import torch
from torch import nn, optim
from torch.nn import functional as F

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class DeepEmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
            
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)

    def fit(self, X_eeg, Z_video, epochs: int, lr: float):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss() 
        losses = []

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            predicted_video_features = self(X_eeg)
            loss = loss_fn(predicted_video_features, Z_video)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        return losses


class DeepEmbeddingClassifier(BaseEstimator, ClassifierMixin):
    """
    An MLP-based equivalent to CCA: Learns a mapping from EEG features 
    to video feature space, allowing zero-shot retrieval on unseen test videos.
    """
    def __init__(self, input_dim: int, video_feature_dim: int, hidden_dims: list[int] = (128, 64), 
                 dropout: float = 0.2, lr: float = 1e-3, epochs: int = 50, random_seed: int = 42):
        self.input_dim = input_dim
        self.video_feature_dim = video_feature_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed

    def fit(self, X_train, Z_train):
        X_train = np.asarray(X_train, dtype=np.float32)
        Z_train = np.asarray(Z_train, dtype=np.float32)

        torch.manual_seed(self.random_seed)

        self.model_ = DeepEmbeddingMLP(
            input_dim=self.input_dim,
            output_dim=self.video_feature_dim,
            hidden_dims=list(self.hidden_dims),
            dropout=self.dropout
        )

        X_tensor = torch.as_tensor(X_train, dtype=torch.float32)
        Z_tensor = torch.as_tensor(Z_train, dtype=torch.float32)

        self.losses_ = self.model_.fit(X_tensor, Z_tensor, epochs=self.epochs, lr=self.lr)
        return self

    def predict(self, eeg_features, candidate_video_features):
        check_is_fitted(self, "model_")
        
        self.model_.eval()
        with torch.no_grad():
            eeg_tensor = torch.as_tensor(eeg_features, dtype=torch.float32)
            predicted_z = self.model_(eeg_tensor).cpu().numpy()
            predicted_z_norm = predicted_z / np.linalg.norm(predicted_z, axis=1, keepdims=True)
            candidate_norm = candidate_video_features / np.linalg.norm(candidate_video_features, axis=1, keepdims=True)
            
            scores = np.dot(predicted_z_norm, candidate_norm.T).squeeze()
            
            if scores.ndim == 1 and len(candidate_video_features) > 1:
                predicted_index = int(np.argmax(scores))
            else:
                predicted_index = int(np.argmax(scores)) # adjust if batch predicting
                
        return predicted_index, scores

def _fit_mlp(X, Y, video_ids, video_features, params, input_dim, video_feature_dim, random_seed=42):
    row_of_id = {video_id: row for row, video_id in enumerate(video_ids)}
    Z = video_features[[row_of_id[video_id] for video_id in Y]]

    model = DeepEmbeddingClassifier(
        input_dim=input_dim,
        video_feature_dim=video_feature_dim,
        random_seed=random_seed,
        **params
    ).fit(X, Z)
    return model 

def _accuracy_mlp(model, X, Y, video_ids, video_features):
    row_of_id = {video_id: row for row, video_id in enumerate(video_ids)}
    candidate_video_features = video_features[[row_of_id[video_id] for video_id in video_ids]]
    
    correct = 0
    for eeg_features, video_id in zip(X, Y):
        predicted_index, _ = model.predict(eeg_features[None, :], candidate_video_features)
        correct += video_ids[predicted_index] == video_id
        
    acc = correct / len(Y)
    print(f"Accuracy: {acc:.4f}")
    return acc

def evaluate_mlp(X_train, Y_train, train_video_ids, train_video_features,
                 X_test, Y_test, test_video_ids, test_video_features, 
                 random_seed: int = 42):
    """
    Data is split into train/validation/test, hyperparameter tuning of the Deep Embedding MLP,
    and returns the best parameters.
    """
    train_video_ids = np.asarray(train_video_ids)
    test_video_ids = np.asarray(test_video_ids)
    train_video_features = np.asarray(train_video_features)
    test_video_features = np.asarray(test_video_features)

    input_dim = X_train.shape[1]
    video_feature_dim = train_video_features.shape[1]

    param_grid = {
        "hidden_dims": [(64,), (128, 64)],
        "dropout": [0.2, 0.4],
        "lr": [1e-3, 1e-4],
        "epochs": [30, 50],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    best_params, best_score = None, -np.inf
    
    # Random or Grid search loop
    for params in ParameterGrid(param_grid):
        fold_scores = []
        for train_idx, val_idx in cv.split(X_train, Y_train):
            model = _fit_mlp(
                X_train[train_idx], Y_train[train_idx], 
                train_video_ids, train_video_features, 
                params, input_dim, video_feature_dim, random_seed
            )
            fold_scores.append(_accuracy_mlp(model, X_train[val_idx], Y_train[val_idx], train_video_ids, train_video_features))

        score = float(np.mean(fold_scores))
        print(f"{params}: CV accuracy {score:.4f}")
        if score > best_score:
            best_params, best_score = params, score

    print("Best parameters:", best_params)
    print("Best CV accuracy:", best_score)

    model = _fit_mlp(X_train, Y_train, train_video_ids, train_video_features, best_params, input_dim, video_feature_dim, random_seed)
    test_accuracy = _accuracy_mlp(model, X_test, Y_test, test_video_ids, test_video_features)
    print("Test accuracy:", test_accuracy)
    print("Chance level:", 1 / len(test_video_ids))

    return best_params, model