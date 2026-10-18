import torch
from torch import nn, optim
from torch.nn import functional as F

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterGrid, KFold
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
                 dropout: float = 0.2, lr: float = 1e-3, epochs: int = 20, random_seed: int = 42):
        self.input_dim = input_dim
        self.video_feature_dim = video_feature_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed
        self.eeg_scaler = StandardScaler()

    def fit(self, X_train, Z_train):
        X_train = np.asarray(X_train, dtype=np.float32)
        Z_train = np.asarray(Z_train, dtype=np.float32)

        X_train_scaled = self.eeg_scaler.fit_transform(X_train)

        torch.manual_seed(self.random_seed)

        self.model_ = DeepEmbeddingMLP(
            input_dim=self.input_dim,
            output_dim=self.video_feature_dim,
            hidden_dims=list(self.hidden_dims),
            dropout=self.dropout
        )

        X_tensor = torch.as_tensor(X_train_scaled, dtype=torch.float32)
        Z_tensor = torch.as_tensor(Z_train, dtype=torch.float32)

        self.losses_ = self.model_.fit(X_tensor, Z_tensor, epochs=self.epochs, lr=self.lr)
        return self

    def predict(self, eeg_features, candidate_video_features):
        check_is_fitted(self, "model_")
        
        self.model_.eval()
        with torch.no_grad():
            eeg_tensor = torch.as_tensor(eeg_features, dtype=torch.float32)
            predicted_z = self.model_(eeg_tensor).cpu().numpy()
            
            # Normalize vectors for cosine similarity scoring
            predicted_z_norm = predicted_z / np.linalg.norm(predicted_z, axis=1, keepdims=True)
            candidate_norm = candidate_video_features / np.linalg.norm(candidate_video_features, axis=1, keepdims=True)
            
            scores = np.dot(predicted_z_norm, candidate_norm.T).squeeze()
            predicted_index = int(np.argmax(scores)) if scores.ndim == 1 else int(np.argmax(scores[0]))
                
        return predicted_index, scores

    def get_score_matrix(self, X_test, candidate_video_features):
        """Computes the cosine similarity score matrix across test EEG samples and candidate videos."""
        check_is_fitted(self, "model_")
        
        self.model_.eval()
        with torch.no_grad():
            X_test_scaled = self.eeg_scaler.transform(np.asarray(X_test, dtype=np.float32))
            eeg_tensor = torch.as_tensor(X_test_scaled, dtype=torch.float32)
            
            predicted_z = self.model_(eeg_tensor).cpu().numpy()
            cand_features = np.asarray(candidate_video_features, dtype=np.float32)
            
            # Normalize vectors for cosine similarity scoring
            predicted_z_norm = predicted_z / np.maximum(np.linalg.norm(predicted_z, axis=1, keepdims=True), 1e-8)
            candidate_norm = cand_features / np.maximum(np.linalg.norm(cand_features, axis=1, keepdims=True), 1e-8)
            
            score_matrix = np.dot(predicted_z_norm, candidate_norm.T)
            
        return score_matrix