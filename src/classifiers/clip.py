from pathlib import Path
import copy
import math
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

import datetime

class EEGVideoDataset(Dataset):
    def __init__(self, eeg_features, labels, video_features, video_ids):
        self.eeg_features = torch.tensor(eeg_features, dtype=torch.float32)
        self.video_features = torch.tensor(video_features, dtype=torch.float32)
        
        id_to_local_idx = {vid: idx for idx, vid in enumerate(video_ids)}
        local_labels = [id_to_local_idx[vid] for vid in labels]
        
        self.labels = torch.tensor(local_labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.eeg_features)
        
    def __getitem__(self, idx):
        eeg = self.eeg_features[idx]
        local_vid_idx = self.labels[idx]
        video = self.video_features[local_vid_idx]
        return eeg, video, local_vid_idx

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return x + self.block(x)
    
class ClipModel(nn.Module):
    def __init__(self, eeg_dim: int, video_dim: int, projection_dim: int, 
                 dropout: float, temperature: float):
        super().__init__()
        self.eeg_dim = eeg_dim
        self.video_dim = video_dim
        hidden_dim = 256
        self.eeg_projection = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
    
    def forward(self, eeg_features, video_features):
        eeg_embeddings = F.normalize(self.eeg_projection(eeg_features), dim=-1)
        video_embeddings = F.normalize(self.video_projection(video_features), dim=-1)
        return eeg_embeddings, video_embeddings

def compute_multi_positive_contrastive_loss(eeg_emb, video_emb, vid_ids, logit_scale):
    logit_scale = logit_scale.exp()
    
    logits_eeg_to_video = logit_scale * torch.matmul(eeg_emb, video_emb.t())
    logits_video_to_eeg = logit_scale * torch.matmul(video_emb, eeg_emb.t())

    match_mask = (vid_ids.unsqueeze(1) == vid_ids.unsqueeze(0)).float()
    row_sum = match_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    match_mask_normalized = match_mask / row_sum
    
    log_probs_eeg = F.log_softmax(logits_eeg_to_video, dim=-1)
    log_probs_video = F.log_softmax(logits_video_to_eeg, dim=-1)
    
    loss_eeg = -(match_mask_normalized * log_probs_eeg).sum(dim=1).mean()
    loss_video = -(match_mask_normalized * log_probs_video).sum(dim=1).mean()
    
    return (loss_eeg + loss_video) / 2.0

class CLIP_Classifier:
    def __init__(self, projection_dim: int = 256, epochs: int = 15, batch_size: int = 32, 
                 lr: int = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.1, 
                 temperature: float = 0.07, device: str = "cpu"):
        self.projection_dim = projection_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.temperature = temperature
        self.device = device
        
        self.model = None
        self.eeg_scaler = StandardScaler()

    def fit(self, X_train, Z_train):
        # 1. Scale EEG and video features
        X_scaled = self.eeg_scaler.fit_transform(np.asarray(X_train, dtype=np.float32))
        Z_scaled = (np.asarray(Z_train, dtype=np.float32))

        # 2. Extract unique video features and map sample indices for multi-positive contrastive loss
        unique_videos, inverse_indices = np.unique(Z_scaled, axis=0, return_inverse=True)
        unique_video_ids = np.arange(len(unique_videos))
        labels = inverse_indices

        eeg_dim = X_scaled.shape[1]
        video_dim = unique_videos.shape[1]

        # 3. Setup dataset and dataloader
        dataset = EEGVideoDataset(X_scaled, labels, unique_videos, unique_video_ids)
        drop_last = len(dataset) >= self.batch_size
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

        # 4. Initialize PyTorch CLIP model and optimizer
        self.model = ClipModel(
            eeg_dim=eeg_dim,
            video_dim=video_dim,
            projection_dim=self.projection_dim,
            dropout=self.dropout,
            temperature=self.temperature
        ).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # 5. Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            for eeg, video, vid_ids in train_loader:
                eeg = eeg.to(self.device)
                video = video.to(self.device)
                vid_ids = vid_ids.to(self.device)

                optimizer.zero_grad()
                eeg_emb, video_emb = self.model(eeg, video)
                loss = compute_multi_positive_contrastive_loss(
                    eeg_emb, video_emb, vid_ids, self.model.logit_scale.clamp(max=100.0)
                )
                loss.backward()
                optimizer.step()

        return self

    def get_score_matrix(self, X_test, candidate_video_features):
        """Computes the similarity score matrix across test EEG samples and candidate videos."""
        self.model.eval()
        
        X_test_scaled = self.eeg_scaler.transform(np.asarray(X_test, dtype=np.float32))
        cand_scaled = (np.asarray(candidate_video_features, dtype=np.float32))

        eeg_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=self.device)
        cand_tensor = torch.tensor(cand_scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            eeg_emb = F.normalize(self.model.eeg_projection(eeg_tensor), dim=-1)
            video_emb = F.normalize(self.model.video_projection(cand_tensor), dim=-1)
            logit_scale = self.model.logit_scale.exp().clamp(max=100.0)
            score_matrix = (eeg_emb @ video_emb.T * logit_scale).cpu().numpy()

        return score_matrix