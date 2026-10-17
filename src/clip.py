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
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
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
        self.eeg_projection = nn.Sequential(
            nn.Linear(eeg_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(512),
            nn.Linear(512, projection_dim)
        )
        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(512),
            nn.Linear(512, projection_dim)
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

def prepare_loaders(dataset, batch_size: int, random_seed: int):
    n = len(dataset)
    train_size = int(0.8 * n)
    validation_size = n - train_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, validation_dataset = random_split(
        dataset, [train_size, validation_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader

@torch.no_grad()
def compute_clip_accuracy(model, data_loader, device):
    """
    Computes global Top-1 and Top-5 retrieval accuracy across the entire 
    set of candidate videos for the given data loader split.
    """
    model.eval()
    logit_scale = model.logit_scale.exp().clamp(max=100.0)

    all_eeg_embs = []
    all_true_indices = []

    # 1. Collect all EEG embeddings and their true local video indices across all batches
    for eeg, _, local_vid_indices in data_loader:
        eeg = eeg.to(device)
        eeg_emb = F.normalize(model.eeg_projection(eeg), dim=-1)
        
        all_eeg_embs.append(eeg_emb)
        all_true_indices.append(local_vid_indices)

    all_eeg_embs = torch.cat(all_eeg_embs, dim=0) # Shape: (total_samples, projection_dim)
    all_true_indices = torch.cat(all_true_indices, dim=0).to(device) # Shape: (total_samples,)

    # 2. Extract the full set of unique candidate video features for this split
    dataset = data_loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset
        
    video_features_tensor = base_dataset.video_features.to(device)
    
    all_video_embs = F.normalize(model.video_projection(video_features_tensor), dim=-1) # Shape: (total_split_videos, projection_dim)
    global_logits = torch.matmul(all_eeg_embs, all_video_embs.T) * logit_scale # Shape: (total_samples, total_split_videos)
    
    total_samples = global_logits.size(0)
    num_candidates = global_logits.size(1)

    # 5. Global Top-1 Accuracy
    preds = global_logits.argmax(dim=1)
    correct_top1 = (preds == all_true_indices).sum().item()

    # 6. Global Top-5 Accuracy
    k = min(5, num_candidates)
    topk_preds = global_logits.topk(k, dim=1).indices # Shape: (total_samples, k)
    correct_top5 = (topk_preds == all_true_indices.unsqueeze(1)).any(dim=1).sum().item()

    top1_acc = correct_top1 / total_samples
    top5_acc = correct_top5 / total_samples
    
    return top1_acc, top5_acc

def evaluate_clip(X_train, Y_train, train_video_ids, train_video_features: np.ndarray, 
                  X_test, Y_test, test_video_ids, test_video_features: np.ndarray, 
                  weight_decay: float, learning_rate: float, 
                  epochs: int, projection_dim: int = 256, batch_size: int = 32, 
                  dropout: float = 0.1, temperature: float = 0.07, random_seed: int = 42,
                  device: str = "cpu", save_path: str | None = None):

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"./weights/clip_model_{timestamp}.pth"

    X_train_scaled, train_v_scaled, X_test_scaled, test_v_scaled  = prepare_clip_inputs(X_train, train_video_features, X_test, test_video_features)

    eeg_dim = X_train.shape[-1]
    video_dim = train_video_features.shape[-1]

    train_dataset = EEGVideoDataset(X_train_scaled, Y_train, train_v_scaled, train_video_ids)
    test_dataset = EEGVideoDataset(X_test_scaled, Y_test, test_v_scaled, test_video_ids)

    train_loader, validation_loader = prepare_loaders(train_dataset, batch_size=batch_size, random_seed=random_seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ClipModel(eeg_dim=eeg_dim, video_dim=video_dim, projection_dim=projection_dim, dropout=dropout, temperature=temperature).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for eeg, video, vid_ids in train_loader:  # Fixed: Unpack vid_ids
            eeg = eeg.to(device)
            video = video.to(device)
            vid_ids = vid_ids.to(device)

            optimizer.zero_grad()
            eeg_emb, video_emb = model(eeg, video)
            # Fixed: Passed vid_ids to loss function
            loss = compute_multi_positive_contrastive_loss(eeg_emb, video_emb, vid_ids, model.logit_scale.clamp(max=100.0))
            loss.backward()
            optimizer.step()

            batch_n = eeg.size(0)
            train_loss_sum += loss.item() * batch_n
            train_samples += batch_n

        training_loss = train_loss_sum / train_samples

        model.eval()
        validation_loss_sum = 0.0
        validation_samples = 0

        with torch.no_grad():
            for eeg, video, vid_ids in validation_loader:  # Fixed: Unpack vid_ids
                eeg = eeg.to(device)
                video = video.to(device)
                vid_ids = vid_ids.to(device)

                eeg_emb, video_emb = model(eeg, video)
                loss = compute_multi_positive_contrastive_loss(eeg_emb, video_emb, vid_ids,  model.logit_scale.clamp(max=100.0))

                batch_n = eeg.size(0)
                validation_loss_sum += loss.item() * batch_n
                validation_samples += batch_n

        validation_loss = validation_loss_sum / validation_samples
        val_top1, val_top5 = compute_clip_accuracy(model, validation_loader, device)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {training_loss:.4f} | "
              f"Val Loss: {validation_loss:.4f} | "
              f"Val Top-1 Acc: {val_top1*100:.2f}% | Val Top-5 Acc: {val_top5*100:.2f}%")
        
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise ValueError("No checkpoint was saved. Check epochs and dataset sizes.")

    model.load_state_dict(best_state)

    model.eval()
    test_loss_sum = 0.0
    test_samples = 0

    with torch.no_grad():
        for eeg, video, vid_ids in test_loader:  # Fixed: Unpack vid_ids
            eeg = eeg.to(device)
            video = video.to(device)
            vid_ids = vid_ids.to(device)

            eeg_emb, video_emb = model(eeg, video)
            loss = compute_multi_positive_contrastive_loss(eeg_emb, video_emb, vid_ids,  model.logit_scale.clamp(max=100.0))

            batch_n = eeg.size(0)
            test_loss_sum += loss.item() * batch_n
            test_samples += batch_n

    test_loss = test_loss_sum / test_samples
    test_top1, test_top5 = compute_clip_accuracy(model, test_loader, device)

    print(f"\nBest Validation Loss: {best_validation_loss:.4f}")
    print(f"Test Top-1 Accuracy: {test_top1*100:.2f}% | Test Top-5 Accuracy: {test_top5*100:.2f}%")

    store_clip_weights(model=model, save_path=save_path)

    return test_loss

def prepare_clip_inputs(X_train, train_video_features, X_test, test_video_features):
    eeg_scaler = StandardScaler()
    video_scaler = StandardScaler()
    
    X_train_scaled = eeg_scaler.fit_transform(np.asarray(X_train, dtype=np.float32))
    Z_train_scaled = video_scaler.fit_transform(np.asarray(train_video_features, dtype=np.float32))

    X_test_scaled = eeg_scaler.transform(np.asarray(X_test, dtype=np.float32))
    Z_test_scaled = video_scaler.transform(np.asarray(test_video_features, dtype=np.float32))
    
    return X_train_scaled, Z_train_scaled, X_test_scaled, Z_test_scaled

def store_clip_weights(model, save_path: str):
    """Save the trained CLIP model weights."""

    sp = Path(save_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), sp)
    print(f"CLIP model weights saved to: {sp}")

def load_clip_model(save_path: str, eeg_dim: int, video_dim: int, projection_dim: int = 256, 
                    dropout: float = 0.1, temperature: float = 0.07, device: str = "cpu"):
    """Load the trained CLIP model weights."""

    path = Path(save_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")
    
    model = ClipModel(eeg_dim=eeg_dim, video_dim=video_dim, projection_dim=projection_dim, dropout=dropout, temperature=temperature)

    state_dict = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"CLIP model loaded from: {path}")
    return model


