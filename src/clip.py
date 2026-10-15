from pathlib import Path
import copy
import math

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import datetime

class CLIPDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class ClipModel(nn.Module):
    def __init__(self, eeg_dim: int, video_dim: int, projection_dim: int, 
                 dropout: float, temperature: float):
        super().__init__()
        self.eeg_dim = eeg_dim
        self.video_dim = video_dim
        self.video_projection = nn.Sequential(
            nn.Linear(self.video_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.eeg_projection = nn.Sequential(
            nn.Linear(self.eeg_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    @property
    def temperature(self):
        return 1.0 / self.logit_scale.exp()
    
    def forward(self, eeg_features, video_features):

        eeg_embeddings = F.normalize(self.eeg_projection(eeg_features), dim=-1)
        video_embeddings = F.normalize(self.video_projection(video_features), dim=-1)

        return eeg_embeddings, video_embeddings

def clip_loss(eeg_embeddings, video_embeddings, logit_scale):
    
    logits = torch.matmul(eeg_embeddings, video_embeddings.T) * logit_scale
    labels = torch.arange(logits.size(0), device=logits.device)

    eeg_loss = F.cross_entropy(logits, labels)
    video_loss = F.cross_entropy(logits.T, labels)

    return (eeg_loss + video_loss) / 2

def prepare_loaders(dataset, batch_size: int, random_seed: int):
    n = len(dataset)
    train_size = int(0.8 * n)
    validation_size = int(0.1 * n)
    test_size = n - train_size - validation_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader

def evaluate_clip(features_path: str, weight_decay: float, learning_rate: float, 
                  epochs: int, projection_dim: int = 256, batch_size: int = 32, 
                  dropout: float = 0.1, temperature: float = 0.07, random_seed: int = 42,
                  device: str = "cpu", save_path: str | None = None):

    if save_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M.%S")
        save_path = f"../weights/clip_model_{timestamp}.pth"

    features_dir = Path(features_path)
    # TODO: load eeg_features, video_features as pairs
    eeg_features, video_features = None # replace -> load_features_as_pair()
    if eeg_features is None or video_features is None:
        raise ValueError(f"load_features_as_pair() not implemented yet!")

    eeg_dim = eeg_features.shape[-1]
    video_dim = video_features.shape[-1]

    dataset = CLIPDataset(eeg_features, video_features)

    train_loader, validation_loader, test_loader = prepare_loaders(dataset, batch_size, random_seed)

    model = ClipModel(eeg_dim=eeg_dim, video_dim=video_dim, projection_dim=projection_dim, dropout=dropout, temperature=temperature).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(epochs):

        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for eeg, video in train_loader:
            eeg = eeg.to(device)
            video = video.to(device)

            optimizer.zero_grad()
            eeg_emb, video_emb = model(eeg, video)
            logit_scale = model.logit_scale.exp().clamp(max=100.0)

            loss = clip_loss(eeg_emb, video_emb, logit_scale)
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
            for eeg, video in validation_loader:
                eeg = eeg.to(device)
                video = video.to(device)

                eeg_emb, video_emb = model(eeg, video)
                logit_scale = model.logit_scale.exp().clamp(max=100.0)
                loss = clip_loss(eeg_emb, video_emb, logit_scale)

                batch_n = eeg.size(0)
                validation_loss_sum += loss.item() * batch_n
                validation_samples += batch_n

        validation_loss = validation_loss_sum / validation_samples

        print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f}")

        # Keep the checkpoint with the best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise ValueError("No checkpoint was saved. Check epochs and dataset sizes.")

    model.load_state_dict(best_state)

    # Final test evaluation — performed only after model selection
    model.eval()
    test_loss_sum = 0.0
    test_samples = 0

    with torch.no_grad():
        for eeg, video in test_loader:
            eeg = eeg.to(device)
            video = video.to(device)

            eeg_emb, video_emb = model(eeg, video)
            logit_scale = model.logit_scale.exp().clamp(max=100.0)
            loss = clip_loss(eeg_emb, video_emb, logit_scale)

            batch_n = eeg.size(0)
            test_loss_sum += loss.item() * batch_n
            test_samples += batch_n

    test_loss = test_loss_sum / test_samples

    print(f"Best Validation Loss: {best_validation_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    store_clip_weights(model=model, save_path=save_path)

    return test_loss

def store_clip_weights(model, save_path: str):
    """Save the trained CLIP model weights."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"CLIP model weights saved to: {save_path}")

def load_clip_model(save_path: str, eeg_dim: int, video_dim: int, projection_dim: int = 256, 
                    dropout: float = 0.1, device: str = "cpu"):
    """Load the trained CLIP model weights."""

    path = Path(save_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")
    
    model = ClipModel(eeg_dim=eeg_dim, video_dim=video_dim, projection_dim=projection_dim, dropout=dropout)

    state_dict = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"CLIP model loaded from: {path}")
    return model


