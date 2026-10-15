from functools import cache

import torch
import numpy as np
import cv2

from transformers import VideoMAEVideoProcessor, VideoMAEModel

@cache
def _pretrained_videomae():
    return VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").eval()

@cache
def _pretrained_videomae_processor():
    return VideoMAEVideoProcessor.from_pretrained("MCG-NJU/videomae-base")

def videomae_embed(data: list[np.ndarray], batch_size: int = 8, device: str = "cpu") -> np.ndarray:
    if len(data) == 0:
        raise ValueError("No videos provided.")

    model = _pretrained_videomae().to(device)
    processor = _pretrained_videomae_processor()
    num_frames = model.config.num_frames

    embeddings = []

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            videos = data[start : start + batch_size]
            sampled_videos = []

            for video in videos:
                if video.ndim != 4 or video.shape[-1] != 3:
                    raise ValueError(
                        f"Expected video shape (T, H, W, 3), "
                        f"got {video.shape}"
                    )

                if len(video) == 0:
                    raise ValueError("Video contains no frames.")

                indices = np.linspace(0, len(video) - 1, num=num_frames, dtype=int,)
                frames = [cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB) for i in indices]
                sampled_videos.append(frames)

            # Preprocess and batch the videos.
            inputs = processor(videos=sampled_videos, return_tensors="pt",)
            pixel_values = inputs["pixel_values"].to(device)

            # (batch, sequence_length, hidden_size).
            features = model(pixel_values=pixel_values).last_hidden_state
            features = features.mean(dim=1)
            
            # (batch, embedding_dim)
            embeddings.append(features.cpu().numpy())

    return np.concatenate(embeddings, axis=0)