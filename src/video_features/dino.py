from functools import cache

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

@cache
def _pretrained_dinov3():
    return AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").eval()

@cache
def _pretrained_dinov3_processor():
    return AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")

def dinov3_embed(data: list[np.ndarray], batch_size: int = 8, num_frames: int = 8, device: str = "cpu") -> np.ndarray:
    """
    :param data: List of videos in format (T, H, W, 3).
    """
    if len(data) == 0:
        raise ValueError("No videos provided.")

    model = _pretrained_dinov3().to(device)
    processor = _pretrained_dinov3_processor()

    embeddings = []

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            videos = data[start : start + batch_size]
            frames_batch = []
            frame_counts = []

            for video in videos:
                if video.ndim != 4 or video.shape[-1] != 3:
                    raise ValueError(
                        f"Expected video shape (T, H, W, 3), "
                        f"got {video.shape}"
                    )
                if len(video) == 0:
                    raise ValueError("Video contains no frames.")

                indices = np.linspace(0, len(video) - 1, num=min(num_frames, len(video)), dtype=int)
                frame_counts.append(len(indices))

                for i in indices:
                    frame = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
                    frames_batch.append(Image.fromarray(frame))

            inputs = processor(images=frames_batch, return_tensors="pt",)
            pixel_values = inputs["pixel_values"].to(device)

            outputs = model(pixel_values=pixel_values)
            features = outputs.last_hidden_state[:, 0, :].float()

            # Average frame embeddings to obtain one per video.
            offset = 0
            video_features = []

            for count in frame_counts:
                video_embedding = features[offset : offset + count].mean(dim=0)
                video_features.append(video_embedding)
                offset += count
                
            # (batch_size, embedding_dim).
            embeddings.append(torch.stack(video_features).cpu().numpy())

    return np.concatenate(embeddings, axis=0)
