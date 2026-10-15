from functools import cache

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

@cache
def _pretrained_openclip():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    return model.eval()

@cache
def _pretrained_openclip_processor():
    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    return preprocess

def openclip_embed(data: list[np.ndarray], batch_size: int = 8, num_frames: int = 8,device: str = "cpu") -> np.ndarray:
    """
    :param data: List of videos in format (T, H, W, 3).
    """
    if len(data) == 0:
        raise ValueError("No videos provided.")

    model = _pretrained_openclip().to(device)
    preprocess = _pretrained_openclip_processor()

    embeddings = []

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            videos = data[start : start + batch_size]
            frames_batch = []

            for video in videos:
                if video.ndim != 4 or video.shape[-1] != 3:
                    raise ValueError(
                        f"Expected video shape (T, H, W, 3), "
                        f"got {video.shape}"
                    )
                if len(video) == 0:
                    raise ValueError("Video contains no frames.")

                indices = np.linspace(0, len(video) - 1, num=min(num_frames, len(video)), dtype=int)

                for i in indices:
                    frame = cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB)
                    frames_batch.append(preprocess(Image.fromarray(frame)))

            pixel_values = torch.stack(frames_batch).to(device)
            features = model.encode_image(pixel_values)
            frame_counts = [min(num_frames, len(video)) for video in videos]
            features = features.float()

            offset = 0
            video_features = []
            
            # Average frame embeddings to obtain one per video.
            for count in frame_counts:
                video_embedding = features[offset : offset + count].mean(dim=0)
                video_features.append(video_embedding)
                offset += count

            # (batch_size, embedding_dim).
            embeddings.append(torch.stack(video_features).cpu().numpy())

    return np.concatenate(embeddings, axis=0)
