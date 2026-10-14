from functools import cache

import numpy as np
import torch
from mne.filter import resample
from braindecode.models import CBraMod


@cache
def _pretrained_cbramod():
    return CBraMod.from_pretrained(
        "braindecode/cbramod-pretrained", return_encoder_output=True
    ).eval()


def cbramod_embed(
    data: np.ndarray,
    sfreq: float,
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """
    :param data: (n_trials, n_channels, n_times) in micro V
    :param sfreq: Sampling frequency in Hz
    """
    data = np.asarray(data, dtype=np.float64)

    sample_freqency = 200
    if sfreq != sample_freqency:
        data = resample(data, up=sample_freqency, down=sfreq, axis=-1)

    # CBraMod splits the signal into 1 s patches, so crop to whole patches
    patch_size = 200
    n_times = (data.shape[-1] // patch_size) * patch_size
    if n_times == 0:
        raise ValueError("Epochs must be at least 1 s long.")
    data = data[..., :n_times]

    # We divide by 100.0, because the paper says so...
    data = data / 100.0
    model = _pretrained_cbramod().to(device)

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            X = torch.as_tensor(
                data[start : start + batch_size], dtype=torch.float32, device=device
            )
            # (batch, n_channels, n_patches, 200)
            features = model(X, return_features=True)["features"]
            embeddings.append(features.mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(embeddings)
