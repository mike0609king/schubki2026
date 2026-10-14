from functools import cache

import numpy as np
import torch
from mne.filter import resample
from braindecode.models import Labram
from braindecode.models.labram import LABRAM_CHANNEL_ORDER


@cache
def _pretrained_labram():
    return Labram.from_pretrained(
        "braindecode/labram-pretrained", use_mean_pooling=True
    ).eval()


def labram_embed(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    batch_size: int = 64,
    device: str = "cpu",
):
    """
    Pretrained LaBraM embedding of EEG epochs, zero-padded to the model's 15 s window.

    :param data: (n_trials, n_channels, n_times) in micro V.
    :param ch_names: Channel names in the order of the data's channel axis.
    Channels not in LABRAM_CHANNEL_ORDER.

    :returns: Returns an array of shape (n_trials, 200).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.shape[1] != len(ch_names):
        raise ValueError(
            f"Data has {data.shape[1]} channels but {len(ch_names)} ch_names were given."
        )

    # Labram has no position embedding for some channels, those channels are droped
    known = {ch.upper() for ch in LABRAM_CHANNEL_ORDER}
    keep = [i for i, ch in enumerate(ch_names) if ch.upper() in known]
    if not keep:
        raise ValueError("None of the channels has a Labram position embedding.")
    print(f"Kept {len(ch_names) - len(keep)}/{len(ch_names)} channels.")
    data = data[:, keep]
    ch_names = [ch_names[i] for i in keep]

    # Labram was pre-trained on 200 Hz EEG. We need to resample to 200 Hz
    sample_freqency = 200
    if sfreq != sample_freqency:
        data = resample(data, up=sample_freqency, down=sfreq, axis=-1)

    model = _pretrained_labram().to(device)
    if data.shape[-1] > model.n_times:
        raise ValueError(
            f"Epochs must be at most {model.n_times / sample_freqency:g} s long."
        )
    data = np.pad(data, ((0, 0), (0, 0), (0, model.n_times - data.shape[-1])))

    # TODO Conversion into correct unit. Not sure whether this one is correct.
    data = data / 100.0

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            X = torch.as_tensor(
                data[start : start + batch_size], dtype=torch.float32, device=device
            )
            embeddings.append(model(X, ch_names=ch_names).cpu().numpy())
    return np.concatenate(embeddings)
