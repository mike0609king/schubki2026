"""
Vibe coded labram which does not require zero padding. No idea whether this works...
Can perhaps be adapted in the future.
"""

from functools import cache
from pathlib import Path

import numpy as np
import torch
from mne.filter import resample
from braindecode.models import Labram
from braindecode.models.labram import LABRAM_CHANNEL_ORDER
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from emd_loader import EmdDataset

# braindecode's export of the original labram-base checkpoint
LABRAM_REPO = "braindecode/labram-pretrained"
LABRAM_SFREQ = 200  # LaBraM was pre-trained on 200 Hz EEG
PATCH_SIZE = 200  # one patch = 1 s at 200 Hz
MAX_PATCHES = 16  # pretrained temporal embedding covers 16 patches (16 s)


@cache
def _load_labram_weights():
    """
    We want this to be caches, because we likely use those weights a ton.
    """
    return load_file(hf_hub_download(LABRAM_REPO, "model.safetensors"))


def _pretrained_labram(n_chans: int, n_times: int):
    n_patches = n_times // PATCH_SIZE
    state = dict(_load_labram_weights())
    # The checkpoint has LaBraM's 16 time slots, braindecode sizes the temporal
    # embedding to n_patches + 1 and only reads the first n_patches of it.
    time_embed = state["temporal_embedding"]
    state["temporal_embedding"] = torch.cat(
        (time_embed, torch.zeros_like(time_embed[:, :1])), dim=1
    )[:, : n_patches + 1]

    model = Labram(
        n_chans=n_chans,
        n_times=n_times,
        sfreq=LABRAM_SFREQ,
        n_outputs=0,
        patch_size=PATCH_SIZE,
        use_mean_pooling=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Mean pooling like in LaBraM fine-tuning: the final norm is unused, fc_norm is fresh
    missing = [k for k in missing if not k.startswith("fc_norm.")]
    unexpected = [k for k in unexpected if not k.startswith("norm.")]
    if missing or unexpected:
        raise RuntimeError(
            f"LaBraM weights did not match. Missing: {missing}, unexpected: {unexpected}"
        )
    return model


def labram_embed(
    data, sfreq: float, ch_names: list[str], batch_size: int = 64, device: str = "cpu"
):
    """
    Pretrained LaBraM embedding of EEG epochs.

    data: (n_trials, n_channels, n_times) in µV.
    ch_names: channel names from LABRAM_CHANNEL_ORDER in the order of the data's channel axis.
    Returns an array of shape (n_trials, 200).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.shape[1] != len(ch_names):
        raise ValueError(
            f"data has {data.shape[1]} channels but {len(ch_names)} ch_names were given."
        )

    if sfreq != LABRAM_SFREQ:
        data = resample(data, up=LABRAM_SFREQ, down=sfreq, axis=-1)

    # LaBraM splits the signal into 1 s patches, so crop to whole patches
    n_times = (data.shape[-1] // PATCH_SIZE) * PATCH_SIZE
    if not 0 < n_times <= MAX_PATCHES * PATCH_SIZE:
        raise ValueError(f"Epochs must be between 1 s and {MAX_PATCHES} s long.")
    data = data[..., :n_times]

    # LaBraM expects input in units of 0.1 mV
    data = data / 100.0

    model = _pretrained_labram(len(ch_names), n_times).to(device).eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            X = torch.as_tensor(
                data[start : start + batch_size], dtype=torch.float32, device=device
            )
            embeddings.append(model(X, ch_names=ch_names).cpu().numpy())
    return np.concatenate(embeddings)


def emd_labram_embeddings(
    emd_dir: Path, sub_id: int, ses_id: int, tmin=0.0, tmax=3.0, device="cpu"
):
    """
    LaBraM embeddings (n_trials, 200) and stimulus ids (n_trials,) of one EMD session.
    tmin, tmax: window relative to video onset in s, EMD epochs span -0.1 s to 3.5 s.
    """
    meta_file = (
        Path(emd_dir)
        / "derivatives"
        / "eeg"
        / f"sub-{sub_id:02d}"
        / f"sub-{sub_id:02d}_eeg_metadata.npy"
    )
    meta = np.load(meta_file, allow_pickle=True).item()
    eeg = EmdDataset(Path(emd_dir)).get_data_by_ids(
        sub_id, ses_id
    )  # (n_trials, n_chans, n_times) in V

    # LaBraM has no position embedding for some of EMD's 128 channels
    known = {ch.upper() for ch in LABRAM_CHANNEL_ORDER}
    chans = [i for i, ch in enumerate(meta["ch_names"]) if ch.upper() in known]
    window = (meta["times"] >= tmin) & (meta["times"] < tmax)
    # convert volt to micro volt
    eeg = eeg[:, chans][..., window] * 1e6

    embeddings = labram_embed(
        eeg, meta["args"].sfreq, [meta["ch_names"][i] for i in chans], device=device
    )
    return embeddings, np.asarray(meta["stimulus_id"][f"ses-{ses_id:02d}"])
