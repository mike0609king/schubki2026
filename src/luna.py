# TODO mostly vibe coded and not checked in detail
# => Check whether adding the layernorm is really necessary. braindecode does
# not seem to have those layernorms implemented, but the checkpoints
# and the luna model repository have additional layernorms.

from functools import cache

import numpy as np
import torch
from torch import nn
from mne.channels import make_standard_montage
from mne.filter import resample
from braindecode.models import LUNA
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# embed_dim, num_queries, depth of the checkpoints in PulpBio/LUNA
LUNA_SIZES = {"base": (64, 4, 8), "large": (96, 6, 10), "huge": (128, 8, 24)}


class _NormMlp(nn.Module):
    """
    timm's Mlp with norm_layer=nn.LayerNorm, which LUNA uses for the channel
    location embedder and the cross attention FFN. braindecode's LUNA leaves this
    norm out, so its weights in the checkpoint would be silently ignored.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.fc1, self.act, self.fc2 = mlp.fc1, mlp.act, mlp.fc2
        self.norm = nn.LayerNorm(mlp.fc1.out_features)

    def forward(self, x):
        return self.fc2(self.norm(self.act(self.fc1(x))))


@cache
def _pretrained_luna(size: str):
    embed_dim, num_queries, depth = LUNA_SIZES[size]
    # n_outputs=0 keeps the pre-training reconstruction head, so that every
    # weight of the checkpoint has a place
    model = LUNA(n_outputs=0, embed_dim=embed_dim, num_queries=num_queries, depth=depth)
    model.channel_location_embedder = _NormMlp(model.channel_location_embedder)
    model.cross_attn.ffn = _NormMlp(model.cross_attn.ffn)

    weights = hf_hub_download("PulpBio/LUNA", f"LUNA_{size}.safetensors")
    model.load_state_dict(load_file(weights), strict=True)
    return model.eval()


@cache
def _electrode_positions():
    """
    Upper case electrode name -> 3D position from the standard 10-05 montage,
    which is what LUNA was pre-trained with.
    """
    positions = make_standard_montage("standard_1005").get_positions()["ch_pos"]
    return {name.upper(): pos for name, pos in positions.items()}


def _luna_latent(model: LUNA, X: torch.Tensor, locations: torch.Tensor):
    """
    LUNA's forward pass up to the latent sequence (batch, n_patches, Q * D), before
    the reconstruction head.
    """
    x, _ = model.prepare_tokens(X, locations)
    x, _ = model.cross_attn(x)
    x = x.reshape(len(X), -1, x.shape[1] * x.shape[2])
    for block in model.blocks:
        x = block(x)
    return model.norm(x)


def luna_embed(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    batch_size: int = 64,
    device: str = "cpu",
    size: str = "base",
):
    """
    Pretrained LUNA embedding of EEG epochs, averaged over patches.

    LUNA is topology agnostic: it maps any set of channels with known 3D
    positions to a fixed number of learned queries, so all channels of the
    standard 10-05 system can be used.

    :param data: (n_trials, n_channels, n_times) EEG. Every channel is z-scored
    per trial, so the unit does not matter.
    :param ch_names: Channel names in the order of the data's channel axis.
    Channels without a position in the standard 10-05 montage are dropped.
    :param size: Pretrained model size, one of "base", "large" or "huge".

    :returns: Returns an array of shape (n_trials, 256) for base, (n_trials, 576)
    for large and (n_trials, 1024) for huge.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.shape[1] != len(ch_names):
        raise ValueError(
            f"Data has {data.shape[1]} channels but {len(ch_names)} ch_names were given."
        )

    # LUNA needs the 3D position of every channel, others are dropped
    positions = _electrode_positions()
    keep = [i for i, ch in enumerate(ch_names) if ch.upper() in positions]
    if not keep:
        raise ValueError("None of the channels has a position in the 10-05 system.")
    print(f"Kept {len(keep)}/{len(ch_names)} channels.")
    data = data[:, keep]
    locations = np.stack([positions[ch_names[i].upper()] for i in keep])

    sample_freqency = 256
    if sfreq != sample_freqency:
        data = resample(data, up=sample_freqency, down=sfreq, axis=-1)

    model = _pretrained_luna(size).to(device)

    # LUNA splits the signal into patches of 40 samples, so crop to whole patches
    n_times = (data.shape[-1] // model.patch_size) * model.patch_size
    if n_times == 0:
        raise ValueError(f"Epochs must be at least {model.patch_size} samples long.")
    data = data[..., :n_times]

    # LUNA z-scores every channel of every epoch
    data = (data - data.mean(axis=-1, keepdims=True)) / (
        data.std(axis=-1, keepdims=True) + 1e-8
    )

    locations = torch.as_tensor(locations, dtype=torch.float32, device=device)
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            X = torch.as_tensor(
                data[start : start + batch_size], dtype=torch.float32, device=device
            )
            batch_locations = locations.expand(len(X), -1, -1)
            latent = _luna_latent(model, X, batch_locations)
            embeddings.append(latent.mean(dim=1).cpu().numpy())
    return np.concatenate(embeddings)
