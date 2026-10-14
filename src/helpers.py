from pathlib import Path

import numpy as np
import h5py
from h5py import Dataset


def save_feature(path: Path, data: np.ndarray, exists_ok: bool = False):
    """
    File is saved into a .h5 file.
    """
    if not str(path).endswith(".h5"):
        raise ValueError("File path needs to end with '.h5'")
    path.unlink(missing_ok=exists_ok)

    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)


def load_feature(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File {str(path)} does not exist!")

    with h5py.File(path, "r") as f:
        dset = f["data"]
        if not isinstance(dset, Dataset):
            raise TypeError(f"File {str(path)} is not formatted correctly.")
        return dset[:]
