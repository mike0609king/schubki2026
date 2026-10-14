from pathlib import Path

import numpy as np
import h5py
from h5py import Dataset


def save_features(path: Path, X: np.ndarray, Y: np.ndarray, exists_ok: bool = False):
    """
    File is saved into a .h5 file.
    """
    if not str(path).endswith(".h5"):
        raise ValueError("File path needs to end with '.h5'")

    if exists_ok:
        path.unlink(missing_ok=True)
    elif not exists_ok and path.exists():
        raise Exception("File would be overwritten, but it is marked as not okay.")

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"File {str(path)} does not exist!")

    with h5py.File(path, "r") as f:
        X, Y = f["X"], f["Y"]
        if not isinstance(X, Dataset) or not isinstance(Y, Dataset):
            raise TypeError(f"File {str(path)} is not formatted correctly.")
        return X[:], Y[:]
