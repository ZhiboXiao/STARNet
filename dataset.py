"""Paired NumPy arrays with explicit shapes and no implicit preprocessing."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_array(path):
    array = np.load(path, allow_pickle=False)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{path}: expected a real numeric array")
    array = np.array(array, dtype=np.float32, copy=True)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"{path}: expected finite [C, S, T] or [S, T] data")
    return torch.from_numpy(array)


class PairedArrayDataset(Dataset):
    def __init__(self, input_dir, target_dir, *, in_channels, sensors, expansion):
        input_dir, target_dir = Path(input_dir), Path(target_dir)
        if not input_dir.is_dir() or not target_dir.is_dir():
            raise ValueError("Input and target directories must exist.")
        inputs = {p.name: p for p in input_dir.glob("*.npy")}
        targets = {p.name: p for p in target_dir.glob("*.npy")}
        if not inputs or inputs.keys() != targets.keys():
            raise ValueError("Input/target folders must contain the same nonempty set of .npy filenames.")
        self.pairs = [(inputs[name], targets[name]) for name in sorted(inputs)]
        self.in_channels, self.sensors, self.expansion = in_channels, sensors, expansion

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]
        x, y = load_array(input_path), load_array(target_path)
        expected = (self.in_channels, self.sensors, x.shape[-1])
        target_shape = (1, self.sensors * self.expansion, x.shape[-1])
        if tuple(x.shape) != expected or x.shape[-1] < 16:
            raise ValueError(f"{input_path}: expected {expected} with T >= 16, got {tuple(x.shape)}")
        if tuple(y.shape) != target_shape:
            raise ValueError(f"{target_path}: expected {target_shape}, got {tuple(y.shape)}")
        return x, y
