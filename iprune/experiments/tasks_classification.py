from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class ToyVisionConfig:
    n_qubits: int = 4
    data_dir: str = "./data"
    seed: int = 42
    download: bool = True

    # pools (paper: D from validation; but readout needs train)
    n_train_pool: int = 4000
    n_val_pool: int = 2000


# -----------------------------
#  Helpers
# -----------------------------
def _subsample(states: np.ndarray, labels: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(states) <= n:
        return states, labels
    idx = rng.choice(len(states), size=n, replace=False)
    return states[idx], labels[idx]


def downsample_to_square(img: np.ndarray, side: int) -> np.ndarray:
    """
    img: (1,H,W) or (H,W) or (C,H,W) with C=1
    Downsample to (side, side) by block-mean with edge padding if needed.
    Works for MNIST/Fashion (28x28) and general image sizes.
    """
    x = np.asarray(img, dtype=np.float64)
    if x.ndim == 3:
        # (C,H,W) -> (H,W) assuming C=1
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D image after squeezing channel, got shape {x.shape}")

    H, W = x.shape

    # Pad to multiples of side (edge padding keeps coarse structure)
    padH = (side - (H % side)) % side
    padW = (side - (W % side)) % side
    if padH > 0 or padW > 0:
        x = np.pad(x, ((0, padH), (0, padW)), mode="edge")
        H, W = x.shape

    bh = H // side
    bw = W // side
    out = np.zeros((side, side), dtype=np.float64)
    for i in range(side):
        for j in range(side):
            block = x[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            out[i, j] = float(block.mean())
    return out


# -----------------------------
#  Encoding: image -> state
# -----------------------------
def amplitude_encode_to_state(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Amplitude encoding into |psi> in C^{2^n}.
    If input is image-like and 2^n is a perfect square, we downsample to side x side
    where side = sqrt(2^n), then flatten to length 2^n.
    """
    dim = 2 ** n_qubits
    side = int(np.sqrt(dim))
    arr = np.asarray(x, dtype=np.float64)

    # If dim is a perfect square and x looks like an image, downsample to side x side
    if side * side == dim and arr.ndim in (2, 3):
        arr = downsample_to_square(arr, side)  # e.g., 4x4 (4q), 16x16 (8q), 32x32 (10q)

    v = arr.reshape(-1)
    # Safety: truncate/pad to dim
    if v.size > dim:
        v = v[:dim]
    elif v.size < dim:
        v = np.pad(v, (0, dim - v.size), mode="constant")

    norm = np.linalg.norm(v)
    if norm < 1e-12:
        v = np.zeros_like(v)
        v[0] = 1.0
        norm = 1.0
    return (v / norm).astype(np.complex128)


# -----------------------------
#  Torchvision loaders (MNIST/Fashion)
# -----------------------------
def _load_torchvision_two_class(
    dataset_name: str,
    cfg: ToyVisionConfig,
    classes: Tuple[int, int],
    train: bool,
    max_keep: int,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError(
            "MNIST/FashionMNIST require torch+torchvision. "
            "Install: pip install torch torchvision"
        ) from e

    tfm = transforms.Compose([transforms.ToTensor()])  # (1,28,28) in [0,1]
    if dataset_name == "MNIST":
        ds = datasets.MNIST(root=cfg.data_dir, train=train, download=cfg.download, transform=tfm)
    elif dataset_name == "FashionMNIST":
        ds = datasets.FashionMNIST(root=cfg.data_dir, train=train, download=cfg.download, transform=tfm)
    else:
        raise ValueError(dataset_name)

    wanted = set(classes)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for img, y in ds:
        yy = int(y)
        if yy in wanted:
            X_list.append(img.numpy())  # (1,28,28)
            y_list.append(yy)
            if len(y_list) >= max_keep:
                break

    X = np.stack(X_list, axis=0)  # (N,1,28,28)
    y = np.array(y_list, dtype=int)
    y = (y == classes[1]).astype(int)  # map to {0,1}
    return X, y


# -----------------------------
#  MNIST 4 vs 9 splits
# -----------------------------
def build_mnist49_splits(cfg: ToyVisionConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xtr, ytr = _load_torchvision_two_class("MNIST", cfg, classes=(4, 9), train=True, max_keep=cfg.n_train_pool)
    Xva, yva = _load_torchvision_two_class("MNIST", cfg, classes=(4, 9), train=False, max_keep=cfg.n_val_pool)

    tr_states = np.stack([amplitude_encode_to_state(Xtr[i], cfg.n_qubits) for i in range(len(Xtr))], axis=0)
    va_states = np.stack([amplitude_encode_to_state(Xva[i], cfg.n_qubits) for i in range(len(Xva))], axis=0)
    return tr_states, ytr, va_states, yva


# -----------------------------
#  Fashion Sandal vs Boot splits
#  labels: Sandal=5, Ankle boot=9
# -----------------------------
def build_fashion_sandal_boot_splits(cfg: ToyVisionConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xtr, ytr = _load_torchvision_two_class("FashionMNIST", cfg, classes=(5, 9), train=True, max_keep=cfg.n_train_pool)
    Xva, yva = _load_torchvision_two_class("FashionMNIST", cfg, classes=(5, 9), train=False, max_keep=cfg.n_val_pool)

    tr_states = np.stack([amplitude_encode_to_state(Xtr[i], cfg.n_qubits) for i in range(len(Xtr))], axis=0)
    va_states = np.stack([amplitude_encode_to_state(Xva[i], cfg.n_qubits) for i in range(len(Xva))], axis=0)
    return tr_states, ytr, va_states, yva


# -----------------------------
#  Bars & Stripes (BAS) splits
# -----------------------------
def generate_bars_and_stripes(n_samples: int, H: int, W: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 1, H, W), dtype=np.float64)
    y = np.zeros((n_samples,), dtype=int)

    for i in range(n_samples):
        is_stripes = int(rng.integers(0, 2))  # 0 bars, 1 stripes
        y[i] = is_stripes

        if is_stripes == 1:
            cols = rng.integers(0, 2, size=(W,))
            img = np.tile(cols[None, :], (H, 1))
        else:
            rows = rng.integers(0, 2, size=(H,))
            img = np.tile(rows[:, None], (1, W))

        X[i, 0] = img

    return X, y


def build_bas_splits(cfg: ToyVisionConfig, H: int = 4, W: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # different seeds to avoid leakage
    Xtr, ytr = generate_bars_and_stripes(cfg.n_train_pool, H, W, seed=cfg.seed + 123)
    Xva, yva = generate_bars_and_stripes(cfg.n_val_pool, H, W, seed=cfg.seed + 999)

    tr_states = np.stack([amplitude_encode_to_state(Xtr[i], cfg.n_qubits) for i in range(len(Xtr))], axis=0)
    va_states = np.stack([amplitude_encode_to_state(Xva[i], cfg.n_qubits) for i in range(len(Xva))], axis=0)
    return tr_states, ytr, va_states, yva


# -----------------------------
#  Paper: sample task ensemble D from validation (uniform), size M
# -----------------------------
def sample_task_ensemble_from_validation(va_states: np.ndarray, va_labels: np.ndarray, M: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    return _subsample(va_states, va_labels, M, seed=seed)
