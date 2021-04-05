from typing import Generator, Tuple

import numpy as np
import torch

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

SIZE_TR = 0.8
SIZE_VAL_OTHER = 0.5


def load_data() -> Tuple[np.ndarray]:
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)
    return X, y


def split_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    X_tr, X_other, y_tr, y_other = train_test_split(
        X, y,
        train_size=SIZE_TR,
        stratify=y
    )

    X_val, X_te, y_val, y_te = train_test_split(
        X_other, y_other,
        train_size=SIZE_VAL_OTHER,
        stratify=y_other
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    return X_tr, X_val, X_te, y_tr, y_val, y_te


def generate_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_epochs: int
) -> Generator:
    # NOTE: Generating a single batch is very fast, so don't use a torch DataLoader here.
    # TODO: Don't allow small final batch for train/valid.
    for _ in range(num_epochs):
        for i_start in range(0, len(X), batch_size):
            i_end = min(i_start + batch_size, len(X))
            Xb = X[i_start:i_end]
            yb = y[i_start:i_end]
            yield torch.from_numpy(Xb).float(), torch.from_numpy(yb).long()


def load_batch_generators(
    batch_size: int
) -> Tuple[Generator]:
    num_epochs = 1_000_000
    X, y = load_data()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)
    bg_tr = generate_batches(X_tr, y_tr, batch_size, num_epochs)
    bg_val = generate_batches(X_val, y_val, batch_size, num_epochs)
    bg_te = generate_batches(X_te, y_te, batch_size, 1)
    return bg_tr, bg_val, bg_te
