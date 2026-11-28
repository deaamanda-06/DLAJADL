"""Pemuat citra tinggi air Tim 02 (data primer meja).

Membaca data/cropped/dataset.npz yang dibuat oleh `python -m src.protokol.pemotong`
dari sesi data/sesi-20261116-095756 (ROI Wadah B -> 224x224).
Data uji = satu blok (siklus pengisian) penuh; validasi dipisah per kondisi
(seed 2026); tidak ada kebocoran frame antar subset.

Kasus: Taksiran tinggi air di wadah B dari citra (regresi mm).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TUGAS = "regresi"
UKURAN = (224, 224)
SEED = 2026


def _akar_proyek() -> Path:
    return Path(__file__).resolve().parents[2]


def muat():
    """Mengembalikan X/y latih, validasi, dan uji; X berbentuk N,C,224,224."""
    path = _akar_proyek() / "data" / "cropped" / "dataset.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan: python -m src.protokol.pemotong "
            "(setelah mengekstrak tim02_sesi-20261116-095756.tar.gz ke data/)."
        )
    d = np.load(path)
    X = {k: np.asarray(d[f"X_{k}"], dtype=np.float32) for k in ("train", "val", "test")}
    y = {k: np.asarray(d[f"y_{k}"], dtype=np.float32).reshape(-1) for k in ("train", "val", "test")}

    for k in X:
        if X[k].ndim != 4 or X[k].shape[1] != 3 or X[k].shape[2:] != UKURAN:
            raise ValueError(f"Bentuk X_{k} tidak sesuai (N,3,224,224): {X[k].shape}")
        if len(X[k]) != len(y[k]) or len(X[k]) == 0:
            raise ValueError(f"Jumlah X_{k} dan y_{k} tidak sesuai/kosong.")

    # Statistik kanal dihitung HANYA dari data latih.
    mu = X["train"].mean(axis=(0, 2, 3), keepdims=True)
    sigma = np.maximum(X["train"].std(axis=(0, 2, 3), keepdims=True), 1e-6)
    for k in X:
        X[k] = ((X[k] - mu) / sigma).astype(np.float32)

    print("Jumlah sampel:", {k: len(v) for k, v in y.items()})
    return X["train"], y["train"], X["val"], y["val"], X["test"], y["test"]