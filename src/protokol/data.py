
"""Pemuat data simulasi untuk pengujian integrasi Sprint 1."""

import numpy as np

from src.common import manifest, split

TUGAS = "regresi"


def muat():
    m = manifest.baca()

    if len(m["berkas"]) != 1:
        raise ValueError("Manifest simulasi harus berisi satu berkas.")

    jalur = manifest.AKAR / m["berkas"][0]

    if not jalur.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {jalur}"
        )

    tabel = np.loadtxt(
        jalur,
        delimiter=",",
        skiprows=1,
        dtype=np.float32
    )

    if tabel.ndim != 2 or tabel.shape[1] != 65:
        raise ValueError(
            "Dataset harus memiliki 64 fitur citra dan 1 target."
        )

    X = tabel[:, :64].reshape(-1, 1, 8, 8)
    y = tabel[:, 64]

    idx_latih, idx_val = split.indeks(
        len(X),
        porsi_val=0.1
    )

    return (
        X[idx_latih],
        y[idx_latih],
        X[idx_val],
        y[idx_val],
        None,
        None
    )
