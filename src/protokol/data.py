"""Pemuat data kasus tim — MILIK TIM, dibekukan setelah Sprint 0 dengan persetujuan pembimbing.

Isi oleh Penjaga Data & Evaluasi. Kontrak yang dipakai penilai dan baseline starter:

    TUGAS = "klasifikasi"  # atau "regresi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)

- X berupa array NumPy (bentuk bebas: [N, fitur], [N, C, H, W], [N, langkah, fitur]);
  y klasifikasi = label bulat 0..K-1, y regresi = float.
- Split latih/validasi WAJIB memakai ``src.common.split`` (seed 2026) agar sama di semua laptop.
- y_test boleh ``None`` bila dataset tidak punya bagian uji; bila ada, data uji hanya dievaluasi
  sekali di sprint terakhir.
- Berkas data dibaca dari ``data/`` dan dicantumkan di ``data/MANIFEST.json`` (sidik data).

Contoh lengkap: ``contoh_fashion_mnist.py`` di folder ini.
"""
from __future__ import annotations

import numpy as np

from src.common import manifest, split

TUGAS = "klasifikasi"


import numpy as np


def muat():
    """Memuat data simulasi citra dan label (tinggi_air_mm) untuk pengujian kerangka model."""
    np.random.seed(42)

    # Simulasi 40 sampel latih, 10 sampel validasi, dan 10 sampel uji
    # Citra simulasi: shape (N, 224, 224, 3)
    Xtr = np.random.rand(40, 224, 224, 3).astype(np.float32)
    ytr = np.random.uniform(10.0, 200.0, size=(40,)).astype(np.float32)

    Xva = np.random.rand(10, 224, 224, 3).astype(np.float32)
    yva = np.random.uniform(10.0, 200.0, size=(10,)).astype(np.float32)

    Xte = np.random.rand(10, 224, 224, 3).astype(np.float32)
    yte = np.random.uniform(10.0, 200.0, size=(10,)).astype(np.float32)

    return Xtr, ytr, Xva, yva, Xte, yte
    
