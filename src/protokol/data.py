"""Pemuat data resmi Tim 02 (Ruang Cerdas 5 Volt).

Kasus: Taksiran tinggi air di wadah B dari citra (regresi mm).
"""
from __future__ import annotations

import numpy as np

from src.protokol import pemotong

TUGAS = "regresi"


def muat():
    """Memuat (X_train, y_train, X_val, y_val, X_test, y_test) hasil pemotongan ROI 224x224."""
    if not pemotong.DATASET_PATH.exists():
        pemotong.potong()
    d = np.load(pemotong.DATASET_PATH)
    return (
        d["X_train"],
        d["y_train"],
        d["X_val"],
        d["y_val"],
        d["X_test"],
        d["y_test"],
    )
