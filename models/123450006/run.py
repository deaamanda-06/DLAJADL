"""Kerangka run.py anggota untuk jalur CNN (Modul 05)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name  # nama folder = NIM
JALUR = "cnn"  # cnn | tl | rnn | trf | ae


# ---- 1. ARSITEKTUR CNN REGRESI TINGGI AIR (MM) ----
class CNNWaterRegressor(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),  # Output 1 nilai regresi (tinggi_air_mm)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        return x


def ke_tensor(X: np.ndarray) -> torch.Tensor:
    """Mengubah numpy array X (N, H, W, C) menjadi PyTorch Tensor (N, C, H, W)."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[-1] in [1, 3]:  # (N, H, W, C) -> (N, C, H, W)
        arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.ndim == 3:  # (N, H, W) -> (N, 1, H, W)
        arr = np.expand_dims(arr, axis=1)
    elif arr.ndim == 2:
        arr = np.expand_dims(arr, axis=1)
    return torch.tensor(arr)


# ---- 2. FUNGSI PELATIHAN & PREDIKSI REGRESI ----
def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Mengembalikan (pred_val, pred_test|None) nilai tinggi_air_mm."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Memuat data dari src/protokol/data.py
    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    # Preprocessing ke Tensor
    X_train = ke_tensor(Xtr)
    y_train = torch.tensor(np.asarray(ytr, dtype=np.float32)).unsqueeze(1)
    X_val = ke_tensor(Xva)
    X_test = ke_tensor(Xte) if final and Xte is not None else None

    # Model, Loss Function (L1/MAE mm), dan Optimizer
    in_channels = X_train.shape[1] if X_train.ndim == 4 else 1
    model = CNNWaterRegressor(in_channels=in_channels)
    criterion = nn.L1Loss()  # Metrik evaluasi utama: MAE (mm)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Pelatihan Model (Simulasi)
    model.train()
    batch_size = 16
    epochs = 3

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            bx = X_train[i : i + batch_size]
            by = y_train[i : i + batch_size]

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    # Prediksi
    model.eval()
    with torch.no_grad():
        pred_val = model(X_val).squeeze(-1).cpu().numpy()
        pred_test = (
            model(X_test).squeeze(-1).cpu().numpy()
            if X_test is not None
            else None
        )

    return pred_val, pred_test


# ---- 3. EKSEKUSI UTAMA (EVALUASI AUTOMATIS HASIL DOSEN) ----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim",
        NIM,
        "--model",
        JALUR,
        "--seed",
        str(a.seed),
        "--pred-val",
        str(out / "pred_val.npy"),
        "--durasi",
        f"{durasi:.1f}",
        "--perangkat",
        a.perangkat,
    ]

    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()