"""Jalur Model CNN dari Nol ? Dea Amanda (123450006).

Menaksir tinggi air wadah B (mm) dari citra kamera meja 224x224.
Jalankan:
    python models/123450006/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "cnn"


class TinggiAirCNN(nn.Module):
    """Arsitektur CNN ringan untuk regresi tinggi permukaan air."""

    def __init__(self) -> None:
        super().__init__()
        self.fitur = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14, 1)),
        )
        self.kepala = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kepala(self.fitur(x)).squeeze(-1)


def _ke_tensor_citra(X: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.transpose(X, (0, 3, 1, 2)), dtype=torch.float32)


def latih_dan_prediksi(seed: int, final: bool) -> tuple[np.ndarray, np.ndarray | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()
    print(f"[CNN] Data dimuat -> Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")

    ds_train = TensorDataset(_ke_tensor_citra(Xtr), torch.tensor(ytr, dtype=torch.float32))
    loader_train = DataLoader(ds_train, batch_size=16, shuffle=True)

    t_val = _ke_tensor_citra(Xva).to(device)
    model = TinggiAirCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.L1Loss()

    best_mae = float("inf")
    best_pred_val = None
    best_state = None

    epochs = 25
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            p_val = model(t_val).cpu().numpy()
            val_mae = float(np.abs(yva - p_val).mean())

        if val_mae < best_mae:
            best_mae = val_mae
            best_pred_val = p_val.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep:02d}/{epochs} | Val MAE: {val_mae:.4f} mm (Terbaik: {best_mae:.4f} mm)")

    pred_test = None
    if final and Xte is not None and best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_test = model(_ke_tensor_citra(Xte).to(device)).cpu().numpy()

    return best_pred_val, pred_test


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
