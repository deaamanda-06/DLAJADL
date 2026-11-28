
"""Transfer Learning Regression - Smoke Test Sprint 1."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"


def latih_dan_prediksi(seed, final=False):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xva = np.asarray(Xva, dtype=np.float32)

    if Xtr.ndim == 3:
        Xtr = Xtr[:, None, :, :]
        Xva = Xva[:, None, :, :]

    ytr = np.asarray(ytr, dtype=np.float32).reshape(-1)

    y_mean = float(ytr.mean())
    y_std = float(ytr.std()) + 1e-6
    y_norm = (ytr - y_mean) / y_std

    # Backbone sementara untuk smoke test.
    # Belum menggunakan bobot pretrained.
    class TLRegresi(nn.Module):
        def __init__(self, channels):
            super().__init__()

            self.backbone = nn.Sequential(
                nn.Conv2d(channels, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )

            self.regressor = nn.Sequential(
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.regressor(
                self.backbone(x)
            ).squeeze(1)

    model = TLRegresi(Xtr.shape[1])

    dataset = TensorDataset(
        torch.tensor(Xtr),
        torch.tensor(y_norm)
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    loss_fn = nn.MSELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/10 "
            f"- Loss: {total_loss / len(loader):.6f}"
        )

    model.eval()

    with torch.no_grad():
        pred_val = (
            model(torch.tensor(Xva)).numpy()
            * y_std + y_mean
        ).astype(np.float32)

        pred_test = None

        if final and Xte is not None:
            Xte = np.asarray(Xte, dtype=np.float32)

            if Xte.ndim == 3:
                Xte = Xte[:, None, :, :]

            pred_test = (
                model(torch.tensor(Xte)).numpy()
                * y_std + y_mean
            ).astype(np.float32)

    return pred_val, pred_test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--perangkat", default="cpu laptop")

    args = parser.parse_args()

    if data.TUGAS != "regresi":
        raise ValueError("Kasus tim harus regresi.")

    print("=" * 45)
    print("TRANSFER LEARNING - REGRESSION")
    print("NIM:", NIM)
    print("Seed:", args.seed)
    print("=" * 45)

    awal = time.time()

    pred_val, pred_test = latih_dan_prediksi(
        args.seed, args.final
    )

    durasi = time.time() - awal

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)

    pred_path = out / "pred_val.npy"
    np.save(pred_path, pred_val)

    perintah = [
        sys.executable,
        "-m", "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(pred_path),
        "--durasi", f"{durasi:.2f}",
        "--perangkat", args.perangkat
    ]

    if pred_test is not None:
        test_path = out / "pred_test.npy"
        np.save(test_path, pred_test)
        perintah += ["--pred-test", str(test_path)]

    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR
    )

    print("\nPENGUJIAN TL BERHASIL")
    print("Prediksi:", pred_path)
    print("Durasi:", round(durasi, 2), "detik")


if __name__ == "__main__":
    main()
