"""Model RNN-GRU (dua arah) untuk regresi tinggi air dari citra 224x224.

Jalankan dari akar repo:
    python models/123450043/run.py --seed 1
    python models/123450043/run.py --seed 1 --final   # hanya sprint terakhir
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data  # noqa: E402  (protokol tim, dibekukan)

NIM = "123450043"
JALUR = "rnn"
EPOCH = 30
BATCH = 16
LR = 1e-3
POOL = 4          # 224 -> 56 langkah waktu
HIDDEN = 64


def _ke_float(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.max() > 1.5:       # jaga-jaga kalau datanya masih 0-255
        X = X / 255.0
    return X


class GRURegresi(nn.Module):
    """Citra dibaca baris demi baris sebagai deret: (N,3,224,224) -> (N,56,168)."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.gru = nn.GRU(3 * (224 // POOL), hidden, batch_first=True, bidirectional=True)
        self.kepala = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        x = F.avg_pool2d(x, POOL)                 # (N,3,56,56)
        x = x.permute(0, 2, 1, 3).contiguous()    # (N,56,3,56)
        x = x.flatten(2)                          # (N,56,168)
        out, _ = self.gru(x)
        return self.kepala(out.mean(dim=1)).squeeze(-1)


def _prediksi(model, X, mu_x, sd_x, mu_y, sd_y, device) -> np.ndarray:
    model.eval()
    hasil = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = torch.from_numpy((X[i:i + 64] - mu_x) / sd_x).to(device)
            hasil.append(model(xb).cpu().numpy())
    return np.concatenate(hasil) * sd_y + mu_y      # kembali ke mm


def latih_dan_prediksi(seed: int, final: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    Xtr, ytr, Xva, yva, Xte, yte = data.muat()
    Xtr, Xva = _ke_float(Xtr), _ke_float(Xva)
    ytr, yva = np.asarray(ytr, dtype=np.float32), np.asarray(yva, dtype=np.float32)
    print(f"frame: train={len(Xtr)} val={len(Xva)} test={0 if Xte is None else len(Xte)}")

    mu_x = Xtr.mean(axis=(0, 2, 3), keepdims=True)
    sd_x = Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    mu_y, sd_y = float(ytr.mean()), float(ytr.std() + 1e-6)

    Xtr_n = (Xtr - mu_x) / sd_x
    ytr_n = (ytr - mu_y) / sd_y

    model = GRURegresi().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.SmoothL1Loss()

    terbaik, pred_val_terbaik, state_terbaik = float("inf"), None, None
    for ep in range(1, EPOCH + 1):
        model.train()
        idx = np.random.permutation(len(Xtr_n))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            xb = torch.from_numpy(Xtr_n[b]).to(device)
            yb = torch.from_numpy(ytr_n[b]).to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        p_val = _prediksi(model, Xva, mu_x, sd_x, mu_y, sd_y, device)
        mae = float(np.abs(yva - p_val).mean())
        if mae < terbaik:
            terbaik, pred_val_terbaik = mae, p_val.copy()
            state_terbaik = {k: v.clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 5 == 0:
            print(f"ep {ep:03d}/{EPOCH} | val MAE {mae:.4f} mm (terbaik {terbaik:.4f} mm)")

    pred_test = None
    if final and Xte is not None:
        model.load_state_dict(state_terbaik)
        pred_test = _prediksi(model, _ke_float(Xte), mu_x, sd_x, mu_y, sd_y, device)

    return pred_val_terbaik, pred_test


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
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat,
    ]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()