
"""Model Bidirectional GRU untuk regresi tinggi air dalam mm.

Dataset:
    data/cropped_blok_mm/dataset.npz

Pembagian data:
    Blok 1: Training
    Blok 2: Validation
    Blok 3: Testing (tidak dievaluasi pada tahap ini)

Jalankan dari akar repo:
    python models/123450043/run_blok.py --seed 1
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data_blok as data


# ============================================================
# KONFIGURASI MODEL
# ============================================================

NIM = "123450043"
JALUR = "rnn"

EPOCH = 30
BATCH = 16
LR = 1e-3
POOL = 4
HIDDEN = 64


# ============================================================
# PREPROCESSING
# ============================================================

def _ke_float(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)

    if X.max() > 1.5:
        X = X / 255.0

    return X


# ============================================================
# ARSITEKTUR BIDIRECTIONAL GRU
# ============================================================

class GRURegresi(nn.Module):
    """Citra dibaca sebagai deret baris.

    Input:
        (N, 3, 224, 224)

    Setelah pooling dan reshape:
        (N, 56, 168)

    Output:
        Prediksi tinggi air dalam skala normalisasi.
    """

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()

        self.gru = nn.GRU(
            input_size=3 * (224 // POOL),
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.kepala = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2 * hidden, 1),
        )

    def forward(self, x):

        # (N, 3, 224, 224) -> (N, 3, 56, 56)
        x = F.avg_pool2d(x, POOL)

        # (N, 3, 56, 56) -> (N, 56, 3, 56)
        x = x.permute(0, 2, 1, 3).contiguous()

        # (N, 56, 3, 56) -> (N, 56, 168)
        x = x.flatten(2)

        out, _ = self.gru(x)

        return self.kepala(
            out.mean(dim=1)
        ).squeeze(-1)


# ============================================================
# FUNGSI PREDIKSI
# ============================================================

def _prediksi(
    model,
    X,
    mu_x,
    sd_x,
    mu_y,
    sd_y,
    device,
) -> np.ndarray:

    model.eval()
    hasil = []

    with torch.no_grad():

        for i in range(0, len(X), 64):

            xb = torch.from_numpy(
                (X[i:i + 64] - mu_x) / sd_x
            ).to(device)

            hasil.append(
                model(xb).cpu().numpy()
            )

    # Mengembalikan prediksi ke satuan mm.
    return np.concatenate(hasil) * sd_y + mu_y


# ============================================================
# PELATIHAN DAN VALIDASI MODEL
# ============================================================

def latih_dan_prediksi(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")

    # Memuat dataset acuan berbasis blok.
    # Data test tidak digunakan pada tahap ini.
    Xtr, ytr, Xva, yva, _, _ = data.muat()

    Xtr = _ke_float(Xtr)
    Xva = _ke_float(Xva)

    ytr = np.asarray(ytr, dtype=np.float32)
    yva = np.asarray(yva, dtype=np.float32)

    info = data.baca_info()

    print("=" * 60)
    print("PELATIHAN BIDIRECTIONAL GRU TIM 02")
    print("=" * 60)

    print("NIM          :", NIM)
    print("Seed         :", seed)
    print("Perangkat    :", device)
    print("Data hash    :", info["data_hash"])
    print("Satuan target: mm")
    print("Train        :", Xtr.shape)
    print("Validation   :", Xva.shape)
    print("Test         : tidak dievaluasi")
    print("Epoch        :", EPOCH)
    print("Batch size   :", BATCH)
    print("Learning rate:", LR)

    # Memastikan dataset yang benar digunakan.
    assert Xtr.shape == (125, 3, 224, 224)
    assert Xva.shape == (123, 3, 224, 224)
    assert ytr.shape == (125,)
    assert yva.shape == (123,)

    assert np.isfinite(Xtr).all()
    assert np.isfinite(Xva).all()
    assert np.isfinite(ytr).all()
    assert np.isfinite(yva).all()

    # Normalisasi dihitung hanya dari data training.
    mu_x = Xtr.mean(
        axis=(0, 2, 3),
        keepdims=True,
    )

    sd_x = Xtr.std(
        axis=(0, 2, 3),
        keepdims=True,
    ) + 1e-6

    mu_y = float(ytr.mean())
    sd_y = float(ytr.std() + 1e-6)

    Xtr_n = (Xtr - mu_x) / sd_x
    ytr_n = (ytr - mu_y) / sd_y

    model = GRURegresi().to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    loss_fn = nn.SmoothL1Loss()

    terbaik = float("inf")
    pred_val_terbaik = None
    epoch_terbaik = None

    print("\nMEMULAI PELATIHAN")
    print("-" * 60)

    for ep in range(1, EPOCH + 1):

        model.train()

        idx = np.random.permutation(
            len(Xtr_n)
        )

        for i in range(0, len(idx), BATCH):

            b = idx[i:i + BATCH]

            xb = torch.from_numpy(
                Xtr_n[b]
            ).to(device)

            yb = torch.from_numpy(
                ytr_n[b]
            ).to(device)

            opt.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            opt.step()

        # Evaluasi pada blok validation.
        p_val = _prediksi(
            model,
            Xva,
            mu_x,
            sd_x,
            mu_y,
            sd_y,
            device,
        )

        if not np.isfinite(p_val).all():
            raise ValueError(
                "Prediksi validation mengandung NaN atau Inf."
            )

        mae = float(
            np.abs(yva - p_val).mean()
        )

        if mae < terbaik:

            terbaik = mae
            pred_val_terbaik = p_val.copy()
            epoch_terbaik = ep

        if ep == 1 or ep % 5 == 0:

            print(
                f"Epoch {ep:03d}/{EPOCH} | "
                f"Val MAE: {mae:.4f} mm | "
                f"Terbaik: {terbaik:.4f} mm"
            )

    if pred_val_terbaik is None:
        raise RuntimeError(
            "Pelatihan gagal menghasilkan prediksi validation."
        )

    print("\nHASIL PELATIHAN")
    print("-" * 60)
    print("Epoch terbaik :", epoch_terbaik)
    print(f"MAE terbaik   : {terbaik:.6f} mm")

    return pred_val_terbaik


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--perangkat",
        default="cpu laptop",
    )

    a = ap.parse_args()

    t0 = time.time()

    pred_val = latih_dan_prediksi(
        a.seed
    )

    durasi = time.time() - t0

    # Hasil baru dipisahkan dari hasil sebelumnya.
    out = AKAR / "results" / NIM / "blok_mm"

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        out / "pred_val.npy",
        np.asarray(pred_val),
    )

    # Evaluasi menggunakan metrik regresi starter,
    # dengan hash dataset acuan yang sebenarnya.
    data.simpan_metrik_validasi(
        pred_val=pred_val,
        seed=a.seed,
        model=JALUR,
        durasi=durasi,
        perangkat=a.perangkat,
        out=out,
    )

    print("\n" + "=" * 60)
    print("PELATIHAN SELESAI")
    print("=" * 60)

    print(f"Durasi     : {durasi:.1f} detik")
    print("Prediksi   :", out / "pred_val.npy")
    print("Metrics    :", out / "metrics.json")
    print("STATUS     : BERHASIL")


if __name__ == "__main__":
    main()
