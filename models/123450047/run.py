
"""
Capstone Deep Learning - Transformer Encoder
NIM: 123450047
Task: Regresi tinggi air dari citra wadah.

Jalankan:
python models/123450047/run.py --seed 1
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ==================================================
# 1. KONFIGURASI PROYEK
# ==================================================

AKAR = Path(__file__).resolve().parents[2]

if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))

from src.protokol import data

import torch
from torch import nn
from torch.nn import functional as F

NIM = Path(__file__).resolve().parent.name
JALUR = "trf"

EPOCH = 5
BATCH_SIZE = 16
D_MODEL = 64
JUMLAH_HEAD = 4
JUMLAH_ENCODER = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT = 0.1


# ==================================================
# 2. MENYIAPKAN FORMAT CITRA
# ==================================================

def ke_nchw(X, nama="data"):
    """
    Mengubah citra menjadi format:
    (jumlah_data, channel, tinggi, lebar)

    Mendukung:
    - (N, H, W)
    - (N, C, H, W)
    - (N, H, W, C)
    """

    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 3:
        X = X[:, None, :, :]

    elif X.ndim == 4:

        if X.shape[1] in (1, 3, 4):
            pass

        elif X.shape[-1] in (1, 3, 4):
            X = np.transpose(X, (0, 3, 1, 2))

        else:
            raise ValueError(
                f"Format channel {nama} tidak dikenali: {X.shape}"
            )

    else:
        raise ValueError(
            f"Dimensi {nama} tidak didukung: {X.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError(f"{nama} mengandung NaN atau Inf.")

    # Jika piksel masih dalam skala 0-255.
    if X.size and X.min() >= 0 and X.max() > 1.5:
        X = X / 255.0

    return np.ascontiguousarray(X)


# ==================================================
# 3. NORMALISASI DATA
# ==================================================

def normalisasi_train_only(Xtr, Xva, Xte):
    """
    Mean dan standar deviasi hanya dihitung
    dari data training untuk mencegah data leakage.
    """

    mean = Xtr.mean(
        axis=(0, 2, 3),
        keepdims=True
    )

    std = Xtr.std(
        axis=(0, 2, 3),
        keepdims=True
    )

    std = np.where(std < 1e-6, 1.0, std)

    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    if Xte is not None:
        Xte = (Xte - mean) / std

    return Xtr, Xva, Xte


# ==================================================
# 4. POSITIONAL ENCODING
# ==================================================

def positional_encoding(jumlah_token, dimensi, device):
    """
    Sinusoidal positional encoding.
    Dibuat dinamis sesuai jumlah patch citra.
    """

    posisi = torch.arange(
        jumlah_token,
        device=device,
        dtype=torch.float32
    ).unsqueeze(1)

    pembagi = torch.exp(
        torch.arange(
            0,
            dimensi,
            2,
            device=device,
            dtype=torch.float32
        ) * (-math.log(10000.0) / dimensi)
    )

    encoding = torch.zeros(
        jumlah_token,
        dimensi,
        device=device
    )

    encoding[:, 0::2] = torch.sin(
        posisi * pembagi
    )

    encoding[:, 1::2] = torch.cos(
        posisi * pembagi[:encoding[:, 1::2].shape[1]]
    )

    return encoding.unsqueeze(0)


# ==================================================
# 5. TRANSFORMER ENCODER
# ==================================================

class PatchTransformerRegressor(nn.Module):

    def __init__(
        self,
        in_channels,
        patch_size,
        d_model=D_MODEL
    ):

        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model

        # Mengubah patch citra menjadi embedding.
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=JUMLAH_HEAD,
            dim_feedforward=128,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=JUMLAH_ENCODER
        )

        self.norm = nn.LayerNorm(d_model)

        # Regression head: menghasilkan satu nilai kontinu.
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Jumlah channel tidak sesuai: "
                f"{x.shape[1]} vs {self.in_channels}"
            )

        tinggi, lebar = x.shape[-2:]

        # Padding agar dimensi dapat dibagi patch_size.
        pad_h = (
            self.patch_size - tinggi % self.patch_size
        ) % self.patch_size

        pad_w = (
            self.patch_size - lebar % self.patch_size
        ) % self.patch_size

        if pad_h or pad_w:
            x = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="replicate"
            )

        # Patch embedding.
        x = self.patch_embedding(x)

        # (B, D, H, W) -> (B, jumlah_patch, D)
        x = x.flatten(2).transpose(1, 2)

        # Positional encoding.
        posisi = positional_encoding(
            x.shape[1],
            self.d_model,
            x.device
        )

        x = x + posisi

        # Transformer encoder.
        x = self.encoder(x)

        # Mean pooling.
        x = x.mean(dim=1)

        x = self.norm(x)

        # Prediksi regresi.
        output = self.regression_head(x)

        return output.squeeze(1)


# ==================================================
# 6. MEMBUAT MODEL DAN MELATIH
# ==================================================

def latih_dan_prediksi(seed, final):

    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.set_num_threads(1)

    # Memastikan tugas adalah regresi.
    if data.TUGAS != "regresi":
        raise ValueError(
            "TUGAS harus regresi. "
            "Periksa protokol data tim."
        )

    # Memuat data melalui protokol resmi.
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    # Konversi citra.
    Xtr = ke_nchw(Xtr, "Xtr")
    Xva = ke_nchw(Xva, "Xva")

    if Xte is not None:
        Xte = ke_nchw(Xte, "Xte")

    ytr = np.asarray(ytr, dtype=np.float32).reshape(-1)
    yva = np.asarray(yva, dtype=np.float32).reshape(-1)

    if len(Xtr) != len(ytr):
        raise ValueError("Jumlah X_train dan y_train berbeda.")

    if len(Xva) != len(yva):
        raise ValueError("Jumlah X_val dan y_val berbeda.")

    if Xtr.shape[1] != Xva.shape[1]:
        raise ValueError("Jumlah channel training dan validasi berbeda.")

    # Normalisasi hanya dari data training.
    Xtr, Xva, Xte = normalisasi_train_only(
        Xtr,
        Xva,
        Xte
    )

    print("\nBENTUK DATA")
    print("Training :", Xtr.shape)
    print("Target   :", ytr.shape)
    print("Validasi :", Xva.shape)

    # Normalisasi target menggunakan statistik training.
    y_mean = float(ytr.mean())
    y_std = float(ytr.std())

    if y_std < 1e-6:
        y_std = 1.0

    ytr_scaled = (ytr - y_mean) / y_std

    # Patch dinamis:
    # 8x8   -> patch 2
    # 224x224 -> patch 16
    patch_size = (
        16 if min(Xtr.shape[-2:]) >= 16 else 2
    )

    print("\nKONFIGURASI TRANSFORMER")
    print("Input channel :", Xtr.shape[1])
    print("Ukuran citra  :", Xtr.shape[-2:])
    print("Patch size    :", patch_size)
    print("Embedding     :", D_MODEL)
    print("Attention     :", JUMLAH_HEAD)
    print("Encoder layer :", JUMLAH_ENCODER)

    device = torch.device("cpu")

    model = PatchTransformerRegressor(
        in_channels=Xtr.shape[1],
        patch_size=patch_size
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Training.
    for epoch in range(EPOCH):

        model.train()

        urutan = np.random.permutation(len(Xtr))

        total_loss = 0.0
        jumlah_batch = 0

        for awal in range(0, len(Xtr), BATCH_SIZE):

            indeks = urutan[
                awal:awal + BATCH_SIZE
            ]

            xb = torch.from_numpy(
                Xtr[indeks]
            ).to(device)

            yb = torch.from_numpy(
                ytr_scaled[indeks]
            ).to(device)

            optimizer.zero_grad()

            prediksi = model(xb)

            loss = F.mse_loss(
                prediksi,
                yb
            )

            loss.backward()

            # Gradient clipping.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            jumlah_batch += 1

        rata_loss = total_loss / max(jumlah_batch, 1)

        print(
            f"Epoch {epoch + 1}/{EPOCH} "
            f"- Loss: {rata_loss:.6f}"
        )

    # Prediksi validasi.
    model.eval()

    def prediksi_data(X):

        if X is None:
            return None

        with torch.no_grad():

            tensor = torch.from_numpy(
                X
            ).to(device)

            hasil = model(tensor).cpu().numpy()

        # Kembalikan ke satuan tinggi air (mm).
        hasil = hasil.reshape(-1) * y_std + y_mean

        return hasil.astype(np.float32)

    pred_val = prediksi_data(Xva)

    pred_test = None

    if final and Xte is not None:
        pred_test = prediksi_data(Xte)

    print("\nBENTUK OUTPUT")
    print("Prediksi validasi:", pred_val.shape)

    if pred_test is not None:
        print("Prediksi testing :", pred_test.shape)

    return pred_val, pred_test


# ==================================================
# 7. EVALUASI DAN PENYIMPANAN HASIL
# ==================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )

    parser.add_argument(
        "--final",
        action="store_true"
    )

    parser.add_argument(
        "--perangkat",
        default="cpu laptop"
    )

    args = parser.parse_args()

    print("=" * 55)
    print("CAPSTONE DEEP LEARNING - TRANSFORMER")
    print("NIM:", NIM)
    print("Seed:", args.seed)
    print("=" * 55)

    waktu_mulai = time.time()

    pred_val, pred_test = latih_dan_prediksi(
        args.seed,
        args.final
    )

    durasi = time.time() - waktu_mulai

    # Folder hasil.
    output = AKAR / "results" / NIM

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    # Simpan prediksi validasi.
    pred_val_path = output / "pred_val.npy"

    np.save(
        pred_val_path,
        pred_val
    )

    print("\nPrediksi validasi disimpan:")
    print(pred_val_path)

    # Perintah evaluator resmi.
    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(pred_val_path),
        "--durasi", f"{durasi:.2f}",
        "--perangkat", args.perangkat
    ]

    # Prediksi testing hanya jika diminta dan datanya tersedia.
    if pred_test is not None:

        pred_test_path = output / "pred_test.npy"

        np.save(
            pred_test_path,
            pred_test
        )

        perintah += [
            "--pred-test",
            str(pred_test_path)
        ]

    # Evaluasi menggunakan modul resmi.
    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR
    )

    print("\n" + "=" * 55)
    print("PENGUJIAN TRANSFORMER BERHASIL")
    print("=" * 55)

    print("NIM      :", NIM)
    print("Model    :", JALUR)
    print("Seed     :", args.seed)
    print("Durasi   :", round(durasi, 2), "detik")
    print("Prediksi :", pred_val_path)
    print("Metrics  :", output / "metrics.json")


if __name__ == "__main__":
    main()
