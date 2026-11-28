
"""Pembanding non-DL Tim 02.

Metode:
    1. Konversi citra RGB ke grayscale.
    2. Hitung profil kecerahan per baris.
    3. Haluskan profil dengan moving average.
    4. Deteksi kandidat garis air dari gradien terbesar.
    5. Regresi linear posisi garis air ke tinggi air (mm).

Dataset:
    data/cropped_blok_mm/dataset.npz

Jalankan:
    python models/123450043/baseline_blok.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data_blok as data


NIM = "123450043"
SEED = 1

SMOOTH = 5
MARGIN = 10


# ============================================================
# EKSTRAKSI POSISI KANDIDAT GARIS AIR
# ============================================================

def posisi_garis_air(img):

    img = np.asarray(
        img,
        dtype=np.float32,
    )

    if img.max() > 1.5:
        img = img / 255.0

    # RGB menjadi grayscale.
    gray = (
        0.299 * img[0]
        + 0.587 * img[1]
        + 0.114 * img[2]
    )

    # Profil kecerahan setiap baris.
    profil = gray.mean(axis=1)

    # Moving average.
    kernel = np.ones(
        SMOOTH,
        dtype=np.float32,
    ) / SMOOTH

    pad = SMOOTH // 2

    profil_halus = np.convolve(
        np.pad(profil, (pad, pad), mode="edge"),
        kernel,
        mode="valid",
    )

    # Perubahan kecerahan antarbaris.
    gradien = np.abs(
        np.gradient(profil_halus)
    )

    # Hindari bagian tepi ROI.
    awal = MARGIN
    akhir = len(gradien) - MARGIN

    posisi = awal + np.argmax(
        gradien[awal:akhir]
    )

    return float(posisi)


def ekstrak_fitur(X):

    hasil = [
        posisi_garis_air(img)
        for img in X
    ]

    return np.asarray(
        hasil,
        dtype=np.float64,
    ).reshape(-1, 1)


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():

    t0 = time.time()

    # Data test tidak digunakan.
    Xtr, ytr, Xva, yva, _, _ = data.muat()

    ytr = np.asarray(
        ytr,
        dtype=np.float64,
    )

    yva = np.asarray(
        yva,
        dtype=np.float64,
    )

    info = data.baca_info()

    print("=" * 60)
    print("PEMBANDING NON-DL TIM 02")
    print("=" * 60)

    print("Metode      : Profil kecerahan + regresi linear")
    print("Data hash   :", info["data_hash"])
    print("Satuan      : mm")
    print("Train       :", len(Xtr))
    print("Validation  :", len(Xva))
    print("Test        : tidak dievaluasi")

    assert Xtr.shape == (125, 3, 224, 224)
    assert Xva.shape == (123, 3, 224, 224)
    assert ytr.shape == (125,)
    assert yva.shape == (123,)

    # Ekstraksi posisi kandidat garis air.
    Ftr = ekstrak_fitur(Xtr)
    Fva = ekstrak_fitur(Xva)

    print("\nFITUR GARIS AIR")
    print("-" * 60)

    print("Shape fitur train:", Ftr.shape)
    print("Shape fitur val  :", Fva.shape)

    print(
        "Rentang posisi train:",
        Ftr.min(),
        "-",
        Ftr.max(),
        "piksel",
    )

    print(
        "Rentang posisi val  :",
        Fva.min(),
        "-",
        Fva.max(),
        "piksel",
    )

    print(
        "Posisi unik train:",
        len(np.unique(Ftr)),
    )

    # Regresi linear dilatih hanya pada blok train.
    model = LinearRegression()
    model.fit(Ftr, ytr)

    pred_val = model.predict(Fva)

    assert pred_val.shape == yva.shape
    assert np.isfinite(pred_val).all()

    mae = float(
        np.abs(yva - pred_val).mean()
    )

    # Pembanding tambahan: tebak rata-rata training.
    pred_rata = np.full(
        len(yva),
        ytr.mean(),
        dtype=np.float64,
    )

    mae_rata = float(
        np.abs(yva - pred_rata).mean()
    )

    durasi = time.time() - t0

    print("\nHASIL REGRESI LINEAR")
    print("-" * 60)

    print(
        "Koefisien :",
        float(model.coef_[0]),
    )

    print(
        "Intercept :",
        float(model.intercept_),
    )

    print(f"MAE regresi linear : {mae:.6f} mm")
    print(f"MAE tebak rata-rata: {mae_rata:.6f} mm")

    # Simpan hasil tanpa menimpa baseline lama.
    out = (
        AKAR
        / "results"
        / "baseline"
        / "blok_mm"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        out / "pred_val.npy",
        pred_val,
    )

    np.save(
        out / "posisi_train.npy",
        Ftr,
    )

    np.save(
        out / "posisi_val.npy",
        Fva,
    )

    data.simpan_metrik_validasi(
        pred_val=pred_val,
        seed=SEED,
        model="baseline",
        durasi=durasi,
        perangkat="cpu laptop",
        out=out,
    )

    print("\n" + "=" * 60)
    print("BASELINE SELESAI")
    print("=" * 60)

    print(f"Durasi  : {durasi:.2f} detik")
    print("Metrics :", out / "metrics.json")
    print("STATUS  : BERHASIL")


if __name__ == "__main__":
    main()
