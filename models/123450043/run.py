
"""Model RNN-GRU untuk regresi tinggi air dari citra 224x224.

Pengujian:
python models/123450043/run.py --seed 1

Data simulasi hanya digunakan untuk smoke test ketika pemuat dataset
tim masih belum tersedia. Hasil simulasi disimpan terpisah.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
JALUR = "rnn"

UKURAN = 224
EPOCH = 5
BATCH_SIZE = 16
HIDDEN_SIZE = 48


# --------------------------------------------------
# 1. DATA SIMULASI
# --------------------------------------------------

def buat_data_simulasi(seed: int):
    """Membuat citra wadah sintetis untuk menguji alur regresi."""

    rng = np.random.default_rng(seed)

    def buat(n):
        posisi_air = rng.integers(24, 200, size=n)

        # Format citra: N, H, W, C
        X = np.empty(
            (n, UKURAN, UKURAN, 3),
            dtype=np.float32
        )

        for i, posisi in enumerate(posisi_air):

            # Bagian atas: udara
            X[i, :posisi, :, :] = (0.78, 0.78, 0.78)

            # Bagian bawah: air
            X[i, posisi:, :, :] = (0.12, 0.35, 0.72)

            # Garis permukaan air
            X[i, max(0, posisi-1):min(UKURAN, posisi+1), :, :] = (
                0.96, 0.96, 0.96
            )

        # Noise kecil agar data tidak terlalu seragam
        X += rng.normal(
            0, 0.01, X.shape
        ).astype(np.float32)

        X = np.clip(X, 0, 1)

        # Simulasi skala: 1 piksel = 1 mm.
        # Bukan kalibrasi penggaris sebenarnya.
        y = (UKURAN - posisi_air).astype(np.float32)

        return X, y

    X_train, y_train = buat(96)
    X_val, y_val = buat(32)

    return X_train, y_train, X_val, y_val, None, None


# --------------------------------------------------
# 2. MEMUAT DATA
# --------------------------------------------------

def siapkan_data(seed: int):

    try:
        hasil = data.muat()
        simulasi = False

    except NotImplementedError:
        print("Pemuat dataset belum tersedia.")
        print("Menggunakan data simulasi untuk smoke test.")

        hasil = buat_data_simulasi(seed)

        # Protokol simulasi: regresi
        data.TUGAS = "regresi"

        # Agar evaluator membaca data simulasi yang sama
        data.muat = lambda: hasil

        # Hash khusus data simulasi, bukan hash dataset asli
        from src.common import manifest

        def hash_simulasi(extra=b""):
            h = hashlib.sha256()

            for arr in hasil:
                if arr is None:
                    h.update(b"None")
                else:
                    a = np.ascontiguousarray(arr)
                    h.update(str(a.shape).encode())
                    h.update(a.dtype.str.encode())
                    h.update(a.tobytes())

            h.update(extra if isinstance(extra, bytes)
                     else str(extra).encode())

            return h.hexdigest()

        manifest.data_hash = hash_simulasi

        simulasi = True

    if data.TUGAS != "regresi":
        raise ValueError(
            "TUGAS harus 'regresi'. "
            "Koordinasikan perubahan protokol dengan Penjaga Data."
        )

    return hasil, simulasi


# --------------------------------------------------
# 3. KONVERSI CITRA MENJADI URUTAN BARIS
# --------------------------------------------------

def citra_ke_urutan(X):

    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 2:

        if X.shape[1] == 224 * 224:
            X = X.reshape(-1, 224, 224, 1)

        elif X.shape[1] == 224 * 224 * 3:
            X = X.reshape(-1, 224, 224, 3)

        else:
            raise ValueError("Ukuran citra flattened tidak sesuai.")

    elif X.ndim == 3:

        X = X[..., None]

    elif X.ndim == 4:

        # Format NCHW -> NHWC
        if X.shape[1] in (1, 3) and X.shape[2:4] == (224, 224):
            X = np.transpose(X, (0, 2, 3, 1))

        # Format NHWC
        elif X.shape[1:3] == (224, 224) and X.shape[3] in (1, 3):
            pass

        else:
            raise ValueError(
                f"Format citra tidak didukung: {X.shape}"
            )

    else:
        raise ValueError(f"Dimensi input tidak didukung: {X.shape}")

    if X.shape[1:3] != (224, 224):
        raise ValueError(f"Ukuran citra harus 224x224: {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("Citra mengandung NaN atau inf.")

    if X.max() > 1.5:
        X = X / 255.0

    # N, 224, 224, C -> N, 224, (224*C)
    # Setiap baris citra menjadi satu timestep.
    X = X.reshape(len(X), 224, -1)

    return np.ascontiguousarray(X, dtype=np.float32)


# --------------------------------------------------
# 4. ARSITEKTUR GRU REGRESSION
# --------------------------------------------------

def buat_model(input_size):

    import torch
    from torch import nn

    class GRURegressor(nn.Module):

        def __init__(self):

            super().__init__()

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=1,
                batch_first=True
            )

            self.dropout = nn.Dropout(0.10)

            self.fc = nn.Linear(
                HIDDEN_SIZE,
                1
            )

        def forward(self, x):

            _, hidden = self.gru(x)

            fitur = self.dropout(hidden[-1])

            output = self.fc(fitur).squeeze(-1)

            # Tinggi air tidak boleh negatif
            return torch.nn.functional.softplus(output)

    return GRURegressor()


# --------------------------------------------------
# 5. TRAINING DAN PREDIKSI
# --------------------------------------------------

def latih_dan_prediksi(seed, final, hasil_data):

    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.set_num_threads(2)

    Xtr, ytr, Xva, yva, Xte, yte = hasil_data

    Xtr = citra_ke_urutan(Xtr)
    Xva = citra_ke_urutan(Xva)

    ytr = np.asarray(ytr, dtype=np.float32).reshape(-1)
    yva = np.asarray(yva, dtype=np.float32).reshape(-1)

    if len(Xtr) != len(ytr) or len(Xva) != len(yva):
        raise ValueError("Jumlah citra dan label tidak sama.")

    if len(Xtr) == 0 or len(Xva) == 0:
        raise ValueError("Data train atau validasi kosong.")

    x_train = torch.from_numpy(Xtr)
    y_train = torch.from_numpy(ytr)

    x_val = torch.from_numpy(Xva)

    model = buat_model(Xtr.shape[-1])

    # Optimizer AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    # Loss regresi
    criterion = nn.MSELoss()

    print("\nMulai training GRU...")
    print("Input:", tuple(Xtr.shape))
    print("Output target: 1 nilai tinggi air (mm)")

    for epoch in range(EPOCH):

        model.train()

        indeks = torch.randperm(len(x_train))

        total_loss = 0.0

        for i in range(0, len(indeks), BATCH_SIZE):

            idx = indeks[i:i + BATCH_SIZE]

            xb = x_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()

            prediksi = model(xb)

            loss = criterion(prediksi, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(idx)

        rata_loss = total_loss / len(x_train)

        print(
            f"Epoch {epoch + 1}/{EPOCH} "
            f"- Loss: {rata_loss:.4f}"
        )

    # Prediksi validasi
    model.eval()

    with torch.no_grad():

        pred_val = model(x_val).cpu().numpy()

        pred_test = None

        if final and Xte is not None and yte is not None:

            x_test = torch.from_numpy(
                citra_ke_urutan(Xte)
            )

            pred_test = model(x_test).cpu().numpy()

    return pred_val, pred_test


# --------------------------------------------------
# 6. EVALUASI DENGAN STARTER DOSEN
# --------------------------------------------------

def jalankan_evaluasi(
    seed,
    pred_val_path,
    pred_test_path,
    durasi,
    perangkat,
    output_metrics
):

    import src.common.evaluate as evaluator

    args = [
        "evaluate.py",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(seed),
        "--pred-val", str(pred_val_path),
        "--durasi", str(round(durasi, 2)),
        "--perangkat", perangkat,
        "--out", str(output_metrics)
    ]

    if pred_test_path is not None:
        args += ["--pred-test", str(pred_test_path)]

    argv_lama = sys.argv

    try:
        sys.argv = args
        evaluator.main()

    finally:
        sys.argv = argv_lama


# --------------------------------------------------
# 7. PROGRAM UTAMA
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument(
        "--final",
        action="store_true"
    )

    parser.add_argument(
        "--perangkat",
        default="cpu laptop"
    )

    a = parser.parse_args()

    print("=" * 50)
    print("CAPSTONE DEEP LEARNING - MODEL RNN GRU")
    print("NIM:", NIM)
    print("Seed:", a.seed)
    print("=" * 50)

    hasil_data, simulasi = siapkan_data(a.seed)

    awal = time.time()

    pred_val, pred_test = latih_dan_prediksi(
        a.seed,
        a.final,
        hasil_data
    )

    durasi = time.time() - awal

    # Simulasi dipisahkan agar tidak menimpa hasil eksperimen asli
    jenis = "simulasi" if simulasi else None

    out = AKAR / "results" / NIM

    if jenis:
        out = out / jenis

    out.mkdir(parents=True, exist_ok=True)

    pred_val_path = out / "pred_val.npy"

    np.save(pred_val_path, pred_val)

    pred_test_path = None

    if pred_test is not None:

        pred_test_path = out / "pred_test.npy"

        np.save(pred_test_path, pred_test)

    output_metrics = out / "metrics.json"

    perangkat = a.perangkat

    if simulasi:
        perangkat += " - DATA SIMULASI, BUKAN HASIL EKSPERIMEN"

    jalankan_evaluasi(
        a.seed,
        pred_val_path,
        pred_test_path,
        durasi,
        perangkat,
        output_metrics
    )

    print("\nPENGUJIAN BERHASIL")
    print("Prediksi:", pred_val_path)
    print("Metrics:", output_metrics)
    print("Durasi:", round(durasi, 2), "detik")


if __name__ == "__main__":
    main()
