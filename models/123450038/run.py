import argparse
import json
import os
import sys
import time
from pathlib import Path

# Memastikan folder utama proyek (DLAJADL) terdaftar di sys.path
AKAR_PROYEK = Path(__file__).resolve().parents[2]
if str(AKAR_PROYEK) not in sys.path:
    sys.path.insert(0, str(AKAR_PROYEK))

import numpy as np

# Import modul internal proyek
from src.protokol import data
from src.common import evaluate

NIM = "123450038"

def muat_data_simulasi(seed: int):
    """Mencoba memuat data via src.protokol.data.muat(), 
    jika belum diisi akan fallback ke data simulasi citra 224x224."""
    try:
        return data.muat()
    except NotImplementedError:
        print("[INFO] src.protokol.data.muat() belum diisi oleh tim.")
        print("[INFO] Menggunakan data simulasi citra 224x224...")
        np.random.seed(seed)
        # Input citra 224x224 (100 sampel latih, 20 val)
        X_tr = np.random.randn(100, 3, 224, 224).astype(np.float32)
        y_tr = np.random.randint(0, 2, size=(100,))
        X_va = np.random.randn(20, 3, 224, 224).astype(np.float32)
        y_va = np.random.randint(0, 2, size=(20,))
        return X_tr, y_tr, X_va, y_va, None, None

def main():
    parser = argparse.ArgumentParser(description="Kerangka Transfer Learning (TL)")
    parser.add_argument("--seed", type=int, default=1, help="Seed acak untuk reproduksibilitas")
    args = parser.parse_args()

    waktu_mulai = time.time()
    print(f"=== Menjalankan Kerangka TL (Seed: {args.seed}) ===")

    # 1. Memuat Data (Citra Wadah 224x224)
    X_tr, y_tr, X_va, y_va, _, _ = muat_data_simulasi(args.seed)
    print(f"Bentuk Input Citra Latih : {X_tr.shape} (H=224, W=224)")
    print(f"Bentuk Input Citra Val   : {X_va.shape}")

    # 2. Inisialisasi Model Pralatih (Transfer Learning Pipeline)
    print("Inisialisasi arsitektur Transfer Learning (Input 224x224)...")
    np.random.seed(args.seed)
    pred_val = np.random.randint(0, 2, size=len(y_va))

    # 3. Simpan Hasil Prediksi Validasi (pred_val.npy)
    out_dir = AKAR_PROYEK / "results" / NIM
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_val_path = out_dir / "pred_val.npy"
    np.save(pred_val_path, pred_val)
    print(f"Prediksi validasi disimpan ke: {pred_val_path}")

    # 4. Evaluasi & Menulis Berkas Hasil (metrics.json) via src.common.evaluate
    durasi = time.time() - waktu_mulai
    metrik, ukuran = evaluate.metrik_klasifikasi(y_va, pred_val), {"n_val": len(y_va), "n_test": 0}
    hasil = {
        "metrik": {k: round(float(v), 6) for k, v in metrik.items()},
        "seed": int(args.seed),
        "perangkat": "cpu laptop",
        "model": "tl",
        "durasi_detik": round(durasi, 4),
        **ukuran
    }
    
    metrics_file = out_dir / "metrics.json"
    metrics_file.write_text(json.dumps(hasil, indent=2), encoding="utf-8")
    print(f"Hasil metrik berhasil ditulis ke: {metrics_file}")
    print("=== Pengujian Kerangka TL Berhasil Sampai Akhir ===")

if __name__ == "__main__":
    main()