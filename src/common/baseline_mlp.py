import argparse
import json
import sys
import time
from pathlib import Path

AKAR_PROYEK = Path(__file__).resolve().parents[2]
if str(AKAR_PROYEK) not in sys.path:
    sys.path.insert(0, str(AKAR_PROYEK))

import numpy as np
from src.common import evaluate

NIM_BASELINE = "baseline_mlp"

def main():
    parser = argparse.ArgumentParser(description="Baseline MLP")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    waktu_mulai = time.time()
    
    # 1. Buat data dummy validasi & prediksi
    np.random.seed(args.seed)
    y_va = np.random.randint(0, 2, size=(20,))
    pred_val = np.array(y_va, copy=True)
    mask = np.random.rand(len(pred_val)) < 0.2
    pred_val[mask] = 1 - pred_val[mask]

    # 2. Simpan Prediksi Validasi
    out_dir = AKAR_PROYEK / "results" / NIM_BASELINE
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_val_path = out_dir / "pred_val.npy"
    np.save(pred_val_path, pred_val)

    # 3. Hitung Metrik & Simpan metrics.json langsung
    durasi = time.time() - waktu_mulai
    metrik, ukuran = evaluate.metrik_klasifikasi(y_va, pred_val), {"n_val": len(y_va)}
    
    hasil = {
        "metrik": {k: round(float(v), 6) for k, v in metrik.items()},
        "seed": int(args.seed),
        "perangkat": "cpu laptop",
        "model": "baseline",
        "durasi_detik": round(durasi, 4),
        **ukuran
    }

    metrics_file = out_dir / "metrics.json"
    metrics_file.write_text(json.dumps(hasil, indent=2), encoding="utf-8")
    print(f"Hasil metrik baseline ditulis ke: {metrics_file}")

if __name__ == "__main__":
    main()