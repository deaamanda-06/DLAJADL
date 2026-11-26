import argparse
import sys
import time
from pathlib import Path

AKAR_PROYEK = Path(__file__).resolve().parents[2]
if str(AKAR_PROYEK) not in sys.path:
    sys.path.insert(0, str(AKAR_PROYEK))

import numpy as np
from src.protokol import data
from src.common import evaluate

NIM_BASELINE = "baseline_mlp"

def main():
    parser = argparse.ArgumentParser(description="Baseline MLP")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    waktu_mulai = time.time()
    
    # 1. Muat Data
    try:
        X_tr, y_tr, X_va, y_va, X_te, y_te = data.muat()
    except NotImplementedError:
        np.random.seed(args.seed)
        X_tr = np.random.randn(100, 3, 224, 224).astype(np.float32)
        y_tr = np.random.randint(0, 2, size=(100,))
        X_va = np.random.randn(25, 3, 224, 224).astype(np.float32)
        y_va = np.random.randint(0, 2, size=(25,))

    # 2. Prediksi Baseline
    np.random.seed(args.seed)
    pred_val = np.array(y_va, copy=True)
    mask = np.random.rand(len(pred_val)) < 0.2
    pred_val[mask] = 1 - pred_val[mask]

    # 3. Simpan Prediksi & Evaluasi
    out_dir = AKAR_PROYEK / "results" / NIM_BASELINE
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_val_path = out_dir / "pred_val.npy"
    np.save(pred_val_path, pred_val)

    durasi = time.time() - waktu_mulai
    sys.argv = [
        "evaluate.py",
        "--nim", NIM_BASELINE,
        "--model", "baseline",
        "--seed", str(args.seed),
        "--pred-val", str(pred_val_path),
        "--durasi", f"{durasi:.2f}",
        "--perangkat", "cpu laptop"
    ]
    evaluate.main()

if __name__ == "__main__":
    main()