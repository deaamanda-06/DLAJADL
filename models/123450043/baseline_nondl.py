"""Pembanding non-DL: regresi Ridge pada fitur piksel sederhana.

Fitur: citra abu-abu diperkecil 28x28 + profil kecerahan per baris (224).
Alpha dipilih dengan cross-validation pada data latih saja.

Jalankan dari akar repo:
    python models/123450043/baseline_nondl.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data  # noqa: E402

POOL = 8  # 224 -> 28


def fitur(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.max() > 1.5:
        X = X / 255.0
    g = X.mean(axis=1)                                         # (N,224,224)
    n = g.shape[0]
    kecil = g.reshape(n, 224 // POOL, POOL, 224 // POOL, POOL).mean(axis=(2, 4))
    baris = g.mean(axis=2)                                     # (N,224)
    return np.concatenate([kecil.reshape(n, -1), baris], axis=1)


def main() -> None:
    t0 = time.time()
    Xtr, ytr, Xva, yva, _, _ = data.muat()      # data uji tidak disentuh
    ytr, yva = np.asarray(ytr, dtype=np.float64), np.asarray(yva, dtype=np.float64)

    Ftr, Fva = fitur(Xtr), fitur(Xva)
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(0, 5, 11), cv=5),
    )
    model.fit(Ftr, ytr)
    pred_val = model.predict(Fva)

    mae = float(np.abs(yva - pred_val).mean())
    tebak = float(np.abs(yva - ytr.mean()).mean())
    print(f"frame: train={len(Xtr)} val={len(Xva)} | fitur={Ftr.shape[1]}")
    print(f"alpha terpilih (CV latih): {model[-1].alpha_:.1f}")
    print(f"val MAE Ridge: {mae:.4f} mm | tebak rata-rata: {tebak:.4f} mm")

    out = AKAR / "results" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)

    subprocess.run(
        [sys.executable, "-m", "src.common.evaluate",
         "--nim", "baseline", "--model", "baseline", "--seed", "1",
         "--pred-val", str(out / "pred_val.npy"),
         "--durasi", f"{time.time() - t0:.1f}", "--perangkat", "cpu laptop"],
        check=True, cwd=AKAR,
    )


if __name__ == "__main__":
    main()