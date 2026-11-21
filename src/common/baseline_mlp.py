"""Baseline regresi (Ridge atas fitur profil spasial 16x16) untuk Tim 02."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))

from src.protokol import data  # noqa: E402

NIM_BASELINE = "baseline_mlp"


def _fitur_ringkas(X: np.ndarray) -> np.ndarray:
    """Rata-rata blok 14x14 pada citra 224x224 -> (N, 3*16*16 = 768) fitur."""
    N, C, H, W = X.shape
    return X.reshape(N, C, 16, H // 16, 16, W // 16).mean(axis=(3, 5)).reshape(N, -1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline Regresi (Ridge)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    Xtr, ytr, Xva, _, _, _ = data.muat()
    Ftr = _fitur_ringkas(np.asarray(Xtr, dtype=np.float32))
    Fva = _fitur_ringkas(np.asarray(Xva, dtype=np.float32))

    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr_n = (Ftr - mu) / sd
    Fva_n = (Fva - mu) / sd

    Xb = np.hstack([Ftr_n, np.ones((len(Ftr_n), 1), np.float32)])
    A = Xb.T @ Xb + 10.0 * np.eye(Xb.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xb.T @ np.asarray(ytr, dtype=np.float32))
    pred_val = np.clip(
        np.hstack([Fva_n, np.ones((len(Fva_n), 1), np.float32)]) @ w, 0.0, None
    ).astype(np.float32)
    durasi = time.time() - t0

    out_dir = AKAR / "results" / NIM_BASELINE
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_val_path = out_dir / "pred_val.npy"
    np.save(pred_val_path, pred_val)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.common.evaluate",
            "--nim",
            NIM_BASELINE,
            "--model",
            "baseline",
            "--seed",
            str(a.seed),
            "--pred-val",
            str(pred_val_path),
            "--durasi",
            f"{durasi:.1f}",
            "--perangkat",
            a.perangkat,
        ],
        check=True,
        cwd=AKAR,
    )


if __name__ == "__main__":
    main()
