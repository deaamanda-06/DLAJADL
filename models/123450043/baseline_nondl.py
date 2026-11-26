"""
Pembanding non-DL (#8): posisi garis air dari profil kecerahan + regresi linear.

1. Citra ROI 224x224 grayscale -> profil kecerahan 1D (rata-rata kolom per baris).
2. Haluskan (moving average), cari baris dengan gradien ekstrem -> posisi garis air.
3. Regresi linear posisi(baris) -> tinggi air (mm), dilatih pada data latih.

Pilihan konfigurasi (rentang kolom, baris awal pencarian, lebar haluskan, jenis
gradien) dipilih di BLOK VALIDASI, lalu model dilatih ulang di blok latih+validasi
dan diuji SEKALI di blok uji. Tanpa GPU/PyTorch.

    python baseline_nondl.py --data-dir sesi-20261116-095756 --out-dir results/123450043
"""
import argparse, itertools, json, os, time
import numpy as np
from sklearn.linear_model import LinearRegression
from data_utils import load_samples, split_by_block, to_arrays


def waterline_row(img, cols, lo, w, mode):
    p = img[:, cols[0]:cols[1]].mean(axis=1)
    p = np.convolve(p, np.ones(w) / w, mode="same")
    g = np.gradient(p)[lo:]
    if mode == "absmax":
        return lo + int(np.argmax(np.abs(g)))
    if mode == "min":
        return lo + int(np.argmin(g))
    return lo + int(np.argmax(g))          # "max"


def feats(X, cfg):
    return np.array([[waterline_row(im, *cfg)] for im in X], dtype=np.float64)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def per_kondisi_mae(pred, y, kond):
    out = []
    for k in np.unique(kond):
        m = kond == k
        out.append(abs(pred[m].mean() - y[m][0]))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="results/123450043")
    ap.add_argument("--test-block", type=int, default=3)
    ap.add_argument("--val-block", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    samples = load_samples(args.data_dir)
    tr, va, te = split_by_block(samples, args.test_block, args.val_block)
    Xtr, ytr = to_arrays(tr); Xva, yva = to_arrays(va); Xte, yte = to_arrays(te)
    kte = np.array([s["kondisi"] for s in te])

    grid = list(itertools.product([(0, 224), (60, 160), (100, 224), (120, 200)],
                                  [60, 100, 140], [3, 5, 9], ["absmax", "min", "max"]))
    best = None
    for cfg in grid:
        m = LinearRegression().fit(feats(Xtr, cfg), ytr)
        v = mae(m.predict(feats(Xva, cfg)), yva)
        if best is None or v < best[0]:
            best = (v, cfg)
    val_mae, cfg = best
    print(f"konfigurasi terpilih (di validasi): kolom={cfg[0]} lo={cfg[1]} w={cfg[2]} mode={cfg[3]} | val MAE={val_mae:.3f} mm")

    Xall = np.concatenate([Xtr, Xva]); yall = np.concatenate([ytr, yva])
    model = LinearRegression().fit(feats(Xall, cfg), yall)
    pred = model.predict(feats(Xte, cfg))
    test_mae = mae(pred, yte)
    cond_mae = per_kondisi_mae(pred, yte, kte)
    mean_mae = mae(np.full_like(yte, yall.mean()), yte)

    t = time.perf_counter()
    for im in Xte[:100]:
        model.predict(feats(im[None], cfg))
    lat = (time.perf_counter() - t) / min(100, len(Xte)) * 1000.0

    metrics = {
        "jalur": "pembanding_non_dl", "nim": "123450043", "seed": None,
        "mae_mm": round(test_mae, 4),
        "mae_mm_per_kondisi": round(cond_mae, 4),
        "mae_mm_tebak_rata2": round(mean_mae, 4),
        "val_mae_mm": round(val_mae, 4),
        "onnx_kb": None, "latency_ms": round(lat, 3),
        "config": {"kolom": list(cfg[0]), "baris_awal": cfg[1], "haluskan": cfg[2], "gradien": cfg[3]},
        "slope_mm_per_baris": round(float(model.coef_[0]), 5),
        "test_block": args.test_block, "n_test_frames": int(len(yte)),
    }
    with open(os.path.join(args.out_dir, "metrics_baseline.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
