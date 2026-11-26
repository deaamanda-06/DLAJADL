"""Pemotong ROI Wadah B dari data sesi meja uji ke dataset 224x224.

    python -m src.protokol.pemotong

Membaca ``data/sesi-20261116-095756/label.csv`` dan ``kamera_index.csv``,
mengambil frame saat air diam (antara ``t_selesai_host + 2.0`` hingga
``min(t_selesai_host + 4.5, t_catat_host - 1.0)``), memotong ROI Wadah B
``(820, 300, 1140, 720)`` lalu mengubah ukuran menjadi ``224x224`` RGB.

Split dilakukan pada tingkat **kondisi** (75 kondisi) menggunakan
``src.common.split.indeks`` (seed 2026) agar tidak ada kebocoran frame dari
kondisi yang sama antara latih, validasi, dan uji.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.common import manifest, split

AKAR = manifest.AKAR
SESI_DIR = AKAR / "data" / "sesi-20261116-095756"
OUT_DIR = AKAR / "data" / "cropped"
DATASET_PATH = OUT_DIR / "dataset.npz"
INDEX_PATH = OUT_DIR / "index.csv"

# ROI tetap untuk Wadah B pada citra 1280x720: (left, top, right, bottom)
ROI_BOX = (820, 300, 1140, 720)
UKURAN = (224, 224)
MAX_FRAME_PER_KONDISI = 5


def _muat_kamera_index(path: Path) -> list[tuple[str, float]]:
    with path.open(encoding="utf-8") as f:
        return [(r["berkas"], float(r["t_host"])) for r in csv.DictReader(f)]


def _pilih_frame(
    frames: list[tuple[str, float]], t_selesai: float, t_catat: float
) -> list[str]:
    akhir = min(t_selesai + 4.5, t_catat - 1.0)
    kandidat = [b for b, t in frames if t_selesai + 2.0 <= t <= akhir]
    if not kandidat:
        kandidat = [b for b, t in frames if t_selesai + 1.5 <= t <= t_catat - 0.5]
    if not kandidat:
        kandidat = [b for b, t in frames if t_selesai <= t <= t_catat]
    if len(kandidat) > MAX_FRAME_PER_KONDISI:
        idx = np.linspace(0, len(kandidat) - 1, MAX_FRAME_PER_KONDISI, dtype=int)
        kandidat = [kandidat[i] for i in idx]
    return kandidat


def _potong_gambar(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as im:
        crop = im.convert("RGB").crop(ROI_BOX).resize(UKURAN, Image.Resampling.BILINEAR)
        arr = np.asarray(crop, dtype=np.float32) / 255.0
    # HWC -> CHW (PyTorch NCHW-friendly)
    return np.transpose(arr, (2, 0, 1))


def potong(simpan_contoh: bool = True) -> Path:
    """Jalankan pemotongan ROI dan simpan hasilnya ke ``data/cropped/dataset.npz``."""
    if not SESI_DIR.exists():
        raise FileNotFoundError(
            f"Folder sesi tidak ditemukan: {SESI_DIR}. "
            "Ekstrak tim02_sesi-20261116-095756.tar.gz ke folder data/."
        )

    with (SESI_DIR / "label.csv").open(encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    frames = _muat_kamera_index(SESI_DIR / "kamera_index.csv")

    # Split pada tingkat kondisi: 75 -> 60 (train+val) + 15 (test)
    idx_tv, idx_te = split.indeks(len(labels), porsi_val=0.2, seed=split.SEED_SPLIT)
    idx_tr_rel, idx_va_rel = split.indeks(len(idx_tv), porsi_val=0.15, seed=split.SEED_SPLIT)
    idx_tr = set(idx_tv[idx_tr_rel].tolist())
    idx_va = set(idx_tv[idx_va_rel].tolist())
    idx_te_set = set(idx_te.tolist())

    bagian: dict[str, tuple[list[np.ndarray], list[float]]] = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }
    baris_index: list[dict[str, str | int | float]] = []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contoh_dir = OUT_DIR / "contoh"
    if simpan_contoh:
        contoh_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(labels):
        split_nama = "train" if i in idx_tr else ("val" if i in idx_va else "test")
        assert (i in idx_tr) or (i in idx_va) or (i in idx_te_set)
        t_selesai = float(row["t_selesai_host"])
        t_catat = float(row["t_catat_host"])
        y = float(row["tinggi_air_mm"].replace(",", "."))

        terpilih = _pilih_frame(frames, t_selesai, t_catat)
        for j, berkas in enumerate(terpilih):
            img_path = SESI_DIR / "kamera" / berkas
            arr = _potong_gambar(img_path)
            bagian[split_nama][0].append(arr)
            bagian[split_nama][1].append(y)
            baris_index.append(
                {
                    "kondisi_no": int(row["no"]),
                    "blok": int(row["ulangan_blok"]),
                    "split": split_nama,
                    "berkas": berkas,
                    "tinggi_air_mm": y,
                }
            )
            if simpan_contoh and j == 0 and i in (0, 10, 24):
                with Image.open(img_path) as im:
                    im.convert("RGB").crop(ROI_BOX).resize(
                        UKURAN, Image.Resampling.BILINEAR
                    ).save(contoh_dir / f"kondisi_{int(row['no']):02d}_y{y}.jpg")

    X_train = np.stack(bagian["train"][0]).astype(np.float32)
    y_train = np.asarray(bagian["train"][1], dtype=np.float32)
    X_val = np.stack(bagian["val"][0]).astype(np.float32)
    y_val = np.asarray(bagian["val"][1], dtype=np.float32)
    X_test = np.stack(bagian["test"][0]).astype(np.float32)
    y_test = np.asarray(bagian["test"][1], dtype=np.float32)

    np.savez_compressed(
        DATASET_PATH,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    with INDEX_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["kondisi_no", "blok", "split", "berkas", "tinggi_air_mm"]
        )
        writer.writeheader()
        writer.writerows(baris_index)

    return DATASET_PATH


def main() -> None:
    out = potong()
    d = np.load(out)
    print(f"ROI Wadah B : {ROI_BOX} -> {UKURAN}")
    print(f"X_train     : {d['X_train'].shape}, y_train: {d['y_train'].shape}")
    print(f"X_val       : {d['X_val'].shape}, y_val  : {d['y_val'].shape}")
    print(f"X_test      : {d['X_test'].shape}, y_test : {d['y_test'].shape}")
    print(f"Tersimpan   : {out.relative_to(AKAR)}")


if __name__ == "__main__":
    main()
