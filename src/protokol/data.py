"""Pemuat citra tinggi air Tim 02.

Struktur yang dibaca:
data/
  sesi_simulasi_*/metadata.csv
  sesi_simulasi_*/frames/*.jpg

Sesi meja nyata dapat dibaca setelah metadata.csv dilengkapi dengan block_id,
condition_id, timestamp_s, condition_start_s, dan tinggi_air_mm.
Pemisahan dilakukan berdasarkan block_id (siklus pengisian), bukan per citra.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

TUGAS = "regresi"
UKURAN = (224, 224)
BUANG_DETIK_PERTAMA = 1.0
SEED = 2026


def _akar_proyek() -> Path:
    return Path(__file__).resolve().parents[2]


def _baca_baris(folder_data: Path):
    baris = []
    for metadata in sorted(folder_data.glob("sesi_*/metadata.csv")):
        with metadata.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_session"] = metadata.parent.name
                row["_metadata_dir"] = str(metadata.parent)
                baris.append(row)
    return baris


def _angka(row, key):
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        raise ValueError(f"Kolom {key!r} kosong pada sesi {row.get('_session')}.")
    return float(value)


def _muat_citra(row) -> np.ndarray:
    path = Path(row["_metadata_dir"]) / row["frame_path"]
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Citra tidak dapat dibaca: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, UKURAN, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def _bagi_blok(block_ids):
    """Urutan blok deterministik; satu block_id hanya boleh masuk satu subset."""
    unik = sorted(set(block_ids), key=lambda x: (str(type(x)), str(x)))
    if len(unik) < 3:
        raise ValueError(
            f"Diperlukan minimal 3 blok siklus terpisah untuk latih/validasi/uji; "
            f"ditemukan {len(unik)}. Tambahkan blok eksperimen."
        )
    rng = np.random.default_rng(SEED)
    urut = list(np.asarray(unik, dtype=object)[rng.permutation(len(unik))])
    n = len(urut)
    n_val = max(1, int(round(n * 0.2)))
    n_test = max(1, int(round(n * 0.2)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = n-2, 1, 1
    return (set(urut[:n_train]), set(urut[n_train:n_train+n_val]),
            set(urut[n_train+n_val:]))


def muat():
    """Mengembalikan X/y latih, validasi, dan uji; X berbentuk N,C,224,224."""
    root = _akar_proyek()
    folder_data = root / "data"
    rows = _baca_baris(folder_data)
    if not rows:
        raise FileNotFoundError(
            f"Tidak ditemukan data/sesi_*/metadata.csv di {folder_data}. "
            "Jalankan meja.py --simulasi jalankan ... terlebih dahulu."
        )

    mode_data = {
        str(r.get("is_simulated", "")).strip().lower() in {"true", "1", "yes"}
        for r in rows
    }
    if len(mode_data) > 1:
        raise ValueError(
            "Folder data mencampur sesi simulasi dan meja nyata. Pisahkan sesi "
            "atau gunakan hanya satu jenis data dalam satu evaluasi."
        )

    terpilih = []
    for row in rows:
        # Mode meja nyata harus memiliki metadata label yang telah diverifikasi.
        if str(row.get("is_simulated", "")).lower() not in {"true", "1", "yes"}:
            if not str(row.get("tinggi_air_mm", "")).strip():
                continue
        waktu = _angka(row, "timestamp_s")
        if waktu < BUANG_DETIK_PERTAMA:
            continue
        if not str(row.get("block_id", "")).strip():
            raise ValueError("block_id kosong: isi metadata berdasarkan siklus pengisian.")
        if not str(row.get("condition_id", "")).strip():
            raise ValueError("condition_id kosong: isi metadata berdasarkan kondisi.")
        target = _angka(row, "tinggi_air_mm")
        row["_target"] = target
        terpilih.append(row)

    if not terpilih:
        raise ValueError("Tidak ada sampel berlabel setelah pembuangan 1 detik pertama.")

    block_train, block_val, block_test = _bagi_blok(
        [r["block_id"] for r in terpilih]
    )
    groups = {
        "train": [r for r in terpilih if r["block_id"] in block_train],
        "val": [r for r in terpilih if r["block_id"] in block_val],
        "test": [r for r in terpilih if r["block_id"] in block_test],
    }
    if any(not groups[k] for k in groups):
        raise ValueError("Salah satu subset kosong; periksa jumlah blok dan metadata.")

    # Pemeriksaan eksplisit: tidak boleh ada block/siklus yang tumpang tindih.
    ids = {k: {r["block_id"] for r in v} for k, v in groups.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])

    def load_group(items):
        images = np.stack([_muat_citra(r) for r in items], axis=0)
        # NHWC -> NCHW
        images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)
        targets = np.asarray([r["_target"] for r in items], dtype=np.float32)
        return images, targets

    X_train, y_train = load_group(groups["train"])
    X_val, y_val = load_group(groups["val"])
    X_test, y_test = load_group(groups["test"])

    # Statistik kanal dihitung HANYA dari data latih.
    mu = X_train.mean(axis=(0, 2, 3), keepdims=True)
    sigma = X_train.std(axis=(0, 2, 3), keepdims=True)
    sigma = np.maximum(sigma, 1e-6)
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
    X_test = (X_test - mu) / sigma

    print("Jumlah sampel:", {k: len(v) for k, v in groups.items()})
    print("Blok latih/validasi/uji:", ids)
    print("Pemeriksaan kebocoran blok: LULUS")
    return X_train, y_train, X_val, y_val, X_test, y_test
