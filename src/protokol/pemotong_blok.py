
"""Dataset acuan Tim 02: split per blok dan target dalam mm.

Jalankan:
    python -m src.protokol.pemotong_blok
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.protokol.pemotong import (
    _potong_gambar,
    ROI_BOX,
    UKURAN,
    MAX_FRAME_PER_KONDISI,
)

ROOT = Path(__file__).resolve().parents[2]

SESI_DIR = ROOT / "data" / "sesi-20261116-095756"
OUT_DIR = ROOT / "data" / "cropped_blok_mm"

DATASET_PATH = OUT_DIR / "dataset.npz"
INDEX_PATH = OUT_DIR / "index.csv"
INFO_PATH = OUT_DIR / "dataset_info.json"

SPLIT_BLOK = {
    1: "train",
    2: "val",
    3: "test",
}


def sha256_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def baca_csv(path):
    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def pilih_frame(frames, selesai, catat):
    awal = selesai + 2.0
    akhir = min(selesai + 4.5, catat - 1.0)

    kandidat = [
        berkas
        for berkas, waktu in frames
        if awal <= waktu <= akhir
    ]

    if len(kandidat) > MAX_FRAME_PER_KONDISI:
        idx = np.linspace(
            0,
            len(kandidat) - 1,
            MAX_FRAME_PER_KONDISI,
            dtype=int,
        )
        kandidat = [kandidat[i] for i in idx]

    return kandidat


def main():
    label_path = SESI_DIR / "label.csv"

    labels = baca_csv(label_path)
    kamera = baca_csv(SESI_DIR / "kamera_index.csv")

    frames = [
        (r["berkas"], float(r["t_host"]))
        for r in kamera
    ]

    labels.sort(key=lambda r: int(r["no"]))

    jumlah_blok = Counter(
        int(r["ulangan_blok"]) for r in labels
    )

    assert len(labels) == 75
    assert len({r["no"] for r in labels}) == 75
    assert jumlah_blok == {1: 25, 2: 25, 3: 25}

    bagian = {
        nama: {"X": [], "y": []}
        for nama in ("train", "val", "test")
    }

    baris_index = []
    frame_dipakai = set()

    print("=" * 60)
    print("PEMBUATAN DATASET ACUAN TIM 02")
    print("=" * 60)

    for row in labels:
        no = int(row["no"])
        blok = int(row["ulangan_blok"])
        split_nama = SPLIT_BLOK[blok]

        # Label sumber menggunakan cm meskipun
        # nama kolom aslinya tinggi_air_mm.
        y_cm = float(
            row["tinggi_air_mm"].replace(",", ".")
        )
        y_mm = y_cm * 10.0

        assert np.isfinite(y_mm)
        assert y_mm >= 0

        terpilih = pilih_frame(
            frames,
            float(row["t_selesai_host"]),
            float(row["t_catat_host"]),
        )

        if not terpilih:
            raise ValueError(
                f"Kondisi {no} tidak memiliki frame "
                "pada rentang waktu utama."
            )

        for berkas in terpilih:
            if berkas in frame_dipakai:
                raise ValueError(
                    f"Frame duplikat: {berkas}"
                )

            frame_dipakai.add(berkas)

            img_path = SESI_DIR / "kamera" / berkas
            arr = _potong_gambar(img_path)

            bagian[split_nama]["X"].append(arr)
            bagian[split_nama]["y"].append(y_mm)

            baris_index.append({
                "kondisi_no": no,
                "blok": blok,
                "split": split_nama,
                "berkas": berkas,
                "tinggi_air_mm": y_mm,
            })

    arrays = {}

    for nama in ("train", "val", "test"):
        arrays[f"X_{nama}"] = np.ascontiguousarray(
            np.stack(bagian[nama]["X"]),
            dtype="<f4",
        )

        arrays[f"y_{nama}"] = np.asarray(
            bagian[nama]["y"],
            dtype="<f4",
        )

    # Verifikasi dataset sebelum disimpan.
    assert len(baris_index) == 373

    assert {
        r["kondisi_no"] for r in baris_index
    } == {
        int(r["no"]) for r in labels
    }

    for nama, target_frame in (
        ("train", 125),
        ("val", 123),
        ("test", 125),
    ):
        X = arrays[f"X_{nama}"]
        y = arrays[f"y_{nama}"]

        assert X.shape == (
            target_frame, 3, 224, 224
        )
        assert y.shape == (target_frame,)
        assert np.isfinite(X).all()
        assert np.isfinite(y).all()
        assert 0 <= X.min() <= X.max() <= 1

    # Hash isi setiap array, bukan metadata ZIP.
    array_info = {}

    for key in sorted(arrays):
        arr = arrays[key]

        array_info[key] = {
            "shape": list(arr.shape),
            "dtype": arr.dtype.str,
            "sha256": hashlib.sha256(
                arr.tobytes(order="C")
            ).hexdigest(),
        }

    # Hash identitas frame dan urutan data.
    index_bytes = json.dumps(
        baris_index,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    index_hash = hashlib.sha256(
        index_bytes
    ).hexdigest()

    identitas = {
        "schema": "tim02_dataset_v1",
        "label_source_sha256": sha256_file(label_path),
        "roi": list(ROI_BOX),
        "ukuran": list(UKURAN),
        "satuan_target": "mm",
        "split_blok": SPLIT_BLOK,
        "array_info": array_info,
        "index_sha256": index_hash,
    }

    canonical = json.dumps(
        identitas,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    full_hash = hashlib.sha256(
        canonical
    ).hexdigest()

    # Mengikuti panjang hash starter (16 karakter).
    identitas["data_hash"] = full_hash[:16]
    identitas["data_hash_sha256"] = full_hash

    identitas["jumlah_kondisi"] = len(labels)
    identitas["jumlah_frame_sumber"] = len(frames)
    identitas["jumlah_frame_dipakai"] = len(baris_index)

    # Dataset lama di data/cropped tidak diubah.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(DATASET_PATH, **arrays)

    with INDEX_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kondisi_no",
                "blok",
                "split",
                "berkas",
                "tinggi_air_mm",
            ],
        )
        writer.writeheader()
        writer.writerows(baris_index)

    identitas["dataset_npz_sha256"] = sha256_file(
        DATASET_PATH
    )

    INFO_PATH.write_text(
        json.dumps(identitas, indent=2),
        encoding="utf-8",
    )

    print(f"Kondisi sumber: {len(labels)}")
    print(f"Frame sumber  : {len(frames)}")
    print(f"Frame dipakai : {len(baris_index)}")
    print(f"ROI           : {ROI_BOX}")
    print("Satuan target : mm")

    for nama in ("train", "val", "test"):
        X = arrays[f"X_{nama}"]
        y = arrays[f"y_{nama}"]

        print(f"\n{nama.upper()}")
        print("Shape X:", X.shape)
        print("Shape y:", y.shape)
        print("Min y  :", float(y.min()))
        print("Max y  :", float(y.max()))

    print("\nDATA HASH:", identitas["data_hash"])
    print("SHA256   :", full_hash)

    print("\nDataset:", DATASET_PATH)
    print("Index  :", INDEX_PATH)
    print("Info   :", INFO_PATH)

    print("\nSTATUS: DATASET ACUAN BERHASIL DIBUAT")


if __name__ == "__main__":
    main()
