
"""Pemuat dan validasi dataset acuan Tim 02 dalam mm."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from src.protokol.pemotong_blok import (
    DATASET_PATH,
    INDEX_PATH,
    INFO_PATH,
    SESI_DIR,
)

TUGAS = "regresi"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def baca_info():
    info = json.loads(
        INFO_PATH.read_text(encoding="utf-8")
    )

    keys = [
        "schema",
        "label_source_sha256",
        "roi",
        "ukuran",
        "satuan_target",
        "split_blok",
        "array_info",
        "index_sha256",
    ]

    identitas = {k: info[k] for k in keys}
    actual = sha256(canonical(identitas))

    assert actual == info["data_hash_sha256"]
    assert actual[:16] == info["data_hash"]
    assert info["satuan_target"] == "mm"

    return info


def baca_array(d, key, info):
    arr = np.ascontiguousarray(d[key])
    expected = info["array_info"][key]

    assert list(arr.shape) == expected["shape"]
    assert arr.dtype.str == expected["dtype"]
    assert sha256(arr.tobytes()) == expected["sha256"]
    assert np.isfinite(arr).all()

    return arr


def verifikasi():
    info = baca_info()

    label_path = SESI_DIR / "label.csv"

    assert sha256(label_path.read_bytes()) == (
        info["label_source_sha256"]
    )

    with label_path.open(encoding="utf-8-sig") as f:
        labels = {
            int(r["no"]): r
            for r in csv.DictReader(f)
        }

    with INDEX_PATH.open(encoding="utf-8") as f:
        rows = [
            {
                "kondisi_no": int(r["kondisi_no"]),
                "blok": int(r["blok"]),
                "split": r["split"],
                "berkas": r["berkas"],
                "tinggi_air_mm": float(r["tinggi_air_mm"]),
            }
            for r in csv.DictReader(f)
        ]

    assert sha256(canonical(rows)) == info["index_sha256"]
    assert len(rows) == 373
    assert len({r["berkas"] for r in rows}) == 373
    assert {r["kondisi_no"] for r in rows} == set(labels)
    assert len(labels) == 75

    for r in rows:
        sumber = labels[r["kondisi_no"]]

        y_cm = float(
            sumber["tinggi_air_mm"].replace(",", ".")
        )

        assert r["tinggi_air_mm"] == y_cm * 10.0
        assert r["blok"] == int(sumber["ulangan_blok"])
        assert r["split"] == info["split_blok"][str(r["blok"])]

    print("=" * 60)
    print("VALIDASI DATASET ACUAN TIM 02")
    print("=" * 60)

    with np.load(DATASET_PATH, allow_pickle=False) as d:
        assert set(d.files) == set(info["array_info"])

        for nama, blok, n in [
            ("train", 1, 125),
            ("val", 2, 123),
            ("test", 3, 125),
        ]:
            X = baca_array(d, f"X_{nama}", info)
            y = baca_array(d, f"y_{nama}", info)

            selected = [
                r for r in rows
                if r["split"] == nama
            ]

            expected_y = np.asarray(
                [r["tinggi_air_mm"] for r in selected],
                dtype=np.float32,
            )

            assert len(selected) == n
            assert len({r["kondisi_no"] for r in selected}) == 25
            assert {r["blok"] for r in selected} == {blok}
            assert X.shape == (n, 3, 224, 224)
            assert y.shape == (n,)
            assert np.array_equal(y, expected_y)
            assert 0 <= X.min() <= X.max() <= 1

            print(f"\n{nama.upper()}")
            print("Blok         :", blok)
            print("Kondisi      : 25")
            print("Frame        :", n)
            print("Rentang label:", y.min(), "-", y.max(), "mm")
            print("Hash array   : MATCH")

    print("\nKondisi lintas split: 0")
    print("Blok lintas split   : 0")
    print("Frame duplikat      : 0")
    print("Hash label sumber   : MATCH")
    print("Hash index          : MATCH")
    print("DATA HASH           :", info["data_hash"])
    print("STATUS              : VALIDASI BERHASIL")

    return info


def muat():
    """Memuat train dan validation tanpa membuka data uji
    untuk pelatihan atau pemilihan model.
    """

    info = baca_info()

    with np.load(DATASET_PATH, allow_pickle=False) as d:
        return (
            baca_array(d, "X_train", info),
            baca_array(d, "y_train", info),
            baca_array(d, "X_val", info),
            baca_array(d, "y_val", info),
            None,
            None,
        )


def simpan_metrik_validasi(
    pred_val,
    seed,
    model,
    durasi,
    perangkat,
    out,
):
    # Gunakan fungsi metrik resmi tanpa mengubah starter.
    from src.common.evaluate import metrik_regresi

    info = baca_info()

    with np.load(DATASET_PATH, allow_pickle=False) as d:
        y_val = baca_array(d, "y_val", info)

    pred_val = np.asarray(pred_val)

    assert pred_val.shape == y_val.shape
    assert np.isfinite(pred_val).all()

    metrik = metrik_regresi(y_val, pred_val)

    hasil = {
        "metrik": {
            f"val_{k}": round(float(v), 6)
            for k, v in metrik.items()
        },
        "seed": int(seed),
        "data_hash": info["data_hash"],
        "perangkat": perangkat,
        "model": model,
        "n_val": int(len(y_val)),
        "n_test": 0,
        "durasi_detik": round(float(durasi), 1),
        "satuan_target": "mm",
    }

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "metrics.json").write_text(
        json.dumps(hasil, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(hasil, indent=2))

    return hasil


if __name__ == "__main__":
    verifikasi()
