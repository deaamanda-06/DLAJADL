import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from src.protokol import data, pemotong

AKAR = Path(__file__).resolve().parents[1]
INDEX = AKAR / "data" / "cropped" / "index.csv"
SESI = AKAR / "data" / "sesi-20261116-095756"


@pytest.fixture(scope="module")
def baris():
    with INDEX.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def per_split(baris, kolom):
    h = defaultdict(set)
    for r in baris:
        h[r["split"]].add(r[kolom])
    return h


def test_semua_subset_terisi(baris):
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()
    jumlah = defaultdict(int)
    for r in baris:
        jumlah[r["split"]] += 1
    for nama, X, y in (("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)):
        assert len(X) > 0 and len(X) == len(y)
        assert len(X) == jumlah[nama]


@pytest.mark.parametrize("kolom", ["kondisi_no", "berkas"])
def test_tanpa_kebocoran(baris, kolom):
    h = per_split(baris, kolom)
    assert not (h["train"] & h["test"])
    assert not (h["train"] & h["val"])
    assert not (h["val"] & h["test"])


def test_uji_dipisah_per_blok(baris):
    h = per_split(baris, "blok")
    assert len(h["test"]) == 1
    assert not (h["test"] & h["train"])
    assert not (h["test"] & h["val"])


def test_normalisasi_dari_latih_saja():
    d = np.load(pemotong.DATASET_PATH)
    mu = d["X_train"].mean(axis=(0, 2, 3), keepdims=True)
    sd = np.maximum(d["X_train"].std(axis=(0, 2, 3), keepdims=True), 1e-6)
    _, _, Xva, _, _, _ = data.muat()
    np.testing.assert_allclose(Xva, (d["X_val"] - mu) / sd, rtol=1e-4, atol=1e-4)


def test_buang_1_detik_pertama(baris):
    waktu = {}
    with (SESI / "kamera_index.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            waktu[r["berkas"]] = float(r["t_host"])
    with (SESI / "label.csv").open(encoding="utf-8") as f:
        t_selesai = {int(r["no"]): float(r["t_selesai_host"]) for r in csv.DictReader(f)}
    for r in baris:
        assert waktu[r["berkas"]] - t_selesai[int(r["kondisi_no"])] >= 1.0