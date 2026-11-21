"""Pemuat data kasus tim ? Taksiran Tinggi Air di Wadah B dari Citra (Regresi)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TUGAS = "regresi"

AKAR = Path(__file__).resolve().parents[2]
DIR_SESI = AKAR / "data" / "sesi_primer" / "sesi-20261116-095756"

# Koordinat ROI Wadah B pada citra asli 1280x720: (x1, y1, x2, y2)
ROI_WADAH_B = (900, 240, 1065, 670)
UKURAN_CITRA = (224, 224)
MAX_FRAME_PER_KONDISI = 8


def _bersihkan_label(val) -> float:
    return float(str(val).replace(",", ".").strip())


def _muat_blok(lbl_df: pd.DataFrame, cam_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    dir_kamera = DIR_SESI / "kamera"

    for _, row in lbl_df.iterrows():
        target_mm = _bersihkan_label(row["tinggi_air_mm"])
        # Ambil citra setelah pompa mati (t_selesai_host) s.d. dicatat (t_catat_host)
        mask_tenang = (cam_df["t_host"] >= row["t_selesai_host"] + 1.0) & (
            cam_df["t_host"] <= row["t_catat_host"]
        )
        frames = cam_df.loc[mask_tenang, "berkas"].tolist()
        if len(frames) < 4:
            mask_all = (cam_df["t_host"] >= row["t_selesai_host"]) & (
                cam_df["t_host"] <= row["t_catat_host"]
            )
            frames = cam_df.loc[mask_all, "berkas"].tolist()
        if not frames:
            continue

        if len(frames) > MAX_FRAME_PER_KONDISI:
            idx = np.linspace(0, len(frames) - 1, MAX_FRAME_PER_KONDISI, dtype=int)
            frames = [frames[i] for i in idx]

        for nama_file in frames:
            path_img = dir_kamera / nama_file
            if not path_img.exists():
                continue
            with Image.open(path_img) as img:
                img_rgb = img.convert("RGB")
                img_crop = img_rgb.crop(ROI_WADAH_B)
                img_resized = img_crop.resize(UKURAN_CITRA, Image.Resampling.BILINEAR)
                arr = np.asarray(img_resized, dtype=np.float32) / 255.0
                X_list.append(arr)
                y_list.append(target_mm)

    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def muat() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Memuat data primer meja dengan pemisahan per blok (Blok 1=Train, Blok 2=Val, Blok 3=Test)."""
    lbl = pd.read_csv(DIR_SESI / "label.csv")
    cam = pd.read_csv(DIR_SESI / "kamera_index.csv")

    Xtr, ytr = _muat_blok(lbl[lbl["ulangan_blok"] == 1], cam)
    Xva, yva = _muat_blok(lbl[lbl["ulangan_blok"] == 2], cam)
    Xte, yte = _muat_blok(lbl[lbl["ulangan_blok"] == 3], cam)

    # Normalisasi hanya dari data latih (Blok 1)
    mean = Xtr.mean(axis=(0, 1, 2), keepdims=True)
    std = Xtr.std(axis=(0, 1, 2), keepdims=True) + 1e-6

    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std
    Xte = (Xte - mean) / std

    return Xtr, ytr, Xva, yva, Xte, yte
