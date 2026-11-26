"""Pemuatan data bersama untuk run.py (RNN) dan baseline_nondl.py."""
import os
import numpy as np


def load_samples(data_dir: str, roi=(850, 290, 1130, 720), skip_after_stop=2.0,
                 max_frames=20, clean=True, mad_k=3.0, diff_floor=0.02):
    """
    Format sesi meja (tim02_sesi-*/): label.csv, kamera_index.csv, kamera/fXXXXXX.jpg
    - blok sebenarnya = kolom `ulangan_blok` (kolom `blok` isinya 1 semua).
    - tinggi_air_mm kadang memakai koma desimal ("4,5") -> diganti titik.
    - frame dipakai: t_selesai_host + skip_after_stop <= t_host <= t_catat_host.
    - clean=True: buang frame "kotor" (tangan masuk, ROI bergeser/tersenggol).
      Per kondisi, hitung selisih rata-rata tiap crop terhadap crop median
      kondisi itu; frame dengan selisih > median + mad_k*MAD (dan > diff_floor)
      dibuang. Sisanya disubsampel rata maksimal `max_frames` per kondisi.
    - roi = (x0, y0, x1, y1) di citra 1280x720; wadah B ada di kanan.
    """
    import pandas as pd
    from PIL import Image

    lab = pd.read_csv(os.path.join(data_dir, "label.csv"))
    idx = pd.read_csv(os.path.join(data_dir, "kamera_index.csv"))
    lab["mm"] = lab["tinggi_air_mm"].astype(str).str.replace(",", ".").astype(float)

    samples, n_before, n_after = [], 0, 0
    for _, r in lab.iterrows():
        f = idx[(idx.t_host >= r.t_selesai_host + skip_after_stop)
                & (idx.t_host <= r.t_catat_host)]
        if len(f) == 0:
            continue
        imgs = []
        for _, fr in f.iterrows():
            im = Image.open(os.path.join(data_dir, "kamera", fr.berkas)).convert("L")
            im = im.crop(roi).resize((224, 224), Image.BILINEAR)
            imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
        imgs = np.stack(imgs)
        n_before += len(imgs)
        if clean and len(imgs) >= 5:
            med = np.median(imgs, axis=0)
            d = np.abs(imgs - med).mean(axis=(1, 2))
            mad = np.median(np.abs(d - np.median(d))) + 1e-8
            keep = d <= max(np.median(d) + mad_k * 1.4826 * mad, diff_floor)
            imgs = imgs[keep]
        n_after += len(imgs)
        if len(imgs) > max_frames:
            imgs = imgs[np.linspace(0, len(imgs) - 1, max_frames).round().astype(int)]
        for im in imgs:
            samples.append({"img": im, "mm": float(r.mm),
                            "blok": int(r.ulangan_blok), "kondisi": int(r.no)})
    print(f"frame jendela label: {n_before} -> setelah pembersihan: {n_after}")
    return samples


def split_by_block(samples, test_block=3, val_block=2):
    """Split per BLOK (bukan acak per citra) agar tidak ada kebocoran."""
    train = [s for s in samples if s["blok"] not in (test_block, val_block)]
    val = [s for s in samples if s["blok"] == val_block]
    test = [s for s in samples if s["blok"] == test_block]
    return train, val, test


def to_arrays(samples):
    x = np.stack([s["img"] for s in samples]).astype(np.float32)
    y = np.array([s["mm"] for s in samples], dtype=np.float32)
    return x, y
