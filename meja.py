#!/usr/bin/env python3
"""Pemotong citra dan pembuat sesi simulasi Tim 02.

Contoh:
  python meja.py --simulasi jalankan rencana/t02-tinggi-air-citra.yaml
  python meja.py jalankan rencana/t02-tinggi-air-citra.yaml --kamera kamera/
  python meja.py jalankan rencana/t02-tinggi-air-citra.yaml --kamera kamera/ --roi 400,80,500,600

ROI harus ditentukan berdasarkan posisi wadah B pada kamera sebenarnya.
Mode simulasi memakai ROI ilustratif internal dan bukan data eksperimen nyata.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
UKURAN = (224, 224)  # lebar, tinggi


def baca_rencana(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("File rencana YAML tidak berisi objek konfigurasi.")
    return cfg


def parse_roi(teks: str | None):
    if not teks:
        return None
    try:
        x, y, w, h = (int(v.strip()) for v in teks.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError("ROI harus berformat x,y,w,h") from exc
    if min(x, y) < 0 or min(w, h) <= 0:
        raise argparse.ArgumentTypeError("Koordinat ROI tidak valid.")
    return x, y, w, h


def crop_resize(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    tinggi, lebar = frame.shape[:2]
    if x + w > lebar or y + h > tinggi:
        raise ValueError(
            f"ROI {(x, y, w, h)} melewati ukuran citra {lebar}x{tinggi}. "
            "Periksa koordinat ROI."
        )
    potong = frame[y:y+h, x:x+w]
    return cv2.resize(potong, UKURAN, interpolation=cv2.INTER_AREA)


def gambar_simulasi(lebar: int, tinggi: int, tinggi_air_mm: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Membuat citra sintetis wadah; bukan representasi kalibrasi meja nyata."""
    img = np.full((tinggi, lebar, 3), (225, 225, 225), dtype=np.uint8)
    # Wadah ilustratif berada pada area tengah gambar simulasi.
    x1, x2 = int(lebar * .31), int(lebar * .69)
    y1, y2 = int(tinggi * .10), int(tinggi * .91)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 70, 70), 5)
    cv2.rectangle(img, (x1+5, y1+5), (x2-5, y2-5), (245, 245, 245), -1)
    # Tinggi target simulasi dipetakan ke fraksi tinggi wadah.
    fraksi = float(np.clip(tinggi_air_mm / 100.0, 0.03, 0.92))
    y_air = int((y2-5) - fraksi * (y2-y1-10))
    cv2.rectangle(img, (x1+7, y_air), (x2-7, y2-7), (190, 125, 45), -1)
    cv2.line(img, (x1+7, y_air), (x2-7, y_air), (220, 190, 120), 3)
    # Garis penggaris sederhana sebagai elemen visual sintetis.
    for mm in range(0, 101, 5):
        yy = int((y2-5) - (mm/100) * (y2-y1-10))
        panjang = 18 if mm % 10 == 0 else 9
        cv2.line(img, (x2+7, yy), (x2+7+panjang, yy), (30, 30, 30), 1)
    noise = rng.normal(0, 1.8, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def jalankan_simulasi(cfg: dict, output: Path) -> Path:
    exp = cfg.get("eksperimen", {})
    cam = exp.get("kamera", {})
    resolucao = cam.get("resolusi", "1280x720").lower().split("x")
    if len(resolucao) != 2:
        raise ValueError("eksperimen.kamera.resolusi harus seperti 1280x720.")
    lebar, tinggi = map(int, resolucao)
    interval = float(cam.get("interval_detik", 0.5))
    durasi = float(exp.get("durasi_kondisi_detik", 4))
    buang = 1.0
    if interval <= 0 or durasi <= buang:
        raise ValueError("Interval harus positif dan durasi kondisi harus > 1 detik.")

    blok = int(exp.get("ulangan_blok", 3))
    kondisi_total = int(exp.get("total_kondisi", 75))
    per_blok = max(1, kondisi_total // max(1, blok))
    # ROI ini hanya untuk kanvas simulasi, bukan koordinat kamera meja sebenarnya.
    roi = (int(lebar*.25), int(tinggi*.05), int(lebar*.50), int(tinggi*.90))
    sesi = output / f"sesi_simulasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    frame_dir = sesi / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    metadata = sesi / "metadata.csv"
    rng = np.random.default_rng(2026)
    kolom = ["frame_path", "block_id", "condition_id", "timestamp_s",
             "condition_start_s", "tinggi_air_mm", "is_simulated"]
    jumlah = 0
    with metadata.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=kolom)
        writer.writeheader()
        for b in range(1, blok+1):
            for c in range(1, per_blok+1):
                # Target simulasi semata-mata untuk menguji alur pemuatan.
                target = float(np.clip((c-1) * (100/max(1, per_blok-1)), 0, 100))
                condition_start = (c-1) * (durasi + float(exp.get("jeda_kondisi_detik", 3)))
                t = buang
                while t < durasi - 1e-9:
                    full = gambar_simulasi(lebar, tinggi, target, rng)
                    cropped = crop_resize(full, roi)
                    name = f"b{b:02d}_c{c:03d}_t{t:.1f}.jpg"
                    rel = Path("frames") / name
                    if not cv2.imwrite(str(sesi / rel), cropped):
                        raise IOError(f"Gagal menyimpan {sesi / rel}")
                    writer.writerow({
                        "frame_path": rel.as_posix(), "block_id": b,
                        "condition_id": f"B{b:02d}_C{c:03d}",
                        "timestamp_s": round(t, 3),
                        "condition_start_s": round(condition_start, 3),
                        "tinggi_air_mm": target, "is_simulated": True
                    })
                    jumlah += 1
                    t += interval
    info = sesi / "SIMULASI.txt"
    info.write_text(
        "DATA SINTETIS UNTUK UJI ALUR PROGRAM SAJA.\n"
        "Bukan hasil pengukuran meja dan tidak boleh dipakai sebagai hasil penelitian.\n"
        f"Jumlah citra: {jumlah}\nROI kanvas simulasi: {roi}\n",
        encoding="utf-8"
    )
    print(f"Sesi simulasi dibuat: {sesi}")
    print(f"Jumlah citra tersimpan: {jumlah}")
    print("Peringatan: label dan citra ini sintetis, bukan data eksperimen asli.")
    return sesi


def jalankan_kamera(cfg: dict, kamera: Path, roi, output: Path) -> Path:
    if roi is None:
        raise ValueError(
            "ROI kamera nyata belum diketahui. Tentukan dengan --roi x,y,w,h "
            "setelah memeriksa posisi wadah B pada citra kamera."
        )
    sesi = output / f"sesi_meja_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    frames = sesi / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    cam_cfg = cfg.get("eksperimen", {}).get("kamera", {})
    ukuran = cam_cfg.get("resolusi", "1280x720").split("x")
    expected = (int(ukuran[0]), int(ukuran[1]))
    rows = []
    for src in sorted(kamera.iterdir()):
        if not src.is_file() or src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        frame = cv2.imread(str(src))
        if frame is None:
            print(f"Lewati file yang tidak terbaca: {src.name}", file=sys.stderr)
            continue
        if (frame.shape[1], frame.shape[0]) != expected:
            raise ValueError(f"{src.name}: resolusi bukan {expected[0]}x{expected[1]}.")
        crop = crop_resize(frame, roi)
        dest = frames / f"{src.stem}_224.jpg"
        cv2.imwrite(str(dest), crop)
        # Data meja perlu metadata label/waktu dan blok dari pencatatan eksperimen.
        rows.append({"frame_path": (Path("frames")/dest.name).as_posix(),
                     "block_id": "", "condition_id": "", "timestamp_s": "",
                     "condition_start_s": "", "tinggi_air_mm": "",
                     "is_simulated": False})
    with (sesi/"metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
            ["frame_path","block_id","condition_id","timestamp_s",
             "condition_start_s","tinggi_air_mm","is_simulated"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Citra terpotong tersimpan: {sesi} ({len(rows)} file)")
    print("Lengkapi metadata blok, kondisi, waktu, dan tinggi_air_mm dari catatan pengukuran.")
    return sesi


def main():
    parser = argparse.ArgumentParser(description="Pemotong citra Tim 02")
    parser.add_argument("--simulasi", action="store_true",
                        help="buat sesi sintetis untuk menguji pipeline")
    parser.add_argument("aksi", choices=["jalankan"])
    parser.add_argument("rencana", type=Path, help="file YAML rencana eksperimen")
    parser.add_argument("--kamera", type=Path, default=Path("kamera/"),
                        help="folder citra kamera nyata")
    parser.add_argument("--roi", type=parse_roi, default=None,
                        help="ROI wadah B: x,y,w,h (wajib untuk kamera nyata)")
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.rencana.is_absolute():
        cfg_path = args.rencana
    else:
        kandidat = [Path.cwd()/args.rencana, ROOT/args.rencana,
                    ROOT/"rencana"/args.rencana.name]
        cfg_path = next((p for p in kandidat if p.exists()), kandidat[-1])
    cfg = baca_rencana(cfg_path.resolve())
    out = args.output if args.output.is_absolute() else (ROOT/args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.simulasi:
        jalankan_simulasi(cfg, out.resolve())
    else:
        kamera = args.kamera if args.kamera.is_absolute() else (ROOT/args.kamera)
        jalankan_kamera(cfg, kamera.resolve(), args.roi, out.resolve())


if __name__ == "__main__":
    main()
