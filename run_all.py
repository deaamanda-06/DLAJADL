
"""Satu perintah reproduksi tim untuk kasus regresi."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parent


def jalankan(perintah: list[str]) -> None:
    print("\n$", " ".join(perintah), flush=True)
    subprocess.run(perintah, check=True, cwd=AKAR)


def baca_metrics(nim: str) -> dict:
    kandidat = [
        AKAR / "results" / nim / "metrics.json",
        AKAR / "results" / nim / "simulasi" / "metrics.json",
    ]

    tersedia = [p for p in kandidat if p.exists()]

    if not tersedia:
        raise FileNotFoundError(
            f"metrics.json tidak ditemukan untuk {nim}"
        )

    # Ambil hasil yang paling baru ditulis.
    jalur = max(tersedia, key=lambda p: p.stat().st_mtime)

    print(f"Membaca hasil: {jalur.relative_to(AKAR)}")

    return json.loads(jalur.read_text(encoding="utf-8"))


def fmt(nilai) -> str:
    if nilai is None:
        return "—"
    return f"{float(nilai):.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--hanya", nargs="*", default=None)
    ap.add_argument("--tanpa-baseline", action="store_true")

    a = ap.parse_args()

    # 1. Jalankan baseline dosen jika diminta.
    if not a.tanpa_baseline:
        jalankan([
            sys.executable,
            "-m",
            "src.common.baseline",
            "--seed",
            str(a.seeds[0])
        ])

    # 2. Temukan folder setiap anggota.
    anggota = sorted(
        p for p in (AKAR / "models").iterdir()
        if p.is_dir()
        and (p / "run.py").exists()
        and p.name != "TEMPLATE"
        and (a.hanya is None or p.name in a.hanya)
    )

    ringkas: dict[str, list[dict]] = {}

    # 3. Jalankan model setiap anggota.
    for m in anggota:
        for s in a.seeds:

            perintah = [
                sys.executable,
                str(m / "run.py"),
                "--seed",
                str(s)
            ]

            if a.final and s == a.seeds[-1]:
                perintah.append("--final")

            jalankan(perintah)

            hasil = baca_metrics(m.name)

            ringkas.setdefault(m.name, []).append(hasil)

    # 4. Susun tabel perbandingan regresi.
    baris = [
        "| NIM | Model | Seed | MAE (mm) | RMSE (mm) | R2 | Durasi (s) | Perangkat |",
        "|---|---|---:|---:|---:|---:|---:|---|"
    ]

    # Hasil baseline.
    bl = AKAR / "results" / "baseline" / "metrics.json"

    if bl.exists():
        h = json.loads(bl.read_text(encoding="utf-8"))
        mt = h["metrik"]

        baris.append(
            f"| baseline | {h.get('model', 'baseline')} | "
            f"{h['seed']} | {fmt(mt.get('val_mae'))} | "
            f"{fmt(mt.get('val_rmse'))} | "
            f"{fmt(mt.get('val_r2'))} | "
            f"{h.get('durasi_detik', 0):.2f} | "
            f"{h.get('perangkat', '')} |"
        )

    # Hasil model anggota.
    for nim, daftar in ringkas.items():
        for h in daftar:
            mt = h["metrik"]

            baris.append(
                f"| {nim} | {h.get('model', '')} | "
                f"{h['seed']} | {fmt(mt.get('val_mae'))} | "
                f"{fmt(mt.get('val_rmse'))} | "
                f"{fmt(mt.get('val_r2'))} | "
                f"{h.get('durasi_detik', 0):.2f} | "
                f"{h.get('perangkat', '')} |"
            )

    # 5. Simpan laporan.
    (AKAR / "reports").mkdir(exist_ok=True)

    out = AKAR / "reports" / "hasil_komparasi.md"

    out.write_text(
        "# Hasil Komparasi Model Regresi\n\n"
        + "\n".join(baris)
        + "\n",
        encoding="utf-8"
    )

    print("\n====================================")
    print("INTEGRASI RUN_ALL BERHASIL")
    print("====================================")
    print("Laporan:", out.relative_to(AKAR))


if __name__ == "__main__":
    main()
