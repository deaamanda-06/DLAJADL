"""Satu perintah reproduksi tim.

    python run_all.py [--seeds 1 2 3] [--final] [--hanya <nim> ...]

Menjalankan baseline dan ``models/<nim>/run.py`` setiap anggota untuk tiap seed, lalu
merakit ``reports/hasil_komparasi.md`` dari ``results/*/metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parent


def jalankan(perintah: list[str]) -> None:
    print("$", " ".join(perintah), flush=True)
    subprocess.run(perintah, check=True, cwd=AKAR)


def main() -> None:
    from src.protokol import data

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--hanya", nargs="*", default=None, help="NIM tertentu saja")
    ap.add_argument("--tanpa-baseline", action="store_true")
    a = ap.parse_args()

    if not a.tanpa_baseline:
        jalankan([sys.executable, "-m", "src.common.baseline_mlp", "--seed", str(a.seeds[0])])
    anggota = sorted(
        p
        for p in (AKAR / "models").iterdir()
        if p.is_dir()
        and (p / "run.py").exists()
        and p.name != "TEMPLATE"
        and (a.hanya is None or p.name in a.hanya)
    )
    ringkas: dict[str, list[dict]] = {}
    for m in anggota:
        for s in a.seeds:
            perintah = [sys.executable, str(m / "run.py"), "--seed", str(s)]
            if a.final and s == a.seeds[-1]:
                perintah.append("--final")
            jalankan(perintah)
            hasil = json.loads((AKAR / "results" / m.name / "metrics.json").read_text("utf-8"))
            ringkas.setdefault(m.name, []).append(hasil)

    tugas = getattr(data, "TUGAS", "klasifikasi")
    if tugas == "regresi":
        baris = [
            "| NIM | Model | seed | val_rmse | val_mae | val_r2 | test_rmse | durasi (s) | perangkat |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for bl_dir in ("baseline_mlp", "baseline"):
            bl = AKAR / "results" / bl_dir / "metrics.json"
            if bl.exists():
                h = json.loads(bl.read_text("utf-8"))
                mt = h["metrik"]
                if "val_rmse" in mt:
                    baris.append(
                        f"| {bl_dir} | ridge/mlp | {h['seed']} | {mt['val_rmse']:.4f} | {mt['val_mae']:.4f} | "
                        f"{mt['val_r2']:.4f} | — | {h.get('durasi_detik', 0):.1f} | {h.get('perangkat', '')} |"
                    )
        for nim, daftar in ringkas.items():
            for h in daftar:
                mt = h["metrik"]
                test_str = f"{mt['test_rmse']:.4f}" if "test_rmse" in mt else "—"
                baris.append(
                    f"| {nim} | {h.get('model', '')} | {h['seed']} | {mt['val_rmse']:.4f} | {mt['val_mae']:.4f} | "
                    f"{mt['val_r2']:.4f} | {test_str} | {h.get('durasi_detik', 0):.1f} | {h.get('perangkat', '')} |"
                )
    else:
        baris = [
            "| NIM | Model | seed | val_acc | val_macro_f1 | test_acc | durasi (s) | perangkat |",
            "|---|---|---|---|---|---|---|---|",
        ]
        bl = AKAR / "results" / "baseline" / "metrics.json"
        if bl.exists():
            h = json.loads(bl.read_text("utf-8"))
            mt = h["metrik"]
            baris.append(
                f"| baseline | mlp dosen | {h['seed']} | {mt['val_acc']:.4f} | {mt['val_macro_f1']:.4f} | — | "
                f"{h.get('durasi_detik', 0):.0f} | {h.get('perangkat', '')} |"
            )
        for nim, daftar in ringkas.items():
            for h in daftar:
                mt = h["metrik"]
                baris.append(
                    f"| {nim} | {h.get('model', '')} | {h['seed']} | {mt['val_acc']:.4f} | {mt['val_macro_f1']:.4f} | "
                    f"{mt.get('test_acc', float('nan')):.4f} | {h.get('durasi_detik', 0):.0f} | {h.get('perangkat', '')} |"
                )

    (AKAR / "reports").mkdir(exist_ok=True)
    out = AKAR / "reports" / "hasil_komparasi.md"
    out.write_text("# Hasil komparasi (dibangkitkan run_all.py)\n\n" + "\n".join(baris) + "\n", encoding="utf-8")
    print("tabel komparasi →", out.relative_to(AKAR))


if __name__ == "__main__":
    main()
