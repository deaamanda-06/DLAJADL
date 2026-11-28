
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL_PATH = ROOT / "data" / "label.csv"


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def parse_tinggi(value):
    try:
        angka = float(str(value).strip().replace(",", "."))
        return angka if math.isfinite(angka) else None
    except (ValueError, TypeError):
        return None


def main():
    with open(LABEL_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print("=" * 60)
    print("PEMERIKSAAN MUTU DATA PRIMER TIM 02")
    print("=" * 60)
    print(f"Jumlah baris aktual : {len(rows)}")
    print("Jumlah target       : 75")
    print(f"Selisih             : {len(rows) - 75}")
    print(f"SHA256 label        : {sha256_file(LABEL_PATH)}")

    blocks = defaultdict(list)

    for row in rows:
        blocks[row["ulangan_blok"]].append(row)

    total_kosong = 0
    total_invalid = 0
    total_penurunan = 0
    total_desimal_koma = 0

    nomor = [row["no"] for row in rows]
    duplikat = len(nomor) - len(set(nomor))

    for block in sorted(blocks, key=int):
        data = sorted(
            blocks[block],
            key=lambda row: int(row["no"])
        )

        kosong = []
        invalid = []
        penurunan = []
        desimal_koma = []
        values = []
        previous = None

        for row in data:
            no = row["no"]
            raw = row["tinggi_air_mm"]

            if "," in str(raw):
                desimal_koma.append((no, raw))

            if raw is None or not str(raw).strip():
                kosong.append(no)
                continue

            tinggi = parse_tinggi(raw)

            if tinggi is None or tinggi < 0:
                invalid.append(no)
                continue

            values.append(tinggi)

            if previous is not None:
                prev_no, prev_tinggi = previous

                if tinggi < prev_tinggi:
                    penurunan.append(
                        (prev_no, no, prev_tinggi, tinggi)
                    )

            previous = (no, tinggi)

        print(f"\nBLOK {block}")
        print("-" * 45)
        print(f"Jumlah kondisi     : {len(data)}/25")
        print(f"Label kosong       : {len(kosong)}")
        print(f"Label tidak valid  : {len(invalid)}")
        print(f"Desimal koma       : {len(desimal_koma)}")
        print(f"Jumlah penurunan   : {len(penurunan)}")

        if values:
            print(f"Tinggi minimum     : {min(values):.2f} cm")
            print(f"Tinggi maksimum    : {max(values):.2f} cm")

        if kosong or invalid:
            status = "BELUM DAPAT DIPASTIKAN"
        elif penurunan:
            status = "TIDAK"
        else:
            status = "YA"

        print(f"Monoton            : {status}")

        for prev_no, no, prev, curr in penurunan:
            print(
                f"  Penurunan kondisi {prev_no} -> {no}: "
                f"{prev} -> {curr} cm"
            )

        for no, raw in desimal_koma:
            print(f"  Desimal koma kondisi {no}: {raw}")

        if kosong:
            print("Kondisi kosong:", kosong)

        if invalid:
            print("Kondisi tidak valid:", invalid)

        total_kosong += len(kosong)
        total_invalid += len(invalid)
        total_penurunan += len(penurunan)
        total_desimal_koma += len(desimal_koma)

    print("\n" + "=" * 60)
    print("RINGKASAN PEMERIKSAAN")
    print("=" * 60)
    print(f"Total kondisi       : {len(rows)}")
    print(f"Total blok          : {len(blocks)}")
    print(f"Nomor duplikat      : {duplikat}")
    print(f"Total label kosong  : {total_kosong}")
    print(f"Total label invalid : {total_invalid}")
    print(f"Total desimal koma  : {total_desimal_koma}")
    print(f"Total penurunan     : {total_penurunan}")

    lulus = (
        len(rows) == 75
        and set(blocks) == {"1", "2", "3"}
        and all(len(b) == 25 for b in blocks.values())
        and duplikat == 0
        and total_kosong == 0
        and total_invalid == 0
        and total_penurunan == 0
    )

    print(
        "STATUS QC LABEL     : "
        + ("LULUS" if lulus else "PERLU PEMERIKSAAN")
    )


if __name__ == "__main__":
    main()
