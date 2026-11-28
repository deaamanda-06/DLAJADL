
from pathlib import Path
import numpy as np

AKAR = Path(__file__).resolve().parents[2]
OUTPUT = AKAR / "data" / "simulasi_smoke.csv"


def main():
    rng = np.random.default_rng(20261110)

    jumlah = 240
    tinggi = rng.uniform(10, 190, jumlah).astype(np.float32)

    baris = np.arange(8, dtype=np.float32)[None, :, None]
    kolom = np.arange(8, dtype=np.float32)[None, None, :]

    batas_air = (
        7.0 - (tinggi / 200.0) * 7.0
    )[:, None, None]

    permukaan = 1.0 / (
        1.0 + np.exp(-2.2 * (baris - batas_air))
    )

    tekstur = 0.025 * np.sin(kolom * 0.9)

    noise = rng.normal(
        0, 0.025, size=(jumlah, 8, 8)
    )

    citra = np.clip(
        0.12 + 0.72 * permukaan + tekstur + noise,
        0, 1
    ).astype(np.float32)

    fitur = citra.reshape(jumlah, -1)

    data = np.column_stack([fitur, tinggi])

    kolom_csv = [
        f"pixel_{i:02d}" for i in range(64)
    ] + ["tinggi_air_mm"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        OUTPUT,
        data,
        delimiter=",",
        header=",".join(kolom_csv),
        comments="",
        fmt="%.6f"
    )

    print("DATA SIMULASI BERHASIL DIBUAT")
    print("Jumlah data:", jumlah)
    print("Ukuran citra: 8 x 8")
    print("Lokasi:", OUTPUT)


if __name__ == "__main__":
    main()
