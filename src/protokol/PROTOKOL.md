# Protokol evaluasi tim — Tim 02 (Ruang Cerdas 5 Volt)

## Identitas dan kasus

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Taksiran tinggi air di wadah B dari citra (regresi mm) |
| Sumber dataset | Data primer meja uji (`tim02_sesi-20261116-095756.tar.gz`) |
| Jenis tugas (`TUGAS`) | regresi |
| Target / label | `tinggi_air_mm` (rentang 0.0 – 9.8 mm, 75 kondisi / 3 blok) |
| Ukuran data latih / validasi / uji | 51 kondisi (255 citra) / 9 kondisi (45 citra) / 15 kondisi (75 citra) |
| Cara split | `split.indeks` seed 2026 pada tingkat kondisi (tanpa kebocoran frame antar-split) |
| Metrik utama untuk komparasi | RMSE & MAE (mm), R² |
| Praproses yang sama untuk semua model | Pemotong ROI Wadah B `(820, 300, 1140, 720)` → resize `224×224` RGB, normalisasi `[0, 1]` |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | dihitung otomatis oleh `src.common.evaluate` |

Disetujui pembimbing pada: Sprint 1 (integrasi data primer meja uji)
