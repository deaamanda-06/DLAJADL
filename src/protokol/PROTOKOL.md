# Protokol evaluasi tim — Tim 02 (Ruang Cerdas 5 Volt)

## Identitas dan kasus

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Taksiran tinggi air di wadah B dari citra (regresi mm) |
| Sumber dataset | Data meja uji, sesi simulasi (`tim02_sesi-20261116-095756.tar.gz`); dijalankan ulang pada data meja asli begitu tersedia |
| Jenis tugas (`TUGAS`) | regresi |
| Target / label | `tinggi_air_mm` (rentang 0.0 – 9.8 mm, 75 kondisi / 3 blok) |
| Masukan | Citra `kamera/` 1280×720, 1 frame per 0,5 detik |
| Pemilihan frame | Frame saat air diam: `t_selesai+2.0` s.d. `min(t_selesai+4.5, t_catat-1.0)`, maksimum 5 frame per kondisi. 1 detik pertama setiap kondisi dibuang |
| Satuan pemisahan | Data uji = satu blok (siklus pengisian) penuh, yaitu blok bernomor tertinggi (25 kondisi). Validasi = 8 kondisi acak dari blok lain (seed 2026). Sisanya latih (42 kondisi). Tidak ada kondisi, frame, atau blok uji yang muncul di subset lain |
| Ukuran data latih / validasi / uji | 208 / 40 / 125 citra (42 / 8 / 25 kondisi) |
| Catatan ukuran | Kondisi 43 dan 48 (latih) hanya 4 frame karena jitter waktu kamera pada batas jendela |
| Praproses yang sama untuk semua model | Potong ROI Wadah B `(820, 300, 1140, 720)` → resize `224×224` RGB, skala `[0, 1]`, lalu standardisasi per kanal dengan mean/std dari **data latih saja** |
| Metrik utama untuk komparasi | **MAE (mm)**; RMSE dan R² sebagai pendamping |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | dihitung otomatis oleh `src.common.evaluate` |
| Pemeriksaan | `tests/test_protokol.py` (6 uji lolos): subset terisi, tanpa kebocoran, uji per blok, normalisasi dari latih, 1 detik pertama dibuang |

Status: diajukan lewat pull request, menunggu persetujuan pembimbing.