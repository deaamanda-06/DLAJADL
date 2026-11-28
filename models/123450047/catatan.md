# Catatan model (benget - 123450047- kelompok 2)

## Alasan optimizer

AdamW dipilih untuk mengoptimalkan parameter Transformer Encoder.
AdamW memisahkan weight decay dari pembaruan gradien (decoupled),
sehingga regularisasi L2 tetap konsisten pada learning rate adaptif.
Konfigurasi Sprint 1: learning rate 0.001, weight decay 0.0001,
batch size 16, loss MSE (regresi tinggi air dalam mm; target
dinormalisasi dengan statistik data latih).

## Regularisasi

- Weight decay 0.0001 (melalui AdamW).
- Dropout 0.1 pada encoder layer dan regression head.
- Gradient clipping (max norm 1.0).
- Normalisasi per kanal dari data latih saja (mencegah kebocoran).
- Split per kondisi, bukan per citra, sehingga tidak ada kebocoran frame.

## Pencarian hyperparameter (tabel)

Sprint 1 hanya menjalankan satu konfigurasi awal (seed 1).
Pencarian hyperparameter dikerjakan di Sprint 2.

| No | LR | Weight decay | Dropout | Batch | Epoch | d_model | Head | Layer | MAE val (mm) | RMSE val (mm) | R2 val |
|----|--------|--------|-----|----|---|----|---|---|-------|-------|-------|
| 1 (awal) | 0.001 | 0.0001 | 0.1 | 16 | 5 | 64 | 4 | 2 | 1.671 | 1.994 | 0.349 |

Catatan: loss latih masih menurun pada epoch terakhir
(0.575 -> 0.421), sehingga jumlah epoch menjadi kandidat utama
untuk dinaikkan di Sprint 2. Penyebab pastinya belum diuji.

## Ablation (>= 2)

Belum dikerjakan (Sprint 2). Kandidat:
1. Dengan vs tanpa positional encoding.
2. Mean pooling vs token kelas (CLS) atau jumlah encoder layer 1 vs 2.

## Hasil 3 seed konfigurasi akhir

Belum dikerjakan (Sprint 2). Hasil sementara baru satu seed
pada konfigurasi awal:

| Seed | MAE val (mm) | RMSE val (mm) | R2 val |
|------|-------|-------|-------|
| 1 | 1.671 | 1.994 | 0.349 |

Pembanding baseline MLP (seed 1): MAE 0.845 mm, RMSE 1.050 mm, R2 0.820.
Data: sesi meja 20261116-095756, ROI Wadah B 224x224
(latih 253 / validasi 45 / uji 75; uji belum dievaluasi).
