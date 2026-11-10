
# Catatan Model RNN - Enggli Rahmadhani

## Arsitektur Model
Menggunakan GRU (Gated Recurrent Unit) untuk memproses citra wadah berukuran 224x224 sebagai urutan 224 timestep.

Setiap timestep merepresentasikan satu baris citra dengan 672 fitur (224 x 3 kanal RGB).

Hidden size: 48
Output: 1 nilai kontinu (tinggi air dalam mm).

## Alasan Optimizer
Menggunakan AdamW dengan learning rate 0.001 dan weight decay 0.0001 karena optimizer ini menggunakan adaptasi learning rate serta regularisasi bobot.

## Regularisasi
- Dropout: 0.1
- Weight decay: 0.0001
- Gradient clipping: 1.0

## Pengaturan Training
- Batch size: 16
- Epoch: 5
- Loss function: Mean Squared Error (MSE)
- Seed: 1

## Pengujian Data Simulasi
Program berhasil dijalankan sampai akhir menggunakan data simulasi.

RMSE: 121.133204
MAE: 109.660160
R2: -4.541463
Durasi: 9.2 detik

Catatan: Hasil ini hanya untuk menguji alur program dan bukan hasil eksperimen data primer.

## Pencarian Hyperparameter
Belum dilakukan. Pengaturan saat ini merupakan konfigurasi awal untuk smoke test.

## Ablation
Belum dilakukan. Direncanakan pada tahap eksperimen berikutnya.

## Hasil 3 Seed
Belum dilakukan. Menunggu data primer dan konfigurasi akhir.
