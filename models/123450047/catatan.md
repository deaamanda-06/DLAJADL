
# Catatan Model Transformer

## Identitas
- NIM: 123450047
- Model: Transformer Encoder
- Jalur: trf
- Tugas: Regresi tinggi air

## Arsitektur
- Input: Citra wadah
- Patch embedding: Conv2D
- Positional encoding: Sinusoidal
- Embedding dimension: 64
- Attention heads: 4
- Encoder layers: 2
- Feed-forward dimension: 128
- Regression head: 64 -> 32 -> 1

## Konfigurasi Training
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 16
- Epoch: 5
- Loss: Mean Squared Error
- Dropout: 0.1
- Seed: 1

## Alasan Pemilihan
AdamW digunakan untuk mengoptimalkan parameter model
dengan regularisasi weight decay.

MSE digunakan karena tugas merupakan regresi yang
menghasilkan prediksi tinggi air dalam satuan mm.

Transformer Encoder digunakan untuk mempelajari
hubungan antarpatch pada citra wadah.

## Pengujian
Data yang digunakan adalah data simulasi untuk
memastikan pipeline dapat berjalan.

Ukuran simulasi: 8 x 8
Ukuran data asli: 224 x 224

Hasil simulasi bukan hasil eksperimen data primer.

## Pengembangan Berikutnya
- Pengujian pada data citra asli.
- Eksperimen hyperparameter.
- Pengujian 3 seed.
- Perbandingan dengan CNN, TL, dan RNN/GRU.
