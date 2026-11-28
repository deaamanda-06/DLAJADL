# Catatan model 123450006

Model `TinggiAirCNN` dibangun dari nol menggunakan PyTorch untuk menaksir tinggi permukaan air (mm) di wadah B dari citra ROI kamera meja berukuran `224 × 224` (`(N, 3, 224, 224)`). Ekstraktor fitur menggunakan 4 blok konvolusi (`3 -> 16 -> 32 -> 64 -> 128` kanal, *kernel* `3×3`, `padding=1`) yang masing-masing diikuti `BatchNorm2d`, `ReLU`, dan `MaxPool2d(2)` (pada 3 blok awal), lalu diringkas memakai `AdaptiveAvgPool2d((14, 1))` agar resolusi spasial vertikal posisi batas air tetap terjaga sebelum masuk ke kepala regresi `Linear(1792, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 1)` dengan fungsi kerugian `L1Loss` (MAE).

## Alasan optimizer

Menggunakan **Adam** (`torch.optim.Adam`) dengan *learning rate* awal `1e-3` karena:
- Melakukan adaptasi *learning rate* per parameter secara otomatis sehingga mempercepat konvergensi pada dataset eksperimen berukuran menengah (`253` sampel latih, `25` *epoch*).
- Lebih stabil dalam meminimalkan fungsi kerugian `nn.L1Loss()` untuk tugas regresi kontinu tanpa memerlukan penjadwalan *momentum* manual seperti pada SGD murni.

## Regularisasi

Beberapa mekanisme diterapkan untuk mencegah *overfitting* pada data latih:
- **Batch Normalization (`BatchNorm2d`):** Ditempatkan setelah setiap lapisan `Conv2d` untuk menormalkan distribusi aktivasi antar-batch dan menjaga kestabilan gradien.
- **Weight Decay (`1e-4`):** Penalti regularisasi L2 pada optimizer Adam untuk membatasi pembesaran bobot model.
- **Dropout (`p = 0.2`):** Menonaktifkan 20% neuron secara acak pada lapisan tersembunyi `Linear(1792, 64)` selama fase pelatihan agar model tidak bergantung pada fitur piksel lokal tertentu.
- **Best Checkpoint Selection:** Bobot model yang disimpan dan dievaluasi adalah *state* dengan nilai `val_mae` terendah selama 25 *epoch*, bukan bobot *epoch* terakhir.

## Pencarian hyperparameter (tabel)

Seluruh percobaan dievaluasi pada data validasi resmi tim (`n_val = 45`, `data_hash: 3988fb57c56bfc12`):

| Eksperimen | Learning Rate | Batch Size | Dropout | Weight Decay | Epoch | Val MAE (mm) | Val RMSE (mm) | Val R² | Keterangan |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Konfigurasi Sprint 1 (Seed 1) | `1e-3` | `16` | `0.2` | `1e-4` | `25` | `0.846` | `1.308` | `0.720` | Baseline CNN utama (`commit eeb3364`) |
| Uji LR Lebih Rendah | `5e-4` | `16` | `0.2` | `1e-4` | `30` | *isi hasil* | *isi hasil* | *isi hasil* | Menguji kehalusan konvergensi |
| Uji Dropout & Decay Lebih Kuat | `1e-3` | `16` | `0.3` | `5e-4` | `25` | *isi hasil* | *isi hasil* | *isi hasil* | Menguji pengurangan varians validasi |
| Uji Batch Size Lebih Kecil | `1e-3` | `8` | `0.2` | `1e-4` | `25` | *isi hasil* | *isi hasil* | *isi hasil* | Pembaruan gradien lebih sering |

## Ablation (>= 2)

1. **Ablasi 1 — Mengganti `AdaptiveAvgPool2d((14, 1))` menjadi `AdaptiveAvgPool2d((1, 1))` (Global Average Pooling):**
   - **Tujuan:** Menguji pengaruh pelestarian struktur spasial vertikal terhadap akurasi taksiran tinggi air.
   - **Hipotesis & Hasil:** Merata-ratakan seluruh peta fitur menjadi vektor `1×1` menghilangkan informasi koordinat vertikal meniskus/permukaan air di dalam wadah B, sehingga galat validasi (`Val MAE`) meningkat dibandingkan mempertahankan 14 bin vertikal (`14×1`). *(Isi angka `Val MAE`, `Val RMSE`, dan `Val R²` setelah dijalankan).*
2. **Ablasi 2 — Menghilangkan Regularisasi (`Dropout(0.0)` dan `weight_decay=0`):**
   - **Tujuan:** Mengukur dampak kombinasi `Dropout(0.2)` dan `weight_decay=1e-4` terhadap generalisasi pada `45` citra validasi.
   - **Hipotesis & Hasil:** Tanpa dropout dan peluruhan bobot, selisih antara galat latih dan galat validasi melebar akibat *overfitting* pada pencahayaan/latar citra meja. *(Isi angka `Val MAE`, `Val RMSE`, dan `Val R²` setelah dijalankan).*

## Hasil 3 seed konfigurasi akhir

Evaluasi konsistensi konfigurasi terbaik pada 3 *random seed* berbeda (`data_hash: 3988fb57c56bfc12`, `n_val = 45`):

| Seed | Val MAE (mm) | Val RMSE (mm) | Val R² | Durasi (detik) |
| :---: | :---: | :---: | :---: | :---: |
| Seed 1 | `0.846` | `1.308` | `0.720` | `437.1` |
| Seed 2 | *isi setelah run* | *isi setelah run* | *isi setelah run* | *isi setelah run* |
| Seed 3 | *isi setelah run* | *isi setelah run* | *isi setelah run* | *isi setelah run* |
| **Rata-rata ± Std** | *mean ± std* | *mean ± std* | *mean ± std* | - |