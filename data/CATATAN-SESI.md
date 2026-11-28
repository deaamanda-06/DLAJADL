# CATATAN SESI DATA PRIMER TIM 02

## 1. Identitas sesi

- Tim: 02.
- Kasus: Taksiran tinggi air di wadah dari citra.
- Pembimbing: Ade Lailani.
- Tanggal sesi: 16 November 2026.
- Arsip sumber: `tim02_sesi-20261116-095756.tar.gz`.
- Target regresi: tinggi air pada wadah B.
- Kolom label sumber: `tinggi_air_mm`.
- Satuan label sumber: sentimeter (cm).
- Satuan target evaluasi: milimeter (mm).
- Rencana eksperimen: 3 ulangan blok × 25 kondisi.
- Jumlah kondisi aktual: 75.
- Selisih terhadap rencana: 0 kondisi.
- Tanggal pemeriksaan: 28 November 2026.
- Pemeriksa: Enggli Rahmadhani (123450043).

## 2. Pemeriksaan struktur data

Berdasarkan pemeriksaan `data/label.csv`, terdapat 75 kondisi tanpa nomor kondisi duplikat.

Kolom `blok` memiliki satu nilai unik, sedangkan kolom `ulangan_blok` memiliki tiga nilai unik dengan masing-masing 25 kondisi. Struktur tersebut sesuai dengan konfigurasi `data/rencana.yaml`.

Oleh karena itu, pengelompokan blok menggunakan kolom `ulangan_blok`, sedangkan urutan pengambilan kondisi menggunakan kolom `no`.

## 3. Pemeriksaan mutu label per blok

| Pemeriksaan | Blok 1 | Blok 2 | Blok 3 | Total |
|---|---:|---:|---:|---:|
| Target kondisi | 25 | 25 | 25 | 75 |
| Kondisi aktual | 25 | 25 | 25 | 75 |
| Label kosong | 0 | 0 | 0 | 0 |
| Label tidak valid | 0 | 0 | 0 | 0 |
| Nomor duplikat | 0 | 0 | 0 | 0 |
| Desimal koma | 2 | 0 | 0 | 2 |
| Minimum (cm) | 0 | 0 | 0 | 0 |
| Maksimum (cm) | 8,4 | 9,8 | 9,4 | 9,8 |
| Minimum (mm) | 0 | 0 | 0 | 0 |
| Maksimum (mm) | 84 | 98 | 94 | 98 |
| Jumlah penurunan | 4 | 6 | 4 | 14 |
| Monoton nonmenurun | Tidak | Tidak | Tidak | Tidak |

Jumlah kondisi dan kelengkapan label telah sesuai dengan rencana eksperimen. Namun, persyaratan monotonisitas belum terpenuhi karena terdapat 14 penurunan tinggi air dalam blok.

## 4. Rincian penurunan tinggi air

| Blok | Kondisi | Sebelum (cm) | Sesudah (cm) |
|---|---|---:|---:|
| 1 | 5 → 6 | 6,9 | 0 |
| 1 | 11 → 12 | 8,0 | 2,6 |
| 1 | 16 → 17 | 8,4 | 2,5 |
| 1 | 22 → 23 | 8,3 | 2,5 |
| 2 | 26 → 27 | 2,4 | 0 |
| 2 | 32 → 33 | 8,0 | 1,0 |
| 2 | 36 → 37 | 7,8 | 5,6 |
| 2 | 40 → 41 | 9,2 | 2,3 |
| 2 | 45 → 46 | 9,7 | 2,4 |
| 2 | 49 → 50 | 9,8 | 2,5 |
| 3 | 54 → 55 | 8,7 | 2,2 |
| 3 | 59 → 60 | 8,9 | 0 |
| 3 | 66 → 67 | 9,4 | 1,2 |
| 3 | 70 → 71 | 8,0 | 2,4 |

Berdasarkan keterangan mengenai pelaksanaan sesi, air dikembalikan dari wadah B ke wadah A beberapa kali ketika wadah mendekati kapasitas maksimum.

Pengosongan wadah berulang menjelaskan pola kenaikan dan penurunan tinggi air dalam satu ulangan blok. Pelaksanaan tersebut berbeda dari rencana awal yang mengharuskan tinggi air meningkat secara monoton dalam setiap blok.

Penurunan label tidak langsung dikategorikan sebagai kesalahan pencatatan. Keterangan pengosongan wadah belum diverifikasi secara individual untuk setiap kejadian penurunan.

Seluruh label asli dipertahankan tanpa manipulasi atau pengurutan ulang berdasarkan tinggi air.

## 5. Penanganan format dan satuan label

Terdapat dua label dengan pemisah desimal koma:

- Kondisi 3: `4,5`.
- Kondisi 9: `5,4`.

Pemisah desimal dikonversi menjadi titik pada saat parsing numerik.

Satuan pengukuran aktual telah dikonfirmasi sebagai sentimeter, meskipun nama kolom sumber adalah `tinggi_air_mm`.

Pada preprocessing dataset acuan, nilai label dikonversi menggunakan:

`tinggi_mm = tinggi_cm × 10`

Rentang label berubah dari 0–9,8 cm menjadi 0–98 mm.

File CSV asli tidak diubah sehingga integritas sumber data tetap terjaga.

## 6. Kondisi dan frame yang dikeluarkan

Tidak terdapat kondisi eksperimen yang dikeluarkan.

| Keterangan | Jumlah |
|---|---:|
| Kondisi awal | 75 |
| Kondisi digunakan | 75 |
| Kondisi dikeluarkan | 0 |
| Frame kamera awal | 4.005 |
| Frame digunakan | 373 |
| Frame tidak dipilih | 3.632 |

Seleksi frame menggunakan rentang waktu berikut:

- Awal: `t_selesai_host + 2.0`.
- Akhir: `min(t_selesai_host + 4.5, t_catat_host - 1.0)`.
- Maksimal lima frame per kondisi.

Pembatasan waktu bertujuan menghindari frame ketika pompa menyala, permukaan air masih beriak, jeda antarkondisi, dan aktivitas pengosongan wadah.

Hasil audit menunjukkan seluruh 75 kondisi memiliki kandidat frame pada rentang waktu utama. Tidak ada kondisi yang membutuhkan rentang alternatif.

Kondisi 43 dan 48 masing-masing memiliki empat frame, sedangkan 73 kondisi lainnya memiliki lima frame.

Frame yang tidak terpilih tidak digunakan pada dataset model karena berada di luar rentang seleksi atau melebihi batas jumlah frame per kondisi.

## 7. Penyelarasan preprocessing dan pembagian dataset

Dataset sebelumnya menggunakan split acak pada tingkat kondisi dengan distribusi 253 frame training, 45 frame validation, dan 75 frame testing.

Hasil audit menunjukkan ketiga ulangan blok tersebar pada beberapa split. Pembagian tersebut belum memenuhi ketentuan pemisahan data uji berdasarkan blok.

Sebagai perbaikan, dibuat pipeline tambahan:

- `src/protokol/pemotong_blok.py`.
- `src/protokol/data_blok.py`.

Pipeline menghasilkan dataset acuan pada `data/cropped_blok_mm/dataset.npz` dengan konfigurasi:

- ROI tetap: `(820, 300, 1140, 720)`.
- Ukuran citra: 224 × 224 piksel.
- Kanal: RGB.
- Rentang piksel: 0–1.
- Satuan target: mm.
- Split berdasarkan kolom `ulangan_blok`.

| Split | Blok | Kondisi | Frame | Rentang label (mm) |
|---|---:|---:|---:|---|
| Train | 1 | 25 | 125 | 0–84 |
| Validation | 2 | 25 | 123 | 0–98 |
| Test | 3 | 25 | 125 | 0–94 |
| Total | 3 blok | 75 | 373 | 0–98 |

Pemilihan blok tersebut mengikuti konfigurasi pembagian blok pada implementasi sebelumnya dan digunakan sebagai usulan protokol acuan.

Normalisasi model GRU dihitung hanya berdasarkan data latih.

Dataset lama dan data primer asli dipertahankan.

## 8. Pemeriksaan integritas data

Algoritma pemeriksaan: SHA-256.

### 8.1 Label sumber

`2a6a959835169f1695051f21b646e9529f55cb2457a2f083af91e6cd0245c124`

Hasil pemeriksaan menunjukkan bahwa `data/label.csv` identik dengan label CSV pada folder sesi asli. Hash label yang diperiksa pada lingkungan Enggli dan Hanna juga identik.

### 8.2 Arsip sumber

`efba34a9cbc5bd24abdcf62c47718d6c4c5b7317e28326390239adab9bc1e31b`

Hash diperoleh melalui pemeriksaan arsip sumber pada lingkungan lokal.

### 8.3 Dataset preprocessing sebelumnya

Ditemukan perbedaan SHA-256 berkas NPZ pada pemeriksaan awal:

- Enggli: `48cf1ec4e3e5379ef9264b950d4cbb351105a0b532be1880c40a330751e0e844`.
- Hanna: `205638f72cbb3149eded1aebb4bad4c400a2b205756d88a26108caec5762954d`.

Perbedaan hash berkas tersebut belum membuktikan perbedaan isi array.

### 8.4 Data hash acuan baru

Data hash:

`417c70ad079cdb5a`

SHA-256 lengkap:

`417c70ad079cdb5a220d2a6ff796144aada403d4a8e8597d65e40e88e8d31e4e`

Hash dihitung berdasarkan identitas label sumber, konfigurasi ROI, satuan target, pembagian blok, hash setiap array, serta identitas dan urutan frame pada indeks dataset.

Hash tidak hanya dihitung berdasarkan seed pembagian data.

Metadata lengkap dicatat pada `data/MANIFEST.json` dan dihasilkan kembali oleh pipeline preprocessing.

## 9. Validasi dan reproduksibilitas dataset acuan

| Pemeriksaan | Hasil |
|---|---|
| Jumlah kondisi | 75 |
| Jumlah blok | 3 |
| Jumlah frame | 373 |
| Kondisi lintas split | 0 |
| Blok lintas split | 0 |
| Frame duplikat | 0 |
| Hash label sumber | MATCH |
| Hash indeks | MATCH |
| Hash seluruh array | MATCH |
| Satuan target | mm |
| Hash sebelum preprocessing ulang | 417c70ad079cdb5a |
| Hash setelah preprocessing ulang | 417c70ad079cdb5a |

Pembuatan ulang dataset menggunakan sumber dan konfigurasi yang sama menghasilkan identitas dataset yang sama pada lingkungan lokal.

Status validasi integritas dataset acuan: BERHASIL.

## 10. Pengujian penggunaan dataset acuan

Dataset acuan telah digunakan untuk pelatihan Bidirectional GRU dan pembanding non-DL berupa posisi kandidat garis air dari profil kecerahan dengan regresi linear.

Pengujian menggunakan seed 1 dengan hasil sebagai berikut.

| Metrik validasi | GRU | Baseline non-DL |
|---|---:|---:|
| MAE (mm) | 10,963981 | 24,913335 |
| RMSE (mm) | 15,235911 | 28,414566 |
| R² | 0,712466 | -0,000081 |
| Durasi (detik) | 25,3 | 1,3 |
| Jumlah frame validasi | 123 | 123 |
| Jumlah frame test dievaluasi | 0 | 0 |
| Data hash | 417c70ad079cdb5a | 417c70ad079cdb5a |

Model GRU mencapai MAE validasi terbaik pada epoch 11 dari 30 epoch.

Baseline non-DL telah berjalan, tetapi masih memiliki keterbatasan dalam mengidentifikasi garis air. Kandidat garis air pada blok validasi terkonsentrasi di bagian bawah ROI.

Hasil tersebut merupakan pengujian validasi awal, bukan hasil komparasi akhir menggunakan blok test.

Artefak pengujian:

- `results/123450043/blok_mm/metrics.json`.
- `results/baseline/blok_mm/metrics.json`.

## 11. Konsistensi data_hash

| Komponen yang diverifikasi | Data hash | Status |
|---|---|---|
| Dataset acuan lokal | 417c70ad079cdb5a | Terverifikasi |
| Preprocessing ulang lokal | 417c70ad079cdb5a | Identik |
| Bidirectional GRU | 417c70ad079cdb5a | Identik |
| Baseline non-DL | 417c70ad079cdb5a | Identik |

Dataset acuan disiapkan agar dapat direproduksi dan digunakan secara seragam oleh seluruh jalur model tim.

Kesamaan hash telah dibuktikan pada dataset dan hasil model yang tercantum dalam tabel. Verifikasi penggunaan dataset acuan pada seluruh jalur anggota belum dinyatakan lengkap.

## 12. Reproduksibilitas pemeriksaan

Pemeriksaan mutu label:

`python models/123450043/cek_mutu_data.py`

Pembuatan dataset acuan:

`python -m src.protokol.pemotong_blok`

Validasi integritas dataset:

`python -m src.protokol.data_blok`

Pelatihan GRU:

`python models/123450043/run_blok.py --seed 1`

Pengujian baseline:

`python models/123450043/baseline_blok.py`

Seluruh perubahan dilakukan melalui pipeline tambahan tanpa memodifikasi file starter pada `src/common/`.

## 13. Kesimpulan

Pemeriksaan data primer menunjukkan bahwa jumlah kondisi dan ulangan blok sesuai rencana, yaitu 75 kondisi dalam tiga blok.

Tidak terdapat label kosong, label tidak valid, maupun nomor kondisi duplikat. Seluruh kondisi dipertahankan dan 373 frame dipilih berdasarkan rentang waktu air diam.

Ditemukan 14 penurunan tinggi air dalam blok dengan penjelasan berupa pengosongan wadah berulang selama pelaksanaan sesi.

Penyelarasan satuan dari cm ke mm serta pembagian dataset berdasarkan blok telah diterapkan pada pipeline acuan baru.

Validasi integritas dan reproduksibilitas dataset lokal berhasil. Pengujian GRU dan baseline menggunakan data hash yang identik.

### Status akhir pemeriksaan

- Jumlah kondisi: MEMENUHI.
- Kelengkapan label: MEMENUHI.
- Monotonisitas: TIDAK MEMENUHI RENCANA AWAL, DENGAN PENJELASAN PENGOSONGAN WADAH.
- Konversi satuan: BERHASIL.
- Pemisahan blok: BERHASIL.
- Integritas dataset acuan lokal: TERVERIFIKASI.
- Reproduksibilitas hash lokal: BERHASIL.
- Konsistensi hash GRU dan baseline: IDENTIK.
- Verifikasi hash seluruh anggota: BELUM LENGKAP.