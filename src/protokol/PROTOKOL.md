# Protokol evaluasi Tim 02 — Ruang Cerdas 5 Volt

## Identitas dan kasus

| Butir | Isi |
|---|---|
| Kasus | Taksiran tinggi air di wadah B dari citra (regresi mm) |
| Sumber dataset | Citra kamera meja; data simulasi hanya untuk menguji pipeline |
| Jenis tugas (`TUGAS`) | Regresi |
| Target / label | `tinggi_air_mm` |
| Ukuran data latih / validasi / uji | Mengikuti jumlah blok yang tersedia; dilaporkan setelah data meja terkumpul |
| Cara split | Berdasarkan `block_id` (satu blok = satu siklus pengisian); seed 2026 untuk pengacakan urutan blok |
| Metrik utama untuk komparasi | MAE (mm) |
| Praproses yang sama untuk semua model | Buang 1 detik pertama setiap kondisi; crop ROI tetap wadah B; resize 224×224; normalisasi kanal dihitung dari data latih saja |
| Resolusi / interval citra | 1280×720 / 0,5 detik |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop, kecuali disetujui pembimbing |
| Evaluasi data uji | Data uji dipisah per blok/siklus pengisian dan dievaluasi sekali pada sprint terakhir |
| `data_hash` saat dibekukan | Diisi setelah data meja terkumpul dan manifest dibekukan |

## Aturan pengolahan dan pemisahan

1. Buang citra yang diambil pada 1 detik pertama sejak awal setiap kondisi.
2. Seluruh citra dari satu `block_id` harus berada pada satu subset saja: latih, validasi, atau uji.
3. Pembagian latih/validasi/uji dilakukan berdasarkan blok siklus pengisian, bukan secara acak per citra.
4. Statistik normalisasi (rata-rata dan simpangan baku kanal) dihitung hanya dari citra latih, lalu diterapkan tanpa dihitung ulang pada validasi dan uji.
5. Bentuk masukan pemuat data adalah `N × 3 × 224 × 224`; target berupa `float32` dalam mm.
6. Metrik utama adalah MAE dalam mm. Jangan menggunakan data simulasi untuk melaporkan performa model eksperimen.

## Label

Target `tinggi_air_mm` berasal dari pembacaan manual penggaris milimeter wadah B oleh dua orang setelah riak air reda (sekitar 2 detik). Catat waktu pengukuran dan rentang citra yang benar-benar sesuai dengan kondisi air diam.

## Persetujuan

Diajukan ke pembimbing melalui *pull request*. Protokol dibekukan setelah disetujui pembimbing; perubahan selanjutnya harus melalui *pull request* baru.

Disetujui pembimbing pada: … (tautan *pull request*)
