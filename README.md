# Proyek Akhir Aljabar Linear - Kelompok 7

## Judul Proyek

**Aplikasi Perbandingan Wajah Menggunakan PCA untuk Kompresi Gambar dan Deteksi Kemiripan Wajah Berbasis Python**

---

## Identitas Kelompok

| Keterangan     | Isi                         |
| -------------- | --------------------------- |
| Mata Kuliah    | Aljabar Linear              |
| Kelompok       | Kelompok 7                  |
| Rombel         | Rombel 1                    |
| Program Studi  | Teknik Informatika          |
| Dosen Pengampu | Dr. Alamsyah, S.Si., M.Kom. |

---

## Anggota Kelompok

| No | Nama Anggota         | NIM        |
| -- | -------------------- | ---------- |
| 1  | Farhan Baihaqi       | 2504130082 |
| 2  | Daniel Kristiawan    | 2504130003 |
| 3  | A Fakhruddin Ar Rozi | 2504130081 |
| 4  | Abu Aqila Bani M     | 2504130143 |

---

## Deskripsi Proyek

Proyek ini merupakan aplikasi berbasis Python dan Streamlit yang digunakan untuk membandingkan dua gambar wajah. Aplikasi ini menerapkan konsep Aljabar Linear pada pengolahan citra digital menggunakan metode **PCA (Principal Component Analysis)**.

Secara umum, aplikasi akan membaca dua gambar wajah yang diunggah pengguna, mendeteksi area wajah menggunakan **Haar Cascade**, mengubah wajah menjadi grayscale, menyamakan ukuran wajah menjadi **200 x 200 piksel**, lalu memproses wajah tersebut menggunakan **PCA**.

PCA digunakan untuk dua tujuan utama, yaitu:

1. **Kompresi gambar wajah**, yaitu mengurangi representasi data gambar dengan mempertahankan komponen utama yang paling penting.
2. **Deteksi kemiripan wajah**, yaitu membandingkan dua wajah berdasarkan jarak fitur setelah diproyeksikan ke ruang komponen utama PCA.

Hasil akhir aplikasi berupa nilai **jarak PCA**, skor kemiripan, kategori kemiripan, kesimpulan apakah kedua gambar kemungkinan berasal dari orang yang sama atau berbeda, serta laporan otomatis yang dapat diunduh.

---

## Latar Belakang

Dalam Aljabar Linear, gambar digital dapat direpresentasikan sebagai matriks. Pada gambar grayscale, setiap piksel memiliki nilai intensitas dari 0 sampai 255. Nilai-nilai piksel tersebut membentuk matriks dua dimensi yang dapat diolah menggunakan operasi matematika.

Konsep ini menjadi dasar dalam pengolahan citra digital. Operasi seperti grayscale, resize, normalisasi, ekstraksi fitur, dan kompresi gambar dapat dipahami sebagai proses manipulasi matriks dan vektor.

Pada proyek ini, metode **Principal Component Analysis (PCA)** digunakan karena PCA merupakan salah satu penerapan penting Aljabar Linear dalam Machine Learning. PCA bekerja dengan mencari komponen utama dari data, yaitu arah yang menyimpan variasi data paling besar. Dalam konteks gambar wajah, PCA dapat digunakan untuk mereduksi dimensi data wajah dan membandingkan ciri utama dari dua gambar wajah.

---

## Tujuan Proyek

Tujuan dari proyek ini adalah:

1. Menerapkan konsep Aljabar Linear dalam pengolahan citra digital.
2. Memahami bahwa gambar digital dapat direpresentasikan sebagai matriks piksel.
3. Mengimplementasikan PCA sebagai metode reduksi dimensi dan ekstraksi fitur.
4. Menggunakan PCA untuk kompresi gambar wajah.
5. Menggunakan PCA untuk mendeteksi tingkat kemiripan dua wajah.
6. Membuat aplikasi berbasis web menggunakan Streamlit.
7. Menampilkan hasil perbandingan wajah dalam bentuk jarak PCA, skor kemiripan, kategori, dan kesimpulan.
8. Menampilkan laporan hasil perbandingan secara otomatis.

---

## Fitur Program

Program ini memiliki beberapa fitur utama, yaitu:

* Upload dua file gambar wajah.
* Menampilkan preview gambar yang dipilih.
* Mendeteksi wajah pada masing-masing gambar menggunakan Haar Cascade.
* Memberi kotak hijau pada wajah yang terdeteksi.
* Menghitung jumlah wajah yang terdeteksi pada setiap gambar.
* Memilih wajah terbesar sebagai wajah utama jika terdapat lebih dari satu wajah.
* Mengubah wajah menjadi grayscale.
* Menyamakan ukuran wajah menjadi 200 x 200 piksel.
* Melakukan histogram equalization agar pencahayaan wajah lebih seimbang.
* Mengubah citra wajah menjadi vektor fitur.
* Menggunakan PCA untuk ekstraksi fitur utama wajah.
* Menggunakan PCA untuk kompresi gambar wajah.
* Menghitung jarak PCA antara dua wajah.
* Menghitung skor kemiripan wajah.
* Menentukan kategori kemiripan.
* Menampilkan kesimpulan kemungkinan orang yang sama atau berbeda.
* Menampilkan laporan hasil perbandingan pada tab laporan.
* Menyediakan tombol download laporan dalam format TXT.

---

## Metode yang Digunakan

### 1. Haar Cascade Face Detection

Haar Cascade digunakan untuk mendeteksi area wajah pada gambar. Program menggunakan file cascade bawaan dari OpenCV, yaitu:

```python
haarcascade_frontalface_default.xml
```

Jika wajah berhasil terdeteksi, program akan mengambil area wajah tersebut untuk diproses lebih lanjut. Jika terdapat lebih dari satu wajah, program akan memilih wajah terbesar sebagai wajah utama.

---

### 2. Grayscale

Gambar berwarna diubah menjadi grayscale agar pemrosesan menjadi lebih sederhana. Dalam grayscale, gambar hanya memiliki satu kanal warna yang berisi nilai intensitas piksel.

Contoh proses pada program:

```python
abu_abu = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2GRAY)
```

---

### 3. Resize Gambar Wajah

Setelah wajah dipotong, ukuran wajah disamakan menjadi:

```text
200 x 200 piksel
```

Tujuannya agar kedua wajah memiliki ukuran matriks yang sama sebelum diproses menggunakan PCA.

---

### 4. Histogram Equalization

Histogram equalization digunakan untuk menyeimbangkan pencahayaan pada gambar wajah. Proses ini membantu agar perbedaan pencahayaan tidak terlalu memengaruhi proses ekstraksi fitur.

Contoh proses pada program:

```python
potongan = cv2.equalizeHist(potongan)
```

---

### 5. Principal Component Analysis (PCA)

PCA atau **Principal Component Analysis** adalah metode dalam Aljabar Linear dan Machine Learning yang digunakan untuk mereduksi dimensi data dengan mempertahankan informasi utama.

Pada proyek ini, citra wajah grayscale berukuran 200 x 200 piksel diubah menjadi vektor berdimensi 40.000. Setelah itu, data wajah dipusatkan dengan mengurangi nilai rata-rata, lalu PCA dihitung menggunakan **SVD (Singular Value Decomposition)**.

Contoh konsep proses PCA pada program:

```python
data_terpusat = data - rata_rata
u, singular_values, vt = np.linalg.svd(data_terpusat, full_matrices=False)
komponen_utama = vt[:jumlah_komponen]
fitur_pca = data_terpusat @ komponen_utama.T
```

Hasil proyeksi PCA kemudian digunakan untuk menghitung jarak antar dua wajah. Semakin kecil jarak PCA, maka kedua wajah dianggap semakin mirip. Sebaliknya, semakin besar jarak PCA, maka kedua wajah dianggap semakin berbeda.

---

## Implementasi Aljabar Linear

Proyek ini berhubungan langsung dengan Aljabar Linear karena seluruh gambar wajah diproses sebagai data numerik berupa matriks dan vektor.

Beberapa konsep Aljabar Linear yang digunakan dalam proyek ini adalah:

1. **Matriks Gambar**
   Gambar grayscale direpresentasikan sebagai matriks dua dimensi yang berisi nilai intensitas piksel.

2. **Vektor Fitur**
   Matriks wajah berukuran 200 x 200 piksel diubah menjadi vektor berdimensi 40.000.

3. **Mean Centering**
   Data wajah dikurangi nilai rata-rata agar data berada di sekitar pusat.

4. **PCA**
   PCA digunakan untuk mencari komponen utama dari data wajah.

5. **SVD**
   Singular Value Decomposition digunakan untuk menghitung komponen utama secara numerik.

6. **Proyeksi Vektor**
   Data wajah diproyeksikan ke ruang komponen utama.

7. **Jarak Euclidean**
   Jarak antara dua fitur wajah dihitung untuk menentukan tingkat kemiripan.

---

## Alur Kerja Program

Alur kerja aplikasi adalah sebagai berikut:

1. Pengguna membuka aplikasi.
2. Pengguna mengunggah Gambar 1.
3. Pengguna mengunggah Gambar 2.
4. Program membaca kedua gambar.
5. Program mengubah gambar menjadi grayscale.
6. Program mendeteksi wajah menggunakan Haar Cascade.
7. Program memilih wajah terbesar sebagai wajah utama.
8. Program memotong area wajah.
9. Program mengubah ukuran wajah menjadi 200 x 200 piksel.
10. Program melakukan histogram equalization.
11. Program mengubah matriks wajah menjadi vektor.
12. Program menghitung PCA menggunakan SVD.
13. Program memproyeksikan wajah ke ruang komponen utama.
14. Program menghitung jarak PCA antara dua wajah.
15. Program menghitung skor kemiripan.
16. Program menentukan kategori kemiripan.
17. Program melakukan kompresi gambar menggunakan PCA.
18. Program menampilkan hasil dan laporan.

---

## PCA untuk Kompresi Gambar

Selain untuk membandingkan kemiripan wajah, PCA juga digunakan untuk kompresi gambar wajah.

Pada proses kompresi, citra grayscale dianggap sebagai matriks. Program kemudian menghitung PCA menggunakan SVD dan hanya mempertahankan beberapa komponen utama. Dengan cara ini, gambar dapat direkonstruksi menggunakan jumlah komponen yang lebih sedikit dibandingkan data asli.

Pada program ini digunakan:

```python
KOMPONEN_PCA_KOMPRESI = 40
```

Artinya, gambar wajah direkonstruksi menggunakan 40 komponen utama. Program juga menghitung beberapa informasi kompresi, yaitu:

* jumlah komponen PCA yang digunakan,
* rasio kompresi,
* variansi yang dipertahankan,
* nilai MSE rekonstruksi,
* nilai PSNR rekonstruksi.

Semakin besar variansi yang dipertahankan, semakin banyak informasi gambar yang masih tersimpan setelah proses kompresi.

---

## PCA untuk Deteksi Kemiripan Wajah

Untuk mendeteksi kemiripan wajah, program membandingkan dua wajah setelah diproyeksikan ke ruang PCA.

Tahapannya adalah:

1. Wajah pertama dan wajah kedua diubah menjadi vektor.
2. Kedua vektor digabung menjadi matriks data.
3. Data dikurangi rata-rata.
4. PCA dihitung menggunakan SVD.
5. Kedua wajah diproyeksikan ke komponen utama.
6. Program menghitung jarak Euclidean antar fitur PCA.
7. Jarak tersebut dinormalisasi menjadi skala 0 sampai 100.

Semakin kecil nilai jarak PCA, semakin mirip wajahnya. Semakin besar nilai jarak PCA, semakin berbeda wajahnya.

---

## Ketentuan Threshold

Pada program ini digunakan batas threshold sebagai berikut:

```python
BATAS_BEDA = 35
```

Ketentuannya:

| Nilai Jarak PCA | Kategori                |
| --------------- | ----------------------- |
| ≤ 20            | Kemiripan Tinggi        |
| ≤ 35            | Kemiripan Sedang        |
| ≤ 50            | Kemiripan Rendah        |
| > 50            | Kemiripan Sangat Rendah |

Jika nilai jarak PCA lebih kecil atau sama dengan 35, maka program menyimpulkan bahwa kedua wajah kemungkinan berasal dari orang yang sama.

Jika nilai jarak PCA lebih besar dari 35, maka program menyimpulkan bahwa kedua wajah kemungkinan berasal dari orang yang berbeda.

---

## Rumus Skor Kemiripan

Program menghitung skor kemiripan dengan rumus:

```python
skor_mirip = max(0.0, min(1.0, 1 - jarak_pca / 100))
```

Keterangan:

* Nilai skor berada pada rentang 0 sampai 1.
* Semakin mendekati 1, maka wajah dianggap semakin mirip.
* Semakin mendekati 0, maka wajah dianggap semakin berbeda.
* Nilai jarak PCA yang kecil menghasilkan skor kemiripan yang lebih tinggi.

---

## Tampilan Aplikasi

Aplikasi memiliki tiga tab utama:

### 1. Tab Input Gambar

Tab ini digunakan untuk memilih dua gambar wajah yang akan dibandingkan.

<img width="1600" height="778" alt="input" src="https://github.com/user-attachments/assets/40e26d62-3c16-49c7-81f6-15a93328f127" />


Fitur pada tab ini:

* Upload file untuk Gambar 1.
* Upload file untuk Gambar 2.
* Preview gambar yang dipilih.
* Tombol mulai bandingkan.

---

### 2. Tab Hasil Perbandingan

Tab ini digunakan untuk menampilkan hasil visual dari proses deteksi wajah.

<img width="1600" height="810" alt="hasil" src="https://github.com/user-attachments/assets/1ddd12cd-d098-4ef7-9c72-0bd0abc4dda3" />


Fitur pada tab ini:

* Menampilkan Gambar 1 dengan kotak wajah.
* Menampilkan Gambar 2 dengan kotak wajah.
* Menampilkan kesimpulan hasil perbandingan.
* Menampilkan nilai jarak PCA.
* Menampilkan skor kemiripan.
* Menampilkan kategori kemiripan.
* Menampilkan waktu proses.

---

### 3. Tab Laporan

Tab ini digunakan untuk menampilkan laporan lengkap hasil perbandingan.

<img width="1117" height="737" alt="laporan" src="https://github.com/user-attachments/assets/9a552651-c5cd-4d50-aba9-a9fa6a456a42" />


Isi laporan meliputi:

* Nama file gambar.
* Resolusi gambar.
* Ukuran file gambar.
* Jumlah wajah yang terdeteksi.
* Waktu proses.
* Tahapan preprocessing.
* Metode yang digunakan.
* Implementasi Aljabar Linear pada PCA.
* Hasil kompresi gambar menggunakan PCA.
* Threshold jarak PCA.
* Nilai jarak PCA.
* Skor kemiripan.
* Kategori kemiripan.
* Interpretasi hasil.
* Kesimpulan akhir.

---

## Library yang Digunakan

Program ini menggunakan beberapa library Python, yaitu:

| Library   | Fungsi                                                          |
| --------- | --------------------------------------------------------------- |
| Streamlit | Membuat tampilan aplikasi web                                   |
| OpenCV    | Membaca gambar dan mendeteksi wajah menggunakan Haar Cascade    |
| NumPy     | Mengolah data numerik, matriks, vektor, dan PCA menggunakan SVD |
| Pillow    | Membaca dan menyesuaikan format gambar                          |
| Time      | Menghitung waktu proses perbandingan                            |

---

## Instalasi Library

Sebelum menjalankan program, pastikan Python sudah terpasang. Setelah itu install library yang dibutuhkan dengan perintah:

```bash
pip install streamlit opencv-python-headless numpy pillow
```

Atau install menggunakan file `requirements.txt`:

```bash
pip install -r requirements.txt
```

Isi file `requirements.txt`:

```txt
streamlit
opencv-python-headless
numpy
pillow
```

---

## Cara Menjalankan Program

1. Pastikan semua library sudah terinstall.
2. Buka folder proyek.
3. Jalankan aplikasi dengan perintah:

```bash
streamlit run app.py
```

4. Upload Gambar 1.
5. Upload Gambar 2.
6. Klik tombol **Mulai Bandingkan**.
7. Lihat hasil pada tab **Hasil Perbandingan**.
8. Lihat laporan lengkap pada tab **Laporan**.
9. Download laporan jika diperlukan.

---


## Struktur Program

Beberapa bagian penting dalam program:

| Bagian Program               | Fungsi                                                                           |
| ---------------------------- | -------------------------------------------------------------------------------- |
| `format_ukuran_file()`       | Mengubah ukuran file menjadi byte, KB, atau MB                                   |
| `kategori_kemiripan()`       | Menentukan kategori kemiripan berdasarkan jarak PCA                              |
| `baca_gambar_upload()`       | Membaca file upload dari Streamlit                                               |
| `gambar_bgr_ke_rgb()`        | Mengubah format warna BGR menjadi RGB                                            |
| `deteksi_dan_potong_wajah()` | Membaca gambar, mendeteksi wajah, memotong wajah, resize, dan equalize histogram |
| `pca_similarity()`           | Menghitung kemiripan wajah menggunakan PCA                                       |
| `kompresi_gambar_pca()`      | Melakukan kompresi gambar wajah menggunakan PCA                                  |
| `bandingkan_wajah()`         | Menjalankan seluruh proses perbandingan wajah                                    |
| `buat_teks_laporan()`        | Membuat laporan hasil perbandingan                                               |
| `st.tabs()`                  | Membuat tab Input, Hasil, dan Laporan pada Streamlit                             |
| `st.file_uploader()`         | Mengunggah gambar dari pengguna                                                  |
| `st.download_button()`       | Mengunduh laporan hasil perbandingan                                             |

---

## Contoh Output Program

Contoh output yang ditampilkan pada laporan:

```text
========== HASIL PERBANDINGAN WAJAH ==========
Nama File 1    : gambar1.jpg
Nama File 2    : gambar2.jpg
Jarak PCA      : 24.5321
Skor Mirip     : 0.7547
Kategori       : Kemiripan Sedang
Threshold PCA  : 35
Waktu Proses   : 0.2134 detik
Kesimpulan     : kemungkinan ORANG YANG SAMA
=============================================
```

---

## Kelebihan Program

Beberapa kelebihan program ini adalah:

1. Berbasis web sehingga mudah dijalankan melalui browser.
2. Tampilan sederhana dan mudah digunakan.
3. Dapat mengunggah dua gambar langsung dari aplikasi.
4. Menampilkan preview gambar.
5. Menampilkan wajah yang terdeteksi dengan kotak hijau.
6. Menggunakan PCA yang berkaitan langsung dengan Aljabar Linear.
7. Dapat digunakan untuk kompresi gambar wajah.
8. Dapat menghitung tingkat kemiripan wajah berdasarkan jarak PCA.
9. Menampilkan laporan hasil perbandingan secara otomatis.
10. Laporan dapat diunduh dalam format TXT.

---

## Keterbatasan Program

Program ini masih memiliki beberapa keterbatasan, yaitu:

1. Hasil perbandingan dapat dipengaruhi oleh pencahayaan gambar.
2. Posisi wajah yang terlalu miring dapat memengaruhi hasil deteksi.
3. Ekspresi wajah yang berbeda dapat memengaruhi nilai jarak PCA.
4. Jika wajah tidak menghadap kamera, deteksi bisa gagal.
5. Program hanya memilih wajah terbesar jika terdapat lebih dari satu wajah.
6. Threshold masih ditentukan secara manual.
7. PCA yang digunakan pada perbandingan dua gambar masih sederhana karena tidak menggunakan dataset wajah yang besar.
8. Program belum menggunakan model deep learning.
9. Hasil program sebaiknya digunakan sebagai simulasi pembelajaran, bukan sebagai sistem identifikasi biometrik resmi.

---

## Kesimpulan

Proyek ini berhasil menerapkan konsep Aljabar Linear dalam bentuk aplikasi perbandingan wajah berbasis Python dan Streamlit. Gambar wajah diproses sebagai matriks piksel, kemudian melalui tahapan grayscale, deteksi wajah, pemotongan area wajah, resizing, histogram equalization, dan ekstraksi fitur menggunakan PCA.

PCA digunakan sebagai metode utama untuk mereduksi dimensi data wajah, melakukan kompresi gambar, dan membandingkan kemiripan dua wajah berdasarkan jarak fitur pada ruang komponen utama. Melalui proyek ini, dapat dipahami bahwa konsep matriks, vektor, proyeksi, SVD, dan reduksi dimensi dalam Aljabar Linear memiliki penerapan nyata dalam bidang pengolahan citra digital dan Machine Learning.

---

## Repository

farhanbaihaqi20/PROJECT-AKHIR-ALJABAR-LINEAR
