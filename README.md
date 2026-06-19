# Proyek Akhir Aljabar Linear - Kelompok 7

## Judul Proyek

**Aplikasi Perbandingan Wajah Menggunakan Haar Cascade dan LBPH Face Recognizer Berbasis Python**

---

## Identitas Kelompok

| Keterangan     | Isi                        |
| -------------- | ---------------------------|
| Mata Kuliah    | Aljabar Linear             |
| Kelompok       | Kelompok 7                 |
| Rombel         | Rombel 1                   |
| Program Studi  | Teknik Informatika         |
| Dosen Pengampu | Dr Alamsyah S.Si,M.Kom,    |

---

## Anggota Kelompok

| No | Nama Anggota | NIM |
|---|---|---|
| 1 | Farhan Baihaqi | 2504130082 |
| 2 | Daniel Kristiawan | 2504130003 |
| 3 | A Fakhruddin Ar Rozi | 2504130081 |
| 4 | Abu Aqila Bani M | 2504130143 |


---

## Deskripsi Proyek

Proyek ini merupakan aplikasi sederhana berbasis Python yang digunakan untuk membandingkan dua gambar wajah. Aplikasi ini dibuat dengan tampilan GUI menggunakan Tkinter sehingga pengguna dapat memilih dua file gambar secara langsung dari komputer.

Program akan mendeteksi wajah pada masing-masing gambar, memotong area wajah yang terdeteksi, mengubahnya ke bentuk grayscale, menyamakan ukuran wajah menjadi 200 x 200 piksel, lalu membandingkan wajah menggunakan metode **LBPH Face Recognizer**.

Hasil perbandingan ditampilkan dalam bentuk nilai **confidence**, skor kemiripan, kategori kemiripan, dan kesimpulan apakah kedua gambar kemungkinan berasal dari orang yang sama atau berbeda.

---

## Latar Belakang

Dalam aljabar linear, gambar digital dapat direpresentasikan sebagai matriks. Setiap piksel pada gambar memiliki nilai numerik tertentu. Pada gambar grayscale, nilai piksel merepresentasikan tingkat intensitas warna dari hitam hingga putih.

Konsep matriks ini digunakan dalam proses pengolahan citra digital, seperti membaca gambar, mengubah gambar menjadi grayscale, melakukan resizing, memotong area wajah, serta membandingkan pola wajah.

Pada proyek ini, library OpenCV digunakan untuk membantu proses deteksi dan pengenalan wajah. Metode **Haar Cascade** digunakan untuk mendeteksi area wajah, sedangkan metode **LBPH Face Recognizer** digunakan untuk membandingkan pola wajah dari dua gambar.

---

## Tujuan Proyek

Tujuan dari proyek ini adalah:

1. Menerapkan konsep aljabar linear dalam pengolahan citra digital.
2. Memahami bahwa gambar digital dapat direpresentasikan dalam bentuk matriks piksel.
3. Membuat aplikasi sederhana untuk mendeteksi dan membandingkan wajah.
4. Menggunakan Python, OpenCV, NumPy, Pillow, dan Tkinter dalam implementasi program.
5. Menampilkan hasil perbandingan wajah dalam bentuk nilai confidence dan skor kemiripan.
6. Menampilkan laporan hasil perbandingan secara otomatis melalui aplikasi.

---

## Fitur Program

Program ini memiliki beberapa fitur utama, yaitu:

* Memilih dua file gambar dari komputer.
* Menampilkan preview gambar yang dipilih.
* Mendeteksi wajah pada masing-masing gambar menggunakan Haar Cascade.
* Memberi kotak hijau pada wajah yang terdeteksi.
* Menghitung jumlah wajah yang terdeteksi pada setiap gambar.
* Memilih wajah terbesar sebagai wajah utama jika terdapat lebih dari satu wajah.
* Mengubah wajah menjadi grayscale.
* Menyamakan ukuran wajah menjadi 200 x 200 piksel.
* Melakukan histogram equalization agar pencahayaan wajah lebih seimbang.
* Membandingkan wajah menggunakan metode LBPH Face Recognizer.
* Menampilkan nilai confidence LBPH.
* Menampilkan skor kemiripan.
* Menampilkan kategori kemiripan.
* Menampilkan kesimpulan kemungkinan orang yang sama atau berbeda.
* Menampilkan laporan hasil perbandingan pada tab laporan.
* Menyediakan tombol untuk menyalin laporan hasil perbandingan.

---

## Metode yang Digunakan

### 1. Haar Cascade Face Detection

Haar Cascade digunakan untuk mendeteksi area wajah pada gambar. Program menggunakan file cascade bawaan dari OpenCV, yaitu:

```python
haarcascade_frontalface_default.xml
```

Jika wajah berhasil terdeteksi, program akan mengambil area wajah tersebut untuk diproses lebih lanjut.

---

### 2. Grayscale

Gambar berwarna diubah menjadi grayscale agar pemrosesan lebih sederhana. Dalam grayscale, gambar hanya memiliki satu kanal warna yang berisi nilai intensitas piksel.

Contoh proses pada program:

```python
abu_abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
```

---

### 3. Resize Gambar Wajah

Setelah wajah dipotong, ukuran wajah disamakan menjadi:

```python
200 x 200 piksel
```

Tujuannya agar kedua wajah memiliki ukuran matriks yang sama sebelum dibandingkan.

---

### 4. Histogram Equalization

Histogram equalization digunakan untuk menyeimbangkan pencahayaan pada gambar wajah. Proses ini membantu agar perbedaan pencahayaan tidak terlalu memengaruhi hasil perbandingan.

Contoh proses pada program:

```python
potongan = cv2.equalizeHist(potongan)
```

---

### 5. LBPH Face Recognizer

LBPH atau **Local Binary Patterns Histogram** digunakan untuk membandingkan pola wajah. Pada program ini, wajah pertama digunakan sebagai data latih, lalu wajah kedua digunakan sebagai data uji.

Contoh konfigurasi LBPH pada program:

```python
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8,
)
```

Hasil dari metode LBPH berupa nilai **confidence**. Semakin kecil nilai confidence, maka wajah dianggap semakin mirip. Semakin besar nilai confidence, maka wajah dianggap semakin berbeda.

---

## Alur Kerja Program

Alur kerja aplikasi adalah sebagai berikut:

1. Pengguna membuka aplikasi.
2. Pengguna memilih Gambar 1.
3. Pengguna memilih Gambar 2.
4. Program membaca kedua gambar.
5. Program mengubah gambar menjadi grayscale.
6. Program mendeteksi wajah menggunakan Haar Cascade.
7. Program memilih wajah terbesar sebagai wajah utama.
8. Program memotong area wajah.
9. Program mengubah ukuran wajah menjadi 200 x 200 piksel.
10. Program melakukan histogram equalization.
11. Program melatih LBPH menggunakan wajah pertama.
12. Program menguji wajah kedua menggunakan model LBPH.
13. Program menghasilkan nilai confidence.
14. Program menghitung skor kemiripan.
15. Program menentukan kategori kemiripan.
16. Program menampilkan hasil dan laporan.

---

## Ketentuan Threshold

Pada program ini digunakan batas threshold sebagai berikut:

```python
BATAS_BEDA = 70
```

Ketentuannya:

| Nilai Confidence | Kategori                |
| ---------------- | ----------------------- |
| ≤ 50             | Kemiripan Tinggi        |
| ≤ 70             | Kemiripan Sedang        |
| ≤ 90             | Kemiripan Rendah        |
| > 90             | Kemiripan Sangat Rendah |

Jika nilai confidence lebih kecil atau sama dengan 70, maka program menyimpulkan bahwa kedua wajah kemungkinan berasal dari orang yang sama.

Jika nilai confidence lebih besar dari 70, maka program menyimpulkan bahwa kedua wajah kemungkinan berasal dari orang yang berbeda.

---

## Rumus Skor Kemiripan

Program menghitung skor kemiripan dengan rumus:

```python
skor_mirip = max(0, 1 - confidence / 100)
```

Keterangan:

* Nilai skor berada pada rentang 0 sampai 1.
* Semakin mendekati 1, maka wajah dianggap semakin mirip.
* Semakin mendekati 0, maka wajah dianggap semakin berbeda.

---

## Tampilan Aplikasi

Aplikasi memiliki tiga tab utama:

### 1. Tab Input Gambar

Tab ini digunakan untuk memilih dua gambar wajah yang akan dibandingkan.


<img width="1600" height="736" alt="input" src="https://github.com/user-attachments/assets/a8917a08-6b58-4168-83a8-14e16230a29a" />


Fitur pada tab ini:

* Tombol pilih file untuk Gambar 1.
* Tombol pilih file untuk Gambar 2.
* Preview gambar yang dipilih.
* Tombol mulai bandingkan.

---

### 2. Tab Hasil Perbandingan

Tab ini digunakan untuk menampilkan hasil visual dari proses deteksi wajah.


<img width="1600" height="792" alt="hasil" src="https://github.com/user-attachments/assets/63cd68e3-630b-49fc-a1be-0a7030948d69" />


Fitur pada tab ini:

* Menampilkan Gambar 1 dengan kotak wajah.
* Menampilkan Gambar 2 dengan kotak wajah.
* Menampilkan kesimpulan hasil perbandingan.
* Menampilkan nilai confidence.
* Menampilkan skor kemiripan.

---

### 3. Tab Laporan

Tab ini digunakan untuk menampilkan laporan lengkap hasil perbandingan.


<img width="1600" height="806" alt="laporan" src="https://github.com/user-attachments/assets/71fcaa0b-1dc3-44dd-ba33-d9e60feb9060" />


Isi laporan meliputi:

* Nama file gambar.
* Resolusi gambar.
* Ukuran file gambar.
* Jumlah wajah yang terdeteksi.
* Waktu proses.
* Tahapan preprocessing.
* Metode yang digunakan.
* Nilai threshold.
* Nilai confidence.
* Skor kemiripan.
* Kategori kemiripan.
* Interpretasi hasil.
* Kesimpulan akhir.

---

## Library yang Digunakan

Program ini menggunakan beberapa library Python, yaitu:

| Library | Fungsi                                                  |
| ------- | ------------------------------------------------------- |
| Tkinter | Membuat tampilan GUI aplikasi                           |
| OpenCV  | Membaca gambar, deteksi wajah, dan LBPH Face Recognizer |
| NumPy   | Mengolah data numerik dan array                         |
| Pillow  | Menampilkan dan menyesuaikan gambar pada GUI            |
| OS      | Mengambil nama file dan ukuran file                     |
| Time    | Menghitung waktu proses perbandingan                    |

---

## Instalasi Library

Sebelum menjalankan program, pastikan Python sudah terpasang. Setelah itu install library yang dibutuhkan dengan perintah:

```bash
pip install opencv-contrib-python numpy pillow
```

Catatan penting:

Program membutuhkan `opencv-contrib-python` karena fitur LBPH Face Recognizer berada pada modul:

```python
cv2.face
```

Jika hanya menginstall `opencv-python`, kemungkinan program akan error karena `cv2.face` tidak tersedia.

---

## Cara Menjalankan Program

1. Pastikan semua library sudah terinstall.
2. Buka file program Python.
3. Jalankan program dengan perintah:

```bash
python ProjectAljabarLinear.py
```

4. Pilih Gambar 1.
5. Pilih Gambar 2.
6. Klik tombol **Mulai Bandingkan**.
7. Lihat hasil pada tab **Hasil Perbandingan**.
8. Lihat laporan lengkap pada tab **Laporan**.

---

## Struktur Program

Beberapa bagian penting dalam program:

| Bagian Program               | Fungsi                                                                           |
| ---------------------------- | -------------------------------------------------------------------------------- |
| `format_ukuran_file()`       | Mengubah ukuran file menjadi byte, KB, atau MB                                   |
| `kategori_kemiripan()`       | Menentukan kategori kemiripan berdasarkan confidence                             |
| `deteksi_dan_potong_wajah()` | Membaca gambar, mendeteksi wajah, memotong wajah, resize, dan equalize histogram |
| `bandingkan_wajah()`         | Membandingkan dua wajah menggunakan LBPH                                         |
| `AplikasiPerbandinganWajah`  | Class utama untuk membuat GUI aplikasi                                           |
| `pilih_gambar_1()`           | Memilih gambar pertama                                                           |
| `pilih_gambar_2()`           | Memilih gambar kedua                                                             |
| `mulai_bandingkan()`         | Menjalankan proses perbandingan                                                  |
| `update_tab_hasil()`         | Menampilkan hasil visual perbandingan                                            |
| `buat_teks_laporan()`        | Membuat laporan hasil perbandingan                                               |
| `salin_laporan()`            | Menyalin laporan ke clipboard                                                    |

---

## Hubungan dengan Aljabar Linear

Proyek ini berhubungan dengan aljabar linear karena gambar digital dapat direpresentasikan sebagai matriks. Setiap piksel pada gambar memiliki nilai numerik, sehingga proses pengolahan gambar dapat dipahami sebagai proses manipulasi matriks.

Beberapa konsep aljabar linear yang berkaitan dengan proyek ini adalah:

1. **Matriks Gambar**
   Gambar grayscale direpresentasikan sebagai matriks dua dimensi yang berisi nilai intensitas piksel.

2. **Transformasi Citra**
   Proses grayscale, resize, dan histogram equalization merupakan bagian dari transformasi data citra.

3. **Vektor Fitur**
   Metode LBPH mengubah pola wajah menjadi representasi fitur yang dapat dibandingkan.

4. **Perbandingan Nilai Numerik**
   Hasil ekstraksi fitur dibandingkan untuk menghasilkan nilai confidence.

5. **Normalisasi Ukuran Matriks**
   Resize wajah menjadi 200 x 200 piksel dilakukan agar kedua data wajah memiliki dimensi yang sama.

---

## Contoh Output Program

Contoh output yang ditampilkan pada terminal:

```text
========== HASIL PERBANDINGAN WAJAH ==========
Nama File 1    : gambar1.jpg
Nama File 2    : gambar2.jpg
Confidence LBPH: 55.4321
Skor mirip     : 0.4457
Kategori       : Kemiripan Sedang
Batas beda     : 70
Waktu proses   : 0.2134 detik
Kesimpulan     : kemungkinan ORANG YANG SAMA
=============================================
```

---

## Kelebihan Program

Beberapa kelebihan program ini adalah:

1. Memiliki tampilan GUI sehingga mudah digunakan.
2. Dapat memilih gambar langsung dari komputer.
3. Menampilkan preview gambar.
4. Menampilkan wajah yang terdeteksi dengan kotak hijau.
5. Menampilkan laporan hasil perbandingan secara otomatis.
6. Menggunakan metode LBPH yang tersedia pada OpenCV.
7. Dapat menghitung waktu proses perbandingan.

---

## Keterbatasan Program

Program ini masih memiliki beberapa keterbatasan, yaitu:

1. Hasil perbandingan dapat dipengaruhi oleh pencahayaan gambar.
2. Posisi wajah yang terlalu miring dapat memengaruhi hasil deteksi.
3. Ekspresi wajah yang berbeda dapat memengaruhi nilai confidence.
4. Jika wajah tidak menghadap kamera, deteksi bisa gagal.
5. Program hanya memilih wajah terbesar jika terdapat lebih dari satu wajah.
6. Threshold masih ditentukan secara manual.
7. Program belum menggunakan dataset besar untuk pelatihan model.

---

## Kesimpulan

Proyek ini berhasil menerapkan konsep aljabar linear dalam bentuk aplikasi perbandingan wajah berbasis Python. Gambar wajah diproses sebagai data matriks, kemudian melalui tahapan grayscale, deteksi wajah, pemotongan area wajah, resizing, histogram equalization, dan perbandingan menggunakan LBPH Face Recognizer.

Melalui proyek ini, dapat dipahami bahwa konsep matriks dan operasi numerik dalam aljabar linear memiliki penerapan nyata dalam bidang pengolahan citra digital dan pengenalan wajah.

farhanbaihaqi20/PROJECT-AKHIR-ALJABAR-LINEAR
