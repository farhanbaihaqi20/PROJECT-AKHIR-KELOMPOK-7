import io
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


UKURAN_WAJAH = (200, 200)
BATAS_BEDA = 35
KOMPONEN_PCA_KEMIRIPAN = 1
KOMPONEN_PCA_KOMPRESI = 40


st.set_page_config(
    page_title="Perbandingan Wajah PCA",
    page_icon="🧠",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #101820;
        color: white;
    }

    .main-title {
        text-align: center;
        color: white;
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .sub-title {
        text-align: center;
        color: #FEE715;
        font-size: 17px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .desc {
        text-align: center;
        color: #C7D0D9;
        font-size: 16px;
        margin-bottom: 24px;
    }

    .result-box {
        padding: 18px;
        border-radius: 14px;
        background: #1E2A36;
        border: 2px solid #FEE715;
        text-align: center;
        margin-top: 18px;
    }

    .success-text {
        color: #30D158;
        font-size: 25px;
        font-weight: 800;
    }

    .danger-text {
        color: #FF453A;
        font-size: 25px;
        font-weight: 800;
    }

    .small-note {
        color: #C7D0D9;
        font-size: 14px;
        text-align: center;
    }

    div[data-testid="stMetric"] {
        background-color: #1E2A36;
        border: 1px solid #FEE715;
        padding: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Fungsi bantu
# =========================

def format_ukuran_file(byte_size):
    if byte_size < 1024:
        return f"{byte_size} byte"
    elif byte_size < 1024 * 1024:
        return f"{byte_size / 1024:.2f} KB"
    else:
        return f"{byte_size / (1024 * 1024):.2f} MB"


def kategori_kemiripan(jarak_pca):
    if jarak_pca <= 20:
        return "Kemiripan Tinggi"
    elif jarak_pca <= BATAS_BEDA:
        return "Kemiripan Sedang"
    elif jarak_pca <= 50:
        return "Kemiripan Rendah"
    else:
        return "Kemiripan Sangat Rendah"


def baca_gambar_upload(uploaded_file):
    """Membaca file upload Streamlit menjadi gambar OpenCV format BGR."""
    byte_data = uploaded_file.getvalue()

    try:
        gambar_pil = Image.open(io.BytesIO(byte_data))
        gambar_pil = ImageOps.exif_transpose(gambar_pil)
        gambar_pil = gambar_pil.convert("RGB")
    except Exception:
        raise ValueError("Gambar tidak bisa dibaca. Pastikan file berupa JPG, JPEG, atau PNG.")

    gambar_rgb = np.array(gambar_pil)
    gambar_bgr = cv2.cvtColor(gambar_rgb, cv2.COLOR_RGB2BGR)

    return gambar_bgr, uploaded_file.name, len(byte_data)


def gambar_bgr_ke_rgb(gambar_bgr):
    return cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)


def deteksi_dan_potong_wajah(gambar_bgr):
    if gambar_bgr is None:
        raise ValueError("Gambar tidak bisa dibaca.")

    tinggi, lebar = gambar_bgr.shape[:2]
    abu_abu = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        raise ValueError("File Haar Cascade tidak ditemukan oleh OpenCV.")

    wajah = face_detector.detectMultiScale(
        abu_abu,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    jumlah_wajah = len(wajah)

    if jumlah_wajah == 0:
        raise ValueError("Tidak ada wajah yang terdeteksi. Coba gunakan foto yang lebih jelas dan menghadap depan.")

    # Ambil wajah terbesar sebagai wajah utama
    x, y, w, h = max(wajah, key=lambda kotak: kotak[2] * kotak[3])

    potongan = abu_abu[y:y + h, x:x + w]
    potongan = cv2.resize(potongan, UKURAN_WAJAH)
    potongan = cv2.equalizeHist(potongan)

    gambar_kotak = gambar_bgr.copy()

    for (fx, fy, fw, fh) in wajah:
        cv2.rectangle(gambar_kotak, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 4)

    return potongan, gambar_kotak, jumlah_wajah, (lebar, tinggi)


def pca_similarity(wajah_1, wajah_2):
    """
    Menghitung kemiripan wajah menggunakan PCA.

    Alur:
    1. Wajah 200x200 diubah menjadi vektor 40.000 piksel.
    2. Dua vektor wajah disusun menjadi matriks data.
    3. Data dikurangi rata-rata, lalu dihitung PCA dengan SVD.
    4. Kedua wajah diproyeksikan ke ruang komponen utama.
    5. Jarak Euclidean pada ruang PCA dinormalisasi menjadi skala 0-100.
    """
    vektor_1 = wajah_1.astype(np.float32).reshape(1, -1) / 255.0
    vektor_2 = wajah_2.astype(np.float32).reshape(1, -1) / 255.0

    data = np.vstack([vektor_1, vektor_2])
    rata_rata = np.mean(data, axis=0)
    data_terpusat = data - rata_rata

    if np.allclose(data_terpusat, 0):
        return {
            "jarak_pca": 0.0,
            "skor_mirip": 1.0,
            "jumlah_komponen": 1,
            "explained_variance": 1.0,
            "fitur_1": np.array([0.0]),
            "fitur_2": np.array([0.0]),
        }

    # SVD digunakan sebagai implementasi PCA yang stabil secara numerik.
    # Dalam PCA, komponen utama berkaitan dengan eigenvector dari matriks kovarians.
    u, singular_values, vt = np.linalg.svd(data_terpusat, full_matrices=False)

    jumlah_komponen = min(KOMPONEN_PCA_KEMIRIPAN, vt.shape[0])
    komponen_utama = vt[:jumlah_komponen]
    fitur_pca = data_terpusat @ komponen_utama.T

    jarak = float(np.linalg.norm(fitur_pca[0] - fitur_pca[1]))
    rmse = jarak / np.sqrt(data.shape[1])
    jarak_pca = rmse * 100
    skor_mirip = max(0.0, min(1.0, 1 - jarak_pca / 100))

    total_variance = float(np.sum(singular_values ** 2))
    if total_variance == 0:
        explained_variance = 1.0
    else:
        explained_variance = float(np.sum(singular_values[:jumlah_komponen] ** 2) / total_variance)

    return {
        "jarak_pca": jarak_pca,
        "skor_mirip": skor_mirip,
        "jumlah_komponen": jumlah_komponen,
        "explained_variance": explained_variance,
        "fitur_1": fitur_pca[0],
        "fitur_2": fitur_pca[1],
    }


def kompresi_gambar_pca(wajah_gray, jumlah_komponen=KOMPONEN_PCA_KOMPRESI):
    """
    Kompresi citra wajah menggunakan PCA.

    Citra grayscale 200x200 dianggap sebagai matriks.
    Baris citra menjadi sampel, kolom citra menjadi fitur.
    Hanya sejumlah komponen utama yang dipakai untuk merekonstruksi gambar.
    """
    matriks = wajah_gray.astype(np.float32) / 255.0
    rata_rata = np.mean(matriks, axis=0)
    matriks_terpusat = matriks - rata_rata

    u, singular_values, vt = np.linalg.svd(matriks_terpusat, full_matrices=False)

    k = min(jumlah_komponen, len(singular_values))
    rekonstruksi = (u[:, :k] @ np.diag(singular_values[:k]) @ vt[:k, :]) + rata_rata
    rekonstruksi = np.clip(rekonstruksi, 0, 1)

    total_variance = float(np.sum(singular_values ** 2))
    if total_variance == 0:
        explained_variance = 1.0
    else:
        explained_variance = float(np.sum(singular_values[:k] ** 2) / total_variance)

    original_values = matriks.shape[0] * matriks.shape[1]
    compressed_values = (matriks.shape[0] * k) + k + (k * matriks.shape[1]) + matriks.shape[1]
    rasio_kompresi = original_values / compressed_values

    mse = float(np.mean((matriks - rekonstruksi) ** 2))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(10 * np.log10(1.0 / mse))

    return {
        "gambar_rekonstruksi": (rekonstruksi * 255).astype(np.uint8),
        "komponen_kompresi": k,
        "variance_kompresi": explained_variance,
        "rasio_kompresi": rasio_kompresi,
        "mse_rekonstruksi": mse,
        "psnr_rekonstruksi": psnr,
    }


def bandingkan_wajah(file_1, file_2):
    waktu_mulai = time.perf_counter()

    gambar_1, nama_file_1, ukuran_file_1 = baca_gambar_upload(file_1)
    gambar_2, nama_file_2, ukuran_file_2 = baca_gambar_upload(file_2)

    wajah_1, gambar_kotak_1, jumlah_wajah_1, resolusi_1 = deteksi_dan_potong_wajah(gambar_1)
    wajah_2, gambar_kotak_2, jumlah_wajah_2, resolusi_2 = deteksi_dan_potong_wajah(gambar_2)

    hasil_pca = pca_similarity(wajah_1, wajah_2)
    kompresi_1 = kompresi_gambar_pca(wajah_1)
    kompresi_2 = kompresi_gambar_pca(wajah_2)

    jarak_pca = hasil_pca["jarak_pca"]
    skor_mirip = hasil_pca["skor_mirip"]
    orang_sama = jarak_pca <= BATAS_BEDA

    waktu_selesai = time.perf_counter()
    waktu_proses = waktu_selesai - waktu_mulai

    return {
        "nama_file_1": nama_file_1,
        "nama_file_2": nama_file_2,
        "ukuran_file_1": format_ukuran_file(ukuran_file_1),
        "ukuran_file_2": format_ukuran_file(ukuran_file_2),
        "resolusi_1": resolusi_1,
        "resolusi_2": resolusi_2,
        "jumlah_wajah_1": jumlah_wajah_1,
        "jumlah_wajah_2": jumlah_wajah_2,
        "gambar_kotak_1": gambar_kotak_1,
        "gambar_kotak_2": gambar_kotak_2,
        "jarak_pca": jarak_pca,
        "skor_akhir": skor_mirip,
        "orang_sama": orang_sama,
        "kategori": kategori_kemiripan(jarak_pca),
        "waktu_proses": waktu_proses,
        "komponen_pca": hasil_pca["jumlah_komponen"],
        "variance_pca": hasil_pca["explained_variance"],
        "kompresi_1": kompresi_1,
        "kompresi_2": kompresi_2,
    }


def buat_teks_laporan(hasil):
    resolusi_1 = f"{hasil['resolusi_1'][0]} x {hasil['resolusi_1'][1]} pixel"
    resolusi_2 = f"{hasil['resolusi_2'][0]} x {hasil['resolusi_2'][1]} pixel"

    if hasil["orang_sama"]:
        interpretasi = (
            f"Nilai jarak PCA sebesar {hasil['jarak_pca']:.2f} berada di bawah atau sama dengan "
            f"threshold {BATAS_BEDA}. Artinya, setelah kedua wajah diproyeksikan ke ruang komponen utama, "
            f"jarak fitur keduanya relatif kecil sehingga sistem menilai pola wajah cukup mirip."
        )
        kesimpulan = "Kemungkinan kedua gambar berasal dari ORANG YANG SAMA."
    else:
        interpretasi = (
            f"Nilai jarak PCA sebesar {hasil['jarak_pca']:.2f} lebih besar dari threshold {BATAS_BEDA}. "
            f"Artinya, setelah kedua wajah diproyeksikan ke ruang komponen utama, jarak fitur keduanya "
            f"cukup besar sehingga sistem menilai pola wajah berbeda."
        )
        kesimpulan = "Kemungkinan kedua gambar berasal dari ORANG YANG BERBEDA."

    rasio_1 = hasil["kompresi_1"]["rasio_kompresi"]
    rasio_2 = hasil["kompresi_2"]["rasio_kompresi"]
    var_kemiripan = hasil["variance_pca"] * 100
    var_kompresi_1 = hasil["kompresi_1"]["variance_kompresi"] * 100
    var_kompresi_2 = hasil["kompresi_2"]["variance_kompresi"] * 100

    psnr_1 = hasil["kompresi_1"]["psnr_rekonstruksi"]
    psnr_2 = hasil["kompresi_2"]["psnr_rekonstruksi"]
    teks_psnr_1 = "Tidak terbatas" if np.isinf(psnr_1) else f"{psnr_1:.2f} dB"
    teks_psnr_2 = "Tidak terbatas" if np.isinf(psnr_2) else f"{psnr_2:.2f} dB"

    laporan = f"""
    
             LAPORAN HASIL PERBANDINGAN WAJAH

✓ Nama File 1
  {hasil['nama_file_1']}

✓ Nama File 2
  {hasil['nama_file_2']}

✓ Resolusi Gambar
  Gambar 1 : {resolusi_1}
  Gambar 2 : {resolusi_2}

✓ Ukuran File
  Gambar 1 : {hasil['ukuran_file_1']}
  Gambar 2 : {hasil['ukuran_file_2']}

✓ Jumlah Wajah Terdeteksi
  Gambar 1 : {hasil['jumlah_wajah_1']} wajah
  Gambar 2 : {hasil['jumlah_wajah_2']} wajah

✓ Waktu Proses
  {hasil['waktu_proses']:.4f} detik


✓ PCA untuk Kompresi Gambar
  Jumlah komponen kompresi : {KOMPONEN_PCA_KOMPRESI}
  Rasio kompresi Gambar 1 : {rasio_1:.2f}x
  Rasio kompresi Gambar 2 : {rasio_2:.2f}x
  Variansi dipertahankan Gambar 1 : {var_kompresi_1:.2f}%
  Variansi dipertahankan Gambar 2 : {var_kompresi_2:.2f}%
  PSNR rekonstruksi Gambar 1 : {teks_psnr_1}
  PSNR rekonstruksi Gambar 2 : {teks_psnr_2}

✓ Threshold Jarak PCA
  {BATAS_BEDA}

✓ Jarak PCA
  {hasil['jarak_pca']:.4f}

✓ Komponen PCA untuk Kemiripan
  {hasil['komponen_pca']} komponen utama

✓ Variansi PCA untuk Kemiripan
  {var_kemiripan:.2f}%

✓ Skor Kemiripan
  {hasil['skor_akhir']:.4f} dari 1.0000

✓ Kategori Kemiripan
  {hasil['kategori']}

✓ Interpretasi Hasil
  {interpretasi}

✓ Kesimpulan Akhir
  {kesimpulan}

"""
    return laporan.strip()


# =========================
# Tampilan Streamlit
# =========================

st.markdown('<div class="sub-title">PROJECT AKHIR ALJABAR LINEAR - KELOMPOK 7</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Perbandingan Wajah Menggunakan PCA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="desc">Upload dua gambar wajah, lalu sistem akan mendeteksi wajah dan membandingkan tingkat kemiripannya dengan PCA.</div>',
    unsafe_allow_html=True,
)

if "hasil" not in st.session_state:
    st.session_state.hasil = None

if "laporan" not in st.session_state:
    st.session_state.laporan = ""


tab_input, tab_hasil, tab_laporan = st.tabs(["1. Input Gambar", "2. Hasil Perbandingan", "3. Laporan"])

with tab_input:
    kolom_1, kolom_vs, kolom_2 = st.columns([1, 0.15, 1])

    with kolom_1:
        st.subheader("Gambar 1")
        file_1 = st.file_uploader(
            "Pilih file gambar pertama",
            type=["jpg", "jpeg", "png"],
            key="file_1",
        )
        if file_1 is not None:
            st.image(file_1, caption=file_1.name, use_container_width=True)

    with kolom_vs:
        st.markdown("<br><br><h1 style='text-align:center;color:#FEE715;'>VS</h1>", unsafe_allow_html=True)

    with kolom_2:
        st.subheader("Gambar 2")
        file_2 = st.file_uploader(
            "Pilih file gambar kedua",
            type=["jpg", "jpeg", "png"],
            key="file_2",
        )
        if file_2 is not None:
            st.image(file_2, caption=file_2.name, use_container_width=True)

    st.markdown("---")

    tombol = st.button("Mulai Bandingkan", type="primary", use_container_width=True)

    if tombol:
        if file_1 is None or file_2 is None:
            st.warning("Pilih Gambar 1 dan Gambar 2 terlebih dahulu.")
        else:
            try:
                with st.spinner("Sedang memproses gambar dan membandingkan wajah menggunakan PCA..."):
                    hasil = bandingkan_wajah(file_1, file_2)
                    laporan = buat_teks_laporan(hasil)

                st.session_state.hasil = hasil
                st.session_state.laporan = laporan
                st.success("Proses selesai. Buka tab Hasil Perbandingan atau Laporan.")

            except Exception as error:
                st.error(f"Terjadi error: {error}")

    st.markdown('<p class="small-note">Kelompok 7 • Rombel 1 • Proyek Akhir Aljabar Linear</p>', unsafe_allow_html=True)


with tab_hasil:
    hasil = st.session_state.hasil

    if hasil is None:
        st.info("Hasil belum tersedia. Silakan pilih dua gambar di tab Input Gambar, lalu klik Mulai Bandingkan.")
    else:
        st.subheader("Wajah yang Terdeteksi")

        kolom_hasil_1, kolom_hasil_2 = st.columns(2)

        with kolom_hasil_1:
            st.image(
                gambar_bgr_ke_rgb(hasil["gambar_kotak_1"]),
                caption=f"Gambar 1: {hasil['nama_file_1']}",
                use_container_width=True,
            )

        with kolom_hasil_2:
            st.image(
                gambar_bgr_ke_rgb(hasil["gambar_kotak_2"]),
                caption=f"Gambar 2: {hasil['nama_file_2']}",
                use_container_width=True,
            )

        st.markdown("---")

        if hasil["orang_sama"]:
            st.markdown(
                '<div class="result-box"><div class="success-text">HASIL: KEMUNGKINAN ORANG YANG SAMA</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-box"><div class="danger-text">HASIL: KEMUNGKINAN ORANG BERBEDA</div></div>',
                unsafe_allow_html=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jarak PCA", f"{hasil['jarak_pca']:.2f}")
        m2.metric("Skor Mirip", f"{hasil['skor_akhir']:.2f}")
        m3.metric("Kategori", hasil["kategori"])
        m4.metric("Waktu Proses", f"{hasil['waktu_proses']:.4f} detik")

        st.caption("Catatan: Semakin kecil nilai jarak PCA, maka semakin mirip wajahnya.")


with tab_laporan:
    if st.session_state.hasil is None:
        st.info("Laporan belum tersedia. Jalankan proses perbandingan terlebih dahulu.")
    else:
        st.subheader("Laporan Hasil Perbandingan")
        st.text_area(
            "Isi laporan",
            value=st.session_state.laporan,
            height=520,
        )

        st.download_button(
            label="Download Laporan TXT",
            data=st.session_state.laporan,
            file_name="laporan_perbandingan_wajah_pca.txt",
            mime="text/plain",
            use_container_width=True,
        )
