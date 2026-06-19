import io
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps


UKURAN_WAJAH = (200, 200)
BATAS_BEDA = 70


st.set_page_config(
    page_title="Perbandingan Wajah LBPH",
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


def kategori_kemiripan(confidence):
    if confidence <= 50:
        return "Kemiripan Tinggi"
    elif confidence <= BATAS_BEDA:
        return "Kemiripan Sedang"
    elif confidence <= 90:
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


def bandingkan_wajah(file_1, file_2):
    waktu_mulai = time.perf_counter()

    gambar_1, nama_file_1, ukuran_file_1 = baca_gambar_upload(file_1)
    gambar_2, nama_file_2, ukuran_file_2 = baca_gambar_upload(file_2)

    wajah_1, gambar_kotak_1, jumlah_wajah_1, resolusi_1 = deteksi_dan_potong_wajah(gambar_1)
    wajah_2, gambar_kotak_2, jumlah_wajah_2, resolusi_2 = deteksi_dan_potong_wajah(gambar_2)

    if not hasattr(cv2, "face"):
        raise ValueError(
            "Modul cv2.face tidak ditemukan. Pastikan requirements.txt memakai opencv-contrib-python-headless."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )

    recognizer.train([wajah_1], np.array([1]))
    label_hasil, confidence = recognizer.predict(wajah_2)

    orang_sama = confidence <= BATAS_BEDA
    skor_mirip = max(0, min(1, 1 - confidence / 100))

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
        "confidence": confidence,
        "skor_akhir": skor_mirip,
        "orang_sama": orang_sama,
        "kategori": kategori_kemiripan(confidence),
        "waktu_proses": waktu_proses,
    }


def buat_teks_laporan(hasil):
    resolusi_1 = f"{hasil['resolusi_1'][0]} x {hasil['resolusi_1'][1]} pixel"
    resolusi_2 = f"{hasil['resolusi_2'][0]} x {hasil['resolusi_2'][1]} pixel"

    if hasil["orang_sama"]:
        interpretasi = (
            f"Nilai confidence sebesar {hasil['confidence']:.2f} berada di bawah atau sama dengan "
            f"threshold {BATAS_BEDA}. Artinya, sistem menilai kedua wajah memiliki pola fitur "
            f"yang cukup mirip berdasarkan metode LBPH."
        )
        kesimpulan = "Kemungkinan kedua gambar berasal dari ORANG YANG SAMA."
    else:
        interpretasi = (
            f"Nilai confidence sebesar {hasil['confidence']:.2f} lebih besar dari threshold {BATAS_BEDA}. "
            f"Artinya, sistem menilai kedua wajah memiliki perbedaan pola fitur yang cukup besar "
            f"berdasarkan metode LBPH."
        )
        kesimpulan = "Kemungkinan kedua gambar berasal dari ORANG YANG BERBEDA."

    laporan = f"""
============================================================
             LAPORAN HASIL PERBANDINGAN WAJAH
============================================================

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

✓ Tahapan Preprocessing
  1. Membaca gambar dari file upload.
  2. Mengubah gambar menjadi grayscale.
  3. Mendeteksi wajah menggunakan Haar Cascade.
  4. Menghitung jumlah wajah yang terdeteksi.
  5. Memilih wajah terbesar sebagai wajah utama.
  6. Memotong area wajah yang terdeteksi.
  7. Mengubah ukuran wajah menjadi {UKURAN_WAJAH[0]} x {UKURAN_WAJAH[1]} pixel.
  8. Melakukan histogram equalization agar pencahayaan lebih seimbang.
  9. Membandingkan hasil preprocessing menggunakan metode LBPH.

✓ Metode yang Digunakan
  Haar Cascade Face Detection + LBPH Face Recognizer
  LBPH = Local Binary Patterns Histogram

✓ Threshold
  {BATAS_BEDA}

✓ Confidence
  {hasil['confidence']:.4f}

✓ Skor Kemiripan
  {hasil['skor_akhir']:.4f} dari 1.0000

✓ Kategori Kemiripan
  {hasil['kategori']}

✓ Interpretasi Hasil
  {interpretasi}

✓ Kesimpulan Akhir
  {kesimpulan}

============================================================
Catatan:
Semakin kecil nilai confidence LBPH, maka semakin mirip wajah
ketika dibandingkan. Sebaliknya, semakin besar nilai confidence,
maka semakin besar perbedaan antara kedua wajah.
============================================================
"""
    return laporan.strip()


# =========================
# Tampilan Streamlit
# =========================

st.markdown('<div class="sub-title">PROJECT AKHIR ALJABAR LINEAR - KELOMPOK 7</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Perbandingan Wajah Menggunakan LBPH</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="desc">Upload dua gambar wajah, lalu sistem akan mendeteksi wajah dan membandingkan tingkat kemiripannya.</div>',
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
                with st.spinner("Sedang memproses gambar dan membandingkan wajah..."):
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
        m1.metric("Confidence LBPH", f"{hasil['confidence']:.2f}")
        m2.metric("Skor Mirip", f"{hasil['skor_akhir']:.2f}")
        m3.metric("Kategori", hasil["kategori"])
        m4.metric("Waktu Proses", f"{hasil['waktu_proses']:.4f} detik")

        st.caption("Catatan: Semakin kecil nilai confidence LBPH, maka semakin mirip wajahnya.")


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
            file_name="laporan_perbandingan_wajah.txt",
            mime="text/plain",
            use_container_width=True,
        )
