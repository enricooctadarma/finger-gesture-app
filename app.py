import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import tempfile
import os
import time

# ---------- Konfigurasi Halaman ----------
st.set_page_config(page_title="Gesture Voice Recognition 🤖", layout="centered")
st.title("🖐️ Gesture Voice Recognition (Cloud Version - Mirror Camera)")
st.markdown("""
Aplikasi ini menggunakan kamera Streamlit (tanpa WebRTC).  
Klik **Ambil Foto** untuk mendeteksi gesture — kamera akan tampil mirror.
""")

# ---------- Input Kalimat ----------
col1, col2 = st.columns(2)
with col1:
    kata1 = st.text_input("✋ Semua jari terbuka", "Halo semuanya!")
    kata2 = st.text_input("👍 Hanya jempol", "Saya senang hari ini!")
with col2:
    kata3 = st.text_input("✌️ Telunjuk & tengah", "Nama saya Enrico!")
    kata4 = st.text_input("🤘 Metal", "Sampai jumpa lagi!")

GESTURES = {
    "✋": kata1,
    "👍": kata2,
    "✌️": kata3,
    "🤘": kata4,
}

# ---------- Fungsi TTS ----------
def speak(text):
    """Text-to-speech dengan file sementara"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        gTTS(text=text, lang="id").save(fp.name)
        st.audio(fp.name, format="audio/mp3", autoplay=True)
        time.sleep(1.5)
        os.remove(fp.name)

# ---------- Fungsi Deteksi Gesture ----------
def detect_gesture(img):
    """Deteksi sederhana gesture berdasarkan area kontur"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gesture = ""

    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        area_contour = cv2.contourArea(contour)
        area_hull = cv2.contourArea(hull)
        if area_contour > 0:
            area_ratio = ((area_hull - area_contour) / area_contour) * 100
        else:
            area_ratio = 0

        # Heuristik yang lebih stabil
        if area_ratio < 10:
            gesture = "✋"
        elif 10 <= area_ratio < 30:
            gesture = "👍"
        elif 30 <= area_ratio < 50:
            gesture = "✌️"
        else:
            gesture = "🤘"

    return gesture

# ---------- Ambil Gambar dari Kamera ----------
st.write("📸 Ambil gambar dari kamera:")
img_file = st.camera_input("Klik tombol di bawah untuk ambil foto (kamera mirror)")

if img_file is not None:
    # Konversi ke format OpenCV
    bytes_data = img_file.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # 🔁 Balik kamera (mirror)
    img = cv2.flip(img, 1)

    # Deteksi gesture
    gesture = detect_gesture(img)

    # Tampilkan hasil
    st.image(img, caption=f"Gesture terdeteksi: {gesture or 'Tidak dikenali'}", channels="BGR")

    if gesture:
        text = GESTURES.get(gesture, "")
        if text:
            st.success(f"Gesture {gesture} → \"{text}\"")
            speak(text)
    else:
        st.warning("Tidak dapat mendeteksi gesture dengan jelas, coba ulangi dengan posisi tangan yang lebih terlihat.")
