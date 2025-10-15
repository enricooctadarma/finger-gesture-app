import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import threading
from gtts import gTTS

# ---------- CONFIG ----------
st.set_page_config(page_title="Gesture Voice Recognition", layout="centered")
st.title("🖐️ Gesture Voice Recognition - Final Stable Version")
st.markdown("""
📸 Ambil foto gesture tanganmu dan biarkan AI mengenali bentuknya!  
🎙️ Aplikasi ini akan mengucapkan teks sesuai gesture yang terdeteksi.  
💡 Kamera otomatis **dibalik (mirror)** agar arah tanganmu sesuai tampilan.
""")

# ---------- USER INPUT ----------
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

# ---------- FUNGSI TEXT-TO-SPEECH ----------
def speak_async(text):
    def _run():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                gTTS(text=text, lang="id").save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
                time.sleep(2)
                os.remove(fp.name)
        except Exception as e:
            st.error(f"Gagal memutar suara: {e}")
    threading.Thread(target=_run, daemon=True).start()

# ---------- DETEKSI GESTURE ----------
def detect_gesture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    _, thresh = cv2.threshold(blur, 70, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    gesture = ""
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        area_contour = cv2.contourArea(contour)
        area_hull = cv2.contourArea(hull)
        area_ratio = ((area_hull - area_contour) / area_contour) * 100 \
            if area_contour > 0 else 0

        # Heuristik sederhana
        if area_ratio < 10:
            gesture = "✋"
        elif 10 <= area_ratio < 30:
            gesture = "👍"
        elif 30 <= area_ratio < 50:
            gesture = "✌️"
        else:
            gesture = "🤘"
    return gesture

# ---------- KAMERA ----------
st.subheader("🎥 Ambil Foto Gesture")
st.caption("Pastikan pencahayaan cukup dan tangan terlihat jelas.")
img_file = st.camera_input("Ambil foto gesture")

if img_file is not None:
    # Baca & mirror gambar
    bytes_data = img_file.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.flip(img, 1)  # 🔁 Mirror

    # Tampilkan preview
    st.image(img, caption="Foto kamu (mirror mode)", channels="BGR")

    # Deteksi gesture
    gesture = detect_gesture(img)
    text_to_speak = GESTURES.get(gesture, "")

    if gesture:
        st.success(f"Gesture terdeteksi: {gesture}")
        st.info(f"Teks: {text_to_speak}")
        if text_to_speak:
            speak_async(text_to_speak)
    else:
        st.warning("Gesture tidak dikenali. Coba foto ulang tanganmu dengan posisi lebih jelas.")
else:
    st.info("📷 Klik tombol di atas untuk mengambil foto gesture tanganmu.")
