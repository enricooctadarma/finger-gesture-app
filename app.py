import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import tempfile
import os
import time

st.set_page_config(page_title="Gesture Voice Recognition 🤖", layout="centered")
st.title("🖐️ Gesture Voice Recognition (Safe Cloud Version - No WebRTC)")
st.markdown("""
Aplikasi ini menggunakan kamera langsung dari Streamlit (tanpa WebRTC).  
Klik **Ambil Foto** untuk mendeteksi gesture!
""")

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

# ===== Fungsi TTS =====
def speak(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        gTTS(text=text, lang="id").save(fp.name)
        st.audio(fp.name, format="audio/mp3", autoplay=True)
        time.sleep(2)
        os.remove(fp.name)

# ===== Deteksi Gesture Sederhana =====
def detect_gesture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    gesture = ""

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        area_contour = cv2.contourArea(contour)
        area_hull = cv2.contourArea(hull)
        area_ratio = ((area_hull - area_contour) / area_contour) * 100 if area_contour > 0 else 0

        if area_ratio < 5:
            gesture = "✋"
        elif area_ratio < 15:
            gesture = "👍"
        elif area_ratio < 25:
            gesture = "✌️"
        else:
            gesture = "🤘"
    return gesture

# ===== Ambil Gambar dari Kamera =====
st.write("📸 Ambil gambar dari kamera:")
img_file = st.camera_input("Klik untuk ambil gambar")

if img_file is not None:
    bytes_data = img_file.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    gesture = detect_gesture(img)

    st.image(img, caption=f"Gesture terdeteksi: {gesture}", channels="BGR")

    if gesture:
        text = GESTURES.get(gesture, "")
        if text:
            st.success(f"Gesture {gesture} → {text}")
            speak(text)
    else:
        st.warning("Tidak dapat mendeteksi gesture, coba ulangi.")
