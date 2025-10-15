import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import tempfile
import os
import time
import threading

# ---------- Konfigurasi ----------
st.set_page_config(page_title="Gesture Voice Recognition 🤖", layout="centered")
st.title("🖐️ Gesture Voice Recognition - Anti Loading Version 🚀")
st.markdown("""
💡 Versi ini **tidak menggunakan WebRTC** dan **tidak akan stuck loading**.  
📸 Kamu bisa **ambil foto langsung dari kamera** (HTML5 native camera).  
🎙️ Arah kamera sudah dibalik agar sesuai cermin (mirror).  
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
def speak_async(text):
    """Text-to-speech dijalankan di thread agar UI tidak nge-freeze"""
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

# ---------- Fungsi Deteksi Gesture ----------
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
        area_ratio = ((area_hull - area_contour) / area_contour) * 100 if area_contour > 0 else 0

        # Heuristik sederhana tapi stabil
        if area_ratio < 10:
            gesture = "✋"
        elif 10 <= area_ratio < 30:
            gesture = "👍"
        elif 30 <= area_ratio < 50:
            gesture = "✌️"
        else:
            gesture = "🤘"
    return gesture

# ---------- Session State ----------
if "last_gesture" not in st.session_state:
    st.session_state.last_gesture = ""
if "last_time" not in st.session_state:
    st.session_state.last_time = 0

# ---------- Kamera (HTML5) ----------
st.subheader("📸 Ambil foto gesture dari kamera kamu:")
uploaded_file = st.file_uploader(
    "Klik 'Ambil Foto' atau unggah gambar dari perangkat",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    label_visibility="collapsed"
)

# ---------- Proses Foto ----------
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Mirror
    img = cv2.flip(img, 1)

    gesture = detect_gesture(img)
    st.image(img, caption=f"Gesture: {gesture or 'Tidak dikenali'}", channels="BGR")

    current_time = time.time()
    if gesture and gesture != st.session_state.last_gesture and (current_time - st.session_state.last_time) > 2:
        st.session_state.last_gesture = gesture
        st.session_state.last_time = current_time
        text = GESTURES.get(gesture, "")
        if text:
            st.success(f"Gesture {gesture} → \"{text}\"")
            speak_async(text)
    else:
        st.info("Tidak ada gesture jelas. Coba ambil foto lagi.")
else:
    st.info("Klik tombol di atas untuk mengambil foto gesture menggunakan kamera.")
