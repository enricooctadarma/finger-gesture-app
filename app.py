import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from gtts import gTTS
import tempfile
import os
import time

# ---------- Konfigurasi Halaman ----------
st.set_page_config(page_title="Gesture Voice Recognition 🤖", layout="centered")
st.title("🖐️ Gesture Voice Recognition Tanpa Mediapipe")
st.markdown("""
Masukkan 4 kalimat yang akan diucapkan sesuai simbol jari berikut:
1. ✋ = Semua jari terbuka  
2. 👍 = Hanya jempol terbuka  
3. ✌️ = Telunjuk & tengah terbuka  
4. 🤘 = Metal (jempol, telunjuk, kelingking)
""")

# ---------- Input Kalimat ----------
col1, col2 = st.columns(2)
with col1:
    kata1 = st.text_input("✋ (Semua jari terbuka)", "Halo semuanya!")
    kata2 = st.text_input("👍 (Hanya jempol)", "Saya senang hari ini!")
with col2:
    kata3 = st.text_input("✌️ (Telunjuk & Tengah)", "Nama saya Enrico!")
    kata4 = st.text_input("🤘 (Metal)", "Sampai jumpa lagi!")

GESTURES = {
    "✋": kata1,
    "👍": kata2,
    "✌️": kata3,
    "🤘": kata4,
}

# ---------- Fungsi TTS ----------
def generate_tts_file(text):
    """Membuat file mp3 dari teks dan mengembalikan path-nya."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text=text, lang="id")
        tts.save(fp.name)
        return fp.name


# ---------- Deteksi Gesture Manual ----------
def detect_gesture(img):
    """Deteksi sederhana gesture tangan berdasarkan area kulit dan jumlah jari."""
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

    return gesture, thresh


# ---------- Video Transformer ----------
class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_gesture = ""
        self.last_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        gesture, mask = detect_gesture(img)

        current_time = time.time()
        if gesture and gesture != self.last_gesture and (current_time - self.last_time) > 2:
            self.last_gesture = gesture
            self.last_time = current_time
            st.session_state["tts_text"] = GESTURES.get(gesture, "")

        cv2.putText(img, f"Gesture: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


# ---------- Tombol Start / Stop Kamera ----------
if "webrtc_running" not in st.session_state:
    st.session_state["webrtc_running"] = False

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("▶️ Start Kamera"):
        st.session_state["webrtc_running"] = True
with col_b:
    if st.button("⏹️ Stop Kamera"):
        st.session_state["webrtc_running"] = False

webrtc_ctx = None
if st.session_state["webrtc_running"]:
    webrtc_ctx = webrtc_streamer(
        key="gesture-voice",
        video_processor_factory=HandGestureTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    st.info("Kamera berhenti. Klik 'Start Kamera' untuk memulai.")

# ---------- Pemutaran TTS ----------
if "tts_text" in st.session_state and st.session_state["tts_text"]:
    text = st.session_state.pop("tts_text")
    try:
        mp3_path = generate_tts_file(text)
        st.audio(mp3_path, format="audio/mp3", autoplay=True)
        st.info(f"🔊 Mengucapkan: \"{text}\"")
    except Exception as e:
        st.error(f"Gagal memutar TTS: {e}")
    finally:
        if "mp3_path" in locals() and os.path.exists(mp3_path):
            os.remove(mp3_path)

st.success("✅ Izinkan akses kamera di browser untuk memulai gesture recognition.")
