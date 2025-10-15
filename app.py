import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from cvzone.HandTrackingModule import HandDetector
from gtts import gTTS
import tempfile
import os
import time

# ---------- Konfigurasi Halaman ----------
st.set_page_config(page_title="Gesture Voice Recognition 🤖", layout="centered")
st.title("🖐️ Gesture Voice Recognition Langsung di Browser")
st.markdown("""
Masukkan 4 kalimat yang akan diucapkan sesuai simbol jari berikut:
1. ✋ = Semua jari terbuka  
2. 👍 = Hanya jempol terbuka  
3. ✌️ = Telunjuk & tengah terbuka  
4. 🤘 = Metal (jempol, telunjuk, dan kelingking terbuka)
""")

# ---------- Input Kalimat ----------
col1, col2 = st.columns(2)

with col1:
    kata1 = st.text_input("✋ (Semua jari terbuka)", "Halo semuanya!")
    kata2 = st.text_input("👍 (Hanya jempol)", "Saya senang hari ini!")

with col2:
    kata3 = st.text_input("✌️ (Telunjuk & Tengah)", "Nama saya Enrico!")
    kata4 = st.text_input("🤘 (Metal)", "Sampai jumpa lagi!")

# ---------- Mapping Gesture ----------
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

# ---------- Video Transformer ----------
class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(detectionCon=0.75, maxHands=1)
        self.last_gesture = ""
        self.last_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        hands, img = self.detector.findHands(img)
        gesture_symbol = ""

        if hands:
            hand = hands[0]
            fingers = self.detector.fingersUp(hand)

            # Logika deteksi gesture
            if fingers == [1, 1, 1, 1, 1]:
                gesture_symbol = "✋"
            elif fingers == [1, 0, 0, 0, 0]:
                gesture_symbol = "👍"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture_symbol = "✌️"
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0:
                gesture_symbol = "🤘"

            # Hindari pengulangan suara beruntun
            current_time = time.time()
            if (
                gesture_symbol
                and gesture_symbol != self.last_gesture
                and (current_time - self.last_time) > 2
            ):
                self.last_gesture = gesture_symbol
                self.last_time = current_time
                st.session_state["tts_text"] = GESTURES.get(gesture_symbol, "")

            cv2.putText(
                img,
                f"Gesture: {gesture_symbol}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                img,
                "Tidak ada tangan terdeteksi",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

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
