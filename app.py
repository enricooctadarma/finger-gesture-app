import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from cvzone.HandTrackingModule import HandDetector
from gtts import gTTS
import tempfile
import os
import time

# === Konfigurasi halaman ===
st.set_page_config(page_title="Gesture Voice Recognition 🤖", layout="centered")
st.title("🖐️ Gesture Voice Recognition Langsung di Browser")
st.markdown("""
Masukkan 4 kalimat yang akan diucapkan sesuai simbol jari berikut:
1. ✋ = Semua jari terbuka  
2. 👍 = Hanya jempol terbuka  
3. ✌️ = Telunjuk & tengah terbuka  
4. 🤘 = Metal (jempol, telunjuk, dan kelingking terbuka)
""")

# === Input teks untuk tiap gesture ===
col1, col2 = st.columns(2)
with col1:
    kata1 = st.text_input("✋ (Semua jari terbuka)", "Halo semuanya!")
    kata2 = st.text_input("👍 (Hanya jempol)", "Saya senang hari ini!")
with col2:
    kata3 = st.text_input("✌️ (Telunjuk & Tengah)", "Nama saya Enrico!")
    kata4 = st.text_input("🤘 (Metal)", "Sampai jumpa lagi!")

# === Mapping gesture ke teks ===
GESTURES = {
    "✋": kata1,
    "👍": kata2,
    "✌️": kata3,
    "🤘": kata4
}

# === Fungsi untuk membuat suara (text-to-speech) ===
def generate_tts(text):
    """Mengubah teks menjadi file suara dan mengembalikan path file mp3-nya."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text=text, lang='id')
        tts.save(fp.name)
        return fp.name

# === Video Transformer untuk deteksi gesture ===
class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(detectionCon=0.8, maxHands=1)
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

            # === Logika deteksi gesture ===
            if fingers == [1, 1, 1, 1, 1]:
                gesture_symbol = "✋"
            elif fingers == [1, 0, 0, 0, 0]:
                gesture_symbol = "👍"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture_symbol = "✌️"
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0:
                gesture_symbol = "🤘"

            # === Cegah pengulangan suara terlalu cepat ===
            current_time = time.time()
            if gesture_symbol and gesture_symbol != self.last_gesture and (current_time - self.last_time) > 2:
                self.last_gesture = gesture_symbol
                self.last_time = current_time
                text_to_speak = GESTURES.get(gesture_symbol, "")
                if text_to_speak:
                    st.session_state["tts_text"] = text_to_speak

            cv2.putText(img, f"Gesture: {gesture_symbol}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Tidak ada tangan terdeteksi", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img

# === Stream kamera ===
webrtc_streamer(
    key="gesture-voice",
    video_processor_factory=HandGestureTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

# === Jika ada teks untuk diucapkan ===
if "tts_text" in st.session_state:
    tts_text = st.session_state.pop("tts_text")
    mp3_path = generate_tts(tts_text)
    st.audio(mp3_path, format="audio/mp3", autoplay=True)
    st.info(f"🔊 Mengucapkan: \"{tts_text}\"")
    os.remove(mp3_path)

st.success("✅ Izinkan akses kamera di browser untuk memulai gesture recognition.")
