# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from cvzone.HandTrackingModule import HandDetector
from gtts import gTTS
import tempfile
import os
import time


# ---------- Page config ----------
st.set_page_config(page_title="Gesture Voice Recognition 🤖", layout="centered")
st.title("🖐️ Gesture Voice Recognition Langsung di Browser")
st.markdown(
"""
Masukkan 4 kalimat yang akan diucapkan sesuai simbol jari berikut:
1. ✋ = Semua jari terbuka
2. 👍 = Hanya jempol terbuka
3. ✌️ = Telunjuk & tengah terbuka
4. 🤘 = Metal (jempol, telunjuk, dan kelingking terbuka)
"""
)


# ---------- Inputs ----------
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


# ---------- TTS helper ----------
def generate_tts_file(text):
    """Generate tts mp3 and return file path (caller must remove file after use)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts = gTTS(text=text, lang="id")
        tts.save(fp.name)
        return fp.name


# ---------- Video transformer ----------
class HandGestureTransformer(VideoTransformerBase):
def __init__(self):
# detector dari cvzone (wrapping MediaPipe)
self.detector = HandDetector(detectionCon=0.75, maxHands=1)
self.last_gesture = ""
self.last_time = time.time()


st.success('✅ Izinkan akses kamera di browser untuk memulai gesture recognition.')


