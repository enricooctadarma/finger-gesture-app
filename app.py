import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import threading
from gtts import gTTS
import mediapipe as mp

# ---------- CONFIG ----------
st.set_page_config(page_title="Gesture Voice Recognition (MediaPipe)", layout="centered")
st.title("🖐️ Gesture Voice Recognition - MediaPipe Edition")
st.markdown("""
📸 Ambil foto gesture tanganmu.  
🤖 Sistem akan mengenali bentuk jari menggunakan **MediaPipe Hands**.  
🎙️ Suara otomatis sesuai gesture yang terdeteksi.  
💡 Kamera & hasil foto akan tampil **mirror** seperti cermin.
""")

# ---------- INPUT TEKS ----------
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

# ---------- FUNGSI SUARA ----------
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

# ---------- MEDIAPIPE SETUP ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_gesture_mediapipe(image):
    """Deteksi gesture dari foto menggunakan MediaPipe"""
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    ) as hands:

        # Konversi BGR ke RGB (karena MediaPipe pakai RGB)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return "", image

        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Tentukan jari mana yang terbuka
        finger_tips = [8, 12, 16, 20]  # Indeks ujung jari selain jempol
        finger_states = []

        landmarks = hand_landmarks.landmark
        for tip_id in finger_tips:
            finger_states.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)
        # Jempol (horizontal)
        thumb_open = landmarks[4].x > landmarks[3].x

        # Deteksi gesture
        if all(finger_states) and thumb_open:
            gesture = "✋"
        elif thumb_open and not any(finger_states):
            gesture = "👍"
        elif finger_states[0] and finger_states[1] and not finger_states[2] and not finger_states[3]:
            gesture = "✌️"
        elif thumb_open and finger_states[0] and finger_states[3] and not finger_states[1] and not finger_states[2]:
            gesture = "🤘"
        else:
            gesture = "Tidak dikenali"

        return gesture, image

# ---------- KAMERA ----------
st.subheader("🎥 Ambil Foto Gesture")
st.caption("Pastikan tangan terlihat jelas dan pencahayaan cukup.")

img_file = st.camera_input("📸 Ambil foto gesture tanganmu")

if img_file is not None:
    # Baca & mirror gambar
    bytes_data = img_file.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    mirror_img = cv2.flip(img, 1)

    # Deteksi gesture menggunakan MediaPipe
    gesture, annotated = detect_gesture_mediapipe(mirror_img)

    # Tampilkan hasil
    st.image(annotated, caption=f"Gesture terdeteksi: {gesture}", channels="BGR", use_column_width=True)

    if gesture in GESTURES:
        st.success(f"Gesture: {gesture} → \"{GESTURES[gesture]}\"")
        speak_async(GESTURES[gesture])
    else:
        st.warning("Gesture tidak dikenali. Coba ambil ulang foto tanganmu.")
else:
    st.info("Klik tombol di atas untuk mengambil foto gesture tanganmu.")
