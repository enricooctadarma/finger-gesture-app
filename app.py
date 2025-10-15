import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from gtts import gTTS
import tempfile
import os

st.set_page_config(page_title="Gesture Image Recognition 🤖", layout="centered")
st.title("🖐️ Gesture Recognition dari Gambar")

st.markdown("""
Unggah gambar tanganmu dan sistem akan mendeteksi simbol jari:
- ✋ Semua jari terbuka  
- 👍 Hanya jempol  
- ✌️ Telunjuk & tengah  
- 🤘 Metal (jempol, telunjuk, kelingking)
""")

uploaded = st.file_uploader("Unggah gambar tangan (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        img_path = tmp.name

    # Deteksi gesture
    from cv2 import cv2
    import numpy as np
    img = cv2.imread(img_path)
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        gesture_symbol = ""

        if fingers == [1,1,1,1,1]:
            gesture_symbol = "✋"
            text = "Halo semuanya!"
        elif fingers == [1,0,0,0,0]:
            gesture_symbol = "👍"
            text = "Saya senang hari ini!"
        elif fingers == [0,1,1,0,0]:
            gesture_symbol = "✌️"
            text = "Nama saya Enrico!"
        elif fingers[0]==1 and fingers[1]==1 and fingers[4]==1 and fingers[2]==0:
            gesture_symbol = "🤘"
            text = "Sampai jumpa lagi!"
        else:
            text = "Tidak dikenali"

        st.image(img, caption=f"Gesture terdeteksi: {gesture_symbol}")
        st.success(f"Teks: {text}")

        # Putar suara
        tts = gTTS(text=text, lang='id')
        sound_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(sound_path)
        st.audio(sound_path, format='audio/mp3', autoplay=True)

        os.remove(img_path)
    else:
        st.warning("Tidak ada tangan terdeteksi dalam gambar.")
