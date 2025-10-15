import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from gtts import gTTS
import os
import time
import threading
from playsound import playsound

# === Fungsi bicara di thread terpisah ===
def speak(text):
    def run_speech():
        try:
            filename = "temp_voice.mp3"
            tts = gTTS(text=text, lang='id')
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("Error suara:", e)
    threading.Thread(target=run_speech, daemon=True).start()

# === Konfigurasi halaman Streamlit ===
st.set_page_config(page_title="Gesture Voice Recognition 🤖", layout="centered")
st.title("🖐️ Gesture Voice Recognition Berbasis Simbol Jari")
st.markdown("""
Masukkan 4 kata yang akan diucapkan sesuai simbol jari berikut:
1. ✋ = Semua jari terbuka  
2. 👍 = Hanya jempol terbuka  
3. ✌️ = Telunjuk & tengah terbuka  
4. 🤘 = Metal (jempol, telunjuk, dan kelingking terbuka)
""")

col1, col2 = st.columns(2)
with col1:
    kata1 = st.text_input("✋ (Semua jari terbuka)", "Halo semuanya!")
    kata2 = st.text_input("👍 (Hanya jempol)", "Saya senang hari ini!")
with col2:
    kata3 = st.text_input("✌️ (Telunjuk & Tengah)", "Nama saya Enrico!")
    kata4 = st.text_input("🤘 (Metal)", "Sampai jumpa lagi!")

start_button = st.button("🚀 Mulai Kamera")

# === Jalankan kamera ===
if start_button:
    st.warning("Tekan tombol **Q** di jendela kamera untuk keluar.")
    st.info("Sedang menjalankan kamera... pastikan pencahayaan cukup ✨")

    gestures_dict = {
        "✋": kata1,
        "👍": kata2,
        "✌️": kata3,
        "🤘": kata4
    }

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    last_gesture = ""
    last_time = time.time()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img)

        gesture_symbol = ""
        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)

            # === Deteksi simbol jari berdasarkan pola ===
            if fingers == [1, 1, 1, 1, 1]:
                gesture_symbol = "✋"
            elif fingers == [1, 0, 0, 0, 0]:
                gesture_symbol = "👍"
            elif fingers == [0, 1, 1, 0, 0]:
                gesture_symbol = "✌️"
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0:
                gesture_symbol = "🤘"

            # === Jika gesture berubah, keluarkan suara ===
            current_time = time.time()
            if gesture_symbol and gesture_symbol != last_gesture and (current_time - last_time) > 1:
                last_gesture = gesture_symbol
                last_time = current_time

                teks = gestures_dict.get(gesture_symbol, "")
                if teks:
                    speak(teks)

            cv2.putText(img, f"Gesture: {gesture_symbol}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Tidak ada tangan terdeteksi", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Gesture Voice Recognition 🤖", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success("Kamera telah dimatikan ✅")
