import streamlit as st
import cv2
import numpy as np
import base64
import time
import tempfile
import os
import threading
from gtts import gTTS
import streamlit.components.v1 as components

# ---------- Page config ----------
st.set_page_config(page_title="Gesture Voice Recognition (Live JS Camera)", layout="centered")
st.title("🖐️ Gesture Voice Recognition — Live Camera (Mirror, Stable Version)")

st.markdown("""
🎥 Kamera langsung di halaman (tanpa WebRTC).  
Frame dikirim ke Python setiap 1.8 detik untuk deteksi gesture dan TTS.  
Gunakan Chrome / Edge untuk hasil terbaik.  
""")

# ---------- User inputs ----------
col1, col2 = st.columns(2)
with col1:
    kata1 = st.text_input("✋ Semua jari terbuka", "Halo semuanya!")
    kata2 = st.text_input("👍 Hanya jempol", "Saya senang hari ini!")
with col2:
    kata3 = st.text_input("✌️ Telunjuk & tengah", "Nama saya Enrico!")
    kata4 = st.text_input("🤘 Metal", "Sampai jumpa lagi!")

GESTURES = {"✋": kata1, "👍": kata2, "✌️": kata3, "🤘": kata4}

# ---------- TTS helper ----------
def speak_async(text):
    def _run():
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                gTTS(text=text, lang="id").save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
                time.sleep(1.5)
                os.remove(fp.name)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()

# ---------- Gesture detection ----------
def detect_gesture_from_bgr(img_bgr):
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (25, 25), 0)
        _, thresh = cv2.threshold(blur, 70, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return ""
        contour = max(contours, key=cv2.contourArea)
        area_contour = cv2.contourArea(contour)
        if area_contour < 2000:
            return ""
        hull = cv2.convexHull(contour)
        area_hull = cv2.contourArea(hull)
        area_ratio = ((area_hull - area_contour) / area_contour) * 100 if area_contour > 0 else 0

        if area_ratio < 12:
            return "✋"
        elif 12 <= area_ratio < 28:
            return "👍"
        elif 28 <= area_ratio < 48:
            return "✌️"
        else:
            return "🤘"
    except Exception:
        return ""

# ---------- Camera component ----------
interval_ms = 1800
html_code = f"""
<div style="display:flex;flex-direction:column;align-items:center;">
  <video id="video" autoplay playsinline style="transform: scaleX(-1); width:640px; height:auto; border:1px solid #ddd;"></video>
  <div style="margin-top:8px;">
    <button id="startBtn">Start Camera</button>
    <button id="stopBtn">Stop Camera</button>
  </div>
  <div id="status" style="margin-top:6px;color:#666">Status: camera stopped</div>
</div>

<script>
const video = document.getElementById('video');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');
let stream = null;
let captureInterval = null;

async function startCamera() {{
  try {{
    stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "user" }}, audio: false }});
    video.srcObject = stream;
    status.innerText = "Status: camera running";
    captureInterval = setInterval(captureFrame, {interval_ms});
  }} catch (err) {{
    status.innerText = "Status: error: " + err.message;
  }}
}}

function stopCamera() {{
  if (captureInterval) {{
    clearInterval(captureInterval);
    captureInterval = null;
  }}
  if (stream) {{
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }}
  video.srcObject = null;
  status.innerText = "Status: camera stopped";
  if (window.Streamlit && window.Streamlit.setComponentValue) {{
    window.Streamlit.setComponentValue(null);
  }}
}}

function captureFrame() {{
  if (!stream) return;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
  if (window.Streamlit && window.Streamlit.setComponentValue) {{
    window.Streamlit.setComponentValue(dataUrl);
  }}
}}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
</script>
"""

component = components.html(html_code, height=560)

# ---------- Safely handle returned value ----------
component_value = None
try:
    # Streamlit sometimes returns dict/None/DeltaGenerator, so we force convert to str safely
    if isinstance(component, str):
        component_value = component
    elif hasattr(component, "_value"):
        component_value = str(component._value)
except Exception:
    component_value = None

# ---------- Process frame ----------
if isinstance(component_value, str) and component_value.startswith("data:image"):
    try:
        header, b64data = component_value.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gesture = detect_gesture_from_bgr(img)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption=f"Gesture: {gesture or 'Tidak dikenali'}",
                 use_column_width=True)

        if "last_gesture" not in st.session_state:
            st.session_state.last_gesture = ""
            st.session_state.last_time = 0

        current_time = time.time()
        if gesture and gesture != st.session_state.last_gesture and (current_time - st.session_state.last_time) > 2:
            st.session_state.last_gesture = gesture
            st.session_state.last_time = current_time
            text = GESTURES.get(gesture, "")
            if text:
                st.success(f"🔊 Mengucapkan: {text}")
                speak_async(text)
    except Exception as e:
        st.error(f"Gagal memproses frame: {e}")
else:
    st.info("Belum menerima frame dari kamera. Tekan 'Start Camera' dan izinkan akses kamera.")
