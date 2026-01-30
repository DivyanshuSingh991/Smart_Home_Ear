import streamlit as st
import requests
import time
import streamlit.components.v1 as components

# ---------------- PAGE CONFIGURATION 
st.set_page_config(
    page_title="Smart Home Ear – CCTV Control Room",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE 
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

# ---------------- CSS 
st.markdown("""
<style>
body { background:#020617; color:#e5e7eb; }

.block {
  background:#020617;
  border:1px solid #1e293b;
  border-radius:18px;
  padding:24px;
  margin-bottom:28px;
}

.header { font-size:2.3rem; font-weight:800; color:#f8fafc; }
.sub { color:#94a3b8; }

.status-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:16px;
}

.status-box {
  background:#020617;
  border:1px solid #1e293b;
  border-radius:14px;
  padding:18px;
  text-align:center;
  font-weight:600;
}

.green{color:#22c55e;} .blue{color:#38bdf8;} .yellow{color:#eab308;}

.camera-frame {
  border-radius:18px;
  padding:10px;
  border:2px solid #22c55e;
}

.alert { padding:22px; border-radius:16px; font-size:1.3rem; font-weight:700; text-align:center; }

.alert-safe { background:linear-gradient(135deg,#064e3b,#16a34a); color:white; }
.alert-alert { background:linear-gradient(135deg,#78350f,#f59e0b); color:white; }
.alert-danger {
  background:linear-gradient(135deg,#7f1d1d,#dc2626);
  color:white;
  animation:pulse 1.5s infinite;
}

/* FLOATING SOS */
.sos-float {
  position:fixed;
  bottom:22px;
  right:22px;
  z-index:9999;
}
.sos-panel {
  background:#020617;
  border:1px solid #dc2626;
  border-radius:12px;
  padding:10px;
  margin-bottom:10px;
  color:white;
  font-size:0.85rem;
  font-weight:600;
}
.sos-btn {
  width:90px;height:90px;border-radius:50%;
  border:none;
  background:radial-gradient(circle,#dc2626,#7f1d1d);
  color:white;
  font-weight:900;
  box-shadow:0 0 30px rgba(220,38,38,.8);
  animation:pulse 1.5s infinite;
}

@keyframes pulse {
  0%{box-shadow:0 0 0 0 rgba(220,38,38,.7);}
  70%{box-shadow:0 0 0 25px rgba(220,38,38,0);}
  100%{box-shadow:0 0 0 0 rgba(220,38,38,0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="block">
  <div class="header">🚨 Smart Home Ear – CCTV Control Room</div>
  <div class="sub">Real-Time Audio & Video Surveillance with AI Threat Detection</div>
</div>
""", unsafe_allow_html=True)

# ---------------- STATUS ----------------
st.markdown("""
<div class="status-grid block">
  <div class="status-box" style= " color:white;">📷 Camera<br><span class="green">ACTIVE</span></div>
  <div class="status-box" style= " color:white;">🎙️ Microphone<br><span class="green">ACTIVE</span></div>
  <div class="status-box"  style="color:white;">🧠 AI Model<br><span class="blue">READY</span></div>
  <div class="status-box" style="color:white;">🖥️ System<br><span class="yellow">MONITORING</span></div>
</div>
""", unsafe_allow_html=True)

# --- CCTV ----
st.markdown("""
<div class="block" style="border-left:6px solid #22c55e;">
<h3 style="color:white;font-weight:900;">📷 Live CCTV Feed – Camera 01</h3>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="camera-frame">', unsafe_allow_html=True)
st.camera_input(" ")
st.markdown('</div>', unsafe_allow_html=True)

# ---- CONTACTS 
st.markdown('<div class="block"><h3  style="color:#38bdf8;">👮 Police & Guardian Contacts</h3></div>', unsafe_allow_html=True)
police_number = st.text_input("🚓 Police Emergency Number", "112")
guardian_number = st.text_input("👨‍👩‍👧 Guardian Contact Number", "+91XXXXXXXXXX")

# MANUAL AUDIO 
st.markdown('<div class="block"><h3 style="color:white">🎧 Manual Audio Test</h3></div>', unsafe_allow_html=True)
audio_file = st.audio_input("Record audio")

danger_triggered = False
risk = None
confidence = 0
sound = ""

if audio_file and st.button("🔍 Analyze Audio"):
    files = {"file":("audio.wav", audio_file.getvalue(),"audio/wav")}
    res = requests.post("http://127.0.0.1:8000/predict", files=files).json()

    risk = res["risk"]
    confidence = res["confidence"]
    sound = res.get("sound","Detected Sound")

    if risk == "danger":
        danger_triggered = True
        st.session_state.monitoring = True   # 🔥 AUTO-ON

        st.markdown(f"""
        <div class="alert alert-danger">
        ⚠️ HIGH RISK DETECTED<br>
        {sound}<br>
        Confidence: {confidence}%<br>
        Immediate action recommended
        </div>
        """, unsafe_allow_html=True)

        st.audio("alert.wav", autoplay=True)

    elif risk == "alert":
        st.markdown(f"""
        <div class="alert alert-alert">
        🟡 POTENTIAL ALERT<br>
        Unusual Sound Detected<br>
        Confidence: {confidence}%
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="alert alert-safe">
        🟢 ENVIRONMENT SAFE<br>
        No Abnormal Sound Detected<br>
        Monitoring continuously • Last checked: Just now
        </div>
        """, unsafe_allow_html=True)

#  CONTINUOUS MONITOR (AUTO-ON)
if danger_triggered:
    st.markdown('<div class="block"><h3>🎙️ Continuous Audio Monitoring</h3></div>', unsafe_allow_html=True)

monitoring = st.toggle(
    "Keep Listening for Further Threats",
    key="monitoring"
)

if monitoring:
    st.info("Listening continuously for abnormal sounds...")
    time.sleep(2)

    try:
        data = requests.get("http://127.0.0.1:8000/live-check", timeout=5).json()
        if data["risk"] == "danger":
            st.audio("alert.wav", autoplay=True)
    except:
        st.warning("Monitoring service unavailable")

# ---------------- LIVE LOCATION ----------------
if danger_triggered:
    components.html("""
    <script>
    navigator.geolocation.getCurrentPosition(
      (pos)=>{
        const link=`https://maps.google.com/?q=${pos.coords.latitude},${pos.coords.longitude}`;
        document.body.innerHTML+=`<p style="color:white;font-size:0.85rem;">
        📍 Live Location: <a href="${link}" target="_blank" style="color:#38bdf8;">Open in Maps</a>
        </p>`;
      }
    );
    </script>
    """, height=60)

# - FLOATING SOS --
if danger_triggered and confidence > 85:
    st.markdown(f"""
    <div class="sos-float">
      <div class="sos-panel">
        ⚠️ HIGH RISK DETECTED<br>
        {sound}<br>
        Confidence: {confidence}%
      </div>
      <div class="sos-panel">
        📞 Police: {police_number}<br>
        📞 Guardian: {guardian_number}<br>
        📍 Share Live Location
      </div>
      <a href="tel:{guardian_number}">
        <button class="sos-btn">SOS</button>
      </a>
    </div>
    """, unsafe_allow_html=True)

#  FOOTER 
st.markdown("""
<div style="text-align:center;color:#38bdf8;;margin-top:40px;">
Smart Home Ear | AI-Powered CCTV Audio-Visual Security System<br>
CODE TRINITY • Anshita Singh | Anshika Pandey | Divyanshu Singh
</div>
""", unsafe_allow_html=True)