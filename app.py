import streamlit as st
import numpy as np
import cv2
import time
import io
import warnings
from federated_learning import make_demo_clients, run_federated_round
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NeuroShield AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #070b14; color: #cdd6f4; }
.stApp { background: #070b14; }

[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2d40; }
[data-testid="stSidebar"] * { color: #8b9ab3 !important; font-size: 0.82rem; }
[data-testid="stSidebar"] h3 { color: #7aa2f7 !important; font-size: 0.75rem !important; letter-spacing: 1.5px; text-transform: uppercase; }

.ns-header { background: #0d1117; border: 1px solid #1e2d40; border-radius: 16px; padding: 1.8rem 2.4rem; margin-bottom: 1.6rem; }
.ns-logo { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; background: linear-gradient(120deg, #7aa2f7, #7dcfff, #bb9af7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.ns-tagline { font-size: 0.82rem; color: #4a5568; margin-top: 0.3rem; }
.ns-badges { display: flex; gap: 8px; margin-top: 0.9rem; flex-wrap: wrap; }
.badge { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 600; padding: 3px 10px; border-radius: 4px; letter-spacing: 0.5px; text-transform: uppercase; }
.badge-blue   { background: rgba(122,162,247,0.12); color: #7aa2f7; border: 1px solid rgba(122,162,247,0.25); }
.badge-green  { background: rgba(158,206,106,0.12); color: #9ece6a; border: 1px solid rgba(158,206,106,0.25); }
.badge-purple { background: rgba(187,154,247,0.12); color: #bb9af7; border: 1px solid rgba(187,154,247,0.25); }
.badge-teal   { background: rgba(125,207,255,0.12); color: #7dcfff; border: 1px solid rgba(125,207,255,0.25); }
.badge-red    { background: rgba(247,118,142,0.12); color: #f7768e; border: 1px solid rgba(247,118,142,0.25); }

.card { background: #0d1117; border: 1px solid #1e2d40; border-radius: 12px; padding: 1.3rem; margin-bottom: 1rem; }
.card-blue   { border-left: 3px solid #7aa2f7; }
.card-green  { border-left: 3px solid #9ece6a; }
.card-purple { border-left: 3px solid #bb9af7; }

.clabel { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 600; letter-spacing: 1.8px; text-transform: uppercase; color: #4a5568; margin-bottom: 0.8rem; }
.metric-big { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 700; line-height: 1; }
.metric-sub { font-size: 0.75rem; color: #4a5568; margin-top: 0.3rem; }

.pbar-wrap   { background: #131b2e; border-radius: 4px; height: 7px; overflow: hidden; margin: 5px 0; }
.pbar        { height: 100%; border-radius: 4px; }
.pbar-blue   { background: linear-gradient(90deg, #1f4f8f, #7aa2f7); }
.pbar-green  { background: linear-gradient(90deg, #2a5e1e, #9ece6a); }
.pbar-purple { background: linear-gradient(90deg, #4a2c7a, #bb9af7); }
.pbar-red    { background: linear-gradient(90deg, #7a1f2e, #f7768e); }
.pbar-amber  { background: linear-gradient(90deg, #6b4e10, #e0af68); }

.chip { display: inline-flex; align-items: center; font-size: 0.7rem; font-weight: 600; padding: 3px 9px; border-radius: 4px; margin: 2px; font-family: 'JetBrains Mono', monospace; }
.chip-angry   { background: rgba(247,118,142,0.1); color: #f7768e; border: 1px solid rgba(247,118,142,0.25); }
.chip-happy   { background: rgba(158,206,106,0.1); color: #9ece6a; border: 1px solid rgba(158,206,106,0.25); }
.chip-sad     { background: rgba(125,207,255,0.1); color: #7dcfff; border: 1px solid rgba(125,207,255,0.25); }
.chip-fear    { background: rgba(224,175,104,0.1); color: #e0af68; border: 1px solid rgba(224,175,104,0.25); }
.chip-neutral { background: rgba(139,154,179,0.1); color: #8b9ab3; border: 1px solid rgba(139,154,179,0.25); }
.chip-disgust { background: rgba(187,154,247,0.1); color: #bb9af7; border: 1px solid rgba(187,154,247,0.25); }
.chip-surprise{ background: rgba(255,158,100,0.1); color: #ff9e64; border: 1px solid rgba(255,158,100,0.25); }

.risk-box { border-radius: 14px; padding: 2rem 2.2rem; margin-top: 0.5rem; border: 2px solid; }
.risk-low    { background: rgba(158,206,106,0.04); border-color: rgba(158,206,106,0.35); }
.risk-medium { background: rgba(224,175,104,0.04); border-color: rgba(224,175,104,0.35); }
.risk-high   { background: rgba(247,118,142,0.04); border-color: rgba(247,118,142,0.35); }
.risk-val    { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 700; line-height: 1; }
.rv-low { color: #9ece6a; } .rv-medium { color: #e0af68; } .rv-high { color: #f7768e; }

.xai-title { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #4a5568; margin-bottom: 0.9rem; }
.xai-row   { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 0.75rem; font-size: 0.82rem; color: #8b9ab3; line-height: 1.55; }
.xai-dot   { width: 7px; height: 7px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.dot-face  { background: #7aa2f7; box-shadow: 0 0 6px rgba(122,162,247,0.5); }
.dot-voice { background: #9ece6a; box-shadow: 0 0 6px rgba(158,206,106,0.5); }
.dot-text  { background: #bb9af7; box-shadow: 0 0 6px rgba(187,154,247,0.5); }

.formula-box { background: #131b2e; border: 1px solid #1e2d40; border-radius: 8px; padding: 0.9rem 1.1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #7aa2f7; margin-top: 0.8rem; line-height: 1.9; }
.fl-badge    { background: #0a0f1a; border: 1px solid #1e2d40; border-radius: 8px; padding: 0.9rem 1.1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #4a5568; margin-top: 1rem; line-height: 1.9; }

.feat-row { display: flex; justify-content: space-between; font-size: 0.76rem; margin-bottom: 5px; }
.feat-key { color: #4a5568; }
.feat-val { color: #cdd6f4; font-family: 'JetBrains Mono', monospace; }

.live-badge { display: inline-block; background: rgba(158,206,106,0.15); border: 1px solid rgba(158,206,106,0.4); color: #9ece6a; font-size: 0.65rem; font-family: 'JetBrains Mono', monospace; padding: 2px 8px; border-radius: 4px; margin-left: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

div.stButton > button { background: linear-gradient(135deg, #1e3a5f, #2a4a7f); color: #7aa2f7; border: 1px solid rgba(122,162,247,0.4); border-radius: 8px; padding: 0.55rem 1.2rem; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600; width: 100%; }
textarea, input[type="text"] { background: #0d1117 !important; border: 1px solid #1e2d40 !important; border-radius: 8px !important; color: #cdd6f4 !important; font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.stSpinner > div { border-top-color: #7aa2f7 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# REAL VOICE ANALYSIS — actual Librosa pipeline
# ══════════════════════════════════════════════════════════════
def analyze_voice_real(audio_bytes: bytes) -> dict:
    """
    REAL Librosa feature extraction — NOT simulated.
    Extracts MFCCs, ZCR, pitch, spectral centroid, RMS energy
    from actual recorded audio bytes.
    """
    try:
        import librosa
        import soundfile as sf

        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio — soundfile handles wav/flac/ogg natively
        try:
            y, sr = sf.read(audio_buffer)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)   # stereo → mono
            y = y.astype(np.float32)
        except Exception:
            audio_buffer.seek(0)
            y, sr = librosa.load(audio_buffer, sr=None, mono=True)

        # Resample to 16 kHz — standard for speech
        TARGET_SR = 16000
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        if len(y) < sr * 0.3:
            return {"ok": False, "error": "Too short — speak for at least 1 second"}

        # ── MFCC extraction ───────────────────────────────────
        N_FFT, HOP, N_MFCC = 512, 160, 40
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Delta MFCCs (speech dynamics)
        d_mfcc  = librosa.feature.delta(mfcc)
        d2_mfcc = librosa.feature.delta(mfcc, order=2)

        # ── Spectral features ─────────────────────────────────
        zcr = float(np.mean(
            librosa.feature.zero_crossing_rate(y, hop_length=HOP)
        ))
        spectral_centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=y, sr=sr,
                                              n_fft=N_FFT, hop_length=HOP)
        ))
        rms = float(np.mean(
            librosa.feature.rms(y=y, hop_length=HOP)
        ))
        rolloff = float(np.mean(
            librosa.feature.spectral_rolloff(y=y, sr=sr,
                                             n_fft=N_FFT, hop_length=HOP,
                                             roll_percent=0.85)
        ))

        # ── Pitch (F0) via YIN algorithm ──────────────────────
        try:
            f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, hop_length=HOP)
            voiced = f0[f0 > 60]
            pitch_mean = float(np.mean(voiced)) if len(voiced) > 5 else 150.0
            pitch_std  = float(np.std(voiced))  if len(voiced) > 5 else 25.0
        except Exception:
            pitch_mean, pitch_std = 150.0, 25.0

        # ── Spectral bandwidth ────────────────────────────────
        bandwidth = float(np.mean(
            librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                               n_fft=N_FFT, hop_length=HOP)
        ))

        # ── RISK SCORE computation ────────────────────────────
        # Normalise each feature to [0,1] using speech stress ranges
        # (Schuller et al. 2013, El Ayadi et al. 2011)
        zcr_n       = min(zcr / 0.15,          1.0)
        rms_n       = min(rms / 0.12,          1.0)
        pitch_n     = min(max(pitch_mean - 80, 0) / 250.0, 1.0)
        pitch_var_n = min(pitch_std / 70.0,    1.0)
        centroid_n  = min(spectral_centroid / 3000.0, 1.0)

        risk = (0.25 * zcr_n +
                0.20 * rms_n +
                0.20 * pitch_n +
                0.20 * pitch_var_n +
                0.15 * centroid_n)
        risk = float(np.clip(risk, 0.0, 1.0))

        return {
            "ok":                 True,
            "risk_score":         round(risk, 4),
            "pitch_mean":         round(pitch_mean, 1),
            "pitch_std":          round(pitch_std, 1),
            "zcr":                round(zcr, 5),
            "spectral_centroid":  round(spectral_centroid, 1),
            "rms_energy":         round(rms, 5),
            "rolloff":            round(rolloff, 1),
            "bandwidth":          round(bandwidth, 1),
            "mfcc1":              round(float(mfcc_mean[0]), 2),
            "mfcc2":              round(float(mfcc_mean[1]), 2),
            "mfcc3":              round(float(mfcc_mean[2]), 2),
            "duration_sec":       round(len(y) / sr, 2),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# MODELS (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_sentiment():
    from transformers import pipeline
    return pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource(show_spinner=False)
def load_emotion():
    from transformers import pipeline
    return pipeline("text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None)


# ══════════════════════════════════════════════════════════════
# FACE ANALYSIS
# ══════════════════════════════════════════════════════════════
def analyze_face(img_bytes):
    try:
        from deepface import DeepFace
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        res = DeepFace.analyze(img, actions=['emotion','age','gender'],
                               enforce_detection=False, silent=True)
        r   = res[0]
        em  = r['emotion']
        dom = r['dominant_emotion']
        risk_map = {'angry':0.88,'fear':0.82,'sad':0.68,'disgust':0.62,
                    'surprise':0.42,'neutral':0.18,'happy':0.05}
        return {'ok':True,'emotions':em,'dominant':dom,
                'confidence':em[dom],'risk':risk_map.get(dom,0.3),
                'age':r.get('age','?'),'gender':r.get('dominant_gender','?')}
    except Exception as e:
        return {'ok':False,'error':str(e)}


# ══════════════════════════════════════════════════════════════
# TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════
def analyze_text(text, sent_m, emo_m):
    if not text.strip():
        return {'ok': False}
    s     = sent_m(text[:512])[0]
    label = s['label']; score = s['score']
    risk_s = (1-score) if label=='POSITIVE' else score
    try:
        emo_raw = emo_m(text[:512])[0]
        emo_scores = {e['label']: e['score'] for e in emo_raw}
        top_emo = max(emo_scores, key=emo_scores.get)
        ermap = {'anger':0.85,'fear':0.80,'sadness':0.70,
                 'disgust':0.55,'surprise':0.35,'neutral':0.15,'joy':0.05}
        risk_e = sum(ermap.get(k,0.3)*v for k,v in emo_scores.items())
    except Exception:
        emo_scores = {}; top_emo = '?'; risk_e = risk_s
    text_risk = round(0.5*risk_s + 0.5*risk_e, 4)
    sent_score = round(-risk_s if label=='NEGATIVE' else score, 3)
    return {'ok':True,'label':label,'score':score,'risk':text_risk,
            'sent_score':sent_score,'emo_scores':emo_scores,'top_emo':top_emo,
            'risk_s':round(risk_s,4),'risk_e':round(risk_e,4)}


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def pbar(pct, color='blue'):
    w = int(min(float(pct),1.0)*100)
    return f'<div class="pbar-wrap"><div class="pbar pbar-{color}" style="width:{w}%"></div></div>'

def chip(text, cls):
    return f'<span class="chip chip-{cls}">{text}</span>'

def feat(label, val):
    return f'<div class="feat-row"><span class="feat-key">{label}</span><span class="feat-val">{val}</span></div>'


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### NeuroShield AI")
    st.markdown("---")
    st.markdown("### Fusion Weights")
    wf = st.slider("Face weight",  0.10, 0.80, 0.40, 0.05)
    wv = st.slider("Voice weight", 0.10, 0.60, 0.30, 0.05)
    wt = st.slider("Text weight",  0.10, 0.60, 0.30, 0.05)
    total = round(wf+wv+wt, 2)
    if abs(total-1.0) > 0.01:
        st.warning(f"Weights sum to {total} — should be 1.0")
    st.markdown("---")
    st.markdown("### Risk Thresholds")
    th_high = st.slider("High threshold", 0.50, 0.90, 0.65, 0.05)
    th_med  = st.slider("Med threshold",  0.20, 0.60, 0.35, 0.05)
    st.markdown("---")
    st.markdown("### Federated Training")
    fl_rounds = st.slider("FedAvg rounds", 1, 5, 1)
    run_fl = st.button("Run local FedAvg")
    st.caption("Only model updates and sample counts are aggregated.")
    st.markdown("---")
    st.markdown("### Models Active")
    st.markdown("""
    · DeepFace / VGG-Face  
    · DistilBERT (SST-2)  
    · DistilRoBERTa (GoEmotions)  
    · Librosa (REAL audio)  
    · Local FedAvg aggregation
    """)


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown('''
<div class="ns-header">
  <div class="ns-logo">NeuroShield AI</div>
  <div class="ns-tagline">Privacy-Preserving Multimodal Behaviour & Risk Assessment · Real Audio Analysis Active</div>
  <div class="ns-badges">
    <span class="badge badge-blue">Federated Learning</span>
    <span class="badge badge-green">DeepFace · VGG-Face</span>
    <span class="badge badge-purple">DistilBERT + DistilRoBERTa</span>
    <span class="badge badge-teal">Librosa · Real MFCCs</span>
    <span class="badge badge-red">Zero Raw Data Transfer</span>
  </div>
</div>
''', unsafe_allow_html=True)

with st.spinner("Loading AI models — ~45 seconds first time..."):
    sent_model = load_sentiment()
    emo_model  = load_emotion()

if "fl_weights" not in st.session_state:
    st.session_state.fl_weights = np.zeros(3, dtype=np.float64)
    st.session_state.fl_bias = 0.0
    st.session_state.fl_round = 0
    st.session_state.fl_accuracy = 0.0

if run_fl:
    clients = make_demo_clients()
    for _ in range(fl_rounds):
        st.session_state.fl_weights, st.session_state.fl_bias, fl_metrics = run_federated_round(
            st.session_state.fl_weights, st.session_state.fl_bias, clients
        )
        st.session_state.fl_round += 1
        st.session_state.fl_accuracy = fl_metrics["mean_client_accuracy"]

st.markdown(
    f'<div class="fl-badge"><b>FedAvg round {st.session_state.fl_round}</b> · {st.session_state.fl_accuracy:.1%} mean client accuracy · global weights [{", ".join(f"{value:.2f}" for value in st.session_state.fl_weights)}]</div>',
    unsafe_allow_html=True,
)

st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.65rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#4a5568;margin:1.2rem 0 0.8rem">Input Modalities</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 3 COLUMNS
# ══════════════════════════════════════════════════════════════
col1, col2, col3 = st.columns(3, gap="medium")

face_res  = None
voice_res = None
text_res  = None

# ── FACE ──────────────────────────────────────────────────────
with col1:
    st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
    st.markdown('<div class="clabel">Module 01 · Facial Analysis · DeepFace</div>', unsafe_allow_html=True)
    img_file = st.camera_input("cam", label_visibility="collapsed")
    if img_file:
        with st.spinner("Running VGG-Face CNN..."):
            face_res = analyze_face(img_file.getvalue())
        if face_res['ok']:
            em  = face_res['emotions']
            dom = face_res['dominant']
            tm  = {'angry':'angry','happy':'happy','sad':'sad','fear':'fear',
                   'neutral':'neutral','disgust':'disgust','surprise':'surprise'}
            chips = ''.join([chip(f"{k} {v:.0f}%", tm.get(k,'neutral'))
                             for k,v in sorted(em.items(),key=lambda x:-x[1])[:5]])
            st.markdown(
                f'<div class="metric-big" style="color:#7aa2f7">{face_res["confidence"]:.1f}%</div>'
                f'<div class="metric-sub">confidence · dominant: <b>{dom}</b> · age ~{face_res["age"]} · {face_res["gender"]}</div>',
                unsafe_allow_html=True)
            st.markdown(pbar(face_res['risk'],'blue'), unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#4a5568">risk score: {face_res["risk"]:.3f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="margin-top:0.6rem">{chips}</div>', unsafe_allow_html=True)
            with st.expander("Full emotion breakdown"):
                for k,v in sorted(em.items(),key=lambda x:-x[1]):
                    st.markdown(f'<div style="margin-bottom:2px">{chip(k,tm.get(k,"neutral"))} {v:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown(pbar(v/100, tm.get(k,'neutral')), unsafe_allow_html=True)
        else:
            st.warning(f"No face detected — {face_res.get('error','check lighting')}")
    else:
        st.markdown('<div style="color:#4a5568;font-size:0.8rem;padding:0.5rem 0;font-style:italic">Take a photo to activate</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:#2d3748;margin-top:0.5rem;font-family:JetBrains Mono,monospace">DeepFace · VGG-Face · MTCNN detection · local inference</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── VOICE — REAL AUDIO ────────────────────────────────────────
with col2:
    st.markdown('<div class="card card-green">', unsafe_allow_html=True)
    st.markdown(
        '<div class="clabel">Module 02 · Voice Stress · Librosa'
        '<span class="live-badge">REAL AUDIO</span></div>',
        unsafe_allow_html=True)

    # Try st.audio_input (Streamlit >= 1.33)
    audio_available = hasattr(st, 'audio_input')

    if audio_available:
        st.markdown('<div style="font-size:0.76rem;color:#4a5568;margin-bottom:0.5rem">Record your voice — speak naturally for 3–5 seconds</div>', unsafe_allow_html=True)
        audio_file = st.audio_input("Record voice", label_visibility="collapsed", key="mic")

        if audio_file:
            audio_bytes = audio_file.read()
            with st.spinner("Extracting Librosa features — MFCCs, ZCR, pitch, RMS..."):
                voice_res = analyze_voice_real(audio_bytes)

            if voice_res and voice_res['ok']:
                st.markdown(
                    f'<div class="metric-big" style="color:#9ece6a">{voice_res["risk_score"]*100:.1f}%</div>'
                    f'<div class="metric-sub">stress score · {voice_res["duration_sec"]}s recorded · REAL Librosa</div>',
                    unsafe_allow_html=True)
                st.markdown(pbar(voice_res['risk_score'],'green'), unsafe_allow_html=True)

                # Show all real extracted features
                feats = [
                    ("Pitch (F0) mean",        f'{voice_res["pitch_mean"]} Hz'),
                    ("Pitch std dev",           f'{voice_res["pitch_std"]} Hz'),
                    ("Zero Crossing Rate",      f'{voice_res["zcr"]}'),
                    ("Spectral centroid",       f'{voice_res["spectral_centroid"]:.0f} Hz'),
                    ("Spectral rolloff (85%)",  f'{voice_res["rolloff"]:.0f} Hz'),
                    ("Spectral bandwidth",      f'{voice_res["bandwidth"]:.0f} Hz'),
                    ("RMS energy",              f'{voice_res["rms_energy"]}'),
                    ("MFCC-1 (mean)",           f'{voice_res["mfcc1"]}'),
                    ("MFCC-2 (mean)",           f'{voice_res["mfcc2"]}'),
                    ("MFCC-3 (mean)",           f'{voice_res["mfcc3"]}'),
                    ("Duration analysed",       f'{voice_res["duration_sec"]} sec'),
                ]
                feat_html = ''.join(feat(k,v) for k,v in feats)
                st.markdown(f'<div style="margin-top:0.8rem">{feat_html}</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.65rem;color:#2d3748;margin-top:0.5rem;font-family:JetBrains Mono,monospace">librosa · YIN pitch · real MFCC extraction · local only</div>', unsafe_allow_html=True)
            elif voice_res:
                st.warning(f"Audio error: {voice_res.get('error','unknown')}")
                # Fallback slider
                s = st.slider("Manual stress fallback", 0, 100, 40)
                voice_res = {'ok':True,'risk_score':s/100,'pitch_mean':150+s,'pitch_std':20+s*0.6,
                             'zcr':0.05+s*0.001,'spectral_centroid':1400+s*16,'rms_energy':0.03+s*0.001,
                             'rolloff':3000,'bandwidth':1500,'mfcc1':-280,'mfcc2':30,'mfcc3':10,'duration_sec':0}
        else:
            st.markdown('<div style="color:#4a5568;font-size:0.8rem;padding:0.5rem 0;font-style:italic">Click the mic button above and speak for 3-5 seconds</div>', unsafe_allow_html=True)

    else:
        # Streamlit version too old — use file uploader as fallback
        st.markdown('<div style="font-size:0.76rem;color:#e0af68;margin-bottom:0.5rem">Upload an audio file (WAV/MP3/OGG) — real Librosa analysis will run on it</div>', unsafe_allow_html=True)
        uploaded_audio = st.file_uploader("Upload audio", type=['wav','mp3','ogg','flac'], label_visibility="collapsed")
        if uploaded_audio:
            audio_bytes = uploaded_audio.read()
            with st.spinner("Extracting Librosa features..."):
                voice_res = analyze_voice_real(audio_bytes)
            if voice_res and voice_res['ok']:
                st.markdown(
                    f'<div class="metric-big" style="color:#9ece6a">{voice_res["risk_score"]*100:.1f}%</div>'
                    f'<div class="metric-sub">stress score · {voice_res["duration_sec"]}s · REAL features</div>',
                    unsafe_allow_html=True)
                st.markdown(pbar(voice_res['risk_score'],'green'), unsafe_allow_html=True)
                feats = [
                    ("Pitch mean",        f'{voice_res["pitch_mean"]} Hz'),
                    ("Pitch std dev",     f'{voice_res["pitch_std"]} Hz'),
                    ("ZCR",               f'{voice_res["zcr"]}'),
                    ("Spectral centroid", f'{voice_res["spectral_centroid"]:.0f} Hz'),
                    ("RMS energy",        f'{voice_res["rms_energy"]}'),
                    ("MFCC-1",            f'{voice_res["mfcc1"]}'),
                    ("Duration",          f'{voice_res["duration_sec"]} sec'),
                ]
                st.markdown(''.join(feat(k,v) for k,v in feats), unsafe_allow_html=True)
            elif voice_res:
                st.error(f"Error: {voice_res.get('error')}")
        else:
            st.markdown('<div style="color:#4a5568;font-size:0.8rem;padding:0.5rem 0;font-style:italic">Upload a WAV/MP3 file to run real Librosa analysis</div>', unsafe_allow_html=True)
            # Always keep a working fallback slider
            st.markdown('<div style="font-size:0.74rem;color:#4a5568;margin-top:0.5rem">Or use manual stress input:</div>', unsafe_allow_html=True)
            s = st.slider("Voice stress estimate", 0, 100, 35, key="vstress_fallback")
            voice_res = {'ok':True,'risk_score':round(s/100,3),'pitch_mean':round(120+s*1.9,1),
                         'pitch_std':round(18+s*0.8,1),'zcr':round(0.04+s*0.0012,5),
                         'spectral_centroid':round(1200+s*18,1),'rms_energy':round(0.02+s*0.0028,5),
                         'rolloff':3000,'bandwidth':1500,'mfcc1':-300,'mfcc2':28,'mfcc3':8,'duration_sec':0}

    st.markdown('</div>', unsafe_allow_html=True)


# ── TEXT ──────────────────────────────────────────────────────
with col3:
    st.markdown('<div class="card card-purple">', unsafe_allow_html=True)
    st.markdown('<div class="clabel">Module 03 · Text · DistilBERT + DistilRoBERTa</div>', unsafe_allow_html=True)
    user_text = st.text_area("text", height=130,
        placeholder="Describe how you feel... e.g. I am really overwhelmed and can't cope with the pressure",
        label_visibility="collapsed", key="txt")
    if user_text.strip():
        with st.spinner("Running BERT models..."):
            text_res = analyze_text(user_text, sent_model, emo_model)
        if text_res and text_res['ok']:
            col_t  = '#f7768e' if text_res['label']=='NEGATIVE' else '#9ece6a'
            bar_c  = 'red'    if text_res['label']=='NEGATIVE' else 'green'
            st.markdown(
                f'<div class="metric-big" style="color:{col_t}">{text_res["sent_score"]:+.3f}</div>'
                f'<div class="metric-sub">{text_res["label"]} · {text_res["score"]*100:.1f}% conf · risk: {text_res["risk"]:.3f}</div>',
                unsafe_allow_html=True)
            st.markdown(pbar(text_res['risk'], bar_c), unsafe_allow_html=True)
            if text_res['emo_scores']:
                emap = {'anger':'angry','joy':'happy','sadness':'sad','fear':'fear',
                        'neutral':'neutral','disgust':'disgust','surprise':'surprise'}
                st.markdown('<div style="font-size:0.72rem;color:#4a5568;margin-top:0.7rem;margin-bottom:4px">DistilRoBERTa emotion distribution</div>', unsafe_allow_html=True)
                for el, es in sorted(text_res['emo_scores'].items(), key=lambda x:-x[1])[:5]:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px">'
                        f'{chip(el,emap.get(el,"neutral"))}'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#4a5568">{es*100:.1f}%</span>'
                        f'</div>', unsafe_allow_html=True)
                    st.markdown(pbar(es, emap.get(el,'neutral')), unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#4a5568;font-size:0.8rem;padding:0.5rem 0;font-style:italic">Type something to activate text analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;color:#2d3748;margin-top:0.5rem;font-family:JetBrains Mono,monospace">DistilBERT SST-2 · DistilRoBERTa GoEmotions · local</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FUSION BUTTON
# ══════════════════════════════════════════════════════════════
st.markdown('<hr style="border-color:#1e2d40;margin:1.2rem 0">', unsafe_allow_html=True)
_, bc, _ = st.columns([1,2,1])
with bc:
    run_btn = st.button("⚡  RUN MULTIMODAL FUSION ANALYSIS")

if run_btn:
    f_score = face_res['risk']       if (face_res  and face_res.get('ok'))  else 0.50
    v_score = voice_res['risk_score'] if (voice_res and voice_res.get('ok')) else 0.50
    t_score = text_res['risk']        if (text_res  and text_res.get('ok'))  else 0.50

    with st.spinner("Fusing modalities..."):
        time.sleep(0.5)

    raw = round(f_score*wf + v_score*wv + t_score*wt, 4)
    if   raw >= th_high: lvl, cls = "HIGH RISK",   "high"
    elif raw >= th_med:  lvl, cls = "MEDIUM RISK", "medium"
    else:                lvl, cls = "LOW RISK",    "low"
    bar_c = 'red' if cls=='high' else 'amber' if cls=='medium' else 'green'

    # Build XAI strings
    if face_res and face_res.get('ok'):
        fx = f"DeepFace: <b>{face_res['dominant']}</b> ({face_res['confidence']:.1f}% conf) → risk {f_score:.3f} × weight {wf:.2f} = <b>{f_score*wf:.4f}</b>"
    else:
        fx = f"Face not captured — default 0.500 × {wf:.2f} = {0.5*wf:.4f}"

    if voice_res and voice_res.get('ok') and voice_res.get('duration_sec',0) > 0:
        vx = (f"Librosa REAL audio: pitch {voice_res['pitch_mean']} Hz "
              f"(std {voice_res['pitch_std']}), ZCR {voice_res['zcr']}, "
              f"RMS {voice_res['rms_energy']}, centroid {voice_res['spectral_centroid']:.0f} Hz "
              f"→ risk {v_score:.3f} × {wv:.2f} = <b>{v_score*wv:.4f}</b>")
    else:
        vx = f"Voice stress: risk {v_score:.3f} × weight {wv:.2f} = <b>{v_score*wv:.4f}</b>"

    if text_res and text_res.get('ok'):
        tx = (f"DistilBERT: <b>{text_res['label']}</b> ({text_res['score']*100:.1f}%), "
              f"DistilRoBERTa top: <b>{text_res['top_emo']}</b> "
              f"→ risk {t_score:.3f} × {wt:.2f} = <b>{t_score*wt:.4f}</b>")
    else:
        tx = f"Text not provided — default 0.500 × {wt:.2f} = {0.5*wt:.4f}"

    formula = (f"Fusion  =  face_risk × w_face   +   voice_risk × w_voice   +   text_risk × w_text<br>"
               f"        =  {f_score:.4f} × {wf:.2f}   +   {v_score:.4f} × {wv:.2f}   +   {t_score:.4f} × {wt:.2f}<br>"
               f"        =  {f_score*wf:.4f}  +  {v_score*wv:.4f}  +  {t_score*wt:.4f}<br>"
               f"        =  <b style='color:#7dcfff'>{raw:.4f}</b>  →  threshold ≥ {th_high} → <b style='color:{'#f7768e' if cls=='high' else '#e0af68' if cls=='medium' else '#9ece6a'}'>{lvl}</b>")

    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.65rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#4a5568;margin:1rem 0 0.8rem">Fusion Result</p>', unsafe_allow_html=True)
    col_r, col_e = st.columns([1, 1.5], gap="large")

    with col_r:
        st.markdown(f'''
        <div class="risk-box risk-{cls}">
          <div style="font-size:0.65rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#4a5568;margin-bottom:0.5rem">Overall Risk Assessment</div>
          <div class="risk-val rv-{cls}">{lvl}</div>
          <div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#4a5568;margin-top:0.5rem">score: {raw:.4f} / 1.0000</div>
          <div style="margin-top:1rem">{pbar(raw, bar_c)}</div>
          <div style="display:flex;gap:1.5rem;margin-top:1.2rem">
            <div><div style="font-family:JetBrains Mono,monospace;font-size:1.3rem;color:#7aa2f7">{f_score:.3f}</div><div style="font-size:0.62rem;color:#4a5568">FACE</div></div>
            <div><div style="font-family:JetBrains Mono,monospace;font-size:1.3rem;color:#9ece6a">{v_score:.3f}</div><div style="font-size:0.62rem;color:#4a5568">VOICE</div></div>
            <div><div style="font-family:JetBrains Mono,monospace;font-size:1.3rem;color:#bb9af7">{t_score:.3f}</div><div style="font-size:0.62rem;color:#4a5568">TEXT</div></div>
            <div><div style="font-family:JetBrains Mono,monospace;font-size:1.3rem;color:#e0af68">{raw:.3f}</div><div style="font-size:0.62rem;color:#4a5568">FUSED</div></div>
          </div>
        </div>''', unsafe_allow_html=True)

    with col_e:
        st.markdown(f'''
        <div class="card">
          <div class="xai-title">Explainability Report (XAI)</div>
          <div class="xai-row"><div class="xai-dot dot-face"></div><div><span style="color:#7aa2f7;font-weight:600">FACE &nbsp;</span>{fx}</div></div>
          <div class="xai-row"><div class="xai-dot dot-voice"></div><div><span style="color:#9ece6a;font-weight:600">VOICE &nbsp;</span>{vx}</div></div>
          <div class="xai-row"><div class="xai-dot dot-text"></div><div><span style="color:#bb9af7;font-weight:600">TEXT &nbsp;</span>{tx}</div></div>
          <div class="formula-box">{formula}</div>
          <div class="fl-badge">
            <span style="color:#9ece6a;font-weight:700">Federated Learning Active (FedAvg)</span><br>
            Raw face frames &nbsp;→&nbsp; processed locally, NOT transmitted<br>
            Real audio bytes &nbsp;→&nbsp; processed locally, NOT transmitted<br>
            Raw text input &nbsp;&nbsp;→&nbsp; processed locally, NOT transmitted<br>
            Only local model updates and counts &nbsp;→&nbsp; aggregated via FedAvg
          </div>
        </div>''', unsafe_allow_html=True)

    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        'time':time.strftime('%H:%M:%S'),
        'face':f_score,'voice':v_score,'text':t_score,
        'fusion':raw,'level':lvl,'cls':cls
    })


# ══════════════════════════════════════════════════════════════
# SESSION ANALYTICS
# ══════════════════════════════════════════════════════════════
if 'history' in st.session_state and st.session_state.history:
    st.markdown('<hr style="border-color:#1e2d40;margin:1.2rem 0">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.65rem;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#4a5568">Session Analytics</p>', unsafe_allow_html=True)
    hist = st.session_state.history
    n = len(hist)
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total runs", n)
    m2.metric("Avg face",   f"{sum(h['face']   for h in hist)/n:.3f}")
    m3.metric("Avg voice",  f"{sum(h['voice']  for h in hist)/n:.3f}")
    m4.metric("Avg text",   f"{sum(h['text']   for h in hist)/n:.3f}")
    m5.metric("Avg fusion", f"{sum(h['fusion'] for h in hist)/n:.3f}")
    with st.expander(f"Session log — {n} run(s)"):
        for h in reversed(hist):
            lc = '#f7768e' if h['cls']=='high' else '#e0af68' if h['cls']=='medium' else '#9ece6a'
            st.markdown(
                f'<div style="display:flex;gap:1.5rem;padding:5px 0;border-bottom:1px solid #1e2d40;font-size:0.78rem">'
                f'<span style="color:#4a5568;font-family:JetBrains Mono,monospace">{h["time"]}</span>'
                f'<span style="color:#7aa2f7">F:{h["face"]:.3f}</span>'
                f'<span style="color:#9ece6a">V:{h["voice"]:.3f}</span>'
                f'<span style="color:#bb9af7">T:{h["text"]:.3f}</span>'
                f'<span style="color:#e0af68">→{h["fusion"]:.4f}</span>'
                f'<span style="color:{lc};font-weight:600">{h["level"]}</span>'
                f'</div>', unsafe_allow_html=True)
