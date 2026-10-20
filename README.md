# 🧠 NeuroShield AI
### Privacy-Preserving Multimodal Mental Health Risk Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red?style=flat-square&logo=streamlit)
![Federated Learning](https://img.shields.io/badge/Federated%20Learning-FedAvg-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Overview

**NeuroShield AI** is a real-time, privacy-preserving multimodal AI system designed to assess mental health risk in individuals — particularly for use in psychiatric pre-visit screening. It fuses three input modalities — **facial expressions**, **voice stress**, and **text sentiment** — into a single unified risk score using a federated learning architecture.

> **Problem it solves:** In India, a significant fraction of people suffering from mental illness hesitate to share their condition even with doctors. NeuroShield AI provides a confidential, on-device screening tool where **no raw personal data ever leaves the user's device** — only aggregated model weight updates are transmitted.

---

## 🎯 Key Features

- 🎭 **Facial Emotion Analysis** — Real-time emotion detection using DeepFace (VGG-Face CNN)
- 🎙️ **Voice Stress Analysis** — Real audio feature extraction using Librosa (MFCCs, pitch, ZCR, spectral features)
- 💬 **Text Sentiment & Emotion** — NLP analysis using DistilBERT (SST-2) and DistilRoBERTa (GoEmotions)
- 🔒 **Federated Learning (FedAvg)** — Privacy-preserving model training across multiple clinic clients
- 🧩 **Multimodal Fusion** — Weighted combination of all three modalities into a unified risk score
- 🔍 **Explainable AI (XAI)** — Human-readable per-modality reasoning for every prediction
- 📊 **Session Analytics** — Real-time tracking of risk scores across multiple runs
- ⚙️ **Configurable Fusion Weights** — Adjustable per-modality weights via sidebar sliders

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NeuroShield AI                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Module 01   │  │  Module 02   │  │    Module 03     │  │
│  │    FACE      │  │    VOICE     │  │      TEXT        │  │
│  │  DeepFace    │  │   Librosa    │  │  DistilBERT +    │  │
│  │  VGG-Face    │  │  Real MFCCs  │  │ DistilRoBERTa   │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬──────────┘  │
│         │                 │                   │             │
│         └─────────────────┼───────────────────┘             │
│                           ▼                                 │
│              ┌────────────────────────┐                     │
│              │   Multimodal Fusion    │                     │
│              │  Score = Σ wᵢ × riskᵢ │                     │
│              └────────────┬───────────┘                     │
│                           ▼                                 │
│              ┌────────────────────────┐                     │
│              │    Risk Assessment     │                     │
│              │  LOW / MEDIUM / HIGH   │                     │
│              └────────────┬───────────┘                     │
│                           ▼                                 │
│              ┌────────────────────────┐                     │
│              │    XAI Report          │                     │
│              │  Per-modality reasons  │                     │
│              └────────────────────────┘                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Federated Learning Layer                │   │
│  │  Clinic-A ──┐                                       │   │
│  │  Clinic-B ──┼──► FedAvg ──► Global Model Update    │   │
│  │  Clinic-C ──┘                                       │   │
│  │  (Only model weights transmitted — NOT raw data)    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Models Used

| Module | Model | Purpose |
|--------|-------|---------|
| Face | DeepFace (VGG-Face + MTCNN) | Emotion detection, age, gender |
| Voice | Librosa | Real MFCC/pitch/spectral feature extraction |
| Text (Sentiment) | DistilBERT (SST-2) | Positive/Negative sentiment classification |
| Text (Emotion) | DistilRoBERTa (GoEmotions) | 7-class emotion classification |
| Federated | Custom FedAvg | Privacy-preserving logistic regression calibrator |

---

## 🔒 Federated Learning — How It Works

NeuroShield AI implements **Federated Averaging (FedAvg)** — the standard privacy-preserving distributed learning algorithm.

### Key Principle:
> **Raw data NEVER leaves the client's device. Only model weight updates are aggregated.**

### FedAvg Algorithm:

```
1. Global model initialised with zero weights
2. For each federated round:
   a. Each clinic client (Clinic-A, Clinic-B, Clinic-C) trains locally
      on their own data for N epochs using gradient descent
   b. Local model updates (weights + bias) computed
   c. Weighted aggregation on server:
      global_weights = Σ (nᵢ / N_total) × local_weights_i
      where nᵢ = number of examples at client i
   d. Global model updated — ready for next round
3. No patient data, no audio, no images ever transmitted
```

### Privacy Guarantees:
- ✅ Raw facial frames — processed locally, NOT transmitted
- ✅ Real audio bytes — processed locally, NOT transmitted
- ✅ Raw text input — processed locally, NOT transmitted
- ✅ Only local model updates and sample counts aggregated via FedAvg

### Federated Client Setup:

```python
# Three simulated clinic clients with different data distributions
clients = make_demo_clients()
# clinic-a: 48 samples, local distribution shift = -0.18
# clinic-b: 48 samples, local distribution shift =  0.00  
# clinic-c: 48 samples, local distribution shift = +0.18

# Run FedAvg aggregation
global_weights, global_bias, metrics = run_federated_round(
    weights, bias, clients, epochs=5, learning_rate=0.35
)
```

---

## 🎙️ Voice Analysis — Real Librosa Pipeline

The voice module performs **genuine audio feature extraction** — not simulated values.

### Features Extracted:

| Feature | Description |
|---------|-------------|
| MFCCs (40 coefficients) | Mel-Frequency Cepstral Coefficients — speech fingerprint |
| Delta MFCCs | First-order MFCC derivatives — speech dynamics |
| Delta-Delta MFCCs | Second-order derivatives — acceleration |
| Zero Crossing Rate (ZCR) | Rate of sign changes — related to pitch and noise |
| Spectral Centroid | Centre of mass of spectrum — brightness indicator |
| Spectral Rolloff (85%) | Frequency below which 85% of energy is contained |
| Spectral Bandwidth | Spread of spectrum around centroid |
| RMS Energy | Root Mean Square energy — loudness measure |
| Pitch (F0) — YIN algorithm | Fundamental frequency — stress correlate |
| Pitch Std Dev | Variability of pitch — emotional arousal indicator |

### Risk Score Computation:

```python
# Normalise to [0,1] using speech stress research ranges
# (Schuller et al. 2013, El Ayadi et al. 2011)
risk = (0.25 × zcr_normalised      +   # zero crossing rate
        0.20 × rms_normalised       +   # energy
        0.20 × pitch_mean_norm      +   # fundamental frequency
        0.20 × pitch_variance_norm  +   # pitch variability
        0.15 × centroid_normalised)     # spectral centroid
```

---

## 🎭 Facial Analysis

Uses **DeepFace** with **VGG-Face** backbone and **MTCNN** face detection.

### Emotion Classes and Risk Mapping:

| Emotion | Risk Score |
|---------|-----------|
| Angry | 0.88 |
| Fear | 0.82 |
| Sad | 0.68 |
| Disgust | 0.62 |
| Surprise | 0.42 |
| Neutral | 0.18 |
| Happy | 0.05 |

---

## 💬 Text Analysis

### Two-Stage NLP Pipeline:

**Stage 1 — Sentiment (DistilBERT SST-2):**
```
Input text → DistilBERT → POSITIVE/NEGATIVE + confidence score
```

**Stage 2 — Emotion (DistilRoBERTa GoEmotions):**
```
Input text → DistilRoBERTa → 7 emotion scores
[anger, fear, sadness, disgust, surprise, neutral, joy]
```

**Combined Text Risk:**
```python
text_risk = 0.5 × sentiment_risk + 0.5 × emotion_risk
```

---

## ⚖️ Multimodal Fusion

Final risk score is a weighted combination of all three modalities:

```
Fusion Score = face_risk × w_face + voice_risk × w_voice + text_risk × w_text

Default weights:
  w_face  = 0.40  (facial expression most reliable)
  w_voice = 0.30  (voice stress analysis)
  w_text  = 0.30  (self-reported text)

Risk Thresholds:
  score ≥ 0.65  →  HIGH RISK   🔴
  score ≥ 0.35  →  MEDIUM RISK 🟡
  score < 0.35  →  LOW RISK    🟢
```

All weights and thresholds are configurable via the sidebar.

---

## 📁 Project Structure

```
Fed_project/
│
├── app.py                    # Main Streamlit application
│   ├── analyze_voice_real()  # Real Librosa audio feature extraction
│   ├── analyze_face()        # DeepFace facial emotion analysis  
│   ├── analyze_text()        # DistilBERT + DistilRoBERTa NLP pipeline
│   └── Multimodal fusion     # Weighted risk score computation
│
├── federated_learning.py     # FedAvg implementation
│   ├── FederatedClient       # Dataclass for clinic client
│   ├── make_demo_clients()   # Creates 3 simulated clinic clients
│   ├── _local_update()       # Local gradient descent training
│   └── run_federated_round() # FedAvg aggregation
│
├── NeuroShield.ipynb         # Development notebook
│
└── README.md                 # This file
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/MehtaSanjana18/NeuroShield-AI.git
cd NeuroShield-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
streamlit>=1.33.0
numpy
opencv-python
librosa
soundfile
deepface
transformers
torch
scikit-learn
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🖥️ Usage

### Step 1 — Facial Analysis (Module 01)
- Allow camera access
- Click the camera button to capture your photo
- DeepFace will detect emotions in real time

### Step 2 — Voice Analysis (Module 02)
- Click the microphone button
- Speak naturally for 3–5 seconds
- Librosa extracts real audio features (MFCCs, pitch, ZCR, etc.)

### Step 3 — Text Analysis (Module 03)
- Type how you are feeling in the text box
- DistilBERT and DistilRoBERTa analyse sentiment and emotions

### Step 4 — Run Fusion
- Adjust modality weights in the sidebar if needed
- Click **⚡ RUN MULTIMODAL FUSION ANALYSIS**
- View the unified risk score and XAI explanation

### Step 5 — Federated Learning (Optional)
- Set number of FedAvg rounds in sidebar
- Click **Run local FedAvg**
- Watch global model weights update across clinic clients

---

## 📊 Output

### Risk Assessment:
```
🔴 HIGH RISK   — score ≥ 0.65 — Immediate psychiatric consultation recommended
🟡 MEDIUM RISK — score ≥ 0.35 — Follow-up screening suggested  
🟢 LOW RISK    — score < 0.35 — No immediate concern detected
```

### XAI Report Example:
```
FACE   → DeepFace: sad (72.3% conf) → risk 0.680 × weight 0.40 = 0.2720
VOICE  → Librosa: pitch 185 Hz (std 42), ZCR 0.089 → risk 0.620 × 0.30 = 0.1860
TEXT   → DistilBERT: NEGATIVE (89.2%), DistilRoBERTa: sadness → risk 0.710 × 0.30 = 0.2130
─────────────────────────────────────────────────────────────
Fusion = 0.2720 + 0.1860 + 0.2130 = 0.6710 → HIGH RISK
```

---

## 🔬 Research Context

This project was developed as a **Final Year BTech Project** exploring:

- **Federated Learning** for healthcare privacy — FedAvg algorithm
- **Multimodal AI** — combining vision, audio, and text modalities
- **Explainable AI** — per-modality reasoning for clinical use
- **Mental health screening** — pre-visit psychiatrist tool

### Why Federated Learning for Mental Health?

Mental health data is among the most sensitive personal data. Traditional centralised ML requires uploading patient data to a server — a major privacy barrier that prevents adoption in clinical settings. Federated Learning solves this by:

1. Training on local device only
2. Transmitting only model weight gradients
3. Aggregating updates without ever seeing raw patient data
4. Enabling collaborative learning across multiple clinics without data sharing

---

## ⚠️ Disclaimer

> NeuroShield AI is a **research prototype** and is **NOT a medical diagnostic tool**. It is designed for research and pre-screening purposes only. All risk assessments should be reviewed by qualified mental health professionals. Do not make clinical decisions based solely on this system's output.

---

## 👩‍💻 Author

**Sanjana Mehta**
- 📧 mehtasanjana137@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/sanjana-mehta-64440728b/)
- 💻 [GitHub](https://github.com/MehtaSanjana18)
- 🎓 MTech AI — IIT Gandhinagar

---

## 📄 License

This project is licensed under the MIT License.

---

*Built with ❤️ for mental health awareness and privacy-preserving AI*
