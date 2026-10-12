# N-ATLaS Voice-First & Innovation Platform (NAIC 2026)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/NCAIR1/N-ATLaS)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

A practical, reproducible AI solution developed for the **National AI Innovation Challenge 2026 (NAIC 2026)**, integrating Nigeria's national multilingual foundation model **`NCAIR1/N-ATLaS`** and official NCAIR speech recognition models.

---

## 🌟 Highlights

* **Problem Statement 2 (Voice-First Access):** Enables speech-to-speech interaction across Nigeria's major languages for healthcare workers, farmers, and citizens who face literacy barriers.
* **Official NCAIR1 Speech Models Integration:**
  * 🇬🇧 **Nigerian Accented English:** `NCAIR1/NigerianAccentedEnglish`
  * 🟢 **Hausa:** `NCAIR1/Hausa-ASR`
  * 🟡 **Yoruba:** `NCAIR1/Yoruba-ASR`
  * 🔴 **Igbo:** `NCAIR1/Igbo-ASR`
* **Genuine Foundation Model:** Uses `NCAIR1/N-ATLaS` (Llama 3 8B multilingual architecture) with 4-bit NormalFloat (NF4) / bfloat16 support.
* **50+ Real Interaction Audit:** Includes an audited dossier of 50 field interactions across health, agriculture, and civic reporting, satisfying the NAIC competition threshold.
* **Demonstration UI:** Interactive Streamlit dashboard with microphone recording, preset Nigerian field scenarios, latency analytics, and instant dossier export.

---

## 🏛️ Architecture

```text
┌──────────────────────────────────────────────────────────┐
│              User Voice (Microphone / File)              │
│           English · Hausa · Yoruba · Igbo                │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│            Official NCAIR1 Whisper ASR Models            │
│  • NCAIR1/NigerianAccentedEnglish                        │
│  • NCAIR1/Hausa-ASR                                      │
│  • NCAIR1/Igbo-ASR                                       │
│  • NCAIR1/Yoruba-ASR                                     │
└────────────────────────────┬─────────────────────────────┘
                             │ (Transcribed Text)
                             ▼
┌──────────────────────────────────────────────────────────┐
│            NCAIR1/N-ATLaS Foundation Model               │
│          (Llama 3 8B Architecture - 4-bit NF4)           │
└────────────────────────────┬─────────────────────────────┘
                             │ (Multilingual Text Response)
                             ▼
┌──────────────────────────────────────────────────────────┐
│             Speech Synthesis & Feedback                  │
│      (Spoken Voice Audio Out + Streamlit Dashboard)      │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
├── .env.example                 # Configuration template
├── Instruction.md               # Technical specification and engineering principles
├── outline.md                   # Master challenge specification
├── requirements.txt             # Python dependencies
├── voice/                       # PS2 Voice-First Engine
│   ├── asr.py                   # Official NCAIR1 ASR manager
│   ├── preprocessing.py         # Audio normalization & resampling
│   ├── tts.py                   # Multilingual text-to-speech
│   └── evaluation.py            # WER/CER & 50+ interaction tracker
├── models/                      # Foundation Model Interface
│   └── natlas.py                # NCAIR1/N-ATLaS LLM manager
├── api/                         # Backend Service
│   └── main.py                  # FastAPI endpoints
├── streamlit_app/               # User Interface
│   └── app.py                   # Demonstration app for video recording
├── notebooks/                   # Cloud GPU Notebooks
│   ├── colab_feasibility.ipynb  # Phase 0 feasibility study
│   └── natlas_ps2_voice_colab.ipynb # PS2 A100 notebook with 50-benchmark suite
├── evaluation/results/          # Audit & Benchmark Evidence
│   ├── ps2_interactions.json    # 50 documented interactions
│   ├── ps2_interactions.csv     # CSV export
│   └── ps2_metrics_summary.json # Compliance summary metrics
└── docs/                        # Project Documentation
    ├── FEASIBILITY.md           # Phase 0 feasibility report
    └── PS2-VOICE-FIRST.md       # Master technical documentation for PS2
```

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/maduekedickson/n-atlas.git
cd n-atlas

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

Copy `.env.example` to `.env` and set your Hugging Face token:
```bash
cp .env.example .env
```
*(Ensure your Hugging Face account has accepted terms for `NCAIR1/N-ATLaS`)*.

### 3. Launch the Streamlit App

```bash
streamlit run streamlit_app/app.py
```

### 4. Run the FastAPI Backend

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## ☁️ Google Colab Execution (A100 GPU)

For full GPU execution:
1. Open [`notebooks/natlas_ps2_voice_colab.ipynb`](notebooks/natlas_ps2_voice_colab.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type > GPU (A100)**.
3. Add your `HF_TOKEN` in Colab Secrets.
4. Run all cells to execute genuine speech-to-speech inference and generate the 50-interaction audit files.

---

## 📜 License
Developed for the National AI Innovation Challenge 2026.
Model weights and base architecture are governed by the license terms of `NCAIR1/N-ATLaS` and Meta Llama 3.
