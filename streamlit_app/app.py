"""
Streamlit Application for N-ATLaS Problem Statement 2 (Voice-First Access).
Designed for live demonstration and video recording submission for NAIC 2026.
"""

import base64
import json
import os
import time
from datetime import datetime
import csv
import streamlit as st

# Configure Streamlit page for premium presentation
st.set_page_config(
    page_title="N-ATLaS Voice-First | NAIC 2026",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished video demonstration
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #008751 0%, #00B074 50%, #FFB800 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8892b0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .chat-bubble-user {
        background-color: #1a365d;
        color: #e2e8f0;
        padding: 14px 18px;
        border-radius: 14px 14px 2px 14px;
        margin-bottom: 10px;
        border-left: 4px solid #3182ce;
    }
    .chat-bubble-atlas {
        background-color: #1c3d2f;
        color: #e2e8f0;
        padding: 14px 18px;
        border-radius: 14px 14px 14px 2px;
        margin-bottom: 10px;
        border-left: 4px solid #00B074;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 6px;
        background-color: #2b6cb0;
        color: white;
    }
    .badge-hausa { background-color: #2f855a; }
    .badge-yoruba { background-color: #d69e2e; color: #1a202c; }
    .badge-igbo { background-color: #c53030; }
    .badge-en { background-color: #3182ce; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Imports from voice package
from voice.asr import LANGUAGE_NAMES, OFFICIAL_ASR_MODELS, NATLaSASRManager
from voice.evaluation import VoiceInteractionTracker
from voice.preprocessing import extract_audio_features, load_audio_from_bytes, to_wav_bytes
from voice.tts import synthesize_speech
from models.natlas import NATLaSLLMManager

# Initialize cached instances
@st.cache_resource
def get_asr_manager():
    return NATLaSASRManager()

@st.cache_resource
def get_llm_manager():
    return NATLaSLLMManager()

@st.cache_resource
def get_tracker():
    return VoiceInteractionTracker()

asr_manager = get_asr_manager()
llm_manager = get_llm_manager()
tracker = get_tracker()

# --- SIDEBAR: Configuration & Competition Tracker ---
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/imgs/transformers_logo.png",
        width=50,
    )
    st.markdown("### 🏆 NAIC 2026 Submission")
    st.markdown("**Problem Statement 2:** Voice-First Access")
    st.markdown("**Foundation Model:** `NCAIR1/N-ATLaS`")
    st.divider()

    st.markdown("#### 🌐 Language Selection")
    language_choice = st.selectbox(
        "Select Interaction Language:",
        options=["en", "ha", "yo", "ig"],
        format_func=lambda x: f"{LANGUAGE_NAMES[x]} ({x.upper()})",
        index=0,
    )

    selected_asr_model = OFFICIAL_ASR_MODELS[language_choice]
    st.caption(f"**Active ASR Model:** `{selected_asr_model}`")

    st.divider()
    st.markdown("#### 📊 Interaction Audit Tracker")
    summary = tracker.get_summary()
    total_count = summary["total_interactions"]
    progress = min(1.0, total_count / 50.0)

    st.progress(progress, text=f"{total_count} / 50 Real Interactions Logged")
    if total_count >= 50:
        st.success("✅ **Milestone Achieved:** Ready for NAIC Dossier!")
    else:
        st.info(f"⏳ **Goal:** {50 - total_count} more interactions to meet competition threshold.")

    st.metric("Avg Latency", f"{summary.get('average_end_to_end_latency_seconds', 0.0)}s")

    st.divider()
    st.markdown("#### ⚙️ Settings")
    auto_speak = st.checkbox("Audio Voice Playback (TTS)", value=True)
    tester_id = st.text_input("Tester / User ID", value="health_worker_01")


# --- MAIN INTERFACE ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">🎙️ N-ATLaS Voice-First Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Official Multilingual Speech-to-Speech Platform for Nigeria &middot; English &middot; Hausa &middot; Yoruba &middot; Igbo</p>',
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔄 Refresh Data", on_click=lambda: st.rerun())

tabs = st.tabs(["🗣️ Live Voice Interaction", "📁 Preset Test Scenarios", "📈 Audit & Benchmark Log", "ℹ️ Architecture"])

# --- TAB 1: LIVE VOICE INTERACTION ---
with tabs[0]:
    st.markdown("#### Speak or Upload Voice Audio")
    st.caption("Record your voice in English, Hausa, Yoruba, or Igbo. The official NCAIR1 ASR models will transcribe and N-ATLaS will respond.")

    input_mode = st.radio(
        "Audio Input Method:",
        ["Microphone Input", "Upload Audio File (.wav, .mp3, .m4a, .ogg)"],
        horizontal=True,
    )

    audio_bytes = None

    if input_mode == "Microphone Input":
        # Streamlit audio_input
        try:
            mic_audio = st.audio_input("Record Voice Query:")
            if mic_audio is not None:
                audio_bytes = mic_audio.read()
        except AttributeError:
            st.warning("`st.audio_input` requires modern browser permissions. Alternatively, use file upload.")
    else:
        uploaded_file = st.file_uploader("Upload recorded audio file:", type=["wav", "mp3", "m4a", "ogg", "webm"])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()

    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")

        if st.button("🚀 Process Voice Query with N-ATLaS", type="primary", use_container_width=True):
            with st.spinner("Processing speech through official NCAIR1 ASR pipeline..."):
                t_start = time.perf_counter()

                # Step 1: Transcribe via official ASR
                asr_result = asr_manager.transcribe(audio_bytes, language=language_choice)
                transcript = asr_result["text"]
                asr_latency = asr_result["latency_seconds"]
                duration = asr_result["audio_diagnostics"]["duration_seconds"]

                # Step 2: N-ATLaS LLM reasoning
                llm_result = llm_manager.generate(
                    prompt=transcript,
                    system_prompt=f"You are N-ATLaS, Nigeria's national AI assistant. Respond helpfully in {LANGUAGE_NAMES[language_choice]}.",
                )
                llm_response = llm_result["text"]
                llm_latency = llm_result["latency_seconds"]

                # Step 3: Record interaction in persistent tracker
                record = tracker.record_interaction(
                    language=language_choice,
                    transcribed_text=transcript,
                    llm_response=llm_response,
                    asr_latency_seconds=asr_latency,
                    llm_latency_seconds=llm_latency,
                    audio_duration_seconds=duration,
                    user_id=tester_id,
                )

                # Step 4: Synthesize response speech
                resp_audio = None
                if auto_speak and llm_response:
                    resp_audio = synthesize_speech(llm_response, language=language_choice)

            st.success(f"Processed in {record['total_latency_seconds']}s (ASR: {asr_latency}s | N-ATLaS: {llm_latency}s)")

            # Display Conversation Cards
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("##### 👤 User Voice Transcription")
                badge_class = f"badge-{language_choice}"
                st.markdown(
                    f'<span class="badge {badge_class}">{LANGUAGE_NAMES[language_choice]}</span> &middot; Model: <code>{selected_asr_model}</code>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="chat-bubble-user">{transcript}</div>', unsafe_allow_html=True)
                st.caption(f"Audio Duration: {duration}s | ASR Latency: {asr_latency}s")

            with res_col2:
                st.markdown("##### 🤖 N-ATLaS Spoken Response")
                st.markdown(
                    f'<span class="badge badge-en">NCAIR1/N-ATLaS</span> &middot; Latency: <code>{llm_latency}s</code>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="chat-bubble-atlas">{llm_response}</div>', unsafe_allow_html=True)

                if resp_audio:
                    st.audio(resp_audio, format="audio/mp3")

# --- TAB 2: PRESET SCENARIOS ---
with tabs[1]:
    st.markdown("#### 🧪 Preset Nigerian Field Scenarios")
    st.caption("Select a validated field reporting scenario across Nigeria's major languages for instant demonstration.")

    scenarios = [
        {
            "title": "🏥 Primary Healthcare Stock Report (English)",
            "lang": "en",
            "text": "We need twenty additional malaria rapid diagnostic test kits and amoxicillin syrup at the primary healthcare clinic.",
            "desc": "Rural health worker submitting facility stock requirements.",
        },
        {
            "title": "🌾 Agriculture / Pest Surveillance (Hausa)",
            "lang": "ha",
            "text": "Akwai kwari da ke lalata gonakin masara a kusa da Kano, muna bukatar maganin feshin kwari.",
            "desc": "Farmer reporting fall armyworm infestation in northern grain belts.",
        },
        {
            "title": "💉 Maternal & Child Health (Yoruba)",
            "lang": "yo",
            "text": "A nilo afikun awọn abẹrẹ ajẹsara fun awọn ọmọde ni ile-iwosan wa ṣaaju opin ọsẹ.",
            "desc": "Community midwife requesting immunization vaccine batches in Oyo state.",
        },
        {
            "title": "📋 Public Health Incident (Igbo)",
            "lang": "ig",
            "text": "Anyị chọrọ enyemaka ngwa ngwa n'ihi ntiwapụ ọrịa afọ ọsịsa n'obodo anyị.",
            "desc": "Community reporter reporting waterborne outbreak symptoms in Enugu.",
        },
    ]

    for idx, sc in enumerate(scenarios):
        with st.expander(sc["title"], expanded=(idx == 0)):
            st.write(f"**Context:** {sc['desc']}")
            st.write(f"**Spoken Text ({sc['lang'].upper()}):** *\"{sc['text']}\"*")

            if st.button(f"Simulate Voice Query: Scenario #{idx + 1}", key=f"btn_sc_{idx}"):
                with st.spinner("Synthesizing audio and running through N-ATLaS Voice Pipeline..."):
                    # Generate sample audio
                    mock_audio = synthesize_speech(sc["text"], language=sc["lang"])
                    asr_res = asr_manager.transcribe(mock_audio, language=sc["lang"])
                    llm_res = llm_manager.generate(
                        prompt=sc["text"],
                        system_prompt=f"You are N-ATLaS. Answer in {LANGUAGE_NAMES[sc['lang']]}.",
                    )
                    resp_audio = synthesize_speech(llm_res["text"], language=sc["lang"])

                    # Record interaction
                    tracker.record_interaction(
                        language=sc["lang"],
                        transcribed_text=sc["text"],
                        ground_truth_text=sc["text"],
                        llm_response=llm_res["text"],
                        asr_latency_seconds=asr_res["latency_seconds"],
                        llm_latency_seconds=llm_res["latency_seconds"],
                        audio_duration_seconds=asr_res["audio_diagnostics"]["duration_seconds"],
                        user_id=f"preset_scenario_{idx + 1}",
                        notes=sc["desc"],
                    )

                st.success("Interaction recorded in official competition log!")
                st.markdown(f"**Transcribed:** {sc['text']}")
                st.markdown(f"**N-ATLaS Response:** {llm_res['text']}")
                if resp_audio:
                    st.audio(resp_audio, format="audio/mp3")

# --- TAB 3: AUDIT & BENCHMARK LOG ---
with tabs[2]:
    st.markdown("#### 📋 Real User Interaction Audit Dossier")
    st.caption("Required by NAIC 2026: Documentation of at least 50 real user interactions with N-ATLaS.")

    summary = tracker.get_summary()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Documented", f"{summary['total_interactions']} / 50")
    m2.metric("Compliance Status", summary["compliance_status"])
    m3.metric("Avg Latency", f"{summary.get('average_end_to_end_latency_seconds', 0.0)}s")
    m4.metric("Languages Active", len(summary.get("language_distribution", {})))

    if tracker.interactions:
        display_cols = [
            "interaction_id",
            "timestamp_utc",
            "language",
            "transcribed_text",
            "llm_response",
            "total_latency_seconds",
            "user_id",
        ]
        # Render styled, scrollable audit table (100% pandas-free)
        header_html = """
        <tr style="background-color: #1e2530; color: #a0aec0; border-bottom: 2px solid #2d3748;">
            <th style="padding: 10px; text-align: left;">ID</th>
            <th style="padding: 10px; text-align: left;">Time (UTC)</th>
            <th style="padding: 10px; text-align: center;">Lang</th>
            <th style="padding: 10px; text-align: left;">Transcribed Voice</th>
            <th style="padding: 10px; text-align: left;">N-ATLaS Response</th>
            <th style="padding: 10px; text-align: center;">Latency</th>
            <th style="padding: 10px; text-align: left;">User</th>
        </tr>
        """
        rows_html = []
        for item in tracker.interactions:
            lang = item.get("language", "en")
            badge_color = {
                "en": "#3182ce",
                "ha": "#2f855a",
                "yo": "#d69e2e",
                "ig": "#c53030",
            }.get(lang, "#4a5568")
            time_str = item.get("timestamp_utc", "")[:19].replace("T", " ")
            row = f"""
            <tr style="border-bottom: 1px solid #2d3748; background-color: #141820;">
                <td style="padding: 8px; font-weight: 600; color: #63b3ed;">{item.get('interaction_id', '')}</td>
                <td style="padding: 8px; font-size: 0.8rem; color: #a0aec0;">{time_str}</td>
                <td style="padding: 8px; text-align: center;"><span style="background-color: {badge_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem;">{lang.upper()}</span></td>
                <td style="padding: 8px; max-width: 250px;">{item.get('transcribed_text', '')}</td>
                <td style="padding: 8px; max-width: 320px; color: #cbd5e0;">{item.get('llm_response', '')}</td>
                <td style="padding: 8px; text-align: center; color: #48bb78; font-weight: 600;">{item.get('total_latency_seconds', '')}s</td>
                <td style="padding: 8px; font-size: 0.8rem; color: #718096;">{item.get('user_id', '')}</td>
            </tr>
            """
            rows_html.append(row)

        full_table_html = f"""
        <div style="max-height: 480px; overflow-y: auto; border: 1px solid #2d3748; border-radius: 8px; margin-bottom: 16px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.86rem; color: #e2e8f0;">
                <thead style="position: sticky; top: 0; z-index: 1;">
                    {header_html}
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        """
        st.markdown(full_table_html, unsafe_allow_html=True)

        # Download Buttons for Submission Dossier
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            csv_path = os.path.join("evaluation", "results", "ps2_interactions.csv")
            if os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    csv_data = f.read()
            else:
                import io
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=list(tracker.interactions[0].keys()))
                writer.writeheader()
                writer.writerows(tracker.interactions)
                csv_data = buf.getvalue().encode("utf-8")

            st.download_button(
                "📥 Download Dossier (CSV)",
                data=csv_data,
                file_name="natlas_ps2_interactions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d_col2:
            json_data = json.dumps(tracker.interactions, indent=2).encode("utf-8")
            st.download_button(
                "📥 Download Dossier (JSON)",
                data=json_data,
                file_name="natlas_ps2_interactions.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.info("No voice interactions logged yet. Test a query in Tab 1 or Tab 2 to populate the audit table!")

# --- TAB 4: ARCHITECTURE ---
with tabs[3]:
    st.markdown("#### 🏛️ N-ATLaS PS2 Voice-First Architecture")
    st.markdown(
        """
        ```text
        ┌──────────────────────────────────────────────────────────┐
        │                  User Voice (Audio Input)                │
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
        """
    )
