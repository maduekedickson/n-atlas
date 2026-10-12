"""
FastAPI Backend Server for N-ATLaS PS2: Voice-First Access.
Provides REST endpoints for official speech recognition, voice chat, TTS synthesis,
and NAIC 2026 interaction logging.
"""

import base64
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from models.natlas import NATLaSLLMManager
from voice.asr import NATLaSASRManager
from voice.evaluation import VoiceInteractionTracker
from voice.tts import synthesize_speech

# Load environment configuration
load_dotenv()

app = FastAPI(
    title="N-ATLaS PS2 Voice-First Access API",
    description="Official speech recognition and multilingual voice interaction powered by NCAIR1 N-ATLaS and Whisper ASR models.",
    version="1.0.0",
)

# Enable CORS for local Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
asr_manager = NATLaSASRManager()
llm_manager = NATLaSLLMManager()
interaction_tracker = VoiceInteractionTracker()


class TextPromptRequest(BaseModel):
    prompt: str
    language: Optional[str] = "en"
    system_prompt: Optional[str] = None
    synthesize_audio: Optional[bool] = True


@app.get("/health")
def health_check():
    """Health check endpoint with system telemetry."""
    summary = interaction_tracker.get_summary()
    return {
        "status": "healthy",
        "competition_track": "Problem Statement 2 (Voice-First Access)",
        "foundation_model": os.getenv("MODEL_NAME", "NCAIR1/N-ATLaS"),
        "asr_models": {
            "en": "NCAIR1/NigerianAccentedEnglish",
            "ha": "NCAIR1/Hausa-ASR",
            "ig": "NCAIR1/Igbo-ASR",
            "yo": "NCAIR1/Yoruba-ASR",
        },
        "device": asr_manager.device,
        "interactions_recorded": summary["total_interactions"],
        "compliance_status": summary["compliance_status"],
    }


@app.post("/voice/transcribe")
async def transcribe_voice(
    file: UploadFile = File(...),
    language: str = Form("en"),
):
    """
    Transcribes audio using the official NCAIR1 ASR model for the selected language.
    """
    try:
        audio_bytes = await file.read()
        result = asr_manager.transcribe(audio_bytes, language=language)
        return {
            "status": "success",
            "filename": file.filename,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/voice/chat")
async def voice_chat_pipeline(
    file: UploadFile = File(...),
    language: str = Form("en"),
    user_id: str = Form("tester_01"),
    notes: str = Form(""),
    synthesize_voice_response: bool = Form(True),
):
    """
    End-to-End Voice Pipeline:
      1. Receives audio from user.
      2. Transcribes via official NCAIR1 ASR model.
      3. Passes transcript to N-ATLaS LLM.
      4. Synthesizes voice audio response.
      5. Logs the interaction to satisfy NAIC 2026 >= 50 interaction audit.
    """
    try:
        audio_bytes = await file.read()

        # Step 1: Speech-To-Text (Official ASR)
        asr_result = asr_manager.transcribe(audio_bytes, language=language)
        transcription_text = asr_result["text"]
        asr_latency = asr_result["latency_seconds"]
        audio_duration = asr_result["audio_diagnostics"]["duration_seconds"]

        # Step 2: Language Model Reasoning (N-ATLaS)
        llm_result = llm_manager.generate(
            prompt=transcription_text,
            system_prompt="You are N-ATLaS, Nigeria's national AI assistant for public health and community intelligence.",
        )
        llm_response_text = llm_result["text"]
        llm_latency = llm_result["latency_seconds"]

        # Step 3: Text-To-Speech Output
        audio_b64 = None
        if synthesize_voice_response and llm_response_text:
            response_audio_bytes = synthesize_speech(llm_response_text, language=language)
            if response_audio_bytes:
                audio_b64 = base64.b64encode(response_audio_bytes).decode("utf-8")

        # Step 4: Record Interaction for NAIC Audit
        record = interaction_tracker.record_interaction(
            language=language,
            transcribed_text=transcription_text,
            llm_response=llm_response_text,
            asr_latency_seconds=asr_latency,
            llm_latency_seconds=llm_latency,
            audio_duration_seconds=audio_duration,
            user_id=user_id,
            notes=notes,
        )

        return {
            "status": "success",
            "interaction_id": record["interaction_id"],
            "language": language,
            "transcription": transcription_text,
            "llm_response": llm_response_text,
            "audio_base64": audio_b64,
            "latencies": {
                "asr_seconds": asr_latency,
                "llm_seconds": llm_latency,
                "total_seconds": record["total_latency_seconds"],
            },
            "audio_diagnostics": asr_result["audio_diagnostics"],
            "competition_progress": interaction_tracker.get_summary()["compliance_status"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice pipeline error: {str(e)}")


@app.post("/voice/synthesize")
def tts_endpoint(req: TextPromptRequest):
    """Generates audio for given text."""
    audio_bytes = synthesize_speech(req.prompt, language=req.language or "en")
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="Failed to synthesize audio")
    return Response(content=audio_bytes, media_type="audio/mp3")


@app.get("/voice/interactions")
def get_interactions():
    """Returns the logged user interactions for NAIC audit."""
    return {
        "summary": interaction_tracker.get_summary(),
        "interactions": interaction_tracker.interactions,
    }
