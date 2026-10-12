"""
Audio Preprocessing Module for N-ATLaS PS2 Voice-First Access.
Handles audio ingestion, normalization, resampling, and format conversion.
"""

import io
import math
import os
import wave
from typing import Dict, Optional, Tuple, Union

import numpy as np

# Whisper standard target sample rate is 16 kHz
TARGET_SAMPLE_RATE = 16000


def load_audio_from_bytes(
    audio_bytes: bytes,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Loads audio from raw bytes (WAV, MP3, OGG, WebM) into a normalized float32 numpy array.
    Tries soundfile first; falls back to standard wave module if soundfile is unavailable.
    """
    try:
        import soundfile as sf
        buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buffer, dtype="float32")
    except Exception:
        # Fallback using standard library wave module for WAV bytes
        try:
            buffer = io.BytesIO(audio_bytes)
            with wave.open(buffer, "rb") as wf:
                sr = wf.getframerate()
                num_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                num_frames = wf.getnframes()
                raw_data = wf.readframes(num_frames)
                
                if sampwidth == 2:
                    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
                elif sampwidth == 1:
                    audio = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")
                
                if num_channels > 1:
                    audio = audio.reshape(-1, num_channels)
        except Exception as e:
            raise RuntimeError(f"Failed to decode audio bytes: {e}")

    # Convert multi-channel to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    # Resample to target_sr if needed
    if sr != target_sr:
        audio = resample_audio(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio.astype(np.float32), sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Resamples a 1D audio waveform to target_sr.
    Uses librosa/scipy if installed, or linear interpolation as fallback.
    """
    if orig_sr == target_sr:
        return audio

    try:
        import scipy.signal
        num_target_samples = int(len(audio) * target_sr / orig_sr)
        return scipy.signal.resample(audio, num_target_samples).astype(np.float32)
    except Exception:
        # High-fidelity linear interpolation fallback
        orig_indices = np.linspace(0, len(audio) - 1, len(audio))
        num_target_samples = int(len(audio) * target_sr / orig_sr)
        target_indices = np.linspace(0, len(audio) - 1, num_target_samples)
        return np.interp(target_indices, orig_indices, audio).astype(np.float32)


def normalize_volume(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalizes RMS energy to target_db to ensure consistent volume into ASR.
    """
    if len(audio) == 0:
        return audio
    rms = np.sqrt(np.mean(np.square(audio)))
    if rms < 1e-6:
        return audio  # Silence or near silence
    current_db = 20 * np.log10(rms)
    gain = 10 ** ((target_db - current_db) / 20)
    normalized = audio * gain
    # Clip to avoid digital clipping
    return np.clip(normalized, -1.0, 1.0)


def extract_audio_features(audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> Dict[str, Union[float, int]]:
    """
    Extracts basic diagnostics (duration, RMS, peak amplitude) for interaction telemetry.
    """
    duration_s = float(len(audio)) / sr if sr > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) > 0 else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    return {
        "duration_seconds": round(duration_s, 2),
        "sample_rate": sr,
        "rms_energy": round(rms, 4),
        "peak_amplitude": round(peak, 4),
        "samples_count": len(audio),
    }


def to_wav_bytes(audio: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> bytes:
    """
    Encodes a 1D float32 numpy array into standard 16-bit PCM WAV bytes.
    """
    clamped = np.clip(audio, -1.0, 1.0)
    int16_data = (clamped * 32767.0).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_data.tobytes())
    return buffer.getvalue()
