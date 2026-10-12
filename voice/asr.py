"""
Official N-ATLaS Speech Recognition (ASR) Module.
Integrates the official NCAIR1 Speech Models for Nigerian Accented English, Hausa, Igbo, and Yoruba:
  - NCAIR1/NigerianAccentedEnglish (English)
  - NCAIR1/Hausa-ASR (Hausa)
  - NCAIR1/Igbo-ASR (Igbo)
  - NCAIR1/Yoruba-ASR (Yoruba)
"""

import os
import time
from typing import Any, Dict, Optional, Union

import numpy as np

from voice.preprocessing import TARGET_SAMPLE_RATE, extract_audio_features, load_audio_from_bytes, normalize_volume

# Official Hugging Face repository IDs from NCAIR (National Centre for AI and Robotics)
OFFICIAL_ASR_MODELS = {
    "en": os.getenv("ASR_MODEL_EN", "NCAIR1/NigerianAccentedEnglish"),
    "ha": os.getenv("ASR_MODEL_HA", "NCAIR1/Hausa-ASR"),
    "ig": os.getenv("ASR_MODEL_IG", "NCAIR1/Igbo-ASR"),
    "yo": os.getenv("ASR_MODEL_YO", "NCAIR1/Yoruba-ASR"),
}

LANGUAGE_NAMES = {
    "en": "Nigerian Accented English",
    "ha": "Hausa (Harshen Hausa)",
    "ig": "Igbo (Asụsụ Igbo)",
    "yo": "Yoruba (Èdè Yorùbá)",
}


class NATLaSASRManager:
    """
    Manages loading and inference for the 4 official NCAIR1 ASR models.
    Supports local GPU/CPU execution as well as fallback test mode.
    """

    def __init__(self, device: Optional[str] = None, hf_token: Optional[str] = None):
        self.device = device or ("cuda" if self._is_cuda_available() else "cpu")
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("huggie_face_token")
        self._loaded_pipelines: Dict[str, Any] = {}

    @staticmethod
    def _is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def get_model_id(self, language: str) -> str:
        lang_key = language.lower()
        if lang_key not in OFFICIAL_ASR_MODELS:
            raise ValueError(f"Unsupported language '{language}'. Supported: {list(OFFICIAL_ASR_MODELS.keys())}")
        return OFFICIAL_ASR_MODELS[lang_key]

    def load_pipeline(self, language: str = "en") -> Any:
        """
        Lazily loads the official Hugging Face ASR pipeline for the requested language.
        """
        lang_key = language.lower()
        if lang_key in self._loaded_pipelines:
            return self._loaded_pipelines[lang_key]

        model_id = self.get_model_id(lang_key)
        print(f"[ASR] Loading official model '{model_id}' on {self.device}...")

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=self.hf_token,
            )
            model.to(self.device)

            processor = AutoProcessor.from_pretrained(model_id, token=self.hf_token)

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=8 if self.device == "cuda" else 1,
                torch_dtype=torch_dtype,
                device=self.device,
            )
            self._loaded_pipelines[lang_key] = pipe
            return pipe
        except Exception as e:
            print(f"[ASR Warning] Unable to load Hugging Face pipeline for '{model_id}': {e}")
            return None

    def transcribe(
        self,
        audio_input: Union[bytes, np.ndarray, str],
        language: str = "en",
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribes the given audio using the official NCAIR1 ASR model for the selected language.

        Returns a dictionary containing:
            - text: Transcribed text
            - language: Language code ('en', 'ha', 'ig', 'yo')
            - language_name: Human readable language name
            - model_id: Official Hugging Face model identifier
            - latency_seconds: Time taken to process the audio
            - audio_diagnostics: Duration, RMS energy, sample rate
        """
        t0 = time.perf_counter()
        lang_key = language.lower()
        model_id = self.get_model_id(lang_key)

        # 1. Parse and preprocess audio
        if isinstance(audio_input, bytes):
            audio, sr = load_audio_from_bytes(audio_input, target_sr=TARGET_SAMPLE_RATE)
        elif isinstance(audio_input, str):
            with open(audio_input, "rb") as f:
                audio, sr = load_audio_from_bytes(f.read(), target_sr=TARGET_SAMPLE_RATE)
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input.astype(np.float32)
            sr = TARGET_SAMPLE_RATE
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        if normalize:
            audio = normalize_volume(audio)

        diagnostics = extract_audio_features(audio, sr=sr)

        # 2. Transcribe via official pipeline if available
        pipe = self.load_pipeline(lang_key)
        transcription_text = ""
        used_fallback = False

        if pipe is not None:
            try:
                result = pipe({"raw": audio, "sampling_rate": sr})
                if isinstance(result, dict) and "text" in result:
                    transcription_text = result["text"].strip()
                elif isinstance(result, list) and len(result) > 0 and "text" in result[0]:
                    transcription_text = result[0]["text"].strip()
            except Exception as e:
                print(f"[ASR Error during inference]: {e}")
                used_fallback = True
        else:
            used_fallback = True

        # Fallback simulation for dry-run/local demo when model weights cannot be downloaded on CPU
        if used_fallback and not transcription_text:
            transcription_text = self._mock_transcription(audio, lang_key)

        latency = round(time.perf_counter() - t0, 3)

        return {
            "text": transcription_text,
            "language": lang_key,
            "language_name": LANGUAGE_NAMES.get(lang_key, lang_key),
            "model_id": model_id,
            "latency_seconds": latency,
            "audio_diagnostics": diagnostics,
            "used_fallback": used_fallback,
        }

    def _mock_transcription(self, audio: np.ndarray, language: str) -> str:
        """
        Provides realistic sample transcriptions for demonstration when running without GPU weights.
        """
        samples = {
            "en": "We need twenty additional malaria test kits and amoxicillin syrup at the primary clinic.",
            "ha": "Ina bukatan magungunan zazzabin cizon sauro da alluran riga-kafi a asibitin mu.",
            "yo": "A nilo ogun ibà ati awọn abẹrẹ ajẹsara ni ile-iwosan wa.",
            "ig": "Anyị chọrọ ọgwụ ịba na ọgwụ mgbochi ọzọ n'ụlọ ọgwụ anyị.",
        }
        return samples.get(language, "Audio transcription completed.")
