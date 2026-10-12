"""
N-ATLaS Language Model (LLM) Manager.
Handles loading, tokenization, and generation with NCAIR1/N-ATLaS.
Supports 4-bit/8-bit quantization on GPU, Colab, or test fallback mode.
"""

import os
import time
from typing import Any, Dict, List, Optional

# Official N-ATLaS Model on Hugging Face
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "NCAIR1/N-ATLaS")


class NATLaSLLMManager:
    """
    Manages loading and text generation for NCAIR1/N-ATLaS.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if self._is_cuda_available() else "cpu")
        self.load_in_4bit = load_in_4bit
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("huggie_face_token")
        self.tokenizer = None
        self.model = None

    @staticmethod
    def _is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def load(self) -> bool:
        """Loads tokenizer and model weights."""
        if self.model is not None and self.tokenizer is not None:
            return True

        print(f"[N-ATLaS] Loading {self.model_name} on {self.device}...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
            )

            # 2. Configure Quantization & Model Load
            kwargs: Dict[str, Any] = {
                "token": self.hf_token,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if self.device == "cuda":
                if self.load_in_4bit:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                else:
                    kwargs["torch_dtype"] = torch.bfloat16
                kwargs["device_map"] = "auto"
            else:
                # CPU fallback
                kwargs["torch_dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
            return True
        except Exception as e:
            print(f"[N-ATLaS Warning] Model loading skipped or failed: {e}")
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Generates text given a user prompt.
        """
        t0 = time.perf_counter()

        formatted_prompt = prompt
        if system_prompt:
            formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\nAssistant:"

        if self.model is not None and self.tokenizer is not None:
            try:
                import torch
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0.0,
                    )
                gen_text = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                latency = round(time.perf_counter() - t0, 3)
                new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]

                return {
                    "text": gen_text,
                    "tokens_generated": int(new_tokens),
                    "latency_seconds": latency,
                    "is_simulated": False,
                }
            except Exception as e:
                print(f"[N-ATLaS Inference Error]: {e}")

        # Fallback response for testing environments without 8B GPU VRAM
        latency = round(time.perf_counter() - t0, 3)
        simulated_response = self._fallback_response(prompt)
        return {
            "text": simulated_response,
            "tokens_generated": len(simulated_response.split()),
            "latency_seconds": latency,
            "is_simulated": True,
        }

    def _fallback_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "malaria" in prompt_lower or "clinic" in prompt_lower or "test" in prompt_lower:
            return (
                "Acknowledged. The request for malaria test kits and pediatric antibiotics has been logged "
                "for the primary health care facility. Stock alerts have been transmitted."
            )
        elif "hausa" in prompt_lower or "ina bukatan" in prompt_lower or "magungunan" in prompt_lower:
            return (
                "An karɓi rahoton ku na asibiti. Mun rubuta buƙatar magungunan zazzabin cizon sauro da alluran riga-kafi "
                "don cibiyar kiwon lafiya ta matakin farko."
            )
        elif "yoruba" in prompt_lower or "ogun" in prompt_lower or "ibà" in prompt_lower:
            return (
                "A ti gba ifitonilete rẹ silẹ. A ti fi ẹsun ibeere fun awọn oogun ati ajẹsara ranṣẹ si ẹka eto ilera."
            )
        elif "igbo" in prompt_lower or "ọgwụ" in prompt_lower or "ịba" in prompt_lower:
            return (
                "Anyị anabatala akụkọ gị. E tinyela arịrịọ maka ọgwụ ịba na ngwa ọgwụgwọ ndị ọzọ n'akwụkwọ ndekọ."
            )
        else:
            return (
                "N-ATLaS has processed your multilingual voice query. The information has been recorded and verified."
            )
