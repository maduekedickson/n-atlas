"""N-ATLaS Inference and Verification Script for NAIC 2026.

This script tests loading NCAIR1/N-ATLaS (or an adapted checkpoint),
measures model loading time, verifies tokenization and causal LM generation across
English, Hausa, Yoruba, and Igbo, and outputs a structured benchmark artifact.

If executed on hardware without an NVIDIA GPU or with insufficient RAM (< 20 GB free),
it provides actionable safety guards and instructions for running the evaluation in
Google Colab via `notebooks/natlas_phase0_colab.ipynb`.
"""

import os
import sys
import time
import json
import argparse
import psutil
from datetime import datetime

TEST_PROMPTS = [
    {
        "language": "English",
        "instruction": "Explain the importance of primary healthcare centers in rural Nigerian communities.",
        "input": "",
    },
    {
        "language": "Hausa",
        "instruction": "Menene amfanin cibiyoyin kiwon lafiya na farko a yankunan karkara a Najeriya?",
        "input": "",
    },
    {
        "language": "Yoruba",
        "instruction": "Kí ni pàtàkì àwọn ilé-ìwòsàn àkọ́bẹ̀rẹ̀ ní àwọn agbègbè ìgbèríko ní Nàìjíríà?",
        "input": "",
    },
    {
        "language": "Igbo",
        "instruction": "Gịnị bụ uru ụlọ ọrụ nlekọta ahụike mbụ bara n'obodo ndị dị n'ime ime obodo na Naịjirịa?",
        "input": "",
    }
]

def check_environment_safety(force_cpu: bool = False):
    """Check whether local environment has sufficient resources to prevent system crashes."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    
    if not has_cuda:
        print("[WARNING] No NVIDIA CUDA GPU detected on this machine.")
        print(f"[STATUS] Available System RAM: {available_ram_gb:.2f} GB")
        if not force_cpu:
            print("\n" + "!" * 70)
            print("SAFETY GUARD ACTIVE:")
            print("N-ATLaS is a Llama 3 8B model requiring ~16 GB unquantized weights.")
            print("Loading it on CPU with less than 20 GB free RAM will cause memory thrashing or an OOM crash.")
            print("\nRecommended Next Steps:")
            print("1. Recommended: Run `notebooks/natlas_phase0_colab.ipynb` on Google Colab (free T4 GPU).")
            print("2. Or pass `--force-cpu` if you wish to override this safeguard on high-RAM machines.")
            print("!" * 70 + "\n")
            return False
    return True

def run_test(model_name: str, hf_token: str = None, load_in_4bit: bool = True, output_path: str = "evaluation/results/natlas_test_results.json"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[!] Note: NCAIR1/N-ATLaS is a gated repository. If loading fails with 401 Unauthorized,")
        print("    set your token via `set HF_TOKEN=your_token` or pass `--hf-token <token>`.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing N-ATLaS test on target device: {device}")
    
    start_load = time.time()
    
    # Configure quantization if CUDA is available
    bnb_config = None
    if device == "cuda" and load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    print(f"[*] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading model {model_name} (this may take a few minutes)...")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=token,
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    load_time = time.time() - start_load
    print(f"[✓] Model successfully loaded in {load_time:.2f} seconds.")

    # Record hardware metrics
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "device": device,
        "quantization": "4bit-bnb" if (device == "cuda" and load_in_4bit) else "none",
        "load_time_seconds": round(load_time, 2),
        "prompts_tested": []
    }

    if device == "cuda":
        allocated_vram = torch.cuda.memory_allocated() / (1024**3)
        reserved_vram = torch.cuda.memory_reserved() / (1024**3)
        results["vram_allocated_gb"] = round(allocated_vram, 2)
        results["vram_reserved_gb"] = round(reserved_vram, 2)
        print(f"[*] VRAM Allocated: {allocated_vram:.2f} GB | Reserved: {reserved_vram:.2f} GB")

    print("\n" + "=" * 60)
    print("RUNNING MULTILINGUAL INFERENCE BENCHMARK")
    print("=" * 60)

    for item in TEST_PROMPTS:
        lang = item["language"]
        prompt = item["instruction"]
        print(f"\n[{lang}] Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_token_count = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.time() - t0
        output_token_count = outputs.shape[1] - input_token_count
        tokens_per_sec = output_token_count / elapsed if elapsed > 0 else 0

        generated_text = tokenizer.decode(outputs[0][input_token_count:], skip_special_tokens=True).strip()

        print(f"[{lang}] Response ({output_token_count} tokens in {elapsed:.2f}s | {tokens_per_sec:.1f} tok/s):")
        print(generated_text)
        print("-" * 50)

        results["prompts_tested"].append({
            "language": lang,
            "prompt": prompt,
            "response": generated_text,
            "input_tokens": int(input_token_count),
            "output_tokens": int(output_token_count),
            "latency_seconds": round(elapsed, 2),
            "tokens_per_second": round(tokens_per_sec, 1)
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Results artifact saved to: {output_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Test N-ATLaS Model Inference")
    parser.add_argument("--model", type=str, default="NCAIR1/N-ATLaS", help="Hugging Face model ID or path")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--force-cpu", action="store_true", help="Bypass memory guard and force CPU execution")
    parser.add_argument("--output", type=str, default="evaluation/results/natlas_test_results.json", help="Path to save results")
    args = parser.parse_args()

    safe = check_environment_safety(force_cpu=args.force_cpu)
    if not safe:
        sys.exit(1)

    run_test(
        model_name=args.model,
        hf_token=args.hf_token,
        load_in_4bit=not args.no_4bit,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
