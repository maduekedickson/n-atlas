# NAIC 2026 — N-ATLaS Technical Feasibility Report

**Project:** National AI Innovation Challenge 2026 — N-ATLaS Innovation Project  
**Date:** September 26, 2026  
**Document Status:** Complete (Phase 0 Milestone)  
**Foundation Model:** `NCAIR1/N-ATLaS` (Llama 3 8B Architecture)  
**Target Environment:** Local Workstation (Windows 11) + Google Colab (Cloud GPU)  

---

## 1. Executive Summary

This feasibility study assesses the hardware, system environment, and execution feasibility of **N-ATLaS** (`NCAIR1/N-ATLaS`) for the National AI Innovation Challenge 2026, adhering strictly to the specifications in `Instruction.md` and `outline.md`.

### Core Verdict
* **Local Windows Execution:** **Not Feasible for Training or Real-Time Inference.** The local machine has integrated Intel Iris Xe Graphics (no dedicated NVIDIA CUDA GPU) and ~0.88 GB of currently available system RAM. Running a 16 GB parameter model on this configuration causes severe memory thrashing or Out-Of-Memory (OOM) failures.
* **Cloud GPU Execution (Google Colab / AWS):** **Highly Feasible & Recommended.** Using Google Colab (with a free 15–16 GB T4 GPU or A100 GPU) allows loading N-ATLaS in 4-bit NormalFloat (`bitsandbytes` NF4) using **~5.5 GB to 6.2 GB VRAM**, enabling rapid baseline benchmarking and parameter-efficient fine-tuning (QLoRA).
* **Selected Problem Statement Direction:** **Problem Statement 3 (Sectoral Fine-Tuning — Healthcare Intelligence)** is technically feasible within the competition timeline via a hybrid pipeline: model training and 500+ held-out evaluation executed on Google Colab GPU, while dataset processing, API serving (`FastAPI`), and interactive validation (`Streamlit`) run locally and in Docker.

---

## 2. Local Environment & Hardware Audit

Execution of `scripts/hardware_check.py` on the development workstation produced the following verified metrics:

| Property | Measured Value | Requirement for 8B Model | Feasibility Status |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 Home (AMD64) | Linux / Windows | Compatible |
| **Python Version** | 3.12.10 (64-bit) | Python 3.11+ | Compatible |
| **CPU Architecture** | 12th Gen Intel (10 physical / 12 logical cores) | Multi-core x86_64 | Adequate for preprocessing |
| **System RAM (Total)** | 15.65 GB | 32 GB+ for CPU FP16 | Insufficient for unquantized CPU inference |
| **System RAM (Available)** | **0.88 GB (94.4% used)** | 20 GB+ free | **Critical Bottleneck (Immediate OOM)** |
| **GPU / Accelerator** | **Intel(R) Iris(R) Xe Graphics** | NVIDIA GPU (Compute Cap. >= 7.0) | **No CUDA / No Tensor Cores** |
| **CUDA Availability** | `False` | Required for bitsandbytes / PyTorch GPU | **Unavailable locally** |
| **Free Storage (Drive C:)** | 167.6 GB | ~25 GB for model + caches | **Sufficient** |

---

## 3. N-ATLaS Model Characteristics & Repositories

Inspection of the official Hugging Face repository (`NCAIR1/N-ATLaS`) revealed:

* **Base Architecture:** `LlamaForCausalLM` (Meta Llama 3 8B multilingual derivative)
* **Parameter Count:** ~8.03 Billion parameters
* **Weights Structure:** 4 shards (`model-00001-of-00004.safetensors` to `00004`)
* **Download Footprint:** ~16.06 GB uncompressed
* **Vocabulary Size:** 128,256 tokens (supporting English, Hausa, Yoruba, and Igbo)
* **Gated Access Requirement:** `gated: auto` (Requires accepting terms on Hugging Face and passing an authenticated `HF_TOKEN`)
* **Default Context Length:** 8,192 tokens

---

## 4. Hardware Requirements & Quantisation Analysis

### 4.1 Memory Footprint by Precision

| Precision Mode | Format | Memory (VRAM / RAM) | Feasibility on Local Machine | Feasibility on Colab (T4 16GB) |
| :--- | :--- | :--- | :--- | :--- |
| **Full / Half (FP16/BF16)** | 16-bit Float | ~16.2 GB | **Impossible** (OOM on CPU & Iris Xe) | **Marginal** (Close to 16GB limit, risk during generation) |
| **8-bit (LLM.int8())** | INT8 | ~8.5 GB | **Not Supported** (Requires CUDA) | **Feasible** (~8.5 GB VRAM utilized) |
| **4-bit (NF4 / bitsandbytes)** | NF4 | **~5.5 – 6.2 GB** | **Not Supported** (Requires CUDA) | **Optimal** (Leaves ~10 GB VRAM headroom for QLoRA & long sequences) |

### 4.2 Quantisation Decision
* **Quantisation is mandatory for consumer and standard cloud GPU setups.**
* The project will standardize on **4-bit NormalFloat (`bitsandbytes` NF4)** with double quantization (`bnb_4bit_use_double_quant=True`) and `torch.float16` compute dtype.
* This delivers >98% of full-precision quality while fitting comfortably within the 15–16 GB VRAM budget of a free Google Colab T4 GPU instance.

---

## 5. Fine-Tuning Feasibility (LoRA / QLoRA)

* **Full Model Fine-Tuning:** Rejected. Full fine-tuning of an 8B model requires >= 64 GB to 80 GB VRAM (e.g., A100/H100 clusters), which violates the project's lightweight, cost-effective requirement.
* **Local Fine-Tuning:** Infeasible without a local NVIDIA GPU.
* **Cloud QLoRA (Google Colab / AWS EC2 g4dn/g5):** **100% Feasible.**
  * Base model frozen in 4-bit.
  * Trainable LoRA adapter injected into attention projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`).
  * Trainable parameters: < 0.2% (~16–32 million parameters).
  * VRAM required during training: **~9.5 GB to 11.5 GB VRAM** (comfortably under the 16 GB T4 limit).
  * Training time for 2,000 domain examples: ~45–90 minutes.
  * Adapter export footprint: Only **~80 MB to 150 MB** (`adapter_model.safetensors`).

---

## 6. Technical Risk Matrix

| Risk | Likelihood | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Colab Session Disconnections** | Medium | Medium | Checkpoint training every 100 steps directly to Google Drive; write lightweight standalone Python training scripts. |
| **Hugging Face Gated Model Access** | Low | High | Ensure Hugging Face account approval on `NCAIR1/N-ATLaS`; store `HF_TOKEN` in Colab Secrets and `.env`. |
| **Language Imbalance in Outputs** | Medium | Medium | Use balanced sampling across English, Hausa, Yoruba, and Igbo in the fine-tuning and evaluation datasets. |
| **Inference Latency for Real-time Demo** | Medium | Low | Deploy the lightweight LoRA adapter onto a FastAPI instance with model quantization or Colab-hosted tunnel / Hugging Face Inference Endpoint. |

---

## 7. Problem Statement Evaluation (PS1 vs. PS2 vs. PS3)

| Criterion | PS1: Developer Infrastructure | PS2: Voice-First Access | PS3: Sectoral Fine-Tuning (Health Intelligence) |
| :--- | :--- | :--- | :--- |
| **Core Deliverable** | SDK, CLI, or test harness | Audio ASR + LLM Voice assistant | Domain Dataset + QLoRA Adapter + 500+ Held-Out Benchmark |
| **Competition Metric** | >= 2 external beta testers | >= 50 real audio interactions | 500+ example benchmark comparing Base vs Adapted |
| **Technical Complexity** | Moderate | High (ASR + LLM pipeline latency) | Moderate to High (Clean data + PEFT + Metrics) |
| **Domain Differentiation** | Medium (Tooling) | High (Accessibility) | **Exceptional (Real-world public health impact)** |
| **Hardware Fit** | Feasible | High bandwidth / ASR compute | **Perfect fit for Colab QLoRA + Local API** |
| **Recommendation** | Backup Option | High barrier (50 voice recordings) | **Recommended Primary Direction** |

---

## 8. Approved Operational Workflow

To satisfy the competition's strict requirement for reproducible, genuine ML development:

1. **Phase 0 Execution:**
   * Notebook: [`notebooks/natlas_phase0_colab.ipynb`](file:///c:/xampp/htdocs/N-Atlas-Challenge2/notebooks/natlas_phase0_colab.ipynb) is executed in Google Colab with GPU.
   * Exports verification artifact: `evaluation/results/natlas_phase0_results.json`.
2. **Phase 1: Dataset & Baseline:**
   * Construct Nigerian health facility reporting benchmark (500+ examples across English, Hausa, Yoruba, Igbo).
   * Run baseline evaluation on Base N-ATLaS in Colab.
3. **Phase 2: Sectoral Fine-Tuning (QLoRA):**
   * Train health intelligence adapter on Colab GPU.
   * Evaluate adapted model on the same 500+ held-out benchmark.
   * Export `comparison.csv` and visualization plots.
4. **Phase 3: Production API & Demonstration:**
   * Package FastAPI service in `api/` and interactive Streamlit UI in `streamlit_app/`.
   * Containerize with Docker for reproducible demonstration.
