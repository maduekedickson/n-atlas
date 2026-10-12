# NAIC 2026 — N-ATLAS Innovation Challenge

## Python-Centred Technical Specification

### 1. Engineering Direction

This project is being developed for the National AI Innovation Challenge 2026 and must integrate genuinely with N-ATLAS.

The project owner prefers **Python as the primary programming language and engineering ecosystem**.

The implementation should therefore maximise the use of the project owner's existing Python, AI/ML and data engineering skills.

The default stack is:

* Python 3.11+
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* TRL where appropriate
* FastAPI
* Pydantic
* Uvicorn
* Streamlit
* Pandas
* NumPy
* scikit-learn
* Matplotlib
* Jupyter
* Docker
* Git/GitHub

Use additional technologies only when there is a clear technical reason.

Do NOT introduce Laravel, PHP, Node.js, React, Kubernetes, Redis, Celery or other infrastructure simply because they are familiar technologies.

The competition project should remain lightweight, reproducible and easy to demonstrate.

---

# 2. Preferred Architecture

The preferred architecture is:

```text
                    ┌──────────────────────┐
                    │      User / Tester   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Streamlit UI     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI API     │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼──────────────────┐
             │                 │                  │
             ▼                 ▼                  ▼
      ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
      │ N-ATLAS     │   │ Domain Logic │   │ Evaluation   │
      │ Inference   │   │ / RAG / NLP  │   │ Pipeline     │
      └─────────────┘   └──────────────┘   └──────────────┘
             │                 │                  │
             └─────────────────┼──────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Structured Response  │
                    └──────────────────────┘
```

The exact architecture may change after the feasibility study.

---

# 3. Python Responsibilities

Python should handle:

### Model

* N-ATLAS loading
* tokenization
* inference
* fine-tuning
* LoRA/QLoRA adapters
* model evaluation
* model benchmarking

### Data

* dataset collection
* cleaning
* transformation
* validation
* train/validation/test splitting
* benchmark generation
* analysis

### API

* model inference endpoint
* domain-specific endpoints
* health/status endpoint
* evaluation endpoint where appropriate
* API documentation

### Application

Streamlit should initially be the preferred UI because it allows rapid development of a working AI application without spending competition time building a complex frontend.

### Evaluation

Python should be used for:

* benchmark execution
* metric calculation
* statistical analysis
* model comparison
* visualisation
* report generation

---

# 4. Python Environment

Use a virtual environment.

Preferred:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Alternatively, use Conda if GPU/CUDA dependency management makes it materially easier.

Do not require Conda unless necessary.

---

# 5. Python Version

Target:

```text
Python 3.11
```

Avoid jumping to the newest Python release if major ML libraries do not yet provide stable support.

The project must record the exact Python version in:

```text
.python-version
```

or:

```text
runtime.txt
```

and in the documentation.

---

# 6. Core Python Dependencies

Initial dependency candidates:

```text
torch
transformers
datasets
accelerate
peft
trl
sentencepiece
safetensors

fastapi
uvicorn
pydantic
python-multipart

streamlit

pandas
numpy
scikit-learn
matplotlib

jupyter
ipykernel

pytest
httpx

python-dotenv
```

Do NOT install everything automatically.

Claude Code must first determine which packages are actually required.

Use a clean:

```text
requirements.txt
```

and, if useful:

```text
requirements-dev.txt
```

---

# 7. Hugging Face Ecosystem

Use Hugging Face as the primary model ecosystem.

N-ATLAS model:

```text
NCAIR1/N-ATLaS
```

Use:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

where compatible with the current model repository.

Use:

```text
datasets
```

for dataset management.

Use:

```text
peft
```

for parameter-efficient fine-tuning where appropriate.

Use:

```text
accelerate
```

for hardware-aware training/inference.

Use:

```text
safetensors
```

for safe model/adapter storage.

---

# 8. First Technical Milestone

Before developing the application, prove that N-ATLAS can actually be loaded and used.

Create:

```text
scripts/
    test_natlas.py
```

The script must:

1. Download/load N-ATLAS.
2. Load its tokenizer.
3. Prepare an appropriate prompt.
4. Run inference.
5. Generate a response.
6. Record inference time.
7. Report the hardware/device being used.
8. Save a small output artifact.

Example command:

```bash
python scripts/test_natlas.py
```

Successful execution of this script is the first major milestone.

---

# 9. Hardware Detection

Create a utility that automatically detects:

```text
CPU
GPU
CUDA availability
GPU memory
RAM
Python version
PyTorch version
Transformers version
```

Example:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "GB"
    )
```

Do not assume that the development machine has sufficient VRAM.

The feasibility report must establish the actual hardware requirements.

---

# 10. Model Loading Strategy

The first implementation should use the simplest reliable inference method.

Do not immediately implement:

* quantisation
* distributed inference
* multi-GPU
* model serving clusters
* Kubernetes

unless the feasibility study proves that they are necessary.

Start with:

```text
Transformers
+
PyTorch
+
device_map="auto"
```

Then optimise only if required.

---

# 11. Quantisation

If available hardware cannot comfortably run N-ATLAS, investigate:

* 8-bit loading
* 4-bit loading
* bitsandbytes
* GPTQ/GPTQ-compatible approaches
* AWQ where supported
* CPU/GPU offloading

Do not choose a quantisation method blindly.

Benchmark:

```text
memory usage
latency
quality
stability
```

and document the decision.

---

# 12. Fine-Tuning Strategy

If PS3 is selected, start with **parameter-efficient fine-tuning** rather than full model training.

Preferred investigation order:

```text
LoRA
↓
QLoRA if necessary
↓
Full fine-tuning only if explicitly justified
```

The purpose is to make the project feasible within the competition timeframe and available compute.

Potential stack:

```python
transformers
datasets
peft
trl
accelerate
```

Do not begin fine-tuning until the baseline model has been evaluated.

---

# 13. Baseline Before Fine-Tuning

This is mandatory for scientific validity.

Create:

```text
evaluation/
    baseline.py
    metrics.py
    benchmark.py
```

The baseline must evaluate:

```text
N-ATLAS base model
```

on the exact task that the adapted model will later perform.

Example:

```text
Base N-ATLAS
      ↓
500+ domain examples
      ↓
Baseline results
      ↓
Fine-tuning
      ↓
Adapted N-ATLAS
      ↓
Same held-out benchmark
      ↓
Comparison
```

Never change the benchmark after seeing the adapted-model results.

---

# 14. Dataset Architecture

Use:

```text
data/
├── raw/
├── interim/
├── processed/
├── train/
├── validation/
├── test/
└── benchmark/
```

Recommended formats:

```text
JSONL
Parquet
CSV
```

Prefer JSONL for instruction-tuning datasets.

Example:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "language": "igbo",
  "domain": "health"
}
```

Dataset schemas must be documented.

---

# 15. Data Processing

Use Python/Pandas for structured data.

Example:

```python
import pandas as pd

df = pd.read_csv("data/raw/data.csv")

df = df.drop_duplicates()
df = df.dropna(subset=["input", "output"])
```

But do not blindly apply generic cleaning.

For language datasets, preserve:

* accents
* diacritics
* punctuation
* language-specific spelling
* meaningful code-switching

Create explicit preprocessing functions.

---

# 16. Dataset Quality

Every training example should be checked for:

* correctness
* relevance
* language
* domain relevance
* toxicity
* duplication
* hallucinated information
* malformed instructions
* personally identifiable information

Where possible, use automated checks followed by human review.

---

# 17. Healthcare Data

If the health direction is selected:

Prefer:

```text
public datasets
synthetic datasets
de-identified datasets
expert-reviewed examples
```

Avoid collecting sensitive patient information merely to increase dataset size.

The project should not require identifiable patient records.

---

# 18. Evaluation Metrics

Do not use generic accuracy for every task.

Select metrics according to the actual task.

For classification:

```text
Accuracy
Precision
Recall
F1
Confusion matrix
```

For structured extraction:

```text
Exact match
Field-level precision
Field-level recall
Field-level F1
```

For generation:

```text
Human evaluation
Task success
Relevance
Factuality
Safety
Language quality
```

For translation:

```text
BLEU
chrF
COMET where appropriate
Human evaluation
```

For response latency:

```text
Average latency
P50
P95
```

---

# 19. Model Comparison

Create a reproducible evaluation script:

```bash
python evaluation/run_benchmark.py \
    --model base \
    --dataset data/benchmark/test.jsonl
```

and:

```bash
python evaluation/run_benchmark.py \
    --model adapted \
    --dataset data/benchmark/test.jsonl
```

Output:

```text
evaluation/results/
    base_results.json
    adapted_results.json
    comparison.csv
    comparison.png
```

The comparison must be generated programmatically.

Do not manually type benchmark numbers.

---

# 20. API

Use FastAPI.

Example structure:

```text
api/
├── main.py
├── routes/
│   ├── health.py
│   ├── inference.py
│   └── model.py
├── schemas/
│   ├── requests.py
│   └── responses.py
└── services/
    └── natlas_service.py
```

Run:

```bash
uvicorn api.main:app --reload
```

The API should automatically expose:

```text
/docs
```

through FastAPI's OpenAPI documentation.

---

# 21. N-ATLAS Service Layer

Do not load the model inside every API request.

Create a singleton/service abstraction.

Conceptually:

```python
class NAtlasService:

    def __init__(self):
        self.tokenizer = ...
        self.model = ...

    def generate(self, messages):
        ...
```

The application should communicate with N-ATLAS through this service.

This makes it easier to:

* test
* benchmark
* replace inference implementation
* add adapters
* deploy

---

# 22. Streamlit

Use Streamlit for the first user interface.

Example:

```text
streamlit_app/
├── app.py
├── components/
├── services/
└── utils/
```

Run:

```bash
streamlit run streamlit_app/app.py
```

The UI should call the FastAPI backend rather than embedding all business logic directly in Streamlit.

Architecture:

```text
Streamlit
    ↓
FastAPI
    ↓
N-ATLAS
```

This also makes the system easier to demonstrate as an actual API product.

---

# 23. Voice

If PS2 or voice is incorporated into the chosen solution, treat voice as a separate module.

Architecture:

```text
Audio
 ↓
N-ATLAS ASR
 ↓
Text
 ↓
N-ATLAS LLM
 ↓
Response
```

Do not substitute another ASR model for the official N-ATLAS ASR where the competition requires official N-ATLAS ASR integration.

Create:

```text
voice/
├── asr.py
├── preprocessing.py
└── evaluation.py
```

---

# 24. Optional RAG

If domain knowledge needs retrieval, investigate a lightweight RAG architecture.

Preferred initial approach:

```text
Documents
 ↓
Chunking
 ↓
Embeddings
 ↓
Vector store
 ↓
Relevant context
 ↓
N-ATLAS
```

Do not add RAG merely because it is fashionable.

First determine whether fine-tuning, prompting or structured retrieval actually solves the problem.

If a vector database is needed, start with a lightweight local option such as:

```text
FAISS
```

or another Python-friendly solution.

Do not introduce a managed vector database unless necessary.

---

# 25. Database

Do not use a database unless the product actually requires persistence.

For prototypes:

```text
SQLite
```

is acceptable.

If production persistence is required:

```text
PostgreSQL
```

is preferred.

Use SQLAlchemy only when database access is actually required.

Do not build a database layer merely because this is a software project.

---

# 26. Docker

Containerise the application once the Python architecture works.

Suggested services:

```text
docker-compose.yml

api
streamlit
```

Model inference may require a separate container depending on hardware.

Example:

```text
Docker
├── FastAPI
├── Streamlit
└── N-ATLAS inference
```

The Docker image must document hardware assumptions.

Do not assume CUDA works inside Docker without testing.

---

# 27. AWS

AWS may be used for deployment because the project owner already has AWS experience.

Possible architecture:

```text
Internet
   ↓
EC2
   ↓
Docker
   ├── FastAPI
   └── Streamlit
```

For GPU inference, investigate appropriate GPU infrastructure only if necessary.

Do not build a complex AWS architecture unless the actual model deployment requires it.

Competition success matters more than infrastructure complexity.

---

# 28. Streamlit vs Full Frontend

Do not build React/Next.js unless there is a demonstrated reason.

For this competition:

```text
Streamlit = default
```

because the priorities are:

```text
AI functionality
N-ATLAS integration
evaluation
validation
technical evidence
```

rather than frontend engineering.

---

# 29. Project Structure

Use this Python-first structure:

```text
naic-natlas/
│
├── api/
│   ├── main.py
│   ├── routes/
│   ├── schemas/
│   └── services/
│
├── streamlit_app/
│   ├── app.py
│   ├── components/
│   └── services/
│
├── model/
│   ├── loader.py
│   ├── inference.py
│   ├── prompts.py
│   ├── adapters/
│   └── config.py
│
├── training/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── config/
│   └── utils.py
│
├── evaluation/
│   ├── benchmark.py
│   ├── metrics.py
│   ├── run_benchmark.py
│   └── reports/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── benchmark/
│
├── voice/
│   ├── asr.py
│   └── evaluation.py
│
├── scripts/
│   ├── test_natlas.py
│   ├── hardware_check.py
│   └── download_model.py
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_data.py
│
├── docs/
│   ├── PROJECT-DEFINITION.md
│   ├── FEASIBILITY.md
│   ├── NATLAS-INTEGRATION.md
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── EVALUATION.md
│   ├── VALIDATION.md
│   ├── API.md
│   └── DEPLOYMENT.md
│
├── notebooks/
│
├── .env.example
├── .gitignore
├── CLAUDE.md
├── README.md
├── requirements.txt
├── requirements-dev.txt
└── docker-compose.yml
```

---

# 30. Notebook Policy

Jupyter notebooks are allowed for:

* exploratory data analysis
* experiments
* visualisation
* model investigation

But final production logic must be moved into Python modules/scripts.

Do not make the competition submission dependent on running a notebook manually.

Preferred:

```text
notebook → discover
Python script → reproduce
API → deploy
Streamlit → demonstrate
```

---

# 31. Data Science Workflow

Use the workflow:

```text
Question
 ↓
Data
 ↓
EDA
 ↓
Baseline
 ↓
Experiment
 ↓
Evaluation
 ↓
Decision
 ↓
Production implementation
```

Use Pandas/NumPy for exploration.

Use Matplotlib for visualisation.

Use scikit-learn where classical ML is appropriate.

Use PyTorch/Hugging Face for N-ATLAS.

---

# 32. Classical ML Is Allowed

Do not force N-ATLAS to perform every task.

A hybrid architecture may be appropriate.

For example:

```text
N-ATLAS
   ↓
Language understanding
   ↓
Structured features
   ↓
scikit-learn classifier
   ↓
Decision support
```

However, N-ATLAS must remain genuinely central to the competition solution.

The use of classical ML should strengthen the solution rather than obscure the N-ATLAS integration.

---

# 33. Reproducibility

Every experiment must be reproducible.

Record:

```text
Python version
Package versions
Model version
Dataset version
Random seed
Hardware
Training configuration
Evaluation configuration
```

Use:

```python
SEED = 42
```

where deterministic behaviour is practical.

---

# 34. Configuration

Use environment variables for deployment configuration.

Example:

```text
.env

MODEL_NAME=NCAIR1/N-ATLaS
MODEL_REVISION=
DEVICE=auto
API_HOST=0.0.0.0
API_PORT=8000
```

Never hardcode credentials.

---

# 35. Logging

Use Python's standard logging framework.

Example:

```python
import logging

logger = logging.getLogger(__name__)
```

Do not use `print()` for production application logging.

Log:

* startup
* model loading
* inference errors
* request IDs
* latency
* evaluation events

Do not log sensitive user data.

---

# 36. Error Handling

The API must handle:

* invalid requests
* model loading failure
* inference failure
* timeout
* malformed model output
* unavailable GPU
* missing environment variables

Return useful HTTP responses.

Never expose stack traces or secrets to users in production.

---

# 37. Performance Measurement

Every model-serving implementation should measure:

```text
model loading time
first-token latency where applicable
total generation latency
tokens/second where available
memory consumption
GPU utilisation where available
```

Do not make performance claims without measurement.

---

# 38. Development Order

Claude Code must follow this sequence:

```text
1. Repository inspection
2. Environment verification
3. N-ATLAS loading
4. Hardware feasibility
5. Baseline inference
6. Problem statement decision
7. Dataset strategy
8. Baseline benchmark
9. Domain adaptation
10. API
11. Streamlit application
12. Validation
13. Docker
14. Deployment
15. Documentation
16. Demo
```

Do not skip directly to step 11.

---

# 39. First Claude Code Prompt

Use this as the first instruction after placing this specification in `CLAUDE.md`:

> Read `CLAUDE.md` completely before modifying the repository.
>
> Do not build the final application yet.
>
> We are developing a Python-first solution for the National AI Innovation Challenge 2026 using N-ATLAS.
>
> Your first task is a technical feasibility sprint.
>
> 1. Inspect the repository and development environment.
> 2. Detect Python version, CPU, RAM, GPU, CUDA and available storage.
> 3. Inspect the official N-ATLaS Hugging Face repository and determine its actual loading/inference requirements.
> 4. Create a clean Python virtual environment if one does not exist.
> 5. Install only the minimum dependencies required for N-ATLAS inference.
> 6. Create `scripts/hardware_check.py`.
> 7. Create `scripts/test_natlas.py`.
> 8. Attempt genuine N-ATLAS inference.
> 9. Measure model loading time, inference latency and memory requirements where possible.
> 10. Do not fake or mock N-ATLAS output.
> 11. Create `docs/FEASIBILITY.md`.
>
> The feasibility report must clearly state:
>
> * whether N-ATLAS runs locally;
> * required RAM;
> * required VRAM;
> * CPU/GPU requirements;
> * inference framework;
> * Python dependencies;
> * approximate inference latency;
> * whether quantisation is necessary;
> * whether LoRA/QLoRA fine-tuning is feasible;
> * what cloud infrastructure would be required if local execution is insufficient;
> * the major technical risks;
> * whether PS1, PS2 or PS3 appears technically feasible within the competition deadline.
>
> Do not implement the final product until this report has been completed.

---

# 40. Core Engineering Principle

The project should demonstrate:

```text
Python
+
AI/ML
+
N-ATLAS
+
Data
+
Evaluation
+
API
+
Real-world validation
```

rather than:

```text
Large frontend
+
many technologies
+
little AI
```

The goal is to make the technical depth visible.

---

# 41. Final Technology Stack

Unless the feasibility study identifies a compelling reason to change it, the preferred stack is:

| Layer            | Technology                                |
| ---------------- | ----------------------------------------- |
| Language         | Python                                    |
| LLM              | N-ATLaS                                   |
| ML framework     | PyTorch                                   |
| Model ecosystem  | Hugging Face Transformers                 |
| Dataset          | Hugging Face Datasets / JSONL / Parquet   |
| Fine-tuning      | PEFT / LoRA / QLoRA                       |
| Training         | Transformers / TRL / Accelerate           |
| Classical ML     | scikit-learn                              |
| Data             | Pandas / NumPy                            |
| API              | FastAPI                                   |
| API server       | Uvicorn                                   |
| UI               | Streamlit                                 |
| Evaluation       | Python                                    |
| Visualisation    | Matplotlib                                |
| Experiments      | Jupyter + Python scripts                  |
| Testing          | Pytest                                    |
| Containerisation | Docker                                    |
| Cloud            | AWS where required                        |
| Version control  | Git/GitHub                                |
| Database         | SQLite initially / PostgreSQL if required |
| Vector search    | FAISS if RAG requires it                  |

The principle is simple:

> **Use the tools you already know to move faster, and introduce new infrastructure only when the project genuinely needs it.**
