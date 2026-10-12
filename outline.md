# NATIONAL AI INNOVATION CHALLENGE 2026

## N-ATLaS Innovation Project — Master Project Specification

**Project Status:** Development Specification
**Primary Language:** Python
**Primary ML Framework:** PyTorch
**Foundation Model:** N-ATLaS
**Model Ecosystem:** Hugging Face Transformers
**Application Framework:** FastAPI + Streamlit
**Deployment:** Docker + AWS where required
**Development Assistant:** Claude Code

---

## 1. PROJECT OBJECTIVE

Build a technically rigorous, genuinely working, reproducible solution for the National AI Innovation Challenge 2026 using N-ATLaS as a core component.

The objective is not to produce a concept, mock-up, or simulated AI system.

The project must produce a working technical artefact with:

* Genuine N-ATLaS integration
* A clearly defined real-world problem
* Measurable technical performance
* Real-world validation
* Reproducible development
* Documented architecture
* Documented evaluation
* A deployable API
* A usable demonstration interface
* Evidence suitable for NAIC submission

The project should be as simple as possible while still satisfying the competition requirements.

---

## 2. NATIONAL AI INNOVATION CHALLENGE CONTEXT

The National AI Innovation Challenge is designed to move N-ATLaS from model availability toward practical national AI infrastructure.

The project must therefore demonstrate more than simply calling an LLM.

It should demonstrate how N-ATLaS can become useful within an actual Nigerian context.

The three official problem areas are:

1. Developer Infrastructure
2. Voice-First Access
3. Sectoral Fine-Tuning

Only one problem statement will be selected for the final submission.

---

## 3. PROBLEM STATEMENT 1 — DEVELOPER INFRASTRUCTURE

PS1 focuses on making N-ATLaS easier for developers to use.

Potential areas include:

* Developer tooling
* SDKs
* Documentation
* Evaluation tools
* Dataset tooling
* NLP components
* Reusable APIs
* Testing infrastructure
* Developer experience

If PS1 is selected, the project must provide a meaningful developer-facing artefact rather than simply another chatbot.

The solution must demonstrate usefulness to external developers.

At least two external beta testers should be incorporated into validation.

---

## 4. PROBLEM STATEMENT 2 — VOICE-FIRST ACCESS

PS2 focuses on making N-ATLaS accessible through voice-first experiences.

Potential users include:

* Nigerians with limited digital literacy
* Rural users
* Health workers
* Farmers
* Students
* Citizens accessing government services
* Users who prefer Nigerian languages

Where voice is used, the official N-ATLaS ASR capability must be used where required.

The solution must document real interactions and demonstrate meaningful user engagement.

The competition requirement of at least 50 documented real user interactions must be satisfied if PS2 is selected.

---

## 5. PROBLEM STATEMENT 3 — SECTORAL FINE-TUNING

PS3 focuses on adapting N-ATLaS to a defined high-impact sector.

Potential sectors include:

* Health
* Agriculture
* Education
* Legal services
* Financial inclusion

The project must demonstrate actual domain adaptation rather than simply prompt engineering.

Requirements include:

* Defined domain
* Domain dataset
* Baseline using base N-ATLaS
* Fine-tuned/adapted N-ATLaS
* Measurable improvement
* 500+ example benchmark
* Held-out evaluation
* Documented API
* Another developer must be able to integrate the resulting service

LoRA/QLoRA should be considered before full fine-tuning because of computational efficiency.

---

## 6. INITIAL DOMAIN HYPOTHESIS — HEALTH

Health is a strong candidate domain for exploration because of the user's existing experience with health data, surveillance, analytics, and facility-level reporting.

A potential direction is a Nigerian health intelligence platform that allows users to communicate health-service information using text or voice and converts that interaction into structured, useful intelligence.

Potential capabilities include:

* Facility reporting
* Health information extraction
* Supply/stock reporting
* Service availability
* Reporting-gap detection
* Facility intelligence
* Health-worker assistance
* Multilingual interaction
* Structured health data extraction

The system must not present itself as an autonomous medical diagnosis or prescription system.

The initial focus should be information processing, reporting, classification, extraction, and decision support.

---

## 7. FINAL PROBLEM-STATEMENT SELECTION

The project must not prematurely lock itself into PS1, PS2, or PS3.

The first technical milestone is feasibility.

After N-ATLaS is successfully loaded and tested, the project should compare:

* Technical feasibility
* Available hardware
* Dataset availability
* Evaluation requirements
* Development time
* Validation requirements
* Deployment requirements
* Differentiation
* Competition compliance

The final problem statement should then be documented in:

`docs/PROJECT-DEFINITION.md`

---

## 8. N-ATLaS FOUNDATION MODEL

N-ATLaS is the foundation model used by the project.

The supplied N-ATLaS documentation describes it as a multilingual model based on Llama 3 8B.

Supported languages include:

* English
* Hausa
* Igbo
* Yoruba

The model was developed for Nigerian linguistic and cultural contexts.

N-ATLaS should remain the actual foundation model used by the project.

No simulated N-ATLaS outputs are permitted.

---

## 9. N-ATLaS TECHNICAL CHARACTERISTICS

The supplied documentation identifies the model architecture as:

* LlamaForCausalLM
* Llama 3 8B
* Hidden size: 4096
* 32 layers
* 32 attention heads
* 8 key/value heads
* Vocabulary size: 128256
* Maximum position embeddings: 131072
* Documented context length: approximately 8092 tokens

The implementation must verify actual runtime characteristics rather than blindly assuming documentation values.

---

## 10. N-ATLaS TRAINING BACKGROUND

The supplied documentation describes approximately 391 million multilingual instruction tokens.

The training data includes English, Hausa, Igbo, and Yoruba examples.

The documented training process includes:

* Supervised fine-tuning
* Multilingual instruction data
* Translated material
* Synthetic Nigerian web content
* Human verification
* Filtering
* Language categorisation
* Translation verification
* Cultural relevance checks

These characteristics should inform evaluation and dataset design.

---

## 11. N-ATLaS EVALUATION BASELINE

The supplied documentation reports human evaluation results for the supported languages.

Reported average scores include:

* English: 4.21
* Hausa: 3.98
* Yoruba: 2.69
* Igbo: 3.87

These values are background information and must not be presented as results produced by this project.

Project-specific benchmarks must be independently generated.

---

## 12. N-ATLaS LIMITATIONS

The project must explicitly acknowledge known limitations.

Relevant limitations include:

* Dialect and accent variation
* Potential language bias
* Reduced accuracy with children's speech
* Limited code-switching capability
* Reduced performance in noisy environments
* Potential differences between languages

The final documentation must transparently report observed limitations.

---

## 13. N-ATLaS LICENSE AND GOVERNANCE

The supplied documentation describes an Open-Source Research and Innovation License with additional restrictions.

The project must comply with the model's actual license.

Attribution must be preserved.

The supplied attribution is:

“N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.”

Any renamed derivative must comply with the documented naming requirements.

---

## 14. N-ATLaS USAGE RESTRICTIONS

The project must not use N-ATLaS for prohibited activities.

The supplied documentation specifically restricts uses involving:

* Surveillance
* Unlawful monitoring
* Discriminatory profiling
* Disinformation
* Impersonation
* Fraud
* Military/intelligence applications
* Weaponisation
* Harmful or unlawful activities

The project must incorporate privacy, human-rights, and responsible-AI considerations.

---

## 15. N-ATLaS USER LIMIT

The supplied documentation specifies an active end-user cap of 1,000 users before additional commercial licensing requirements apply.

The architecture and deployment plan must account for this restriction.

Actual production scale must not be represented beyond what the license permits.

---

## 16. PRIMARY PROGRAMMING LANGUAGE

Python is the primary programming language.

Python should be used for:

* Model inference
* Data processing
* Training
* Evaluation
* APIs
* Automation
* Testing
* Dataset preparation
* Benchmarking
* Supporting application logic

The project should maintain a clean Python package structure.

---

## 17. PRIMARY MACHINE-LEARNING FRAMEWORK

PyTorch is the primary machine-learning framework.

This is a deliberate project decision.

N-ATLaS should be integrated using its native Hugging Face/PyTorch implementation.

The project must not convert N-ATLaS to TensorFlow simply for framework consistency.

TensorFlow/Keras may only be introduced later if a specific independently developed component genuinely benefits from it.

PyTorch is therefore the default ML framework for this project.

---

## 18. HUGGING FACE STACK

The core model stack should use:

* Transformers
* Datasets
* PEFT
* Accelerate
* TRL where required

The project should use native Hugging Face mechanisms for:

* Tokenization
* Model loading
* Inference
* Fine-tuning
* Dataset handling
* Model configuration
* Evaluation support

---

## 19. MODEL LOADING

The initial model-loading implementation should follow the documented N-ATLaS approach using:

`AutoTokenizer`

and

`AutoModelForCausalLM`

The project must verify:

* Model download
* Tokenizer loading
* Model loading
* Device placement
* Data type
* Memory requirements
* Generation
* Inference latency
* Output quality

No mocked model should be used as a substitute for the real model.

---

## 20. HARDWARE FEASIBILITY

Before significant application development begins, the environment must be assessed.

The system should detect:

* Operating system
* Python version
* CPU
* RAM
* GPU
* VRAM
* CUDA availability
* PyTorch version
* Disk space

The project should determine whether local inference is practical.

If local resources are insufficient, cloud GPU execution should be evaluated.

---

## 21. PHASE 0 — FEASIBILITY FIRST

The first Claude Code implementation task is a feasibility assessment.

Claude Code must:

1. Inspect the repository.
2. Inspect the development environment.
3. Detect hardware.
4. Create a clean Python environment.
5. Install minimum required dependencies.
6. Inspect N-ATLaS requirements.
7. Attempt genuine N-ATLaS loading.
8. Run genuine inference.
9. Measure loading time.
10. Measure inference latency.
11. Measure memory usage.
12. Document the result.

The result must be written to:

`docs/FEASIBILITY.md`

No final product should be built before this phase is complete.

---

## 22. PROJECT STRUCTURE

The initial structure should be:

```text
naic-natlas/
│
├── api/
│   ├── main.py
│   ├── routes/
│   ├── schemas/
│   └── services/
│
├── model/
│   ├── loader.py
│   ├── inference.py
│   ├── prompts.py
│   ├── config.py
│   └── adapters/
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
├── streamlit_app/
│
├── scripts/
│   ├── hardware_check.py
│   ├── test_natlas.py
│   └── download_model.py
│
├── tests/
│
├── notebooks/
│
├── docs/
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
└── CLAUDE.md
```

Directories should only be created when they are required.

---

## 23. DATA STRATEGY

The dataset strategy must be designed around the selected problem statement.

Potential sources include:

* Public datasets
* Government/open datasets
* Domain-specific datasets
* Carefully generated synthetic examples
* Human-created examples
* Expert-reviewed examples
* Existing Nigerian datasets

All datasets must have documented provenance.

Personal or sensitive data must not be collected without proper justification, safeguards, and permissions.

---

## 24. DATASET PROCESSING

Dataset preparation should include:

* Cleaning
* Deduplication
* Language identification
* Normalisation
* Validation
* Quality checks
* Train/validation/test splitting
* Leakage prevention
* Metadata tracking

The final dataset methodology must be documented in:

`docs/DATASET.md`

---

## 25. DOMAIN ADAPTATION

If PS3 is selected, domain adaptation should begin with a baseline.

The process should be:

```text
Base N-ATLaS
      ↓
Baseline Benchmark
      ↓
Domain Dataset
      ↓
LoRA / QLoRA Adaptation
      ↓
Adapted N-ATLaS
      ↓
Held-out Benchmark
      ↓
Comparison
```

The project must demonstrate measurable improvement rather than assuming that fine-tuning improves performance.

---

## 26. PEFT / LoRA / QLORA

Parameter-efficient fine-tuning should be considered before full model fine-tuning.

Preferred initial approach:

* LoRA
* QLoRA where hardware constraints require it
* Hugging Face PEFT
* Accelerate

The configuration must record:

* Base model
* Dataset
* LoRA parameters
* Quantisation configuration
* Learning rate
* Epochs
* Batch size
* Gradient accumulation
* Evaluation strategy
* Checkpoint strategy

---

## 27. BENCHMARKING

Benchmarking is a core requirement.

The project must compare the base model against the developed system where applicable.

Metrics should be selected according to the task.

Possible metrics include:

* Accuracy
* Precision
* Recall
* F1
* Exact match
* ROUGE
* BLEU where appropriate
* Semantic similarity
* Structured extraction accuracy
* Human evaluation
* Latency
* Resource consumption

Metrics must correspond to the actual problem.

---

## 28. 500+ EXAMPLE BENCHMARK

For PS3, the benchmark must contain at least 500 examples.

The benchmark should contain:

* Input
* Expected output/reference
* Relevant metadata
* Language
* Domain category
* Evaluation criteria

The benchmark must be held out from training.

Results must compare:

`Base N-ATLaS`

against

`Adapted N-ATLaS`

using the same benchmark.

---

## 29. EVALUATION REPRODUCIBILITY

Every reported metric must be reproducible.

The project should record:

* Dataset version
* Model version
* Code version
* Configuration
* Random seeds
* Evaluation script
* Environment
* Hardware
* Date of evaluation

No manually typed performance numbers should be presented as experimental results unless generated by the evaluation pipeline.

---

## 30. FASTAPI BACKEND

FastAPI should provide the production API layer.

Initial endpoints may include:

```text
GET  /health
GET  /api/v1/model
POST /api/v1/generate
POST /api/v1/analyze
POST /api/v1/evaluate
```

Actual endpoints should be adapted to the final problem.

The API must expose a clear contract.

API documentation should be automatically generated where possible.

---

## 31. STREAMLIT APPLICATION

Streamlit should be used as the primary demonstration interface.

The Streamlit application should communicate with the FastAPI backend rather than embedding all production logic directly inside Streamlit.

Potential UI features include:

* Text input
* Voice input where applicable
* Language selection
* N-ATLaS response
* Structured analysis
* Confidence/quality indicators where meaningful
* Evidence or source information where applicable
* Feedback collection
* Evaluation/demo mode

---

## 32. VOICE ARCHITECTURE

If PS2 is selected:

```text
User Voice
    ↓
Official N-ATLaS ASR
    ↓
Transcribed Text
    ↓
N-ATLaS
    ↓
Domain/Application Logic
    ↓
Response
```

The official N-ATLaS ASR requirement must be respected.

The system must separately evaluate:

* Speech recognition
* Language understanding
* Response quality
* End-to-end latency

---

## 33. HEALTH APPLICATION ARCHITECTURE

If the health direction is selected, the architecture may follow:

```text
User / Health Worker
        ↓
Text or Voice
        ↓
N-ATLaS
        ↓
Structured Health Information
        ↓
Domain Intelligence Layer
        ↓
Validation / Classification
        ↓
FastAPI
        ↓
Dashboard / Alerts / API
```

The system should prioritise health information intelligence rather than autonomous clinical decision-making.

---

## 34. API-FIRST DESIGN

The final system must not depend exclusively on the Streamlit interface.

The core functionality must be accessible through an API.

This enables:

* Mobile integration
* Web integration
* Other developers
* Government systems
* Third-party applications
* Future production clients

For PS3, the API must be sufficiently documented for another developer to integrate.

---

## 35. SECURITY AND PRIVACY

The application must implement reasonable security controls.

These should include:

* Environment variables for secrets
* No API keys committed to Git
* Input validation
* Authentication where required
* Rate limiting where required
* Logging without unnecessary sensitive data
* Secure file handling
* Dependency management
* Basic abuse protection

Sensitive health or personal information must receive additional safeguards.

---

## 36. TESTING

Testing must include:

* Unit tests
* API tests
* Model integration tests
* Dataset validation tests
* Evaluation tests
* Basic end-to-end tests

At minimum, the project must verify:

* N-ATLaS loads
* API starts
* API receives valid input
* Model generates output
* Invalid requests are handled
* Evaluation pipeline runs
* Application communicates with backend

---

## 37. DOCKER AND DEPLOYMENT

The application should be containerised.

Docker should provide reproducible execution.

The project should define:

* Dockerfile
* Environment variables
* Runtime configuration
* Dependency installation
* Startup commands

AWS may be used where cloud resources are required.

The deployment architecture must match available resources and budget.

---

## 38. REAL-WORLD VALIDATION

The project must be tested with actual users where required.

Validation should document:

* Who participated
* Number of participants/interactions
* Date
* Location/context
* Task performed
* User feedback
* Observed failures
* Improvements made
* Final results

Validation evidence must never be fabricated.

For PS1, at least two external beta testers are required.

For PS2, at least 50 documented real user interactions are required.

For PS3, benchmark and domain-validation requirements must be satisfied.

---

## 39. DOCUMENTATION

The repository must maintain the following documents:

```text
docs/
├── FEASIBILITY.md
├── PROJECT-DEFINITION.md
├── NATLAS-INTEGRATION.md
├── ARCHITECTURE.md
├── DATASET.md
├── EVALUATION.md
├── VALIDATION.md
├── API.md
└── DEPLOYMENT.md
```

Documentation must describe what was actually built.

It must distinguish:

* Verified facts
* Measurements
* Assumptions
* Hypotheses
* Future work

---

## 40. NAIC SUBMISSION PACKAGE

The final submission should contain the required evidence, including:

* Working artefact URL
* N-ATLaS integration evidence PDF
* Real-world validation PDF
* Technical documentation PDF
* 3–5 minute demonstration video
* Team profile PDF
* CAC/identity documentation where applicable

The submission must demonstrate genuine N-ATLaS integration.

The project must not rely on screenshots alone where the competition requires a functioning artefact.

---

## 41. DEVELOPMENT RULES AND CLAUDE CODE INSTRUCTIONS

Claude Code must treat this document as the governing project specification.

### Core rules

1. Read `CLAUDE.md` completely before modifying the project.
2. Do not fabricate N-ATLaS outputs.
3. Do not mock N-ATLaS integration and present it as genuine.
4. Do not fabricate benchmarks.
5. Do not fabricate user-validation results.
6. Do not fabricate deployment evidence.
7. Do not claim a model was fine-tuned unless it actually was.
8. Do not report unmeasured performance.
9. Keep experiments reproducible.
10. Prefer the smallest working architecture.
11. Avoid unnecessary dependencies.
12. Keep production logic outside notebooks.
13. Use notebooks for exploration only.
14. Keep secrets outside source control.
15. Document important technical decisions.
16. Test changes before declaring them complete.
17. Preserve a clean Git history.
18. Use Python as the primary programming language.
19. Use PyTorch as the primary ML framework.
20. Use Hugging Face Transformers for N-ATLaS.
21. Do not convert N-ATLaS to TensorFlow.
22. Use PEFT/LoRA/QLoRA where appropriate.
23. Use FastAPI for the production API.
24. Use Streamlit for the primary demonstration UI.
25. Use Docker for reproducibility.
26. Use AWS only where it provides a practical deployment or compute advantage.
27. Validate against real data.
28. Use held-out evaluation data.
29. Respect N-ATLaS licensing restrictions.
30. Preserve required N-ATLaS attribution.
31. Respect privacy and responsible-AI requirements.
32. Do not introduce autonomous high-risk decision-making without appropriate safeguards.
33. Make language support explicit.
34. Record model and dataset versions.
35. Record hardware and environment for benchmarks.
36. Keep evaluation scripts executable.
37. Make API documentation clear enough for external developers.
38. Build validation evidence during development rather than at the end.
39. Prioritise actual working functionality over visual complexity.
40. Do not begin major product development until N-ATLaS feasibility is established.
41. Every major technical claim must be supported by code, measurement, documentation, or an identified source.

---

# IMMEDIATE CLAUDE CODE TASK

Before building the final product, Claude Code must execute the following task:

> Read this entire specification before making changes.
>
> Do not build the final application yet.
>
> First inspect the repository and development environment.
>
> Detect:
>
> * Python version
> * CPU
> * RAM
> * GPU
> * VRAM
> * CUDA
> * PyTorch
> * available disk space
>
> Create a clean Python environment and install only the minimum dependencies required to test N-ATLaS.
>
> Inspect the actual N-ATLaS repository/model configuration and verify the correct loading and inference process.
>
> Attempt genuine N-ATLaS inference using the native Hugging Face/PyTorch implementation.
>
> Do not mock or simulate model output.
>
> Measure:
>
> * Model download size
> * Model loading time
> * Inference latency
> * Memory consumption
> * GPU/CPU utilisation where practical
> * Maximum practical input/output configuration
>
> Create:
>
> `scripts/hardware_check.py`
>
> `scripts/test_natlas.py`
>
> `docs/FEASIBILITY.md`
>
> The feasibility report must state:
>
> 1. Whether N-ATLaS runs locally.
> 2. Required RAM.
> 3. Required VRAM.
> 4. CPU/GPU configuration.
> 5. PyTorch configuration.
> 6. Inference latency.
> 7. Whether quantisation is required.
> 8. Whether LoRA/QLoRA is realistically possible locally.
> 9. Whether cloud GPU resources are likely to be required.
> 10. Risks and limitations.
> 11. Initial feasibility of PS1, PS2, and PS3.
>
> Do not proceed to major application development until the feasibility findings have been documented.

---

# FINAL ARCHITECTURAL PRINCIPLE

The project should follow this principle:

**N-ATLaS is the AI foundation. PyTorch is the ML runtime. Python is the primary language. FastAPI is the service layer. Streamlit is the demonstration layer. Docker provides reproducibility. AWS provides scalable infrastructure where necessary.**

The goal is not to build the largest system.

The goal is to build a **real, useful, measurable, demonstrable Nigerian AI solution powered by N-ATLaS**.
