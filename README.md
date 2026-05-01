# 🤲 nedo.ai | नेदो
### *Real-Time Indian Sign Language to Text — Breaking Silence, Building Bridges*

<div align="center">

<!-- HERO IMAGE PLACEHOLDER -->
<!-- Replace with your actual banner: 1280×640px recommended -->
<!-- Suggested: A split-panel showing a signing hand on the left, translated medical text on the right -->
![nedo.ai Banner](https://placehold.co/1280x400/0f172a/38bdf8?text=nedo.ai+%E2%80%94+Real-Time+ISL+Translation&font=raleway)

<br/>

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![CUDA 12.6+](https://img.shields.io/badge/CUDA-12.6+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![Status: Research](https://img.shields.io/badge/Status-Active%20Research-f59e0b?style=for-the-badge)]()

<br/>

*"Because every patient deserves to be heard — regardless of the words they can speak."*

</div>

---

## 🎯 The Mission

In India, over **18 million people** live with speech and hearing impairments. When they enter a hospital — a triage ward, an emergency room, a consultation desk — the existing communication infrastructure fails them completely. A translator is rarely present. Critical time is lost. Care is compromised.

**nedo.ai** is a privacy-first, real-time system that translates **Continuous Indian Sign Language (ISL)** into text directly at the point of care. Built for clinical environments, it empowers deaf and mute patients to communicate their symptoms, history, and needs — *instantly, accurately, and without intermediaries.*

We believe assistive technology isn't a feature. It's a fundamental right.

---

## 🪷 Etymology — The Name Behind the Mission

> *"Aneḍamūka"* — **अनेडमूक** (Sanskrit)
> Meaning: *"One who is deaf and mute"*. A term found in classical Sanskrit texts, used not as a pejorative, but as a descriptor rooted in the observation of lived experience.

The name **nedo.ai** is a deliberate contraction of **Aneḍamūka** — preserving the cultural and linguistic heritage of the Sanskrit origin while adopting the lean, modern syntax of a technology startup. The `.ai` suffix reflects our commitment to accessible, intelligent systems built around human need.

We chose this name as an act of recognition: the community this project serves has a name, a history, and a language of their own. *ISL is not a limitation — it is a language.*

---

## 🏛️ Technical Architecture

<!-- ARCHITECTURE DIAGRAM PLACEHOLDER -->
<!-- Replace with your actual system diagram -->
<!-- Suggested tools: Excalidraw, Lucidchart, or draw.io -->
<!-- Dimensions: 1200×600px, dark background recommended -->

<br/>

### 🦴 Pillar I — The Skeletal Pipeline (Privacy-Preserving Landmark Extraction)

Raw video of patients is a **privacy liability**. nedo.ai never stores it.

Instead, we use **MediaPipe Holistic** to extract a structured skeleton of **543 spatial landmarks** per frame in real-time. This includes:

| Landmark Group | Count | Description |
|---|---|---|
| Pose | 33 | Full upper-body joint positions |
| Left Hand | 21 | Per-finger and palm keypoints |
| Right Hand | 21 | Per-finger and palm keypoints |
| Face Mesh | 468 | Facial expression and lip-reading support |

Every downstream model, training run, and inference call operates **only on these normalised `(x, y, z, visibility)` coordinate tensors** — never on pixel data. This design choice is non-negotiable and is enforced at the pipeline level.

```
Input Frame (RGB) ──► MediaPipe Holistic ──► 543 × 4 Landmark Tensor
                                                      │
                                          NO VIDEO STORED ✓
                                          NO FACES RETAINED ✓
                                          GDPR/HIPAA ALIGNED ✓
```

---

### 🧠 Pillar II — Self-Supervised Learning for Motion Representation (SSL)

Labelled ISL datasets are scarce and expensive to produce. nedo.ai addresses this with a **Self-Supervised Learning (SSL)** pre-training strategy that learns rich motion representations from *unlabelled* landmark sequences.

The core idea: teach the model to **predict masked or future motion states** from context — much like how BERT learns language by predicting hidden tokens. This gives us:

- ✅ **Data efficiency** — the model pre-trains on large volumes of unlabelled signing video landmarks
- ✅ **Transfer learning** — pre-trained weights fine-tune rapidly on small labelled ISL corpora
- ✅ **Robustness** — the model learns the *geometry of human motion*, not superficial visual textures

The pre-training objective is a **masked joint prediction task** over spatio-temporal graph sequences, followed by supervised fine-tuning on the ISL gloss vocabulary.

---

### 🔤 Pillar III — Continuous Sequence Decoding with CTC

Sign language is continuous — one sign flows into the next with no explicit boundary marker. Frame-level classification fails here. We use **Connectionist Temporal Classification (CTC) Loss** to handle this directly.

CTC allows the model to:
1. Output a probability distribution over the ISL vocabulary at *every time step*
2. Collapse duplicate and blank tokens via the CTC decoding algorithm
3. Produce the most probable word/gloss sequence *without requiring segmented training data*

The decoder head sits atop a **Bidirectional GRU (Bi-GRU)**, which aggregates the spatio-temporal features extracted by the **ST-GCN** backbone — giving the model both spatial graph structure awareness and temporal sequence memory.

```
Landmark Tensor
      │
  [ST-GCN Backbone]        ← Spatio-temporal graph convolutions over joint skeleton
      │
  [Bi-GRU Sequence Model]  ← Forward + backward temporal context fusion
      │
  [CTC Decoder Head]       ← Continuous, unsegmented ISL sequence → Text
      │
  ISL Text Output ✓
```

---

## 🛠️ Tech Stack

| Layer | Tool | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.11 | Core runtime |
| **Deep Learning** | PyTorch | 2.x | Model training & inference |
| **Training Framework** | PyTorch Lightning | 2.x | Reproducible training loops, logging, checkpointing |
| **Pose Estimation** | MediaPipe Holistic | 0.10.x | Real-time 543-landmark extraction |
| **Graph Network** | ST-GCN | Custom | Spatio-temporal skeleton modelling |
| **Sequence Model** | Bi-GRU | PyTorch built-in | Temporal sequence encoding |
| **Loss Function** | CTC Loss | PyTorch built-in | Continuous sequence alignment |
| **Mixed Precision** | AMP / FP16 | PyTorch built-in | VRAM optimisation for RTX 2050 |
| **Inference Runtime** | ONNX Runtime | 1.18.x | Optimised deployment inference |
| **Pre-training** | SSL (Masked Joint Prediction) | Custom | Unlabelled data representation learning |
| **Experiment Tracking** | Weights & Biases | Latest | Loss curves, metrics, model versioning |
| **Environment** | Conda / venv | — | Dependency isolation |

---

## ⚙️ Installation

> **Hardware Target:** NVIDIA RTX 2050 (4GB VRAM) | CUDA 12.6+ | Driver 525+
> These steps are verified for this configuration. Adjust `batch_size` and `mixed_precision` flags for other hardware.

### 1. Prerequisites

Ensure the following are installed on your system before proceeding:

- **OS:** Ubuntu 22.04 LTS / Windows 11 (WSL2 recommended for Linux parity)
- **GPU Driver:** NVIDIA Driver ≥ 525.x
- **CUDA Toolkit:** 12.6+ → [Download](https://developer.nvidia.com/cuda-downloads)
- **cuDNN:** 9.x compatible with CUDA 12.6 → [Download](https://developer.nvidia.com/cudnn)
- **Python:** 3.11 (via `pyenv` or `conda` recommended)

Verify your CUDA setup:
```bash
nvidia-smi
nvcc --version
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/nedo.ai.git
cd nedo.ai
```

---

### 3. Create and Activate Virtual Environment

**Using `conda` (recommended):**
```bash
conda create -n nedo python=3.11 -y
conda activate nedo
```

**Using `venv`:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# OR
.venv\Scripts\activate      # Windows
```

---

### 4. Install PyTorch with CUDA 12.6 Support

> ⚠️ **Critical:** Do NOT install PyTorch via `requirements.txt` first. Install it manually with the correct CUDA index.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> *Note: PyTorch uses `cu124` as the closest stable wheel for CUDA 12.x at time of writing. Check [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the latest.*

Verify the install:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True | NVIDIA GeForce RTX 2050
```

---

### 5. Install Project Dependencies

```bash
pip install -r requirements.txt
```

Key packages included in `requirements.txt`:
```
pytorch-lightning>=2.0
mediapipe>=0.10
onnxruntime-gpu>=1.18
wandb
numpy
opencv-python
scikit-learn
```

---

### 6. Configure for RTX 2050 (4GB VRAM)

The project ships with a configuration file optimised for the RTX 2050's 4GB VRAM constraint. Open `configs/hardware/rtx2050.yaml` and confirm:

```yaml
# configs/hardware/rtx2050.yaml
trainer:
  accelerator: gpu
  devices: 1
  precision: "16-mixed"        # AMP FP16 — critical for 4GB VRAM
  accumulate_grad_batches: 4   # Simulates larger effective batch size

dataloader:
  batch_size: 16               # Tune down to 8 if OOM errors occur
  num_workers: 4
  pin_memory: true

model:
  use_onnx_export: true        # Enable ONNX export after training
```

---

### 7. Run a Quick Smoke Test

```bash
python scripts/smoke_test.py --config configs/hardware/rtx2050.yaml
```

You should see landmark extraction running live from your webcam and a `[PASS]` confirmation in the terminal.

---

## 🗺️ Roadmap

```
┌─────────────────────────────────────────────────────────────────────┐
│                        nedo.ai Development Phases                   │
└─────────────────────────────────────────────────────────────────────┘
```

### ✅ Phase 0 — Foundation: Landmark Pipeline *(In Progress)*
- [x] MediaPipe Holistic integration (543 landmarks)
- [x] Real-time webcam inference loop
- [x] Privacy-preserving data serialisation (no raw video)
- [ ] ISL dataset landmark extraction & preprocessing scripts
- [ ] Data augmentation pipeline (joint dropout, spatial jitter, speed perturbation)

---

### 🔬 Phase 1 — Representation Learning: SSL Pre-training
- [ ] Design masked joint prediction pretext task
- [ ] Build spatio-temporal graph data loader
- [ ] Implement ST-GCN backbone architecture
- [ ] Pre-train on large unlabelled ISL motion corpus
- [ ] Evaluate learned representations via downstream probing

---

### 🧩 Phase 2 — Supervised Fine-Tuning: CTC Decoder
- [ ] Attach Bi-GRU temporal encoder to pre-trained ST-GCN
- [ ] Implement CTC loss head and beam search decoder
- [ ] Fine-tune on labelled ISL gloss dataset
- [ ] Evaluate: WER (Word Error Rate), CER (Character Error Rate)
- [ ] Mixed precision training on RTX 2050 with gradient accumulation

---

### 🚀 Phase 3 — Optimisation & Deployment
- [ ] Export best checkpoint to ONNX format
- [ ] Benchmark ONNX Runtime inference latency (target: <100ms end-to-end)
- [ ] Build lightweight FastAPI / WebSocket real-time inference server
- [ ] Develop clinical UI prototype (web-based, touch-friendly for triage settings)
- [ ] Conduct pilot evaluation with ISL interpreter feedback
- [ ] Open-source model weights on Hugging Face Hub

---

## 📁 Project Structure

```
nedo.ai/
├── configs/
│   ├── hardware/
│   │   └── rtx2050.yaml        # Hardware-specific training config
│   └── model/
│       └── stgcn_bigru.yaml    # Model architecture config
├── data/
│   ├── raw/                    # Raw landmark tensors (no video)
│   └── processed/              # Augmented, split datasets
├── nedo/
│   ├── models/
│   │   ├── stgcn.py            # ST-GCN backbone
│   │   ├── bigru.py            # Bi-GRU temporal encoder
│   │   └── ctc_head.py         # CTC decoder head
│   ├── pipeline/
│   │   ├── extractor.py        # MediaPipe landmark extractor
│   │   └── preprocessor.py     # Normalisation & augmentation
│   ├── ssl/
│   │   └── masked_joint.py     # SSL pre-training objective
│   └── utils/
│       └── metrics.py          # WER / CER evaluation
├── scripts/
│   ├── extract_landmarks.py    # Run landmark extraction on video corpus
│   ├── pretrain.py             # SSL pre-training entry point
│   ├── finetune.py             # CTC fine-tuning entry point
│   ├── export_onnx.py          # Export to ONNX for deployment
│   └── smoke_test.py           # Quick hardware & pipeline validation
├── tests/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🤝 Contributing

Contributions, especially from the **Deaf and hard-of-hearing community**, ISL interpreters, and accessibility researchers, are warmly welcomed.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit with clear messages: `git commit -m 'feat: add masked joint SSL objective'`
4. Open a Pull Request with a description of motivation and changes

Please read `CONTRIBUTING.md` for our code standards and `CODE_OF_CONDUCT.md` for community guidelines.

---

## 📄 License

```
MIT License

Copyright (c) 2025 nedo.ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

See [LICENSE](LICENSE) for the full text.

---

## 🙏 Acknowledgements

- The **Indian Sign Language Research and Training Centre (ISLRTC)** for ISL standardisation efforts
- The **MediaPipe** team at Google for the open-source pose estimation framework
- The researchers behind **ST-GCN** (Yan et al., 2018) and **CTC** (Graves et al., 2006)
- Every deaf and mute patient whose unheard voice is the reason this project exists

---

<div align="center">

**nedo.ai | नेदो**
*Breaking Silence. Building Bridges. One Sign at a Time.*

<br/>

Built with 🤍 for Healthcare Accessibility · India

</div>
