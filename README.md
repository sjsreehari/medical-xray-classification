# 🫁 Pediatric Chest X-Ray Disease Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![ROC-AUC 96.82%](https://img.shields.io/badge/Test%20ROC--AUC-96.82%25-brightgreen.svg)]()
[![Recall 94.36%](https://img.shields.io/badge/Pneumonia%20Recall-94.36%25-success.svg)]()

An end-to-end clinical AI triage and visual diagnostic system for automated detection of **Pediatric Pneumonia** from anterior-posterior chest radiographs. Built with **Deep Residual Networks (ResNet-18)**, **Dual-Level Class Imbalance Compensation**, and **Gradient-Weighted Class Activation Mapping (Grad-CAM)** for transparent "RGB-like" thermal visual interpretability.

Includes a high-performance **FastAPI REST API** backend and an interactive **Streamlit** clinician dashboard.

---

## 📌 Table of Contents
- [Clinical Background & Overview](#-clinical-background--overview)
- [System Architecture](#-system-architecture)
- [Scientific Methodology](#-scientific-methodology)
  - [Neural Network Model](#1-neural-network-model)
  - [Dataset & Cohort Partitioning](#2-dataset--cohort-partitioning)
  - [Input Features & Augmentation Pipeline](#3-input-features--augmentation-pipeline)
  - [Training & Optimization Strategy](#4-training--optimization-strategy)
- [Benchmark Evaluation & Results](#-benchmark-evaluation--results)
- [Interpretability Engine (Grad-CAM & RGB Heatmap)](#-interpretability-engine-grad-cam--rgb-heatmap)
- [Project Directory Structure](#-project-directory-structure)
- [Quick Start Guide](#-quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [One-Command Launch](#one-command-launch)
  - [Manual Execution](#manual-execution)
- [REST API Endpoints](#-rest-api-endpoints)
- [Detailed Scientific Report](#-detailed-scientific-report)
- [Citation & License](#-citation--license)

---

## 🩺 Clinical Background & Overview
Pneumonia is the leading infectious cause of mortality in children under five years of age worldwide. Accurate radiographic triage reduces diagnostic latency and prevents both inappropriate antibiotic administration and delayed critical care.

This system provides:
1. **High Diagnostic Sensitivity (94.36% Recall)**: Minimizes life-threatening false-negative omissions.
2. **Robust Discriminative Capacity (96.82% ROC-AUC)**: Delivers reliable probability estimates across diverse patient populations.
3. **Transparent Explainability**: Produces localized pseudo-color RGB heatmaps that expose where the neural network detects pulmonary infiltrates and consolidation.

---

## 🏗️ System Architecture

```
[ Raw Chest X-Ray (JPEG/PNG) ]
             │
             ▼
[ Feature Pipeline: Grayscale Broadcast (3-Ch) + Z-Score Norm ]
             │
             ▼
[ Deep Residual CNN Backbone: ResNet-18 (ImageNet Pretrained) ]
             │
             ├─────────────────────────────────────────────────┐
             ▼ (Forward Pass)                                  ▼ (Backward Gradient Hooks)
[ Dropout (p=0.5) + Linear Head (2-Class) ]             [ Grad-CAM Layer-4 Convolution Hooks ]
             │                                                 │
             ▼                                                 ▼
[ Logits / Softmax Probabilities ]               [ GAP Gradient Weights + ReLU Feature Maps ]
             │                                                 │
             │                                                 ▼
             │                                   [ Bilinear Upsampling + JET Colormap (BGR->RGB) ]
             │                                                 │
             │                                                 ▼
             │                                   [ Alpha Blending (α=0.45) with Original X-Ray ]
             │                                                 │
             └─────────────────┬───────────────────────────────┘
                               ▼
            [ FastAPI Production Server (Port 8000) ]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [ Streamlit Clinician UI (8501) ]       [ Client Applications ]
```

---

## 🔬 Scientific Methodology

### 1. Neural Network Model
- **Backbone**: Deep Residual Network (**ResNet-18**; He et al., CVPR 2016).
  - Employs residual skip connections ($H(x) = F(x) + x$) to solve the vanishing gradient and feature degradation problems.
  - Consists of 4 residual stages with 8 `BasicBlock` modules totaling 18 parameterized layers.
  - Initialized with **ImageNet-1K V1** weights (`models.ResNet18_Weights.IMAGENET1K_V1`), giving the network instant awareness of edge, boundary, and textural features.
- **Classification Head**:
  - Global Average Pooling (GAP) collapsing spatial dimensions to 512 dimensions.
  - Dropout layer (`p=0.5`) to suppress feature co-adaptation and mitigate overfitting.
  - Linear classifier (`nn.Linear(512, 2)`) generating logits for `[NORMAL, PNEUMONIA]`.

### 2. Dataset & Cohort Partitioning
- **Source**: Guangzhou Women and Children's Medical Center Pediatric Chest X-Ray Dataset (Kermany et al., Cell 2018 / Mooney, Kaggle).
- **Cohort Split**:
  - **Train**: 5,216 images (1,341 Normal, 3,875 Pneumonia — ~1:2.89 imbalance).
  - **Val**: 16 images (8 Normal, 8 Pneumonia).
  - **Test**: 624 clinical radiographs (234 Normal, 390 Pneumonia).

### 3. Input Features & Augmentation Pipeline
- **Channel Adaptation**: Radiographs are grayscale. To leverage 3-channel pretrained filters without modifying initial weights, images are replicated into 3 channels via `transforms.Grayscale(num_output_channels=3)`.
- **Z-Score Normalization**: Channel-wise ImageNet standardisation:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
- **Training Data Augmentations**:
  - Scale & Crop: Resize to $244 \times 244$, followed by `RandomCrop(224)`.
  - Bilateral Symmetry: `RandomHorizontalFlip(p=0.5)`.
  - Radiographic Tilt: `RandomRotation(degrees=10)`.
  - Photometric Jitter: `ColorJitter(brightness=0.2, contrast=0.2)` to model X-ray beam voltage ($kV_p$) and sensor variations.

### 4. Training & Optimization Strategy
- **Dual-Level Class Imbalance Compensation**:
  1. *Data Sampling*: `WeightedRandomSampler` with inverse class frequency weights ($w_c = 1 / N_c$).
  2. *Loss Objective*: Weighted Cross-Entropy Loss:
     $$\mathcal{L} = -\sum_{c} w_c \cdot y_c \cdot \log(\hat{p}_c)$$
     with $w_{\text{NORMAL}} = 1.0$ and $w_{\text{PNEUMONIA}} = 0.333$.
- **Optimizer**: Adam ($\alpha = 1 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, Weight Decay $\lambda = 1 \times 10^{-4}$).
- **Scheduler**: `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)` monitoring validation accuracy.
- **Early Stopping**: Patience = 4 epochs. Model converged at Epoch 6 with best validation accuracy of **100.00%**.

---

## 📊 Benchmark Evaluation & Results

Evaluated on the independent 624-image test set:

| Diagnostic Metric | Score | Clinical Relevance |
| :--- | :---: | :--- |
| **Accuracy** | **91.99%** | Overall correct triage rate across both cohorts |
| **Pneumonia Sensitivity (Recall)** | **94.36%** | **Critical safety metric**: Correctly captures 368 / 390 pneumonia cases |
| **Pneumonia Precision** | **92.93%** | Minimizes false positive alarms and overtreatment |
| **Specificity (Normal Recall)** | **88.03%** | Correctly identifies 206 / 234 normal healthy lungs |
| **F1-Score** | **93.64%** | Harmonic balance between precision and sensitivity |
| **ROC-AUC** | **96.82%** | Near-optimal discrimination regardless of decision threshold |
| **Average Precision (PR-AUC)**| **97.01%** | Area under precision-recall curve under class skew |

### Confusion Matrix Breakdown (N = 624)
- **True Negatives (TN)**: 206 (Normal correctly identified)
- **False Positives (FP)**: 28 (Normal flagged as Pneumonia)
- **False Negatives (FN)**: 22 (Pneumonia missed)
- **True Positives (TP)**: 368 (Pneumonia correctly identified)

### Evaluation Artifacts
All high-resolution diagnostic plots are saved in `outputs/`:
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`
- `outputs/pr_curve.png`
- `outputs/training_history.png`
- `outputs/metrics.json`
- `outputs/classification_report.txt`

---

## 🎨 Interpretability Engine (Grad-CAM & RGB Heatmap)

### Mathematical Formulation
Grad-CAM (Selvaraju et al., ICCV 2017) computes neuron importance weights $\alpha_k^c$ via Global Average Pooling of the gradients of class score $Y^c$ with respect to the last convolutional feature map $A^k$ (`model.layer4[-1].conv2`):

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial Y^c}{\partial A_{i, j}^k}$$

The localized class activation map is obtained via rectified linear combination:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k=1}^{512} \alpha_k^c A^k\right)$$

The $\text{ReLU}$ ensures only features that *positively* support the predicted disease class are visualized.

### The "RGB-Like" Pseudo-Color Superimposition
1. **Spatial Interpolation**: Upsamples the $7 \times 7$ normalized activation matrix to $224 \times 224$ via bilinear interpolation (`cv2.resize`).
2. **JET Colormap Application**: Quantizes to $[0, 255]$ and applies `cv2.applyColorMap(..., cv2.COLORMAP_JET)`:
   - **Deep Blue**: Inactive baseline regions (healthy tissue, mediastinum).
   - **Cyan / Green**: Moderate background activation.
   - **Yellow / Red**: Peak gradient concentration indicating pulmonary consolidations and infiltrates.
3. **BGR to RGB Conversion**: Converts OpenCV BGR channels to standard RGB (`cv2.cvtColor`).
4. **Alpha Blending**: Superimposes the RGB heatmap over the original radiograph:
   $$\text{Overlay}_{\text{RGB}} = \alpha \cdot \text{Heatmap}_{\text{RGB}} + (1 - \alpha) \cdot \text{X-Ray}_{\text{RGB}}$$
   with $\alpha = 0.45$.

---

## 📁 Project Directory Structure

```
├── .gitignore               # Excludes virtualenvs, 2.3GB dataset & 128MB checkpoints
├── README.md                # System documentation
├── report.txt               # Full scientific and technical specification report
├── requirements.txt         # Pinned Python package dependencies
├── start.sh                 # Unified startup script for backend and UI
├── train.py                 # End-to-end training and evaluation CLI runner
├── api.py                   # FastAPI REST API serving endpoints
├── app.py                   # Streamlit clinical dashboard
├── src/
│   ├── __init__.py
│   ├── config.py            # Global paths, hyperparameters, device auto-selection (MPS/CUDA/CPU)
│   ├── dataset.py           # Kaggle downloader, dataset classes, and weighted sampler
│   ├── model.py             # ResNet-18 architecture with custom head & checkpoint manager
│   ├── train.py             # PyTorch training loop with ReduceLROnPlateau & early stopping
│   ├── evaluate.py          # Metric calculations, confusion matrix, ROC/PR curves
│   └── gradcam.py           # Manual PyTorch hook Grad-CAM & RGB overlay generator
├── outputs/                 # Evaluation plots and metrics JSON
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── training_history.png
│   ├── metrics.json
│   └── classification_report.txt
└── logs/                    # Training execution logs
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+
- Hardware: Apple Silicon (M1/M2/M3/M4 with MPS), NVIDIA GPU (CUDA), or multi-core CPU.

### One-Command Launch
To automatically set up the virtual environment, verify dependencies, and launch both FastAPI and Streamlit concurrently:

```bash
chmod +x start.sh
./start.sh
```

- **FastAPI Backend**: `http://localhost:8000` (Swagger docs at `/docs`)
- **Streamlit Frontend**: `http://localhost:8501`

### Manual Execution

1. **Activate virtual environment & install requirements**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train the model & run evaluation**:
   ```bash
   python train.py
   ```

3. **Start the FastAPI backend**:
   ```bash
   python api.py
   ```

4. **Launch the Streamlit clinician dashboard** (in a new terminal):
   ```bash
   streamlit run app.py
   ```

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service health status, model architecture, and class labels |
| `POST`| `/predict` | Multipart upload of X-ray image (`file`). Returns predicted class, confidence, probability distribution, and Base64-encoded Grad-CAM RGB heatmap |
| `GET` | `/metrics` | Returns test accuracy, sensitivity, precision, F1-score, and ROC-AUC |

### Example Inference Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_chest_xray.jpeg"
```

---

## 📄 Detailed Scientific Report
For an in-depth mathematical breakdown, epoch logs, and medical discussions, see:
- [`report.txt`](./report.txt)

---

## 📜 Citation & References
1. **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR.
2. **Grad-CAM**: Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. ICCV.
3. **Dataset**: Kermany, D. S., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122-1131.
