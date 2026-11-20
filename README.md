# Network-Aware Adaptive Image Deblurring
### Bridging Deep Learning and Classical Digital Image Processing for Resource-Constrained Environments

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1%2Bcu121-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview
This repository contains the complete implementation of **Network-Aware Adaptive Image Deblurring** (B.Tech Capstone Project). 

Traditional image restoration frameworks assume a single massive deep network should solve all blur types. In reality:
* **Motion blur** is directional and trajectory-dependent (requiring deep learning transformers to trace motion vectors).
* **Defocus blur** is an optical depth-of-field phenomenon governed by aperture geometry (solvable with exact, zero-weight classical physics).

This project introduces a **dual-branch, network-aware architecture** that dynamically routes computation across physical degradation types and hardware tiers:
1. **Motion Blur Branch (Deep Learning):**
   * **Cloud Tier:** High-capacity **Restormer Transformer** ($26.1\text{M}$ parameters, GPU) with Multi-Dconv Head Transposed Attention (MDTA), Gated Feed-Forward Networks (GDFN), and memory-safe overlapping tile inference.
   * **Edge Tier:** Ultra-compact **Lightweight Residual CNN** ($94,243$ parameters; $0.37\text{ MB}$, CPU), delivering a **$277.2\times$ parameter reduction** and $\sim 1.99\text{ s}$ CPU execution.
   * **Multi-Domain Triplet Loss:** $\mathcal{L}_{\text{total}} = 0.5\mathcal{L}_{\text{charb}} + 0.3\mathcal{L}_{\text{SSIM}} + 0.2\mathcal{L}_{\text{edge}}$.
2. **Defocus Blur Branch (Classical DIP):**
   * **Tikhonov-Regularized Wiener Deconvolution:** Frequency-domain inversion featuring a 2D discrete Laplacian prior ($\lambda |P|^2$) that completely suppresses the high-frequency noise grain typical of unregularized inverse filters.
   * **White's Damped Richardson-Lucy:** Iterative spatial Expectation-Maximization with noise-floor thresholding ($\tau = 0.003$) and non-negativity preservation.
3. **Adaptive Decision Engine:** Client-side telemetry layer balancing image resolution, server availability, and network serialization overhead (which accounts for $\sim 37\%$ of total cloud latency).

---

## 📊 Benchmark Results

### 1. Motion Deblurring Benchmark (GoPro Test Split)
| Model / Pipeline | Target Hardware | Parameter Count | Checkpoint Size | MSE ↓ | PSNR (dB) ↑ | SSIM ↑ | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Degraded Blurred Input** | — | — | — | 0.001734 | 28.30 | 0.8712 | — |
| **Edge Lightweight CNN** | **Local Edge CPU** | **94,243** | **0.37 MB** | 0.001540 | **28.88** | **0.8795** | **1.99 s (CPU)** |
| **Custom-Loss Restormer** | **Cloud GPU** | 26,126,644 | 99.84 MB | 0.000365 | **34.79** | **0.9580** | 19.75 s (1.21 s GPU) |
| **Original SOTA Restormer** | **Cloud GPU** | 26,126,644 | 99.85 MB | 0.000361 | **34.82** | **0.9584** | 20.05 s (1.20 s GPU) |

### 2. Defocus Deblurring Benchmark (54 Parameter Conditions)
| Defocus Radius ($R$) | Noise Level ($\sigma$) | Degraded Input PSNR | Tikhonov Wiener PSNR | White's Damped RL PSNR | Winner |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$R = 2.5\text{ px}$** | $\sigma = 0.000\text{ (Noiseless)}$ | 30.44 dB | **40.56 dB** (SSIM: 0.9881) | 33.99 dB (SSIM: 0.9570) | **Wiener (+10.12 dB)** |
| **$R = 2.5\text{ px}$** | $\sigma = 0.005\text{ (Low Noise)}$ | 30.31 dB | 25.40 dB (SSIM: 0.5026) | **32.94 dB** (SSIM: 0.9103) | **Damped RL (+2.63 dB)** |
| **$R = 2.5\text{ px}$** | $\sigma = 0.010\text{ (Med Noise)}$ | 29.96 dB | 19.83 dB (SSIM: 0.2698) | **30.85 dB** (SSIM: 0.8033) | **Damped RL (+0.89 dB)** |

---

## 🚀 Quickstart & Execution

### Option A: 1-Click Launch (Windows)
Double-click `start_project.bat` in the root folder. It automatically boots the FastAPI GPU server and launches the Streamlit web interface.

### Option B: Manual Launch (Two Terminals)

1. **Terminal 1: Start GPU Server (FastAPI)**
   ```powershell
   cd Restormer/Motion_Deblurring
   python -m uvicorn server:app --host 127.0.0.1 --port 8000
   ```

2. **Terminal 2: Start Interactive UI (Streamlit)**
   ```powershell
   cd Restormer/Motion_Deblurring
   streamlit run app.py --server.port 8501
   ```
   Open **`http://localhost:8501`** in your browser.

---

## 📂 Repository Structure
```
Image_Deblurring_Project/
├── start_project.bat                                # 1-click Windows launcher
├── test_defocus.py                                  # Unit tests for classical DIP package
├── requirements.txt                                 # Core Python dependencies
├── restoration/                                     # Classical Digital Image Processing Branch
│   └── defocus/
│       ├── psf.py                                   # Sub-pixel anti-aliased circular disk PSF
│       ├── wiener.py                                # Tikhonov-regularized Wiener deconvolution
│       ├── richardson_lucy.py                       # White's Damped Richardson-Lucy deconvolution
│       ├── blur.py                                  # Controlled synthetic defocus convolution
│       └── evaluate_defocus.py                      # Automated 54-condition parameter sweep
├── results/                                         # Benchmark outputs and evaluation data
│   └── defocus_eval/
│       └── defocus_benchmark_results.csv            # 54-condition quantitative results
└── Restormer/                                       # Deep Learning Motion Deblurring Branch
    ├── basicsr/
    │   └── models/losses/custom_loss.py             # Differentiable Charbonnier + SSIM + Sobel loss
    └── Motion_Deblurring/
        ├── app.py                                   # Dual-branch Streamlit web application
        ├── server.py                                # Memory-safe tiled FastAPI GPU server
        └── lightweight/                             # Edge client micro-architecture
            ├── model.py                             # 94k-parameter 5-block Residual CNN
            └── checkpoints_continued/               # Lightweight trained model checkpoint
```

---

## 👤 Author
* **Tanishk Raj** – B.Tech Computer Science & Engineering (Capstone Project, November 2026).
