# PHI-Twin Reproducibility & Benchmark Execution Guide

This document provides complete instructions for reproducing all benchmark figures, tables, and empirical evaluations reported in the manuscript:
> *"PHI-Twin: An Edge Digital Twin Framework for Aircraft Landing Gear Prognostics via Physics-Based Simulation and Explainable AI"*

---

## 1. System Requirements & Environment Setup

* **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
* **Python Version:** 3.10+
* **Hardware:** Any standard CPU workstation (GPU with CUDA support is optional but will accelerate training).

### Installation:
```bash
# Clone the repository
git clone https://github.com/Sm-bello/PHI-Twin-LandingGear-B787-version.git
cd PHI-Twin-LandingGear-B787-version

# Create and activate a virtual environment
conda create -n phitwin python=3.10 -y
conda activate phitwin

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Fast Evaluation (Under 5 Seconds)

To verify the pre-trained weights (`models/best_model.pt`) and fitted zero-leakage scalers without re-training:
```bash
python src/evaluate.py
```
Expected output:
* **Accuracy:** $\approx 80.4\% - 82.8\%$
* **Catastrophic Detection (Class 10 AOG & Class 11 Bogie):** 100% Precision & Recall.

---

## 3. Reproducing Stress Tests & Robustness Sweeps

### A. AWGN Noise Robustness Sweeps (Table 2 & Figure 5):
Evaluates the model under Additive White Gaussian Noise from $\infty$ down to 0 dB SNR:
```bash
python src/noise_stress_test.py
```

### B. Operational Domain-Shift Stress Tests (Figure 6):
Evaluates diagnostic invariance across CS-25.473 sink rates, landing weights, and ambient thermal regimes:
```bash
python src/domain_shift_test.py
```

### C. Explainable AI & Pearson Alignment ($r = 0.781$, Figure 7):
Runs GradientExplainer SHAP and computes the Pearson correlation with internal temporal self-attention weights:
```bash
python src/xai_engine.py
```

---

## 4. Live MRO Dashboard & FlightGear Telemetry Bridge

The Level 4 Predictive MRO Dashboard can be launched standalone or connected live to FlightGear:

### Step 1: Launch Local Ollama (Optional for Offline LLM Reasoning)
```bash
# Start Ollama server with Llama-3.2 pulled
ollama serve
ollama pull llama3.2
```
*(If Ollama is not running, the dashboard automatically falls back to deterministic rule-based AMM logic).*

### Step 2: Launch MRO Dashboard GUI
```bash
python src/mro_dashboard.py
```

### Step 3: Stream Live FlightGear Telemetry (Optional)
Launch FlightGear with the custom UDP output socket:
```bash
fgfs --aircraft=787-8 --generic=socket,out,30,localhost,16662,udp,output
```
Touchdowns detected by the Weight-On-Wheels (WOW) switch will automatically trigger the ODE multi-physics state estimator, PyTorch neural inference, and Ollama LLM maintenance briefing in under 1.0 second.
