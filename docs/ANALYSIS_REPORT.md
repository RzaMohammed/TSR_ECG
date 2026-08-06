# TSRNet: Real-time ECG Anomaly Detection - Project Analysis Report

## Project Overview

This repository (`TSRNet-main`) is the official implementation for the paper: **TSRNet: Simple Framework for Real-time ECG Anomaly Detection with Multimodal Time and Spectrogram Restoration Network (ISBI 2024)**.

The project tackles the challenge of distinguishing between normal and abnormal electrocardiogram (ECG) signals. It introduces a specialized restoration-based anomaly detection network called **Multimodal Time and Spectrogram Restoration Network (TSRNet)**.

### Core Philosophy
The model is trained **solely on normal ECG data**. During inference, if the model struggles to accurately restore a given ECG signal (resulting in a high restoration error), it flags the signal as an anomaly. To enhance feature extraction and robustness, the network simultaneously leverages both:
1. **Time Series Domain**
2. **Time-Frequency (Spectrogram) Domain**

By extracting representations from both domains, TSRNet learns robust features with superior discrimination abilities. Furthermore, a novel inference method called **Peak-based Error** is introduced, focusing on the reconstruction error specifically around the critical ECG peaks.

## Repository Structure
```
c:\TESTING PROJECT\TSRNet-main\TSRNet-main\
├── dataloader.py        # Dataset loading and preprocessing scripts
├── Images/              # Contains architecture diagrams (main_architecture.png, cross_attention.png, etc.)
├── lib/                 # Contains the core neural network model definitions (TSRNet, TSRNet_time, etc.)
├── README.md            # Canonical landing page & architectural guide
├── test.py              # Inference script for evaluating the model on test data
├── train.py             # Script for training the model
└── utils.py             # Helper utilities (time formatting, average meters, normalization)
```

## Key Components

### 1. Training Pipeline (`train.py`)
The training script supports two primary modes:
- **Time Domain Only Mode** (default): Uses the `TSRNet_time` model.
- **Multimodal (Time + Spectrogram) Mode** (`--spec` flag): Uses the full `TSRNet` model.

**Training Mechanism:**
- The model employs a **Self-Restoration** mechanism by applying masks to the input data.
- **Time Branch Masking**: Randomly masks patches of the time-series ECG signal based on the `--mask_ratio_time` (default 30%).
- **Spectrogram Branch Masking**: Randomly masks patches of the spectrogram based on the `--mask_ratio_spec` (default 20%).
- The model is tasked with restoring these masked regions. The loss function minimizes the mean squared error between the generated/restored signal and the original signal.

### 2. Testing & Inference Pipeline (`test.py`)
The testing script evaluates the trained model and computes the anomaly detection performance using the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

**Inference Mechanism:**
- It iteratively masks different segments of the signal (sliding window approach) and reconstructs them.
- It computes the restoration error (difference between original and reconstructed).
- **Peak-based Error (`--mask_loss` flag)**: If enabled, it calculates the restoration error *only* around the R-peaks of the ECG signal (specifically a window of $\pm240$ time steps around the peak). This focuses the anomaly detection on the most morphologically significant parts of the ECG wave.

### 3. Model Architecture (`lib/`)
While the internal code of the `lib` folder was not fully exposed, the architecture relies on:
- An encoder-decoder structure for restoration.
- A Cross-Attention mechanism (as suggested by the image references) to fuse features between the Time and Spectrogram domains.

### 4. Data Loading (`dataloader.py`)
- Designed to work with the large-scale **PTB-XL** dataset.
- The dataloader returns the time-series ECG, the spectrogram ECG, and the R-peak indices required for the Peak-based Error inference.

## How to Run (Usage)

### Prerequisites
- Pytorch, Torchvision, Numpy, SciPy, HeartPy, PyWavelets

### Training
**Without Spectrogram Branch:**
```bash
python train.py --dims <dimension> --data_path <path> --save_path <path>
```
**With Spectrogram Branch:**
```bash
python train.py --dims <dimension> --data_path <path> --save_path <path> --spec
```

### Testing
**Standard Inference:**
```bash
python test.py --dims <dimension> --data_path <path> --load_path <path>
```
**With Peak-based Error (Recommended for higher accuracy):**
```bash
python test.py --dims <dimension> --data_path <path> --load_path <path> --mask_loss
```

---
*Analysis conducted on: 2026-07-31*
*Based on TSRNet ISBI 2024 implementation*
