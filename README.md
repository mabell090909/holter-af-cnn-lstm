# End-to-End Deep Learning Framework for Continuous Atrial Fibrillation Detection from Single-Lead ECG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.13.0](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://tensorflow.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23666224.svg)](https://doi.org/10.5281/zenodo.23666224)

This repository contains the official, fully reproducible implementation of the machine learning pipeline, statistical validation protocols, and real-time inference server described in our article submitted to ***MethodsX*** (Elsevier):

> **"Continuous Atrial Fibrillation Screening via Lightweight CNN-LSTM Architecture with Dual-Stage Preprocessing and Patient-Leakage-Free Validation"**

---

## Key Highlights & Methodological Rigor

1. **Strict Zero Patient Leakage Guarantee:**
   * Partitioned using a 5-Fold Stratified Group K-Fold scheme ($N = 1,487,081$ clean 10-second windows across 173 unique patients). All longitudinal recordings from the same individual are strictly constrained to the same fold.
   * Full patient-to-fold mapping manifest provided in [`fold_splits_audit.json`](fold_splits_audit.json) (*Supplementary File S1*).

2. **Physiologically Grounded Preprocessing Cascade:**
   * Dual-stage 3rd-order Butterworth bandpass (0.5–40.0 Hz) applied continuously to uninterrupted 24-hour recordings before segmentation (ensuring complete filter-state continuity).
   * Pan-Tompkins QRS peak detector quality gate ($< 3$ QRS peaks per 10 s rejected as unreadable leads-off / artifact; rejection rate $\sim 1.45\%$, physiologically neutral).

3. **Cost-Sensitive Class Weighting vs. SMOTE:**
   * Uses balanced loss weighting ($w_c = N / (2 \cdot N_c)$) rather than Euclidean synthetic oversampling (SMOTE). Avoids synthetic time-series morphological distortions that inflate false alarm rates.

4. **Independent External Validation:**
   * Single-shot evaluation on the held-out **MIT-BIH Atrial Fibrillation Database (AFDB)** ($N = 84,334$ windows across 23 patients) at the cross-validated threshold ($T^* = 0.5763$), achieving **ROC-AUC of 0.8996** and **89.51% mean patient sensitivity**.

---

## Directory Structure

```text
├── 1_preprocess.py               # Raw WFDB parsing, filtering, resampling & quality gate
├── 2_train_model.py              # 5-fold Stratified Group CV & Random Forest baseline
├── 3_api_server.py               # Production Flask REST API for edge inference
├── 4_evaluate_external_afdb.py   # External validation on MIT-BIH AFDB
├── run_ablation_study.py         # Multi-seed ablation study (Raw vs BP vs Full DSP vs SMOTE)
├── download_physionet.py         # Automated WFDB downloader for open benchmark cohorts
├── fold_splits_audit.json        # Supplementary File S1: Patient allocation per fold
├── final_model_metadata.json     # Calibrated operating threshold (T*) & CV metadata
├── requirements.txt              # Pinned software dependencies
├── LICENSE                       # MIT License
└── README.md                     # Documentation & replication manual
```

---

## Environment Setup

The pipeline is tested and verified on **Python 3.10.11** under both Linux (Ubuntu 22.04 LTS) and Windows 10/11 x64:

```bash
# Clone the repository
git clone https://github.com/mabell090909/holter-af-cnn-lstm.git
cd holter-af-cnn-lstm

# Create and activate a clean virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: .\venv\Scripts\activate

# Install exact pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step-by-Step Reproduction Guide

### 1. Download Open Benchmark Datasets
Download the PhysioNet Long-Term AF Database (LTAFDB) and MIT-BIH AFDB automatically:
```bash
python download_physionet.py
```

### 2. Preprocessing & Quality Gate Audit
Run the continuous filtering, resampling (to 128 Hz), QRS peak detection, and quality gate segmentation:
```bash
python 1_preprocess.py
```
*Outputs are saved to `processed/` as memory-mapped `.npy` arrays (`X.npy`, `y.npy`, `subjects.npy`, `rr_features.npy`).*

### 3. Model Training & 5-Fold Cross-Validation
Execute the 5-Fold Stratified Group K-Fold cross-validation and symmetric Random Forest baseline:
```bash
python 2_train_model.py
```
*Outputs fold checkpoints (`fold_checkpoint_seed_42_f*.json`), ROC curves, and saved models.*

### 4. Independent External Validation
Evaluate the final trained model on the external MIT-BIH AFDB without retraining:
```bash
python 4_evaluate_external_afdb.py
```

### 5. Multi-Seed Ablation Study
Run the 3-shuffle ablation experiment evaluating raw vs bandpass-only vs full DSP cascade, with and without SMOTE:
```bash
python run_ablation_study.py
```

### 6. Launch Inference REST API
Start the real-time inference server (default port `5003`):
```bash
python 3_api_server.py
```
*Endpoint `/predict` accepts a JSON payload containing single-lead ECG arrays (`ecg_signal`) and returns clinical episode classifications, thresholded at $T^* = 0.5763$, with an average latency of $< 170\text{ ms}$.*

---

## Supplementary Materials & Audit Records

* **Supplementary File S1:** [`fold_splits_audit.json`](fold_splits_audit.json) lists every patient assigned to each of the 5 cross-validation folds across both LTAFDB (84 patients) and SHDB-AF (89 patients).
* **Threshold Calibration:** Validation-tuned threshold $T^* = 0.5763$ is stored in [`final_model_metadata.json`](final_model_metadata.json).

---

## Citation

If you use this code or methodology in your research, please cite our article:

```bibtex
@article{MethodsX_AF_Detection_2026,
  title={Continuous Atrial Fibrillation Screening via Lightweight CNN-LSTM Architecture with Dual-Stage Preprocessing and Patient-Leakage-Free Validation},
  author={The Authors},
  journal={MethodsX},
  year={2026},
  publisher={Elsevier},
  doi={10.5281/zenodo.23666224}
}
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
