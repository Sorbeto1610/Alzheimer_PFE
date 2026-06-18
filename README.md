# Early Detection of Alzheimer's Disease through Oral Expression Analysis

**Détection précoce de la maladie d'Alzheimer par étude de l'expression orale retranscrite à l'écrit**

> End-of-study research project (PFE) — Gabriel CHABREDIER & Valentine GOBERT

---

## Overview

This project presents an interactive Streamlit application that showcases our research on the automatic early detection of Alzheimer's disease. The approach combines two complementary signals extracted from audio recordings of patients:

1. **Transcription-based model** — SBERT embeddings of spoken text to capture linguistic patterns (lexical diversity, syntactic structure, language errors)
2. **Silence-based model** — sequential analysis of pauses and silence durations extracted from raw audio
3. **Stacking metamodel** — linear regression fusion of both models for improved predictions

The dataset consists of 166 training audio recordings and 71 test recordings from two groups: Alzheimer patients (AD) and healthy controls (CN).

---

## Repository Structure

```
Alzheimer_PFE/
├── app.py              # Streamlit app (French)
├── app_en.py           # Streamlit app (English)
├── requirements.txt
│
├── assets/
│   ├── audio/          # Demo audio samples (adrso077.wav, adrso312.wav)
│   └── images/         # Profile photos and silence analysis charts
│
├── audio_test/         # 71 test audio files (.wav)
│
├── data/
│   ├── stacking_train.xlsx               # Training set (transcriptions + silence features)
│   ├── stacking_test.xlsx                # Test set
│   ├── transcriptions_finale.xlsx        # Full transcription dataset with silence metadata
│   ├── entire_model_info.xlsx            # Performance metrics for all trained models
│   ├── linear_regression_metamodel_metrics.csv  # Stacking metamodel metrics
│   └── train_scraped_encadrant.csv       # Supervisor's original transcriptions
│
├── models/
│   ├── Linear_Regression_model_Silences.pth                        # Silence regression model
│   ├── 2nd_embedding_linear_regression_SBERT_encadrant_loss.pth    # SBERT regression model
│   ├── stacked_scaler.pkl                                          # Feature scaler
│   └── meta_model.pkl                                              # Stacking metamodel
│
└── plots/              # Training curves, confusion matrices, SHAP plots
```

---

## Application Content

The app walks through the full research methodology:

- **Introduction** — context and objectives
- **Initial Ambition** — target: >80% accuracy and F1-score
- **Data Analysis** — exploration of the two audio groups
- **Transcription** — comparison between supervisor transcriptions and our custom model (which preserves silence markers)
- **Silence Dataset** — extraction of silence duration/count features and statistical analysis
- **Model Comparisons** — BERT vs SBERT, 1st vs 2nd embedding strategy, early stopping criterion selection
- **Silence Sequential Model** — standalone binary classifier on silence features + SHAP analysis
- **Stacking Fusion** — linear regression metamodel combining both modalities
- **Live Demo** — select any test patient to run inference and compare predictions against ground truth

---

## Getting Started

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the French version:**
```bash
streamlit run app.py
```

**Run the English version:**
```bash
streamlit run app_en.py
```

---

## Key Results

| Model | Accuracy | F1-Score |
|---|---|---|
| BERT (supervisor transcription) | — | — |
| SBERT 1st embedding | — | — |
| SBERT 2nd embedding (best) | — | — |
| Silence classifier | — | — |
| **Stacking metamodel** | **—** | **—** |

> Refer to the `data/entire_model_info.xlsx` file and the in-app performance tables for detailed metrics.

---

## Tech Stack

- **Streamlit** — interactive web application
- **PyTorch** — neural network models (BERT/SBERT classifiers, regression heads)
- **Sentence-Transformers** — `all-MiniLM-L6-v2` for SBERT embeddings
- **scikit-learn** — stacking metamodel, scaler
- **NLTK** — sentence tokenization for text chunking
- **pandas / openpyxl** — data handling
