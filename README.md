# Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-4B8BBE)](https://imbalanced-learn.org/)

An end-to-end machine learning system for detecting fraudulent credit card transactions. The project trains and compares multiple classifiers on a highly imbalanced dataset, persists the best model, and exposes predictions through a Flask REST API and a modern web dashboard (**FraudGuard Pro**).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [Web Interface](#web-interface)
- [REST API Reference](#rest-api-reference)
- [Configuration](#configuration)
- [Risk Scoring Logic](#risk-scoring-logic)
- [Performance Considerations](#performance-considerations)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgments](#acknowledgments)

---

## Overview

Credit card fraud is rare but costly. In typical production data, fraudulent transactions often represent **well under 1%** of all records, which makes accuracy alone a misleading metric. This project addresses that imbalance with:

| Capability | Description |
|------------|-------------|
| **Multi-model training** | Trains Logistic Regression, Random Forest, Gradient Boosting, SVM, and KNN in parallel |
| **Automatic selection** | Picks the best classifier by **ROC AUC** on the held-out test set |
| **Class imbalance handling** | Applies **SMOTE** oversampling on the training split |
| **Feature scaling** | `StandardScaler` fit on training data only |
| **Inference service** | Flask API with validation, probability scores, and tiered risk labels |
| **Interactive UI** | Responsive dashboard with form validation, sample data, and JSON export |

The system is designed as a **mini-project / portfolio demonstration**: train offline, serve predictions online. It is not a production payment gateway integration out of the box.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE (Training)                              │
├─────────────────────────────────────────────────────────────────────────┤
│  data/Creditcard.csv                                                    │
│       │                                                                 │
│       ▼                                                                 │
│  FraudDetector (src/models/fraud_detector.py)                           │
│    • Stratified train/test split (80/20)                                │
│    • StandardScaler → SMOTE (train only)                                │
│    • Train 5 classifiers → evaluate → select best by ROC AUC            │
│       │                                                                 │
│       ▼                                                                 │
│  model.pkl  { model, scaler, feature_names, metrics }                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ONLINE (Inference)                              │
├─────────────────────────────────────────────────────────────────────────┤
│  run.py / app.py (Flask)                                                │
│    • Load model.pkl at startup                                          │
│    • POST /api/predict → scale → predict_proba → risk tier              │
│    • GET  /api/model-info, /api/health                                  │
│       │                                                                 │
│       ▼                                                                 │
│  Web UI (src/web/templates + static/)                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Request flow (prediction):**

1. Client sends transaction features (`time`, `amount`, `v1`–`v28`) as JSON.
2. Server validates types and required fields.
3. Features are reordered to match training column names, scaled with the saved scaler.
4. Model outputs fraud probability; server maps probability to risk level and recommendation.

---

## Dataset

The project uses the widely cited **[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** dataset (ULB MLG, Kaggle).

| Property | Value |
|----------|-------|
| **Records** | 284,807 transactions |
| **Features** | 30 inputs + 1 binary target |
| **Fraud rate** | ~0.17% (492 fraud / 284,315 legitimate) |
| **Time** | Seconds elapsed between transaction and first transaction in dataset |
| **Amount** | Transaction amount (not transformed) |
| **V1–V28** | PCA-transformed components (original features anonymized) |
| **Class** | `0` = legitimate, `1` = fraud |

> **Important:** Because V1–V28 are PCA components, individual features are **not directly interpretable** (e.g. “high V5 means overseas purchase”). The model learns patterns in this anonymized space—appropriate for benchmarking, not for explaining decisions to cardholders without additional feature engineering.

Place the CSV at:

```
data/Creditcard.csv
```

The file is listed in `.gitignore` due to size; download it from Kaggle and add it locally before training.

---

## Machine Learning Pipeline

### Preprocessing

1. **Stratified split** — 80% train / 20% test (`random_state=42`) preserves fraud ratio in both sets.
2. **Scaling** — `StandardScaler` fit on training features; same transform applied to test and inference.
3. **Resampling** — SMOTE applied **only on training data** to reduce majority-class bias.

### Models evaluated

| Algorithm | Role in pipeline |
|-----------|-------------------|
| Logistic Regression | Linear baseline, fast inference |
| Random Forest | Non-linear ensemble, robust default |
| Gradient Boosting | Sequential boosting, strong AUC potential |
| SVM (RBF) | High-dimensional boundary learning |
| KNN | Distance-based baseline |

Hyperparameters are centralized in `config/settings.py` under `MODEL_PARAMS`.

### Model selection

The trainer selects the model with the highest **ROC AUC** on the test set. For imbalanced fraud detection, ROC AUC is generally more informative than raw accuracy (a naive “always legitimate” classifier would score ~99.83% accuracy but fail completely on fraud).

### Serialized artifact

`model.pkl` is a Python pickle bundle:

```python
{
    "model": <fitted sklearn estimator>,
    "scaler": <StandardScaler>,
    "feature_names": ["Time", "Amount", "V1", ..., "V28"],
    "model_name": "<best algorithm name>",
    "metrics": { "accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_mean", "cv_std" }
}
```

> **Note:** `model.pkl` is gitignored. You must train locally before starting the web app.

---

## Project Structure

```
Credit card/
├── app.py                      # Flask application & API routes
├── run.py                      # Entry point (loads model, starts server)
├── train_model.py              # Full training pipeline (5 models)
├── quick_train.py              # Fast training (Random Forest only)
├── setup.py                    # Dependency installer helper
├── requirements.txt            # Python dependencies
├── model.pkl                   # Trained model (generated, not in git)
│
├── config/
│   └── settings.py             # Model params, Flask, risk thresholds
│
├── data/
│   └── Creditcard.csv          # Dataset (download separately)
│
├── src/
│   ├── models/
│   │   └── fraud_detector.py   # FraudDetector class (train/eval/save)
│   └── web/
│       ├── templates/
│       │   └── index.html        # FraudGuard Pro dashboard
│       └── static/
│           ├── style.css         # UI styles (glass-morphism theme)
│           └── script.js         # Form handling, API client, export
│
└── model/
    └── train_model.py          # Legacy script (depends on removed modules)
```

---

## Prerequisites

- **Python** 3.8 or newer
- **pip** package manager
- **~150 MB disk space** for dependencies
- **Kaggle dataset** placed in `data/Creditcard.csv`

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd "Credit card"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or use the setup helper:

```bash
python setup.py
```

### 4. Add the dataset

Download [Creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/Creditcard.csv`.

---

## Training the Model

You must generate `model.pkl` before running the web application.

### Option A — Full pipeline (recommended)

Trains all five algorithms, compares metrics, and saves the best model:

```bash
python train_model.py
```

Expected output includes per-model accuracy, ROC AUC, F1 score, and the selected winner.

### Option B — Quick training

Faster iteration with a single Random Forest (50 trees, reduced depth):

```bash
python quick_train.py
```

Use this for development smoke tests; use **Option A** for best model quality.

### After training

Confirm `model.pkl` exists in the project root:

```bash
# Windows PowerShell
Test-Path model.pkl

# macOS / Linux
ls -la model.pkl
```

---

## Running the Application

```bash
python run.py
```

| Resource | URL |
|----------|-----|
| Web dashboard | http://localhost:5000 |
| Health check | http://localhost:5000/api/health |
| Model metadata | http://localhost:5000/api/model-info |

The development server binds to `0.0.0.0:5000` with debug mode enabled. For production, use a WSGI server (see [Deployment](#deployment)).

---

## Web Interface

The **FraudGuard Pro** dashboard (`src/web/templates/index.html`) provides:

- **Transaction form** — `Time`, `Amount`, and `V1`–`V28` inputs
- **Quick Fill** — loads representative sample values for testing
- **Real-time validation** — numeric checks before submission
- **Results panel** — fraud probability, risk tier, and approve/review/block recommendation
- **Export** — download analysis results as JSON
- **Model status** — live indicator from `/api/model-info`

---

## REST API Reference

### `GET /api/health`

Returns service status and whether the model loaded successfully.

**Response (200):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-05-15T12:00:00.000000"
}
```

---

### `GET /api/model-info`

Returns metrics for the loaded classifier.

**Response (200):**

```json
{
  "success": true,
  "model_info": {
    "name": "Random Forest",
    "accuracy": 0.9987,
    "precision": 0.91,
    "recall": 0.88,
    "f1_score": 0.89,
    "roc_auc": 0.97,
    "cv_mean": 0.96,
    "cv_std": 0.01
  }
}
```

---

### `POST /api/predict`

Scores a single transaction.

**Headers:** `Content-Type: application/json`

**Body:**

```json
{
  "time": 406.0,
  "amount": 220.0,
  "v1": -1.23,
  "v2": 0.45,
  "v3": -0.78,
  "v4": 2.1,
  "v5": -0.5,
  "v6": -1.2,
  "v7": -0.3,
  "v8": 0.1,
  "v9": -0.4,
  "v10": 0.3,
  "v11": -2.5,
  "v12": 0.9,
  "v13": -0.7,
  "v14": -1.9,
  "v15": 0.2,
  "v16": -0.3,
  "v17": -1.8,
  "v18": 0.7,
  "v19": -0.2,
  "v20": 0.1,
  "v21": -0.4,
  "v22": 0.3,
  "v23": -0.1,
  "v24": 0.2,
  "v25": -0.3,
  "v26": 0.1,
  "v27": -0.2,
  "v28": 0.01
}
```

All fields `time`, `amount`, and `v1`–`v28` are **required**.

**Success response (200):**

```json
{
  "success": true,
  "prediction": {
    "is_fraud": false,
    "confidence": 12.45,
    "risk_level": "LOW",
    "recommendation": "APPROVE"
  },
  "model_info": {
    "name": "Random Forest",
    "accuracy": 0.9987,
    "roc_auc": 0.97
  },
  "timestamp": "2026-05-15T12:00:00.000000"
}
```

> `confidence` is the **fraud class probability × 100**, not calibrated “certainty” in a legal/compliance sense.

**Error responses:**

| Status | Cause |
|--------|--------|
| 400 | Missing fields, invalid JSON, or non-numeric values |
| 500 | Model not loaded or internal prediction error |

**Python example:**

```python
import requests

payload = {
    "time": 406,
    "amount": 220.0,
    "v1": -1.23, "v2": 0.45, "v3": -0.78, "v4": 2.1, "v5": -0.5,
    "v6": -1.2, "v7": -0.3, "v8": 0.1, "v9": -0.4, "v10": 0.3,
    "v11": -2.5, "v12": 0.9, "v13": -0.7, "v14": -1.9, "v15": 0.2,
    "v16": -0.3, "v17": -1.8, "v18": 0.7, "v19": -0.2, "v20": 0.1,
    "v21": -0.4, "v22": 0.3, "v23": -0.1, "v24": 0.2, "v25": -0.3,
    "v26": 0.1, "v27": -0.2, "v28": 0.01
}

response = requests.post("http://localhost:5000/api/predict", json=payload)
result = response.json()

if result["success"]:
    p = result["prediction"]
    print(f"Fraud: {p['is_fraud']} | Confidence: {p['confidence']}% | Action: {p['recommendation']}")
```

**cURL example:**

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"time\":406,\"amount\":220,\"v1\":-1.23,\"v2\":0.45,\"v3\":-0.78,\"v4\":2.1,\"v5\":-0.5,\"v6\":-1.2,\"v7\":-0.3,\"v8\":0.1,\"v9\":-0.4,\"v10\":0.3,\"v11\":-2.5,\"v12\":0.9,\"v13\":-0.7,\"v14\":-1.9,\"v15\":0.2,\"v16\":-0.3,\"v17\":-1.8,\"v18\":0.7,\"v19\":-0.2,\"v20\":0.1,\"v21\":-0.4,\"v22\":0.3,\"v23\":-0.1,\"v24\":0.2,\"v25\":-0.3,\"v26\":0.1,\"v27\":-0.2,\"v28\":0.01}"
```

---

## Configuration

Edit `config/settings.py` to tune behavior:

| Section | Purpose |
|---------|---------|
| `MODEL_CONFIG` | Test split ratio, random seed, CV folds, file paths |
| `MODEL_PARAMS` | Per-algorithm hyperparameters |
| `FLASK_CONFIG` | Host, port, debug, upload limits |
| `FEATURE_CONFIG` | Required API fields and validation ranges |
| `PERFORMANCE_THRESHOLDS` | Minimum acceptable metrics (documentation targets) |
| `RISK_LEVELS` | Thresholds for CRITICAL / HIGH / MEDIUM / LOW |

Environment variables for production:

```bash
export FLASK_ENV=production
export SECRET_KEY=<strong-random-secret>
```

---

## Risk Scoring Logic

Fraud probability drives both the binary label and operational recommendation:

| Fraud probability | Risk level | Recommendation |
|-------------------|------------|----------------|
| ≥ 80% | `CRITICAL` | `BLOCK_IMMEDIATELY` |
| ≥ 60% | `HIGH` | `BLOCK` |
| ≥ 40% | `MEDIUM` | `REVIEW` |
| < 40% | `LOW` | `APPROVE` |

Binary classification uses a **0.5 threshold** on fraud probability (`is_fraud = proba > 0.5`). Thresholds are hardcoded in `app.py` and can be adjusted for your cost matrix (false positive vs false negative trade-off).

---

## Performance Considerations

| Topic | Guidance |
|-------|----------|
| **Training time** | Full pipeline trains 5 models with 5-fold CV—expect several minutes on CPU |
| **Quick train** | ~1–3 minutes for Random Forest with 50 estimators |
| **Inference** | Single-row prediction is typically milliseconds on CPU |
| **Memory** | Full dataset loads into RAM (~70 MB CSV); ensure sufficient memory |
| **Class imbalance** | Monitor precision/recall for fraud class, not accuracy alone |

Dependencies such as `xgboost` and `lightgbm` are listed in `requirements.txt` for extension but are **not used** in the current training code.

---

## Deployment

### Production WSGI (Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Ensure `model.pkl` is present and `load_model()` runs before serving (wrap in application factory or preload hook for multi-worker setups).

### Docker (example)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Copy or mount model.pkl and data/ as needed

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

### Security checklist for production

- [ ] Replace default `SECRET_KEY` in `app.py` / `config/settings.py`
- [ ] Disable Flask `debug=True`
- [ ] Put the API behind HTTPS and authentication
- [ ] Do not expose pickle files publicly (arbitrary code execution risk if tampered)
- [ ] Add rate limiting and request logging
- [ ] Never commit real cardholder data

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Failed to load model` | Run `python train_model.py` or `python quick_train.py` to create `model.pkl` |
| `Data file not found` | Download dataset to `data/Creditcard.csv` |
| `Missing required fields` | API expects lowercase keys: `time`, `amount`, `v1`–`v28` |
| Training is slow | Use `quick_train.py` for development; reduce `n_estimators` in config |
| Import errors | Activate venv and run `pip install -r requirements.txt` |
| Port 5000 in use | Change port in `run.py` or set `FLASK_CONFIG["PORT"]` |

---

## Limitations & Future Work

**Current limitations**

- Features are PCA-anonymized; no merchant/category interpretability.
- Single static threshold (0.5); no cost-sensitive learning or PR-AUC optimization in the API layer.
- Pickle serialization is convenient but not ideal for production model registry (consider ONNX or sklearn-pipeline with versioned artifacts).
- No automated test suite in the repository yet.
- Legacy `model/train_model.py` references removed `python scripts/` modules and should not be used.

**Suggested enhancements**

- Probability calibration (Platt scaling / isotonic regression)
- Threshold tuning on validation PR curve
- Batch prediction endpoint and model versioning
- SHAP or permutation importance for explainability within PCA space
- CI pipeline with unit tests for validation and API contracts
- Integration with real-time streaming (Kafka, etc.) for production patterns

---

## Acknowledgments

- Dataset: [Machine Learning Group - ULB](https://www.researchgate.net/publication/319867396_Calibrating_Probability_with_Under-sampling_for_Unbalanced_Classification) — Credit Card Fraud Detection ([Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- [scikit-learn](https://scikit-learn.org/) — Machine learning algorithms and metrics
- [imbalanced-learn](https://imbalanced-learn.org/) — SMOTE resampling
- [Flask](https://flask.palletsprojects.com/) — Web framework

---

## License

Specify your license in a `LICENSE` file at the repository root. If none is provided, assume all rights reserved by the repository owner.

---

**Built for educational and portfolio use.** For production fraud systems, pair ML scores with rules engines, human review queues, regulatory compliance, and continuous model monitoring.
