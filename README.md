<div align="center">

# Heart Disease Prediction using Supervised ML

**Benchmarks 4 classification algorithms on 13 clinical features from the UCI Heart Disease dataset — Random Forest achieves 78.6% accuracy, with feature correlation analysis and confusion matrix evaluation.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Heart%20Disease-red?style=flat)](https://archive.ics.uci.edu/ml/datasets/heart+disease)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

</div>

---

## The Problem

Heart disease is the world's leading cause of death — 17.9 million lives lost annually (WHO). Early diagnosis depends on clinical measurements that are often analyzed manually. This project applies supervised ML to automate binary diagnosis prediction (heart disease present / absent) from 13 clinical attributes, comparing four algorithms to identify which delivers the best decision support for practitioners.

---

## Dataset — UCI Heart Disease (14 attributes)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `age` | Continuous | Patient age |
| 2 | `sex` | Binary | 1 = Male · 0 = Female |
| 3 | `cp` | Ordinal | Chest pain type (1–4): typical angina · atypical angina · non-anginal · asymptomatic |
| 4 | `trestbps` | Continuous | Resting blood pressure (mmHg) |
| 5 | `chol` | Continuous | Serum cholesterol (mg/dl) |
| 6 | `fbs` | Binary | Fasting blood sugar > 120 mg/dl |
| 7 | `restecg` | Ordinal | Resting ECG results (0 · 1 · 2) |
| 8 | `thalach` | Continuous | Maximum heart rate achieved |
| 9 | `exang` | Binary | Exercise-induced angina |
| 10 | `oldpeak` | Continuous | ST depression induced by exercise |
| 11 | `slope` | Ordinal | Slope of peak exercise ST segment |
| 12 | `ca` | Ordinal | Major vessels colored by fluoroscopy (0–3) |
| 13 | `thal` | Ordinal | Thalassemia: 3 = normal · 6 = fixed defect · 7 = reversible defect |
| Y | `target` | Binary | **1 = Heart disease present · 0 = Absent** |

---

## Approach

### Pipeline

| Stage | Method | Purpose |
|-------|--------|---------|
| 1. EDA | pandas · seaborn | Distribution plots · correlation heatmap · class balance check |
| 2. Preprocessing | scikit-learn | Feature scaling (StandardScaler) · train/test split |
| 3. Modeling | 4 classifiers | Train and evaluate each algorithm on identical splits |
| 4. Evaluation | Confusion matrix · accuracy | Compare algorithms; identify best performer |
| 5. Feature importance | Random Forest importances | Rank which clinical features drive prediction most |

### Algorithms Benchmarked

| Algorithm | Approach |
|-----------|----------|
| **Random Forest** | Ensemble of decision trees — reduces overfitting via bagging |
| **SVM** | Finds optimal hyperplane separating classes in feature space |
| **Naive Bayes** | Probabilistic classifier assuming feature independence |
| **Decision Tree** | Rule-based splits on clinical feature thresholds |

---

## Results

| Algorithm | Accuracy |
|-----------|----------|
| **Random Forest** | **78.6%** ← best |
| SVM | — |
| Naive Bayes | — |
| Decision Tree | — |

| Metric | Value |
|--------|-------|
| Best model | **Random Forest** |
| Best accuracy | **78.6%** |
| Algorithms benchmarked | **4** |
| Clinical features used | **13** (continuous · ordinal · binary) |
| Dataset | UCI Heart Disease (303 records) |
| Evaluation | Confusion matrix · accuracy score · feature correlation |

---

## Key Findings

- **Random Forest** outperforms SVM, Naive Bayes, and Decision Tree on this dataset
- **Chest pain type (cp)**, **thalassemia (thal)**, and **maximum heart rate (thalach)** are the strongest predictors
- Feature correlation analysis reveals clusters of related clinical measurements

---

## Demo

### Video Walkthrough
> *2-minute walkthrough: data exploration → model training → confusion matrix → feature importance.*

[![Watch the Demo](https://img.shields.io/badge/Watch%20Demo-Coming%20Soon-red?style=for-the-badge&logo=youtube)](#)

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python |
| ML | scikit-learn (Random Forest · SVM · Naive Bayes · Decision Tree) |
| Data wrangling | pandas · NumPy |
| Visualization | matplotlib · seaborn |
| Dataset | [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease) |

---

## Setup & Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the notebook
jupyter notebook heart_disease_prediction.ipynb
