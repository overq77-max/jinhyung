# Adolescent Suicide Risk Prediction using Explainable Machine Learning and Correlation Network Modeling

This repository provides a clean, reproducible analysis pipeline for predicting adolescent suicidal behaviors using Korean Youth Risk Behavior Survey (KYRBS) data (2015–2024).  
The project integrates weighted machine learning, model explainability, and correlation-based network analysis to identify key risk factors and their interconnections.

---

## 🔍 Project Overview

This repository includes:

- **Data preprocessing**  
  Clean and harmonize KYRBS variables, compute derived features (BMI categories, sleep duration, etc.)

- **Weighted machine learning models**  
  Train XGBoost models using survey weights and PSU-based cross-validation.

- **Explainability-ready outputs**  
  Models output probabilities suitable for SHAP interpretation.

- **Correlation network analysis**  
  Compute weighted Pearson correlations and generate network graphs with centrality metrics.

This project is designed for public release, reproducibility, and further research extension.

---

## 📁 Project Structure

```
kyrbs-suicide-risk/
│
├── README.md
├── requirements.txt
│
├── src/
│ ├── preprocess_kyrbs.py # Data cleaning and feature construction
│ ├── train_weighted_xgb.py # Weighted ML model pipeline
│ └── build_corr_network.py # Weighted correlation network
│
└── outputs/ # Auto-generated results (models, plots, tables)
```
---
## ⚙️ Installation

### 1. Clone the repository:

```bash
git clone https://github.com/<your-username>/kyrbs-suicide-risk.git
cd kyrbs-suicide-risk
```

### 2. Install dependencies:
```
pip install -r requirements.txt 
```

#### 🧹 1. Data Preprocessing

```
Script: src/preprocess_kyrbs.py

This script:

Loads the raw KYRBS dataset (kyrbs_merged.csv)

Computes BMI values & categories

Computes sleep duration and weekday sleep categories

Recodes suicidal variables (2 = No, 3 = Yes)

Selects final analysis variables

Outputs: analysis_variables_only.csv

Run: python src/preprocess_kyrbs.py 
```

#### 🤖 2. Weighted Machine Learning (XGBoost)

```
Script: src/train_weighted_xgb.py

This script:

Computes adjusted survey weights

Splits train/test while respecting PSU clusters

Applies imputation

Performs hyperparameter search with StratifiedGroupKFold

Evaluates model with AUROC, AUPRC, F1, calibration curves

Saves model artifacts and evaluation outputs

Run:python src/train_weighted_xgb.py

To switch outcomes:TARGET_COL = "M_SUI_PLAN"   # suicidal plan
                   TARGET_COL = "M_SUI_ATT"    # suicidal attempt
```

#### 🔗 3. Correlation Network Analysis

```
Script: src/build_corr_network.py

This script:

Computes weighted Pearson correlations (|r| ≥ 0.10)

Builds an undirected network graph

Computes centrality metrics

Saves:

correlation matrix

edge list

centrality tables

network visualizations

Run: python src/build_corr_network.py
```
🧪 Requirements
```
pandas
numpy
scikit-learn
xgboost
networkx
matplotlib
```
---

## 📄 Notes

The repository does not include raw KYRBS data due to usage restrictions.
Users must obtain the data independently and place kyrbs_merged.csv in the project root.

All scripts are prepared for open-source release and reproducible research.

---
## 📝 License

MIT License
© CHA University