# 🏦 Loan Eligibility Prediction

A complete end-to-end Machine Learning project to predict whether a loan application will be **Approved** or **Rejected** based on applicant financial and personal details.

---

## 📌 Project Overview

Financial institutions receive thousands of loan applications daily. Manually reviewing each one is time-consuming and prone to bias. This project builds an ML classification model to automate loan eligibility prediction with high accuracy, and deploys it as an interactive **Streamlit web application**.

---

## 📂 Project Structure

```
loan-prediction/
│
├── 📓 loan_prediction.ipynb   # Full ML pipeline (EDA → Training → Evaluation)
├── 🐍 app.py                  # Streamlit web application
├── 📊 data.csv                # Dataset (4269 records)
├── 📄 requirements.txt        # Python dependencies
└── 📄 README.md               # Project documentation
```

> ⚠️ `.pkl` model files are **not included**. Run `loan_prediction.ipynb` to generate them.

---

## 📊 Dataset Description

| Property | Value |
|---|---|
| **Total Records** | 4,269 |
| **Total Features** | 12 input + 1 target |
| **Missing Values** | None |
| **Target Classes** | Approved (62.2%) / Rejected (37.8%) |

### 🗂️ Features

| Column | Type | Description |
|---|---|---|
| `loan_id` | int | Unique identifier (dropped during training) |
| `no_of_dependents` | int | Number of dependents (0–5) |
| `education` | categorical | Graduate / Not Graduate |
| `self_employed` | categorical | Yes / No |
| `income_annum` | int | Annual income in ₹ (2L – 99L) |
| `loan_amount` | int | Requested loan amount in ₹ (3L – 3.95Cr) |
| `loan_term` | int | Loan duration in months (2–20) |
| `cibil_score` | int | Credit score (300–900) |
| `residential_assets_value` | int | Value of residential property in ₹ |
| `commercial_assets_value` | int | Value of commercial property in ₹ |
| `luxury_assets_value` | int | Value of luxury assets in ₹ |
| `bank_asset_value` | int | Bank balance / fixed deposits in ₹ |
| `loan_status` ⭐ | categorical | **Target** — Approved / Rejected |

### 📈 Key Statistics

| Feature | Min | Mean | Max |
|---|---|---|---|
| Annual Income | ₹2,00,000 | ₹50,59,124 | ₹99,00,000 |
| Loan Amount | ₹3,00,000 | ₹1,51,33,450 | ₹3,95,00,000 |
| CIBIL Score | 300 | 599.9 | 900 |
| Loan Term | 2 months | 10.9 months | 20 months |
| Dependents | 0 | 2.5 | 5 |

---

## 🤖 ML Pipeline

### 1. Exploratory Data Analysis
- Target class distribution (Approved vs Rejected)
- Feature distributions by loan status
- Correlation heatmap
- CIBIL score analysis

### 2. Feature Engineering
Four new features are derived:

| New Feature | Formula |
|---|---|
| `total_assets` | Sum of all 4 asset values |
| `loan_to_income_ratio` | loan_amount / income_annum |
| `asset_to_loan_ratio` | total_assets / loan_amount |
| `income_per_dependent` | income_annum / (dependents + 1) |

### 3. Models Trained & Compared

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble of 150 trees |
| Gradient Boosting | Sequential boosting ensemble |
| XGBoost | Optimized gradient boosting |
| SVM (RBF kernel) | Support vector classifier |

Evaluated using: **5-Fold Cross Validation**, **Test Accuracy**, **ROC-AUC**

---

## 🖥️ Streamlit App Features

- 📋 Sidebar form with all input fields
- 📊 Live applicant summary cards
- 📈 CIBIL score visual gauge
- ✅ Approved / ❌ Rejected result banner with confidence %
- 📊 Probability breakdown bars
- 💡 Smart key insights (risk flags, ratio analysis)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-prediction.git
cd loan-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
Open and run all cells in `loan_prediction.ipynb`. This generates the `.pkl` model files.

### 4. Launch the app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
streamlit
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📸 App Preview

> The web app takes applicant inputs and predicts loan eligibility in real time with confidence scores and risk insights.

---

## 🙏 Acknowledgements

- Dataset inspired by publicly available loan approval datasets
- Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [Streamlit](https://streamlit.io/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
