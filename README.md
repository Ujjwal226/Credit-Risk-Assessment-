# Credit Risk Assessment using Machine Learning

An end-to-end **Credit Risk Assessment system** that predicts the probability of loan default using **XGBoost**, optimized decision thresholds, and a production-ready **FastAPI API** with an interactive web interface.

The project covers the complete ML workflow — from data preprocessing and model selection to probability calibration, threshold optimization, API development, and cloud deployment.

---

## 🚀 Live Demo

**Live Application:**  
https://YOUR-RENDER-URL.onrender.com

**API Documentation:**  
https://YOUR-RENDER-URL.onrender.com/docs

> Replace `YOUR-RENDER-URL` with your actual Render service URL.

---

## 📌 Project Overview

Credit risk assessment is the process of estimating the likelihood that a borrower will default on a loan.

This project builds a machine learning system that takes applicant and loan information such as:

- Age
- Annual income
- Home ownership
- Employment length
- Loan purpose
- Loan grade
- Loan amount
- Interest rate
- Loan-to-income ratio
- Previous default history
- Credit history length

and produces:

1. **Probability of default**
2. **Binary risk prediction**
3. **High Risk / Low Risk decision**

The prediction threshold is optimized separately instead of blindly using the default `0.5` classification threshold.

---

## 🧠 Machine Learning Pipeline

```text
                    Credit Risk Dataset
                           │
                           ▼
                 Data Preprocessing
                           │
              ┌────────────┴────────────┐
              │                         │
        Numerical Features       Categorical Features
              │                         │
       Median Imputation          Missing Category
                                      │
                               One-Hot Encoding
              │                         │
              └────────────┬────────────┘
                           ▼
                  Model Comparison
                           │
              ┌────────────┴────────────┐
              │                         │
      Logistic Regression           XGBoost
              │                         │
              └────────────┬────────────┘
                           ▼
                    Hyperparameter
                       Search
                           │
                           ▼
                       XGBoost
                           │
                           ▼
                   Probability
                    Calibration
                           │
                           ▼
                  Validation Set
                           │
                           ▼
                 Optimal Threshold
                    Selection
                           │
                           ▼
                 Final Calibrated
                      Model
                           │
                           ▼
                     Test Set
                           │
                           ▼
                  FastAPI REST API
                           │
                           ▼
                 Interactive Web UI
                           │
                           ▼
                        Render
```

---

## 📊 Model Performance

The final calibrated XGBoost model was evaluated on a held-out test set.

| Metric | Score |
|---|---:|
| ROC-AUC | **95.23%** |
| Accuracy | **93.32%** |
| Precision | **91.09%** |
| Recall | **76.58%** |
| F1 Score | **83.21%** |
| Decision Threshold | **0.4891** |

### Confusion Matrix

```text
                 Predicted
               0          1
Actual  0    4841       102
        1     319      1043
```

The threshold was selected using validation data rather than directly optimizing it on the final test set.

---

## 🔍 Model Comparison

Two baseline approaches were evaluated before selecting the final model.

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 87.05% | 81.16% | 54.48% | 77.81% | 64.81% |
| XGBoost | **94.71%** | **91.80%** | **81.91%** | **79.63%** | **80.73%** |

XGBoost was subsequently calibrated and retrained for the final evaluation.

---

## 🎯 Why Threshold Optimization?

A classification model does not necessarily need to use:

```text
threshold = 0.50
```

For credit-risk prediction, the trade-off between false positives and false negatives is important.

The project therefore:

1. Trains the model.
2. Calibrates its probability outputs.
3. Uses a validation set to evaluate different thresholds.
4. Selects the threshold that maximizes validation F1.
5. Retrains the calibrated model on the full training set.
6. Evaluates the final model once on the untouched test set.

The resulting threshold was:

```text
0.48913299523060355
```

Therefore:

```text
Probability >= 0.4891  →  High Risk
Probability < 0.4891   →  Low Risk
```

---

## ⚙️ Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- HTML5
- CSS3
- JavaScript
- SVG-based risk visualization

### Deployment

- GitHub
- Render

---

## 📁 Project Structure

```text
Credit-Risk-Assessment/
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── best_threshold.pkl
├── credit_risk_dataset.csv
├── credit_risk_model.pkl
├── Credit_Risk.ipynb
├── main.py
├── requirements.txt
├── .python-version
├── runtime.txt
└── .gitignore
```

---

## 🔌 API

### Health Check

```http
GET /health
```

Example response:

```json
{
    "status": "healthy",
    "model_loaded": true,
    "threshold": 0.48913299523060355
}
```

### Credit Risk Prediction

```http
POST /predict
```

Example request:

```json
{
    "person_age": 35,
    "person_income": 85000,
    "person_home_ownership": "MORTGAGE",
    "person_emp_length": 10,
    "loan_intent": "PERSONAL",
    "loan_grade": "A",
    "loan_amnt": 10000,
    "loan_int_rate": 6.5,
    "loan_percent_income": 0.12,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 12
}
```

Example response:

```json
{
    "default_probability": 0.02,
    "default_prediction": 0,
    "threshold": 0.48913299523060355,
    "Result": "Low Risk"
}
```

---

## 🌐 Interactive Web Interface

The project includes a browser-based interface for submitting loan applications.

The interface provides:

- Applicant information form
- Loan information form
- Credit history information
- Automatic loan-to-income ratio calculation
- API health indicator
- Animated prediction result
- Default probability visualization
- Decision threshold visualization
- High Risk / Low Risk verdict

The frontend communicates directly with the FastAPI `/predict` endpoint.

---

## 🧪 Example Risk Scenarios

### High Risk Example

```json
{
    "person_age": 40,
    "person_income": 90000,
    "person_home_ownership": "MORTGAGE",
    "person_emp_length": 12,
    "loan_intent": "DEBTCONSOLIDATION",
    "loan_grade": "D",
    "loan_amnt": 60000,
    "loan_int_rate": 16.0,
    "loan_percent_income": 0.67,
    "cb_person_default_on_file": "Y",
    "cb_person_cred_hist_length": 15
}
```

Model output:

```text
Default Probability ≈ 91.19%
Prediction: High Risk
```

---

### Low Risk Example

```json
{
    "person_age": 45,
    "person_income": 120000,
    "person_home_ownership": "MORTGAGE",
    "person_emp_length": 15,
    "loan_intent": "HOMEIMPROVEMENT",
    "loan_grade": "A",
    "loan_amnt": 10000,
    "loan_int_rate": 6.0,
    "loan_percent_income": 0.08,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 20
}
```

Model output:

```text
Default Probability ≈ 1.38%
Prediction: Low Risk
```

---

## 🛠️ Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Ujjwal226/Credit-Risk-Assessment.git
cd Credit-Risk-Assessment
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn main:app --reload
```

### 5. Open the application

```text
http://127.0.0.1:8000
```

### API documentation

```text
http://127.0.0.1:8000/docs
```

### Health check

```text
http://127.0.0.1:8000/health
```

---

## ☁️ Deployment

The application is deployed as a **Render Web Service**.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Health Check

```text
/health
```

The Python version is pinned using:

```text
.python-version
```

with:

```text
3.11.9
```

---

## 🔐 Model Serving

The trained model and optimized threshold are serialized using Joblib:

```text
credit_risk_model.pkl
best_threshold.pkl
```

At application startup:

```python
model = joblib.load(MODEL_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))
```

The API then uses the loaded model to generate default probabilities for incoming applications.

---

## 📚 Dataset Features

The model uses the following features:

| Feature | Description |
|---|---|
| `person_age` | Applicant age |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Requested loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount relative to income |
| `cb_person_default_on_file` | Historical default indicator |
| `cb_person_cred_hist_length` | Length of credit history |

---

## 🧠 Key Machine Learning Concepts Demonstrated

- Exploratory data analysis
- Missing-value handling
- Numerical feature preprocessing
- Categorical feature encoding
- Logistic Regression baseline
- XGBoost classification
- Hyperparameter optimization
- Cross-validation
- Class imbalance handling
- Probability calibration
- Decision threshold optimization
- Precision / Recall analysis
- F1-score optimization
- ROC-AUC evaluation
- Confusion matrix analysis
- False-positive / false-negative analysis
- Model serialization
- REST API model serving
- Cloud deployment

---

## ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes**.

The predictions are machine learning estimates and should not be treated as an actual lending decision, financial advice, or a substitute for a regulated credit-risk assessment process.

---

## 👨‍💻 Author

**Ujjwal Khanna**

B.Tech — Computer Science Engineering

GitHub:  
https://github.com/Ujjwal226

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub.