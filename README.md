
# Customer Churn Prediction

This repository contains a production-oriented implementation of a customer churn prediction pipeline for a retail bank. It includes:

- Data exploration and analysis to understand feature distributions and relationships.
- A repeatable preprocessing and feature engineering pipeline.
- Model training, selection, evaluation, and persistence (model artifact saved for inference).

Developed for CodeChef Weekend Dev Challenge 14: "DL Projects" — attempted on 6 Sep 2025.

### Key achievements

- Delivered an end-to-end pipeline: raw CSV → cleaned dataset → trained model → serialized artifact for inference (see `Part 3`).
- Evaluated model performance using standard classification metrics (accuracy, precision/recall, F1, ROC AUC). Numerical results and plots are produced by `Part 3/main.py`.
- Provided utilities (`model.py`) for easy integration with a web service (Flask/FastAPI) for real-time inference.

## Problem definition

Predict whether a customer will churn (binary classification) using features such as credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, active membership, and estimated salary.

## Project structure

- Part 1/: Data exploration and initial analysis
  - `main.py` — exploratory analysis, summary statistics, and baseline model experiments
  - `Churn_Modelling.csv` — original dataset sample used in Part 1

- Part 2/: Data preprocessing and feature engineering
  - `main.py` — cleaning steps, encoding categorical variables, feature transformations
  - `Churn_Modelling.csv` — dataset used for intermediate steps

- Part 3/: Final model, evaluation, and integration utilities
  - `main.py` — final training pipeline, evaluation (confusion matrix, accuracy, AUC), and example inference
  - `model.py` — reusable functions: build_model(), train(), save_model(), load_model(), predict()
  - `Modified_Churn_Modelling.csv` — cleaned dataset produced during preprocessing

## Dataset

- The dataset contains customer demographic and account-related fields such as credit score, geography, gender, age, tenure, balance, number of products, has credit card, is active member, estimated salary, and churn label.
- The cleaned `Modified_Churn_Modelling.csv` (Part 3) is provided for quick reproduction of the final training step.

## Approach and pipeline

1. Exploratory Data Analysis (Part 1)
   - Inspect distributions, missing values, and correlations.
   - Establish a baseline model and metrics.

2. Preprocessing & Feature Engineering (Part 2)
   - Handle missing values (if any), encode categorical features, scale numeric features when required.
   - Create new features if helpful and select features for training.

3. Model Training & Evaluation (Part 3)
   - Train a classifier (e.g., logistic regression / tree-based / simple NN depending on the script).
   - Evaluate using accuracy, precision/recall, F1 score, and ROC AUC. Save the best model to disk.

## How to run (recommended)

1. From the repository root, create and activate a virtual environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. To reproduce the final model training and evaluation (Part 3):

```powershell
Set-Location -LiteralPath "Customer Churn Prediction project"
python "Part 3\main.py"
```

3. To run only prediction utilities (example):

```powershell
Set-Location -LiteralPath "Customer Churn Prediction project\Part 3"
python -c "from model import load_model, predict; m=load_model('best_model.pkl'); print(predict(m, sample_input))"
```

### Evaluation

- The scripts print evaluation metrics (accuracy, confusion matrix, classification report). Check `Part 3/main.py` for exact metric outputs and saved model filename (e.g., `best_model.pkl`).

### Web App / Integration notes

- Part 3 includes functions that can be called from a lightweight Flask or FastAPI app to accept JSON input and return churn predictions.
- For a production demo, build a small web form that posts customer features to an endpoint which calls `model.predict()` and returns the result.

### Tips and assumptions

- Scripts assume CSV files are present under each Part folder (paths are relative). If you move files, update paths in the scripts.
- If you encounter missing packages, install them with pip (the main `requirements.txt` at repo root should cover the environment used when I developed the solutions).
- Random seeds are used in training where reproducibility is desired; exact outputs may vary with environment and package versions.

### Sample prediction JSON

Use this example as the POST body for a `/predict` endpoint or to pass to the `predict()` utility in `model.py`.

```json
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000.0
}
```


