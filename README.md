# Customer Churn Prediction

This folder contains my solution for the "Customer Churn Prediction" problem from Weekend Dev Challenge 14 (attempted on 6 Sep 2025).

Overview
- The project demonstrates preprocessing, feature engineering, model training, and saving for a customer churn prediction task.
- It is organized into three incremental parts (Part 1, Part 2, Part 3). The final model and utilities live in Part 3.

Contents
- Part 1/
  - `main.py` — initial data loading and baseline preprocessing.
  - `Churn_Modelling.csv` — original dataset copy used for Part 1.
- Part 2/
  - `main.py` — enhanced preprocessing and feature engineering.
  - `Churn_Modelling.csv` — dataset.
- Part 3/
  - `main.py` — final training, evaluation, and example inference steps.
  - `model.py` — reusable model-building and persistence utilities.
  - `Modified_Churn_Modelling.csv` — cleaned/modified dataset used for training in Part 3.

Quick start
1. Create and activate a virtual environment and install dependencies (from repo root):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
```

2. Run the final training script:

```powershell
python "Part 3\main.py"
```

Notes
- Paths in the scripts are relative to this folder; run commands from this folder for the simplest path behavior.
- If you only want to run the final model, use the cleaned CSV `Part 3/Modified_Churn_Modelling.csv`.
- Add a license or author info as needed.

Contact
- Repository: https://github.com/SanyamBK/Customer-Churn-Prediction
