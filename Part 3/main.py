from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    prediction_proba = None
    
    if request.method == 'POST':
        # Retrieve input values from form
        geography = request.form.get("geography")
        gender = request.form.get("gender")
        credit_score = float(request.form.get("credit_score"))
        age = int(request.form.get("age"))
        tenure = int(request.form.get("tenure"))
        balance = float(request.form.get("balance"))
        num_of_products = int(request.form.get("num_of_products"))
        has_cr_card = int(request.form.get("has_cr_card"))
        is_active_member = int(request.form.get("is_active_member"))
        estimated_salary = float(request.form.get("estimated_salary"))

        # One-hot encoding for geography & gender
        geography_germany = 1 if geography == "Germany" else 0
        geography_spain = 1 if geography == "Spain" else 0
        gender_male = 1 if gender == "Male" else 0

        # Create DataFrame in correct feature order
        input_data = pd.DataFrame([[
            credit_score, age, tenure, balance, num_of_products,
            has_cr_card, is_active_member, estimated_salary,
            geography_germany, geography_spain, gender_male
        ]], columns=[
            "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary",
            "Geography_Germany", "Geography_Spain", "Gender_Male"
        ])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]  # churn probability

    return render_template('index.html', prediction=prediction, prediction_proba=prediction_proba)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
