
from flask import Flask, render_template, request
import pickle, numpy as np, pandas as pd

app = Flask(__name__)

model    = pickle.load(open("model.pkl",   "rb"))
columns  = pickle.load(open("columns.pkl", "rb"))

print("MODEL COLUMNS:")
print(columns)
encoders = pickle.load(open("encoders.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    raw = {
        "Age": int(form["Age"]),
        "BusinessTravel": form["BusinessTravel"],
        "DailyRate": 800,
        "HourlyRate": 65,
        "MonthlyRate": 15000,
        "Department": form["Department"],
        "DistanceFromHome": int(form["DistanceFromHome"]),
        "Education": int(form["Education"]),
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": int(form["EnvironmentSatisfaction"]),
        "Gender": form["Gender"],
        "JobInvolvement": int(form["JobInvolvement"]),
        "JobLevel": int(form["JobLevel"]),
        "JobRole": form["JobRole"],
        "JobSatisfaction": int(form["JobSatisfaction"]),
        "MaritalStatus": form["MaritalStatus"],
        "MonthlyIncome": int(form["MonthlyIncome"]),
        "NumCompaniesWorked": int(form["NumCompaniesWorked"]),
        "OverTime": "Yes" if form["OverTime"] == "1" else "No",
        "PercentSalaryHike": int(form["PercentSalaryHike"]),
        "PerformanceRating": 3,
        "RelationshipSatisfaction": int(form["RelationshipSatisfaction"]),
        "StockOptionLevel": int(form["StockOptionLevel"]),
        "TotalWorkingYears": int(form["TotalWorkingYears"]),
        "TrainingTimesLastYear": int(form["TrainingTimesLastYear"]),
        "WorkLifeBalance": int(form["WorkLifeBalance"]),
        "YearsAtCompany": int(form["YearsAtCompany"]),
        "YearsInCurrentRole": int(form["YearsAtCompany"]) // 2,
        "YearsSinceLastPromotion": int(form["YearsSinceLastPromotion"]),
        "YearsWithCurrManager": int(form["YearsWithCurrManager"])
    }

    for col, le in encoders.items():
        if col in raw and col != "Attrition":
            try:
                raw[col] = le.transform([raw[col]])[0]
            except:
                raw[col] = 0

    input_df = pd.DataFrame([raw])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    prediction_label = "Yes" if model.predict(input_df)[0] == 1 else "No"
    probability = model.predict_proba(input_df)[0]

    return render_template(
        "result.html",
        prediction=prediction_label,
        prob_yes=round(probability[1] * 100),
        prob_no=round(probability[0] * 100)
    )
if __name__ == "__main__":
    app.run(debug=True)