import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import train_rsf, explain_model_with_shap
from predict import calculate_brier_score, simulate_intervention
from utils import load_and_preprocess

st.image("pic.png", width=120)
st.title("HR Analytics Dashboard: Employee Retention Risk on IBM Dataset")

# Load and preprocess data
data_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = load_and_preprocess(data_path)

X = df.drop(["Time", "Event"], axis=1)
y = np.array(list(zip(df["Event"], df["Time"])), dtype=[("event", bool), ("time", float)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RSF model
st.subheader("Model Training")
rsf_model = train_rsf(X_train, y_train)

# SHAP Explanation
if st.button("Run SHAP Feature Explanation"):
    explain_model_with_shap(rsf_model, X_train, X_test)

# Model Evaluation
st.subheader("Model Evaluation")
rsf_brier = calculate_brier_score(rsf_model, X_test, y_test, model_type='rsf')
st.write(f"RSF Integrated Brier Score: {rsf_brier:.3f}")

# Intervention Simulation
st.subheader("Simulated Interventions")
interventions = {
    "EngagementScore": {"func": lambda x: x * 1.2, "cost_per_unit": 50},
    "MonthlyIncome": {"func": lambda x: x * 1.2, "cost_per_unit": 10000},
    "FlexibleScheduleFlag": {"func": lambda x: 1, "cost_per_unit": 200},
    "AnnualIncentiveAmount": {"func": lambda x: x * 1.3, "cost_per_unit": 5000},
    "SupervisorRating": {"func": lambda x: np.clip(x + 1, 1, 5), "cost_per_unit": 1000},
}

results, time_points, original_median = simulate_intervention(rsf_model, X_test, interventions)
roi_df = pd.DataFrame.from_dict({k: v["roi"] for k, v in results.items()}, orient='index', columns=["ROI"])
st.dataframe(roi_df.sort_values("ROI", ascending=False))

# Retention Prediction
st.subheader("🎯 Predict Retention for Custom Employee")

with st.form("retention_form"):
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000, 500)
    flexible = st.selectbox("Flexible Schedule", ["No", "Yes"]) == "Yes"
    engagement = st.slider("Engagement Score", -2.0, 2.0, 0.0, 0.1)
    remote_days = st.slider("Hybrid Work Days", 0, 5, 2)
    bonus = st.number_input("Annual Incentive", 0, 20000, 3000, 500)
    supervisor_rating = st.slider("Supervisor Rating", 1, 5, 3)
    commute_time = st.slider("Commute Time (min)", 0, 120, 45)
    growth_flag = st.selectbox("Career Growth Available", ["No", "Yes"]) == "Yes"
    target_year = st.slider("Year to Predict Retention At", 1, 10, 2)
    predict = st.form_submit_button("Predict Retention")

if predict:
    input_data = pd.DataFrame([{ 
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "FlexibleScheduleFlag": int(flexible),
        "EngagementScore": engagement,
        "HybridWorkDays": remote_days,
        "AnnualIncentiveAmount": bonus,
        "SupervisorRating": supervisor_rating,
        "CommuteTime": commute_time,
        "CareerGrowthFlag": int(growth_flag),
    }])

    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[X.columns]

    surv_func = rsf_model.predict_survival_function(input_data, return_array=True)[0]
    survival_at_target = np.interp(target_year, rsf_model.unique_times_, surv_func)
    st.success(f"Estimated Retention Chance at {target_year} Years: {survival_at_target * 100:.2f}%")

