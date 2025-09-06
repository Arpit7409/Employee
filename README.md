# 📊 HCL HR Analytics Dashboard

A predictive HR analytics dashboard using **Random Survival Forests (RSF)** to estimate employee retention likelihood and simulate interventions for improved outcomes. Built with **Streamlit** and **scikit-survival**.

## 🎯 Features

- Predict employee retention over time
- SHAP-based feature importance explanation
- Simulate the impact of HR interventions (e.g., salary increase, flexibility)
- Real-time prediction form for individual employee scenarios

## 🧠 Model
- Algorithm: Random Survival Forest (RSF)
- Evaluated using Integrated Brier Score
- Feature explanations via SHAP values

## 🧪 Usage

```bash
pip install -r requirements.txt
streamlit run main.py
```

Make sure to place the required HR dataset file at:  
📁 `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`

## 📂 Project Structure

```
main.py              # Streamlit dashboard entry
model.py             # RSF training + SHAP explainability
predict.py           # Brier score + intervention simulation
utils.py             # Preprocessing and feature engineering
pic.png              # Dashboard logo image
requirements.txt     # Required Python packages
```

## 📈 Inputs for Custom Prediction
- Job Satisfaction
- Monthly Income
- Engagement Score
- Incentive Bonus
- Commute Time
- Supervisor Rating
- Career Growth Flag
- Flexible Work Schedule
