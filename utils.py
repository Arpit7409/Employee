import pandas as pd
import numpy as np

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df["Event"] = df["Attrition"].map({"Yes": 1, "No": 0})
    df["Time"] = df["YearsAtCompany"]
    df = add_synthetic_features(df)
    df = df.drop(["EmployeeNumber", "Attrition", "Over18", "StandardHours"], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

def add_synthetic_features(df):
    np.random.seed(42)
    df["EngagementScore"] = np.random.normal(loc=df["JobSatisfaction"]*0.5, scale=0.5)
    df["FlexibleScheduleFlag"] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
    df["HybridWorkDays"] = np.random.randint(0, 5, len(df))
    df["AnnualIncentiveAmount"] = np.abs(np.random.normal(loc=5000, scale=2000, size=len(df))).astype(int)
    df["SupervisorRating"] = np.random.randint(1, 6, len(df))
    df["CommuteTime"] = np.random.randint(10, 120, len(df))
    df["CareerGrowthFlag"] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    return df