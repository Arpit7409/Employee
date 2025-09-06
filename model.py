import shap
from sklearn.model_selection import GridSearchCV
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt
import streamlit as st


def train_rsf(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5],
        'min_samples_split': [2, 5]
    }
    rsf = RandomSurvivalForest(random_state=42)
    grid_search = GridSearchCV(
        estimator=rsf,
        param_grid=param_grid,
        cv=3,
        scoring='neg_log_loss',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def explain_model_with_shap(model, X_train, X_test):
    sample_background = shap.sample(X_train, 50)
    explainer = shap.KernelExplainer(model.predict, sample_background)
    sample_X_test = X_test.iloc[:50]
    shap_values = explainer.shap_values(sample_X_test)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, sample_X_test, plot_type="dot", show=False)
    plt.title("SHAP Feature Importance for Tuned RSF Model", fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)  
    plt.clf()

    