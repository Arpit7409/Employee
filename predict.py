import numpy as np
import pandas as pd
from sksurv.metrics import integrated_brier_score


def calculate_brier_score(model, X, y, model_type='rsf'):
    from sksurv.metrics import integrated_brier_score

    times = np.quantile(y["time"][y["event"]], [0.25, 0.5, 0.75])

    surv_funcs = model.predict_survival_function(X)

    surv_prob_matrix = np.asarray([
        fn(times) for fn in surv_funcs 
    ])

    return integrated_brier_score(y, y, surv_prob_matrix, times)


def simulate_intervention(rsf, X_test, interventions):
    results = {}
    original_survival = rsf.predict_survival_function(X_test, return_array=True)
    original_median = np.median(original_survival, axis=0)

    for feature, params in interventions.items():
        X_int = X_test.copy()
        X_int[feature] = params["func"](X_int[feature])
        surv_array = rsf.predict_survival_function(X_int, return_array=True)
        intervention_median = np.median(surv_array, axis=0)

        improvement = np.trapz(intervention_median - original_median, rsf.unique_times_)
        roi = improvement / (params["cost_per_unit"] / 1000)

        results[feature] = {
            "survival": surv_array,
            "improvement": improvement,
            "roi": roi
        }

    return results, rsf.unique_times_, original_median