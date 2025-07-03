# info_static.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

# These are the "ideal" pairs we found by analyzing all 750 days of data.
IDEAL_PAIRS = [
    (2, 6), (8, 34), (11, 23), (12, 49), (18, 29), (20, 22), 
    (26, 45), (33, 35), (37, 41), (40, 45), (47, 49), (48, 49)
]

# The period used to calculate the initial, stable hedge ratios.
STATIC_CALCULATION_PERIOD = 500
HEDGE_ESTIMATOR = 'ridge' # The winning estimator

def get_hedge_ratio(y, x, estimator: str):
    """Calculates a hedge ratio using a specified model."""
    if estimator == 'ols':
        model = sm.OLS(y, sm.add_constant(x)).fit()
        return model.params[1]
    elif estimator == 'ridge':
        model = Ridge(alpha=1.0)
        # scikit-learn expects X to be 2D and y to be 1D
        model.fit(x, y.ravel())
        return model.coef_[0]
    else:
        raise ValueError(f"Estimator {estimator} not recognized.")

def create_initial_static_ratios(price_file_path: str):
    """
    Calculates a single, static hedge ratio for each ideal pair based on an
    initial period of data and saves it to a file.
    """
    print(f"--- Calculating initial static hedge ratios using '{HEDGE_ESTIMATOR}' on first {STATIC_CALCULATION_PERIOD} days ---")
    
    prices_df = pd.read_csv(price_file_path, header=None, sep=r'\s+')
    static_period_df = prices_df.iloc[:STATIC_CALCULATION_PERIOD]

    initial_ratios = []
    for pair in IDEAL_PAIRS:
        asset1_idx, asset2_idx = pair
        
        y = static_period_df.iloc[:, asset1_idx].values
        x = static_period_df.iloc[:, asset2_idx].values.reshape(-1, 1)
        hedge_ratio = get_hedge_ratio(y, x, HEDGE_ESTIMATOR)
        
        initial_ratios.append({
            'pair': pair,
            'static_hedge_ratio': hedge_ratio
        })
        
    with open("initial_static_ratios.py", "w") as f:
        f.write(f"INITIAL_STATIC_RATIOS = {initial_ratios}")
        
    print(f"--- Static ratio data saved to initial_static_ratios.py ---")

if __name__ == "__main__":
    create_initial_static_ratios('prices.txt')