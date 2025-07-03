# info_static.py
import numpy as np
import pandas as pd
import statsmodels.api as sm

IDEAL_PAIRS = [
    (2, 6), (8, 34), (11, 23), (12, 49), (18, 29), (20, 22), 
    (26, 45), (33, 35), (37, 41), (40, 45), (47, 49), (48, 49)
]

# The period used to calculate the initial, stable hedge ratios.
STATIC_CALCULATION_PERIOD = 500

def create_initial_static_ratios(price_file_path: str):
    """
    Calculates a single, static hedge ratio for each ideal pair based on an
    initial period of data.
    """
    print(f"--- Calculating initial static hedge ratios using first {STATIC_CALCULATION_PERIOD} days ---")
    
    prices_df = pd.read_csv(price_file_path, header=None, delim_whitespace=True)
    static_period_df = prices_df.iloc[:STATIC_CALCULATION_PERIOD]

    initial_ratios = []
    for pair in IDEAL_PAIRS:
        asset1_idx, asset2_idx = pair
        
        y = static_period_df.iloc[:, asset1_idx]
        x = static_period_df.iloc[:, asset2_idx]
        model = sm.OLS(y, sm.add_constant(x)).fit()
        hedge_ratio = model.params.iloc[1]
        
        initial_ratios.append({
            'pair': pair,
            'static_hedge_ratio': hedge_ratio
        })
        
    # Save this data to a new python file so our main strategy can import it
    with open("initial_static_ratios.py", "w") as f:
        f.write(f"INITIAL_STATIC_RATIOS = {initial_ratios}")
        
    print(f"--- Static ratio data saved to initial_static_ratios.py ---")

if __name__ == "__main__":
    # Run this file once from your terminal to generate the data file.
    create_initial_static_ratios('prices.txt')