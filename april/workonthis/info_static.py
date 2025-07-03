# info_static.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import sys

# --- PATH SETUP ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
except NameError:
    parent_dir = ".."


# These are the "ideal" pairs we found by analyzing all 750 days of data.
IDEAL_PAIRS = [
    (2, 6), (8, 34), (11, 23), (12, 49), (18, 29), (20, 22), 
    (26, 45), (33, 35), (37, 41), (40, 45), (47, 49), (48, 49)
]

# --- MODIFIED: The period used to calculate the stable hedge ratios ---
STATIC_CALCULATION_PERIOD = 500

def create_initial_static_ratios(price_file_path: str):
    """
    Calculates a single, static hedge ratio for each ideal pair based on an
    initial period of data and saves it to a file.
    """
    print(f"--- Calculating initial static hedge ratios using first {STATIC_CALCULATION_PERIOD} days (OLS) ---")
    
    prices_df = pd.read_csv(price_file_path, header=None, sep=r'\s+')
    static_period_df = prices_df.iloc[:STATIC_CALCULATION_PERIOD]

    initial_ratios = []
    for pair in IDEAL_PAIRS:
        asset1_idx, asset2_idx = pair
        
        # Using OLS as determined by our validation
        y = static_period_df.iloc[:, asset1_idx].values
        x = static_period_df.iloc[:, asset2_idx].values
        model = sm.OLS(y, sm.add_constant(x)).fit()
        
        initial_ratios.append({
            'pair': pair,
            'static_hedge_ratio': model.params[1]
        })
        
    # Save this data to a new python file so our main strategy can import it
    with open("initial_static_ratios.py", "w") as f:
        f.write(f"INITIAL_STATIC_RATIOS = {initial_ratios}")
        
    print(f"--- Static ratio data has been recalculated and saved to initial_static_ratios.py ---")

if __name__ == "__main__":
    # Create the path to prices.txt assuming it's in the parent directory
    prices_path = os.path.join(parent_dir, "prices.txt")
    create_initial_static_ratios(prices_path)