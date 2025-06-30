# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 2: HELPER FUNCTION
# ==============================================================================

def estimate_ou_parameters(spread_series: np.ndarray):
    """
    Estimates the Ornstein-Uhlenbeck parameters (mu, theta, sigma_sq)
    for a given spread series by running a regression on its discretized form.
    """
    if len(spread_series) < 2: return None
    
    spread_lagged = spread_series[:-1]
    spread_diff = spread_series[1:] - spread_lagged
    
    x_reg = sm.add_constant(spread_lagged)
    model = sm.OLS(spread_diff, x_reg).fit()
    alpha, beta = model.params

    if beta >= 0: return None
    
    theta = -np.log(1 + beta)
    mu = alpha / theta
    sigma_sq = np.var(model.resid)
    
    if theta <= 0 or sigma_sq <= 0: return None

    return {'mu': mu, 'theta': theta, 'sigma_sq': sigma_sq}

# ==============================================================================
# SECTION 3: MAIN ANALYSIS BLOCK
# ==============================================================================
if __name__ == '__main__':
    # --- CONFIGURATION ---
    TRAINING_PERIOD_END = 600
    
    FINAL_PORTFOLIO_PAIRS = [
        (48, 49), (8, 34), (11, 23), (2, 22), (18, 30), (29, 42),
        (9, 31), (7, 13), (25, 37), (3, 44), (26, 45), (0, 40),
        (20, 35), (15, 27), (19, 33), (41, 46)
    ]

    print(f"--- Generating Final Strategy Models ---")
    print(f"Using training data up to day: {TRAINING_PERIOD_END}")

    # --- Load Data ---
    try:
        prices_df = pd.read_csv('prices.txt', header=None, sep=r'\s+')
    except FileNotFoundError:
        print("FATAL ERROR: prices.txt not found. Please ensure it is in the same directory.")
        sys.exit(1)
        
    training_data_df = prices_df.iloc[:TRAINING_PERIOD_END]
    
    final_models = []
    
    # --- Process Each Pair ---
    for pair in FINAL_PORTFOLIO_PAIRS:
        print(f"Processing pair {pair}...", end="")
        asset1_idx, asset2_idx = pair

        # Step 1: Calculate the single, static hedge ratio using OLS
        y = training_data_df.iloc[:, asset1_idx]
        x = training_data_df.iloc[:, asset2_idx]
        model = sm.OLS(y, sm.add_constant(x)).fit()
        
        # --- THIS IS THE FIX ---
        # Access the hedge ratio by its integer position using .iloc[1]
        hedge_ratio = model.params.iloc[1]
        # --- END OF FIX ---

        # Step 2: Create the spread based on this static ratio
        spread = y - hedge_ratio * x
        
        # Step 3: Estimate the OU parameters for this stable spread
        ou_params = estimate_ou_parameters(spread.to_numpy())
        
        if ou_params:
            final_models.append({
                'pair': pair,
                'hedge_ratio': hedge_ratio,
                'ou_mu': ou_params['mu'],
                'ou_theta': ou_params['theta'],
                'ou_sigma_sq': ou_params['sigma_sq']
            })
            print(" -> SUCCESS: Model created.")
        else:
            print(" -> SKIPPED: Spread not mean-reverting in training period.")

    # --- Save the final models to a file ---
    if final_models:
        with open("final_strategy_data.py", "w") as f:
            f.write(f"FINAL_STRATEGY_MODELS = {final_models}")
        print(f"\n--- Analysis Complete ---")
        print(f"Successfully created and saved {len(final_models)} robust pair models to 'final_strategy_data.py'")
    else:
        print("\n--- Analysis Complete ---")
        print("Could not create any valid models from the provided pairs and training period.")