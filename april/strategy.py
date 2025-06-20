import pandas as pd
from pandas import DataFrame
from typing import List, Dict, Any
import numpy as np
from numpy import ndarray
import statsmodels.api as sm
from scipy.stats import norm

COINTEGRATED_PAIRS = [
    (18, 20), # p-value in trading period: 0.002775
    (15, 18)  # p-value in trading period: 0.002145
]

TRADING_PAIRS_PARAMS: List[Dict[str, Any]] = []
prices_df_full = pd.read_csv("./prices.txt", sep=r"\s+", header=None)
prices_formation = prices_df_full.iloc[:250]

for pair in COINTEGRATED_PAIRS:
    stock_y_idx, stock_x_idx = pair
    X_formation = sm.add_constant(prices_formation.iloc[:, stock_x_idx])
    y_formation = prices_formation.iloc[:, stock_y_idx]
    model = sm.OLS(y_formation, X_formation).fit()
    TRADING_PAIRS_PARAMS.append({
        'pair': pair, 'y_idx': stock_y_idx, 'x_idx': stock_x_idx,
        'hedge_ratio': model.params[stock_x_idx], 'intercept': model.params['const']
    })

print("--- Strategy Parameters Calculated for STABLE Pairs ---")
for params in TRADING_PAIRS_PARAMS:
    print(f"Pair {params['pair']}: HR={params['hedge_ratio']:.4f}, Intercept={params['intercept']:.4f}")

current_positions_april = np.zeros(50)

def get_aprils_positions(prices_so_far: DataFrame) -> ndarray:
    global current_positions_april
    
    z_score_window = 60
    if prices_so_far.shape[0] < z_score_window:
        return current_positions_april.astype(int)

    ideal_dollar_positions = np.zeros(50) 
    
    entry_threshold = 0.75 
    asset_position_limit = 10000.0 

    for pair_params in TRADING_PAIRS_PARAMS:
        y_idx = pair_params['y_idx']
        x_idx = pair_params['x_idx']
        hedge_ratio = pair_params['hedge_ratio']
        intercept = pair_params['intercept']

        spread = prices_so_far.iloc[:, y_idx] - hedge_ratio * prices_so_far.iloc[:, x_idx] - intercept
        rolling_mean = spread.rolling(window=z_score_window).mean().iloc[-1]
        rolling_std = spread.rolling(window=z_score_window).std().iloc[-1]

        if rolling_std == 0 or np.isnan(rolling_std):
            continue
            
        z_score = (spread.iloc[-1] - rolling_mean) / rolling_std

        if abs(z_score) < entry_threshold:
            continue 
        target_dollar_position_for_pair = -np.sign(z_score) * asset_position_limit

        # Add this pair's maxed-out dollar positions to the aggregate total
        ideal_dollar_positions[y_idx] += target_dollar_position_for_pair
        ideal_dollar_positions[x_idx] -= target_dollar_position_for_pair * hedge_ratio


    final_dollar_positions = np.zeros(50)
    for i in range(50):
        if abs(ideal_dollar_positions[i]) > asset_position_limit:
            final_dollar_positions[i] = np.sign(ideal_dollar_positions[i]) * asset_position_limit
        else:
            final_dollar_positions[i] = ideal_dollar_positions[i]
    final_share_positions = np.zeros(50)
    latest_prices = prices_so_far.iloc[-1]
    for i in range(50):
        if latest_prices[i] > 0:
            final_share_positions[i] = final_dollar_positions[i] / latest_prices[i]

    current_positions_april = final_share_positions
    return final_share_positions.astype(int)
