import numpy as np
import pandas as pd
import statsmodels.api as sm 

#2sls Hedge Ratios
STATIC_PAIRS = [
    {'pair': (3, 4), 'hedge_ratio': -0.494726},
    {'pair': (48, 49), 'hedge_ratio': 0.950505},
    {'pair': (6, 23), 'hedge_ratio': -0.313472},
    {'pair': (17, 29), 'hedge_ratio': 1.266978},
    {'pair': (20, 35), 'hedge_ratio': 1.064861},
    {'pair': (7, 47), 'hedge_ratio': -1.518148},
    {'pair': (18, 38), 'hedge_ratio': 0.680895},
    {'pair': (0, 39), 'hedge_ratio': -0.381576},
    {'pair': (31, 34), 'hedge_ratio': 1.047837},
    {'pair': (11, 19), 'hedge_ratio': -3.305927},
    {'pair': (15, 24), 'hedge_ratio': 0.588129},
    {'pair': (26, 43), 'hedge_ratio': -0.365806},
    {'pair': (8, 40), 'hedge_ratio': 0.472898},
    {'pair': (25, 37), 'hedge_ratio': 0.864921},
    {'pair': (5, 30), 'hedge_ratio': 0.483024},
    {'pair': (45, 46), 'hedge_ratio': -0.070321},
    {'pair': (12, 13), 'hedge_ratio': 1.077101},
    {'pair': (1, 33), 'hedge_ratio': 1.899301}
]
# --- CONSTANTS ---
LOOKBACK_DAYS = 85
Z_ENTRY_THRESHOLD = 0.08
POSITION_SIZE = 10000

def get_aprils_positions(prcSoFar):
    final_positions = np.zeros(50)
    current_day_index = prcSoFar.shape[1]
    
    if current_day_index < LOOKBACK_DAYS:
        return final_positions

    for pair_info in STATIC_PAIRS:
        asset1_idx, asset2_idx = pair_info['pair']
        hedge_ratio = pair_info['hedge_ratio']
        
        asset1_prices = prcSoFar[int(asset1_idx), -LOOKBACK_DAYS:]
        asset2_prices = prcSoFar[int(asset2_idx), -LOOKBACK_DAYS:]

        spread_series = asset1_prices - hedge_ratio * asset2_prices
        
        if spread_series.std() > 1e-6:
            rolling_mean = spread_series.mean()
            rolling_std = spread_series.std()
            current_spread_value = spread_series[-1]
            z_score = (current_spread_value - rolling_mean) / rolling_std
            
            current_price_asset1 = prcSoFar[int(asset1_idx), -1]
            position_asset1 = POSITION_SIZE / current_price_asset1
            
            
            # 2. If not stopped out, check for a normal entry signal.
            if z_score > Z_ENTRY_THRESHOLD:
                final_positions[int(asset1_idx)] = -position_asset1
                final_positions[int(asset2_idx)] = position_asset1 * hedge_ratio
            
            elif z_score < -Z_ENTRY_THRESHOLD:
                final_positions[int(asset1_idx)] = position_asset1
                final_positions[int(asset2_idx)] = -position_asset1 * hedge_ratio
            
            # 3. If the z-score is between the entry thresholds, our desired position is also zero.
            # The backtester will automatically flatten any open trades.
            
    return final_positions.astype(int)
