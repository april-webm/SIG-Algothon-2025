# strategy.py (Final Version with Dual Parameter Sets)

import numpy as np
import statsmodels.api as sm
import sys
import os

# --- PATH SETUP ---
try:
    strategy_dir = os.path.dirname(os.path.abspath(__file__))
    if strategy_dir not in sys.path:
        sys.path.append(strategy_dir)
except NameError:
    pass

# --- DATA IMPORT ---
try:
    # This file contains the pair definitions and their initial STATIC hedge ratios
    from april.initial_static_ratios import INITIAL_STATIC_RATIOS
except ImportError:
    print("ERROR: initial_static_ratios.py not found. Please run info_static.py first.")
    INITIAL_STATIC_RATIOS = []

# ==============================================================================
# SECTION 1: DUAL PARAMETER SETS
# ==============================================================================

# These parameters were optimized for the STATIC hedge ratio period (Days ~90-500)
STATIC_PARAMS = {
    "ZSCORE_LOOKBACK": 130,
    "ENTRY_THRESHOLD": 2.1,
    "STOP_LOSS_THRESHOLD": 3.75,
}

# These parameters should be the best you found for the ADAPTIVE period (Days 501-750)
ADAPTIVE_PARAMS = {
    "ZSCORE_LOOKBACK": 110,
    "ENTRY_THRESHOLD": 0.8, # Example value, update with your best result
    "STOP_LOSS_THRESHOLD": 3.5,  # Example value, update with your best result
}

# --- GLOBAL STRATEGY SETTINGS ---
ADAPTIVE_START_DAY = 501       # Day to switch to expanding window HR and ADAPTIVE_PARAMS
POSITION_SIZE = 10000          # Dollar value for the primary asset in a pair

# --- STATE MANAGEMENT ---
# A single state dictionary tracks positions across both periods
PAIR_POSITION_STATE = {str(info['pair']): 0 for info in INITIAL_STATIC_RATIOS}

# ==============================================================================
# SECTION 2: FINAL STRATEGY FUNCTION
# ==============================================================================
def get_aprils_positions(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Implements a hybrid pairs trading strategy that uses different optimized
    parameters for the static and adaptive trading periods.
    """
    global PAIR_POSITION_STATE

    final_positions = np.zeros(50)
    current_day = prcSoFar.shape[1]
    
    # --- Select the appropriate parameter set for the current day ---
    if current_day < ADAPTIVE_START_DAY:
        params = STATIC_PARAMS
    else:
        params = ADAPTIVE_PARAMS

    # Do not trade until we have enough data for the lookback period
    if current_day < params['ZSCORE_LOOKBACK']:
        return final_positions

    for pair_info in INITIAL_STATIC_RATIOS:
        pair = pair_info['pair']
        asset1_idx, asset2_idx = pair

        # --- Hedge Ratio Selection Logic ---
        if current_day < ADAPTIVE_START_DAY:
            hedge_ratio = pair_info['static_hedge_ratio']
        else:
            # Use the robust expanding window calculation
            regression_window = prcSoFar[:, :current_day]
            y = regression_window[asset1_idx, :]
            x = regression_window[asset2_idx, :]
            model = sm.OLS(y, sm.add_constant(x)).fit()
            hedge_ratio = model.params[1]

        # --- Z-Score and Signal Generation ---
        zscore_window = prcSoFar[:, -params['ZSCORE_LOOKBACK']:]
        spread_series = zscore_window[asset1_idx, :] - hedge_ratio * zscore_window[asset2_idx, :]
        if np.std(spread_series) < 1e-6: continue
        
        current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
        z_score = (current_spread - np.mean(spread_series)) / np.std(spread_series)

        # --- Stateful Trading Logic (with Stop-Loss) ---
        pair_key = str(pair)
        current_pos_state = PAIR_POSITION_STATE.get(pair_key, 0)
        new_pos_state = current_pos_state

        if current_pos_state == 0:
            if z_score < -params['ENTRY_THRESHOLD']: new_pos_state = 1
            elif z_score > params['ENTRY_THRESHOLD']: new_pos_state = -1
        elif current_pos_state == 1:
            if z_score >= 0 or z_score < -params['STOP_LOSS_THRESHOLD']: new_pos_state = 0
        elif current_pos_state == -1:
            if z_score <= 0 or z_score > params['STOP_LOSS_THRESHOLD']: new_pos_state = 0
        
        PAIR_POSITION_STATE[pair_key] = new_pos_state

        # --- Calculate Final Positions ---
        if new_pos_state != 0:
            pos_asset1 = new_pos_state * (POSITION_SIZE / prcSoFar[asset1_idx, -1])
            final_positions[asset1_idx] += pos_asset1
            pos_asset2 = - (pos_asset1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]
            final_positions[asset2_idx] += pos_asset2
            
    return final_positions.astype(int)