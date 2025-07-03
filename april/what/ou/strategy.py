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

# --- PATH SETUP ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
except NameError:
    parent_dir = ".."

# --- DATA IMPORT ---
try:
    from april.initial_static_ratios import INITIAL_STATIC_RATIOS
except ImportError:
    print("ERROR: initial_static_ratios.py not found. Please run info_static.py first.")
    INITIAL_STATIC_RATIOS = []

# ==============================================================================
# SECTION 2: FINAL OPTIMIZED PARAMETERS
# ==============================================================================

# ** PLEASE REPLACE THESE WITH YOUR WINNING PARAMETERS **
# These were the best parameters found for the Z-Score model in the early period
EARLY_PERIOD_PARAMS = {
    "ZSCORE_LOOKBACK": 150,      # From your previous Z-score run
    "ENTRY_THRESHOLD": 0.5,      # From your previous Z-score run
    "STOP_LOSS_THRESHOLD": 4.0,  # From your previous Z-score run
}

# These were the best parameters found for the OU model in the late period
LATE_PERIOD_PARAMS = {
    "OU_LOOKBACK": 250,
    "ENTRY_THRESHOLD": 0.5,
    "STOP_LOSS_THRESHOLD": 3.5,
}

# --- GLOBAL STRATEGY SETTINGS ---
TRANSITION_DAY = 600  # The optimal start day for the winning OU strategy
POSITION_SIZE = 10000

# --- STATE MANAGEMENT ---
PAIR_POSITION_STATE = {str(info['pair']): 0 for info in INITIAL_STATIC_RATIOS}

# ==============================================================================
# SECTION 3: HELPER FUNCTIONS
# ==============================================================================

def estimate_ou_mu(spread_series: np.ndarray) -> float:
    """Estimates the long-term mean (mu) of the Ornstein-Uhlenbeck process."""
    spread_lagged = spread_series[:-1]
    spread_diff = spread_series[1:] - spread_lagged
    x_reg = sm.add_constant(spread_lagged)
    model = sm.OLS(spread_diff, x_reg).fit()
    alpha, beta = model.params
    if abs(beta) < 1e-6: return np.mean(spread_series)
    return alpha / -beta

# ==============================================================================
# SECTION 4: THE FINAL HYBRID STRATEGY FUNCTION
# ==============================================================================

def get_aprils_positions(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Implements the final hybrid strategy, switching from a Z-Score model
    to an Ornstein-Uhlenbeck model at the optimal transition day.
    """
    global PAIR_POSITION_STATE

    final_positions = np.zeros(50)
    current_day = prcSoFar.shape[1]
    
    # --- MODEL SELECTION LOGIC ---
    if current_day < TRANSITION_DAY:
        # --- EXECUTE EARLY PERIOD STRATEGY (Z-SCORE) ---
        params = EARLY_PERIOD_PARAMS
        if current_day < params['ZSCORE_LOOKBACK']: return final_positions

        for pair_info in INITIAL_STATIC_RATIOS:
            pair = pair_info['pair']
            asset1_idx, asset2_idx = pair

            # Use static hedge ratio for the early period
            hedge_ratio = pair_info['static_hedge_ratio']

            zscore_window = prcSoFar[:, -params['ZSCORE_LOOKBACK']:]
            spread = zscore_window[asset1_idx, :] - hedge_ratio * zscore_window[asset2_idx, :]
            if np.std(spread) < 1e-6: continue
            
            current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
            signal_score = (current_spread - np.mean(spread)) / np.std(spread)
            
            # --- Trading Logic (identical for both models) ---
            pair_key = str(pair)
            current_pos = PAIR_POSITION_STATE.get(pair_key, 0)
            new_pos = current_pos

            if current_pos == 0:
                if signal_score < -params['ENTRY_THRESHOLD']: new_pos = 1
                elif signal_score > params['ENTRY_THRESHOLD']: new_pos = -1
            elif current_pos == 1:
                if signal_score >= 0 or signal_score < -params['STOP_LOSS_THRESHOLD']: new_pos = 0
            elif current_pos == -1:
                if signal_score <= 0 or signal_score > params['STOP_LOSS_THRESHOLD']: new_pos = 0
            
            PAIR_POSITION_STATE[pair_key] = new_pos

            if new_pos != 0:
                pos1 = new_pos * (POSITION_SIZE / prcSoFar[asset1_idx, -1])
                final_positions[asset1_idx] += pos1
                final_positions[asset2_idx] += - (pos1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]
    
    else:
        # --- EXECUTE LATE PERIOD STRATEGY (ORNSTEIN-UHLENBECK) ---
        params = LATE_PERIOD_PARAMS
        if current_day < params['OU_LOOKBACK']: return final_positions

        for pair_info in INITIAL_STATIC_RATIOS:
            pair = pair_info['pair']
            asset1_idx, asset2_idx = pair

            # Use expanding window OLS for the hedge ratio in the adaptive period
            y = prcSoFar[asset1_idx, :]
            x = prcSoFar[asset2_idx, :]
            model = sm.OLS(y, sm.add_constant(x)).fit()
            hedge_ratio = model.params[1]

            ou_window = prcSoFar[:, -params['OU_LOOKBACK']:]
            spread = ou_window[asset1_idx, :] - hedge_ratio * ou_window[asset2_idx, :]
            spread_std = np.std(spread)
            if spread_std < 1e-6: continue

            try:
                mu = estimate_ou_mu(spread)
                current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
                signal_score = (current_spread - mu) / spread_std

                # --- Trading Logic (identical for both models) ---
                pair_key = str(pair)
                current_pos = PAIR_POSITION_STATE.get(pair_key, 0)
                new_pos = current_pos

                if current_pos == 0:
                    if signal_score < -params['ENTRY_THRESHOLD']: new_pos = 1
                    elif signal_score > params['ENTRY_THRESHOLD']: new_pos = -1
                elif current_pos == 1:
                    if signal_score >= 0 or signal_score < -params['STOP_LOSS_THRESHOLD']: new_pos = 0
                elif current_pos == -1:
                    if signal_score <= 0 or signal_score > params['STOP_LOSS_THRESHOLD']: new_pos = 0
                
                PAIR_POSITION_STATE[pair_key] = new_pos

                if new_pos != 0:
                    pos1 = new_pos * (POSITION_SIZE / prcSoFar[asset1_idx, -1])
                    final_positions[asset1_idx] += pos1
                    final_positions[asset2_idx] += - (pos1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]
            
            except Exception:
                continue
            
    return final_positions.astype(int)