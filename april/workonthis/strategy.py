# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================
import os
import sys
import numpy as np
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# --- PATH SETUP ---
try:
    strategy_dir = os.path.dirname(os.path.abspath(__file__))
    if strategy_dir not in sys.path:
        sys.path.append(strategy_dir)
except NameError:
    pass

# --- DATA IMPORT ---
try:
    # This file contains the pair definitions and their initial static hedge ratios
    from april.initial_static_ratios import INITIAL_STATIC_RATIOS
except ImportError:
    print("ERROR: initial_static_ratios.py not found. Please run info_static.py first.")
    INITIAL_STATIC_RATIOS = []

# ==============================================================================
# SECTION 2: FINAL OPTIMIZED PARAMETERS FROM ALL GRID SEARCHES
# ==============================================================================

# These parameters were found to be optimal for the Z-Score model in the STATIC period.
EARLY_PERIOD_PARAMS = {
    "ZSCORE_LOOKBACK": 150,
    "ENTRY_THRESHOLD": 0.5,
    "STOP_LOSS_THRESHOLD": 4.0,
}

# These parameters were found to be optimal for the OU model in the ADAPTIVE period.
LATE_PERIOD_PARAMS = {
    "OU_LOOKBACK": 460,
    "RISK_AVERSION": 0.15,
    "TRANSACTION_COST": 0.0005, # Kept fixed as a reasonable assumption
}

# The optimal day to switch models, discovered via grid search.
TRANSITION_DAY = 651
POSITION_SIZE = 10000

# --- STATE MANAGEMENT ---
PAIR_POSITION_STATE = {str(info['pair']): 0 for info in INITIAL_STATIC_RATIOS}

# ==============================================================================
# SECTION 3: HELPER FUNCTIONS
# ==============================================================================

def estimate_ou_parameters(spread_series: np.ndarray):
    """Estimates the Ornstein-Uhlenbeck parameters (mu, theta, sigma_ou)."""
    if len(spread_series) < 2: return None
    spread_lagged = spread_series[:-1]
    spread_diff = spread_series[1:] - spread_lagged
    model = sm.OLS(spread_diff, sm.add_constant(spread_lagged)).fit()
    alpha, beta = model.params
    if beta >= 0: return None
    
    theta = -np.log(1 + beta)
    mu = alpha / theta
    sigma = np.std(model.resid)
    sqrt_term = -2 * np.log(1 + beta)
    if sqrt_term <= 0: return None
    sigma_ou = sigma / np.sqrt(sqrt_term)
    
    return {'mu': mu, 'theta': theta, 'sigma_ou': sigma_ou}

def calculate_optimal_boundaries(theta, sigma_ou, risk_aversion, transaction_cost):
    """Calculates the optimal trading boundaries from OU parameters."""
    if theta <= 0 or sigma_ou <= 0: return None
    profit_target = (sigma_ou / np.sqrt(2 * theta)) * risk_aversion
    entry_boundary = profit_target + transaction_cost * sigma_ou
    exit_boundary = transaction_cost * sigma_ou
    return {'entry': entry_boundary, 'exit': exit_boundary}

# ==============================================================================
# SECTION 4: THE FINAL HYBRID STRATEGY FUNCTION
# ==============================================================================

def get_aprils_positions(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Implements the final hybrid strategy, switching from a Z-Score model
    to an Ornstein-Uhlenbeck model at the optimal transition day, with all
    parameters optimized via comprehensive grid searches.
    """
    global PAIR_POSITION_STATE

    final_positions = np.zeros(50)
    current_day = prcSoFar.shape[1]
    
    # --- MODEL & PARAMETER SELECTION LOGIC ---
    if current_day < TRANSITION_DAY:
        # --- EXECUTE EARLY PERIOD STRATEGY (Z-SCORE) ---
        params = EARLY_PERIOD_PARAMS
        if current_day < params['ZSCORE_LOOKBACK']: return final_positions

        for pair_info in INITIAL_STATIC_RATIOS:
            pair = pair_info['pair']
            hedge_ratio = pair_info['static_hedge_ratio'] # Use pre-calculated static hedge ratio
            asset1_idx, asset2_idx = pair

            # Generate signal with Z-Score
            zscore_window = prcSoFar[:, -params['ZSCORE_LOOKBACK']:]
            spread = zscore_window[asset1_idx, :] - hedge_ratio * zscore_window[asset2_idx, :]
            if np.std(spread) < 1e-6: continue
            
            current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
            signal_score = (current_spread - np.mean(spread)) / np.std(spread)
            
            # Z-Score Trading Logic
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
    
    else:
        # --- EXECUTE LATE PERIOD STRATEGY (ORNSTEIN-UHLENBECK) ---
        params = LATE_PERIOD_PARAMS
        if current_day < params['OU_LOOKBACK']: return final_positions

        for pair_info in INITIAL_STATIC_RATIOS:
            pair = pair_info['pair']
            asset1_idx, asset2_idx = pair

            try:
                # Use the robust expanding window OLS for the hedge ratio in the adaptive phase
                y = prcSoFar[asset1_idx, :]
                x = prcSoFar[asset2_idx, :]
                model = sm.OLS(y, sm.add_constant(x)).fit()
                hedge_ratio = model.params[1]

                # Generate signal with OU Model
                ou_window = prcSoFar[:, -params['OU_LOOKBACK']:]
                spread = ou_window[asset1_idx, :] - hedge_ratio * ou_window[asset2_idx, :]
                
                ou_params = estimate_ou_parameters(spread)
                if ou_params is None: continue
                
                boundaries = calculate_optimal_boundaries(
                    ou_params['theta'], ou_params['sigma_ou'], 
                    params['RISK_AVERSION'], params['TRANSACTION_COST']
                )
                if boundaries is None: continue

                upper_entry = ou_params['mu'] + boundaries['entry']
                lower_entry = ou_params['mu'] - boundaries['entry']
                upper_exit = ou_params['mu'] + boundaries['exit']
                lower_exit = ou_params['mu'] - boundaries['exit']
                
                current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
                pair_key = str(pair)
                current_pos = PAIR_POSITION_STATE.get(pair_key, 0)
                new_pos = current_pos

                if current_pos == 0:
                    if current_spread < lower_entry: new_pos = 1
                    elif current_spread > upper_entry: new_pos = -1
                elif current_pos == 1:
                    if current_spread >= lower_exit: new_pos = 0
                elif current_pos == -1:
                    if current_spread <= upper_exit: new_pos = 0
                
                PAIR_POSITION_STATE[pair_key] = new_pos
            except Exception:
                continue

    # --- Position Sizing (Common for both periods) ---
    for pair_key, position in PAIR_POSITION_STATE.items():
        if position != 0:
            pair = eval(pair_key)
            asset1_idx, asset2_idx = pair
            
            # Determine the correct hedge ratio for the day before sizing
            if current_day < TRANSITION_DAY:
                hedge_ratio = next(item['static_hedge_ratio'] for item in INITIAL_STATIC_RATIOS if str(item["pair"]) == pair_key)
            else:
                y = prcSoFar[asset1_idx, :]
                x = prcSoFar[asset2_idx, :]
                hedge_ratio = sm.OLS(y, sm.add_constant(x)).fit().params[1]

            pos1 = position * (POSITION_SIZE / prcSoFar[asset1_idx, -1])
            final_positions[asset1_idx] += pos1
            final_positions[asset2_idx] += - (pos1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]
            
    return final_positions.astype(int)