# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================
import os
import sys
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# --- PATH SETUP ---
# Ensures the script can be loaded by a backtester in a different directory.
try:
    strategy_dir = os.path.dirname(os.path.abspath(__file__))
    if strategy_dir not in sys.path:
        sys.path.append(strategy_dir)
except NameError:
    pass

# --- DATA IMPORT ---
# This is the crucial step: importing our pre-calculated models.
try:
    from april.workonthis.final_strategy_data import FINAL_STRATEGY_MODELS
except ImportError:
    print("ERROR: final_strategy_data.py not found. Please run info_generator.py first.")
    FINAL_STRATEGY_MODELS = []

# ==============================================================================
# SECTION 2: HELPER FUNCTION
# ==============================================================================

def calculate_optimal_boundaries(theta, sigma_sq, risk_aversion):
    """
    Calculates the optimal trading boundaries based on pre-calculated OU parameters.
    sigma_sq is the variance of the residuals, not the volatility of the process.
    """
    if theta <= 0 or sigma_sq <= 0: return None
    
    # This formula is a simplified standard for the optimal boundary
    # It balances the speed of reversion (theta) against variance (sigma_sq) and risk appetite
    entry_boundary = np.sqrt(sigma_sq / (1 - np.exp(-2 * theta))) * risk_aversion
    
    # We set a simple exit rule: revert to the mean
    return {'entry': entry_boundary, 'exit': 0}

# ==============================================================================
# SECTION 3: FINAL STRATEGY DEFINITION
# ==============================================================================

# These are the winning hyperparameters from our final grid search.
FINAL_PARAMS = {
    "TRADING_START_DAY": 601, # Start trading after the 600-day training period
    "RISK_AVERSION": 0.20,
}

POSITION_SIZE = 10000
# State management dictionary to track positions for each pair
PAIR_POSITION_STATE = {str(p['pair']): 0 for p in FINAL_STRATEGY_MODELS}

# ==============================================================================
# SECTION 4: THE FINAL STRATEGY FUNCTION
# ==============================================================================

def get_aprils_positions(prcSoFar: np.ndarray) -> np.ndarray:
    """
    Implements the final, theoretically-grounded statistical arbitrage strategy.
    
    This strategy trades a portfolio of pairs using a pure "closed-form" approach:
    1.  It uses a single, pre-calculated hedge ratio for each pair.
    2.  It uses a single, pre-calculated set of Ornstein-Uhlenbeck parameters.
    3.  It trades using optimal boundaries derived from these stable parameters.
    """
    global PAIR_POSITION_STATE
    final_positions = np.zeros(50)
    current_day = prcSoFar.shape[1]
    
    # Do not trade before the specified start day
    if current_day < FINAL_PARAMS['TRADING_START_DAY']:
        return final_positions

    # --- Trading Logic using Pre-Calculated Models ---
    for model_params in FINAL_STRATEGY_MODELS:
        pair = model_params['pair']
        asset1_idx, asset2_idx = pair

        try:
            # --- Step 1: Extract the Stable, Pre-Calculated Model ---
            hedge_ratio = model_params['hedge_ratio']
            mu = model_params['ou_mu']
            theta = model_params['ou_theta']
            sigma_sq = model_params['ou_sigma_sq']
            
            # --- Step 2: Calculate Optimal Trading Boundaries ---
            boundaries = calculate_optimal_boundaries(theta, sigma_sq, FINAL_PARAMS['RISK_AVERSION'])
            if boundaries is None: continue

            # --- Step 3: Make the Trading Decision ---
            upper_entry = mu + boundaries['entry']
            lower_entry = mu - boundaries['entry']
            
            current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
            pair_key = str(pair)
            current_pos = PAIR_POSITION_STATE.get(pair_key, 0)
            new_pos = current_pos

            if current_pos == 0:
                if current_spread < lower_entry: new_pos = 1      # Go Long
                elif current_spread > upper_entry: new_pos = -1 # Go Short
            elif current_pos == 1:
                if current_spread >= mu: new_pos = 0 # Exit when it reverts back to the mean
            elif current_pos == -1:
                if current_spread <= mu: new_pos = 0 # Exit when it reverts back to the mean
            
            PAIR_POSITION_STATE[pair_key] = new_pos

            # --- Step 4: Translate Position into Asset Shares ---
            if new_pos != 0:
                pos1 = new_pos * (POSITION_SIZE / prcSoFar[asset1_idx, -1])
                final_positions[asset1_idx] += pos1
                pos2 = - (pos1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]
                final_positions[asset2_idx] += pos2
        
        except Exception:
            # For safety, if any calculation fails for a single pair, skip it for that day.
            continue
            
    return final_positions.astype(int)