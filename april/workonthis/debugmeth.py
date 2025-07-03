# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
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

from backtester import Backtester, Params

# ==============================================================================
# SECTION 2: INITIAL DATA & HELPER FUNCTIONS
# ==============================================================================
try:
    from april.initial_static_ratios import INITIAL_STATIC_RATIOS
except ImportError:
    print("ERROR: initial_static_ratios.py not found. Please run info_static.py first.")
    INITIAL_STATIC_RATIOS = []

def estimate_ou_parameters(spread_series: np.ndarray):
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

def calculate_optimal_boundaries(theta, sigma_ou, risk_aversion, transaction_cost=0.05):
    if theta <= 0 or sigma_ou <= 0: return None
    profit_target = (sigma_ou / np.sqrt(2 * theta)) * risk_aversion
    entry_boundary = profit_target + transaction_cost * sigma_ou
    exit_boundary = transaction_cost * sigma_ou
    return {'entry': entry_boundary, 'exit': exit_boundary}

# ==============================================================================
# SECTION 3: REFACTORED STRATEGY CLASS FOR DEBUGGING
# ==============================================================================
class DebugHybridStrategy:
    def __init__(self, static_params, adaptive_params, transition_day, position_size):
        self.static_params = static_params
        self.adaptive_params = adaptive_params
        self.transition_day = transition_day
        self.position_size = position_size
        
        # --- THIS IS THE FIX ---
        # We now use the correct key 'static_hedge_ratio' to read the data
        self.static_ratios_map = {str(info['pair']): info['static_hedge_ratio'] for info in INITIAL_STATIC_RATIOS}
        # --- END OF FIX ---
        
        self.pair_position_state = {pair_key: 0 for pair_key in self.static_ratios_map.keys()}

    def getMyPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        final_positions = np.zeros(50)
        current_day = prcSoFar.shape[1]
        
        is_static_period = current_day < self.transition_day
        
        if is_static_period:
            active_params = self.static_params
            if current_day < active_params['ZSCORE_LOOKBACK']: return final_positions
        else:
            active_params = self.adaptive_params
            if current_day < active_params['OU_LOOKBACK']: return final_positions

        for pair_key, static_hr in self.static_ratios_map.items():
            pair = eval(pair_key)
            asset1_idx, asset2_idx = pair
            new_pos = self.pair_position_state.get(pair_key, 0)
            
            try:
                hedge_ratio = 0 
                if is_static_period:
                    hedge_ratio = static_hr
                    zscore_window = prcSoFar[:, -active_params['ZSCORE_LOOKBACK']:]
                    spread = zscore_window[asset1_idx, :] - hedge_ratio * zscore_window[asset2_idx, :]
                    if np.std(spread) < 1e-6: continue
                    
                    current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
                    signal_score = (current_spread - np.mean(spread)) / np.std(spread)
                    
                    current_pos = self.pair_position_state.get(pair_key, 0)
                    if current_pos == 0:
                        if signal_score < -active_params['ENTRY_THRESHOLD']: new_pos = 1
                        elif signal_score > active_params['ENTRY_THRESHOLD']: new_pos = -1
                    elif current_pos == 1:
                        if signal_score >= 0 or signal_score < -active_params['STOP_LOSS_THRESHOLD']: new_pos = 0
                    elif current_pos == -1:
                        if signal_score <= 0 or signal_score > active_params['STOP_LOSS_THRESHOLD']: new_pos = 0
                
                else: # ADAPTIVE PERIOD
                    y, x = prcSoFar[asset1_idx, :], prcSoFar[asset2_idx, :].reshape(-1, 1)
                    hedge_ratio = Ridge(alpha=1.0).fit(x, y.ravel()).coef_[0]

                    ou_window = prcSoFar[:, -active_params['OU_LOOKBACK']:]
                    spread = ou_window[asset1_idx, :] - hedge_ratio * ou_window[asset2_idx, :]
                    
                    ou_params = estimate_ou_parameters(spread)
                    if ou_params is None: continue
                    
                    boundaries = calculate_optimal_boundaries(ou_params['theta'], ou_params['sigma_ou'], active_params['RISK_AVERSION'])
                    if boundaries is None: continue

                    current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
                    current_pos = self.pair_position_state.get(pair_key, 0)
                    
                    if current_pos == 0:
                        if current_spread < (ou_params['mu'] - boundaries['entry']): new_pos = 1
                        elif current_spread > (ou_params['mu'] + boundaries['entry']): new_pos = -1
                    elif current_pos == 1:
                        if current_spread >= ou_params['mu']: new_pos = 0
                    elif current_pos == -1:
                        if current_spread <= ou_params['mu']: new_pos = 0
                
                self.pair_position_state[pair_key] = new_pos
                if new_pos != 0:
                    pos1 = new_pos * (self.position_size / prcSoFar[asset1_idx, -1])
                    final_positions[asset1_idx] += pos1
                    final_positions[asset2_idx] += - (pos1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]

            except Exception as e:
                continue
                
        return final_positions.astype(int)

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK FOR DEBUGGING
# ==============================================================================
if __name__ == '__main__':
    # --- Hardcode a single set of promising parameters to test ---
    test_params_config = {
        "TRANSITION_DAY": 501,
        "POSITION_SIZE": 10000,
        "STATIC_PARAMS": {
            "ZSCORE_LOOKBACK": 150,
            "ENTRY_THRESHOLD": 0.5,
            "STOP_LOSS_THRESHOLD": 4.0,
        },
        "ADAPTIVE_PARAMS": {
            "OU_LOOKBACK": 350,
            "RISK_AVERSION": 0.2,
        }
    }

    print("--- Running a single backtest for debugging ---")
    print(f"Parameters: {test_params_config}")

    # --- Run the backtest ---
    try:
        strategy = DebugHybridStrategy(
            static_params=test_params_config['STATIC_PARAMS'],
            adaptive_params=test_params_config['ADAPTIVE_PARAMS'],
            transition_day=test_params_config['TRANSITION_DAY'],
            position_size=test_params_config['POSITION_SIZE']
        )
        
        bt_params = Params(strategy_function=strategy.getMyPosition, start_day=1, end_day=750)
        backtester = Backtester(bt_params)
        results = backtester.run(start_day=bt_params.start_day, end_day=bt_params.end_day)
        
        start_idx = 151 
        target_pnl = results['daily_pnl'][start_idx:]
        
        if len(target_pnl) > 0 and np.std(target_pnl) > 0:
            score = np.mean(target_pnl) - 0.1 * np.std(target_pnl)
            print("\n--- Backtest Successful ---")
            print(f"Final Score: {score:.4f}")
            backtester.show_dashboard(results, ["cum-pnl", "daily-pnl"])
        else:
            print("\n--- Backtest Completed but No Trades Were Made or PnL was Flat ---")

    except Exception as e:
        print("\n--- BACKTEST FAILED WITH AN ERROR ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        import traceback; traceback.print_exc()