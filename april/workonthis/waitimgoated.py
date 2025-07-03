# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================
import os
import sys
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
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
# SECTION 3: THE STABLE TRADING STRATEGY CLASS (FROM DEBUGGING)
# ==============================================================================
class StableHybridStrategy:
    def __init__(self, static_params, adaptive_params, transition_day, position_size):
        self.static_params = static_params
        self.adaptive_params = adaptive_params
        self.transition_day = transition_day
        self.position_size = position_size
        self.static_ratios_map = {str(info['pair']): info['static_hedge_ratio'] for info in INITIAL_STATIC_RATIOS}
        self.pair_position_state = {pair_key: 0 for pair_key in self.static_ratios_map.keys()}

    def getMyPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        final_positions = np.zeros(50)
        current_day = prcSoFar.shape[1]
        
        is_static_period = current_day < self.transition_day
        
        if is_static_period:
            active_params = self.static_params
            min_lookback = active_params['ZSCORE_LOOKBACK']
        else:
            active_params = self.adaptive_params
            min_lookback = active_params['OU_LOOKBACK']

        if current_day < min_lookback:
            return final_positions

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

            except Exception:
                continue
                
        return final_positions.astype(int)

# ==============================================================================
# SECTION 4: PARALLEL BACKTESTING WORKER
# ==============================================================================
def run_backtest_worker(params, optimization_target):
    try:
        strategy = StableHybridStrategy(**params)
        bt_params = Params(strategy_function=strategy.getMyPosition, start_day=1, end_day=750)
        backtester = Backtester(bt_params)
        results = backtester.run(start_day=bt_params.start_day, end_day=bt_params.end_day)
        
        pnl = results['daily_pnl']
        
        if optimization_target == 'STATIC':
            start_idx, end_idx = 151, params['transition_day']
        else: # ADAPTIVE
            start_idx = params['transition_day']
            end_idx = 750
        
        target_pnl = pnl[start_idx:end_idx]
        
        if len(target_pnl) > 0 and np.std(target_pnl) > 0:
            score = np.mean(target_pnl) - 0.1 * np.std(target_pnl)
        else: score = -np.inf
            
        return {'params': params, 'score': score}
    except Exception:
        return {'params': params, 'score': -np.inf}

# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # --- CHOOSE YOUR CAMPAIGN ---
    OPTIMIZATION_TARGET = 'ADAPTIVE' # Change to 'STATIC' for the other campaign

    # Define fixed placeholders and the grid for the target period
    if OPTIMIZATION_TARGET == 'ADAPTIVE':
        placeholder_static_params = {
            "ZSCORE_LOOKBACK": 150, "ENTRY_THRESHOLD": 0.5, "STOP_LOSS_THRESHOLD": 4.0,
        }
        adaptive_grid = {
            "OU_LOOKBACK": (250, 601, 10),
            "RISK_AVERSION": (0.1, 1.51, 0.05),
        }
        transition_day_grid = [301, 401, 501, 601, 651]
        
        adaptive_combos = [dict(zip(adaptive_grid.keys(), v)) for v in product(*{k: np.arange(*v).round(2).tolist() if isinstance(v, tuple) else v for k, v in adaptive_grid.items()}.values())]
        
        final_params_list = []
        for start_day in transition_day_grid:
            for adaptive_p in adaptive_combos:
                final_params_list.append({
                    "transition_day": start_day,
                    "position_size": 10000,
                    "static_params": placeholder_static_params,
                    "adaptive_params": adaptive_p
                })
    else: # STATIC
        placeholder_adaptive_params = {
             "OU_LOOKBACK": 350, "RISK_AVERSION": 0.2,
        }
        static_grid = {
            "ZSCORE_LOOKBACK": (90, 181, 15),
            "ENTRY_THRESHOLD": (0.25, 1.51, 0.25),
            "STOP_LOSS_THRESHOLD": (3.0, 4.01, 0.5),
        }
        transition_day_grid = [301, 401, 501, 601]
        
        static_combos = [dict(zip(static_grid.keys(), v)) for v in product(*{k: np.arange(*v).round(2).tolist() if isinstance(v, tuple) else v for k, v in static_grid.items()}.values())]

        final_params_list = []
        for start_day in transition_day_grid:
            for static_p in static_combos:
                final_params_list.append({
                    "transition_day": start_day,
                    "position_size": 10000,
                    "static_params": static_p,
                    "adaptive_params": placeholder_adaptive_params
                })

    print(f"--- Starting Grid Search for {OPTIMIZATION_TARGET} Period ---")
    print(f"--- Total combinations: {len(final_params_list)} ---")
    results_list = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_backtest_worker, params, OPTIMIZATION_TARGET) for params in final_params_list]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results_list.append(result)
            print(f"Completed {i+1}/{len(final_params_list)} -> Score: {result['score']:.2f}")

    if results_list:
        results_df = pd.DataFrame(results_list)
        flat_params = pd.json_normalize(results_df['params'])
        scores_df = results_df[['score']]
        full_results_df = pd.concat([flat_params, scores_df], axis=1)
        best_results_df = full_results_df.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        filename = f"grid_search_results_{OPTIMIZATION_TARGET.lower()}.csv"
        best_results_df.to_csv(filename, index=False)
        
        print("\n\n--- Grid Search Complete ---")
        print(f"\nSUCCESS: All results saved to '{filename}'")
        print(f"\n--- Top 5 Best Parameter Sets for the {OPTIMIZATION_TARGET} Period ---")
        print(best_results_df.head(5).to_string())
    else:
        print("No successful backtest results were obtained.")