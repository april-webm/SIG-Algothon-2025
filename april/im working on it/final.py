# final_grid_search_OU_control.py

import os
import sys
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
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

from backtester import Backtester, Params

# ==============================================================================
# SECTION 1: HELPER FUNCTIONS
# ==============================================================================
IDEAL_PAIRS = [
    (2, 6), (8, 34), (11, 23), (12, 49), (18, 29), (20, 22), 
    (26, 45), (33, 35), (37, 41), (40, 45), (47, 49), (48, 49)
]

def estimate_ou_parameters(spread_series: np.ndarray):
    spread_lagged = spread_series[:-1]
    spread_diff = spread_series[1:] - spread_lagged
    x_reg = sm.add_constant(spread_lagged)
    model = sm.OLS(spread_diff, x_reg).fit()
    alpha, beta = model.params
    if beta >= 0: return None
    theta = -np.log(1 + beta)
    mu = alpha / theta
    sigma = np.std(model.resid)
    sigma_ou = sigma / np.sqrt(-2 * np.log(1 + beta))
    return {'mu': mu, 'theta': theta, 'sigma_ou': sigma_ou}

def calculate_optimal_boundaries(theta, sigma_ou, risk_aversion, transaction_cost):
    if theta <= 0 or sigma_ou <= 0: return None
    profit_target = (sigma_ou / np.sqrt(2 * theta)) * risk_aversion
    entry_boundary = profit_target + transaction_cost * sigma_ou
    exit_boundary = transaction_cost * sigma_ou
    return {'entry': entry_boundary, 'exit': exit_boundary}

# ==============================================================================
# SECTION 2: THE TRADING STRATEGY CLASS
# ==============================================================================
class OptimalControlStrategy:
    def __init__(self, params: dict):
        self.params = params
        self.pair_position_state = {str(p): 0 for p in IDEAL_PAIRS}

    def getMyPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        final_positions = np.zeros(50)
        current_day = prcSoFar.shape[1]
        
        lookback = self.params['OU_LOOKBACK']
        trading_start_day = self.params['TRADING_START_DAY']
        risk_aversion = self.params['RISK_AVERSION']
        
        if current_day < trading_start_day or current_day < lookback:
            return final_positions

        for pair in IDEAL_PAIRS:
            asset1_idx, asset2_idx = pair
            y = prcSoFar[asset1_idx, :]
            x = prcSoFar[asset2_idx, :]
            model = sm.OLS(y, sm.add_constant(x)).fit()
            hedge_ratio = model.params[1]

            ou_window = prcSoFar[:, -lookback:]
            spread_series = ou_window[asset1_idx, :] - hedge_ratio * ou_window[asset2_idx, :]
            
            try:
                ou_params = estimate_ou_parameters(spread_series)
                if ou_params is None: continue
                
                boundaries = calculate_optimal_boundaries(
                    ou_params['theta'], ou_params['sigma_ou'], risk_aversion, 0.05
                )
                if boundaries is None: continue
                
                upper_entry = ou_params['mu'] + boundaries['entry']
                lower_entry = ou_params['mu'] - boundaries['entry']
                upper_exit = ou_params['mu'] + boundaries['exit']
                lower_exit = ou_params['mu'] - boundaries['exit']
                
                current_spread = spread_series[-1]
                pair_key = str(pair)
                current_pos = self.pair_position_state.get(pair_key, 0)
                new_pos = current_pos

                if current_pos == 0:
                    if current_spread < lower_entry: new_pos = 1
                    elif current_spread > upper_entry: new_pos = -1
                elif current_pos == 1 and current_spread >= lower_exit:
                    new_pos = 0
                elif current_pos == -1 and current_spread <= upper_exit:
                    new_pos = 0
                
                self.pair_position_state[pair_key] = new_pos

                if new_pos != 0:
                    pos1 = new_pos * (self.params['POSITION_SIZE'] / prcSoFar[asset1_idx, -1])
                    final_positions[asset1_idx] += pos1
                    final_positions[asset2_idx] += - (pos1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]
            except Exception:
                continue
                
        return final_positions.astype(int)

# ==============================================================================
# SECTION 3: PARALLEL BACKTESTING WORKER
# ==============================================================================
def run_backtest_worker(params, prices_path):
    try:
        strategy = OptimalControlStrategy(params=params)
        bt_params = Params(strategy_function=strategy.getMyPosition, start_day=1, end_day=750, prices_filepath=prices_path)
        backtester = Backtester(bt_params)
        results = backtester.run(start_day=bt_params.start_day, end_day=bt_params.end_day)
        
        start_index = params['TRADING_START_DAY'] - 1
        target_pnl = results['daily_pnl'][start_index:]
        
        if len(target_pnl) > 0 and np.std(target_pnl) > 0:
            score = np.mean(target_pnl) - 0.1 * np.std(target_pnl)
        else:
            score = -np.inf
            
        return {'params': params, 'score': score}
    except Exception:
        return {'params': params, 'score': -np.inf}

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    PRICES_PATH = os.path.join(parent_dir, "prices.txt")

    # Final grid search for the optimal control model's hyperparameters
    grid = {
        "TRADING_START_DAY": (50,651,50),
        "OU_LOOKBACK": (50, 601, 25),
        "RISK_AVERSION": (0.05, 1.01, 0.05),
        "POSITION_SIZE": [10000],
    }
    
    processed_grid = {k: np.arange(*v) if isinstance(v, tuple) else v for k, v in grid.items()}
    param_combinations = list(product(*processed_grid.values()))
    param_names = list(processed_grid.keys())
    
    print(f"--- Starting Optimal Control Grid Search: {len(param_combinations)} combinations ---")
    results_list = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_backtest_worker, dict(zip(param_names, comb)), PRICES_PATH) for comb in param_combinations]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results_list.append(result)
            print(f"Completed {i+1}/{len(param_combinations)} -> Score: {result['score']:.2f} | Params: {result['params']}")

    if results_list:
        results_df = pd.DataFrame(results_list)
        params_df = pd.DataFrame(results_df['params'].tolist())
        scores_df = results_df[['score']]
        full_results_df = pd.concat([params_df, scores_df], axis=1)
        best_results_df = full_results_df.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        filename = "final_optimal_control_results.csv"
        best_results_df.to_csv(filename, index=False)
        
        print("\n\n--- Grid Search Complete ---")
        print(f"\nSUCCESS: All results saved to '{filename}'")
        print("\n--- Top 5 Best Overall Parameter Sets (Optimal Control Strategy) ---")
        print(best_results_df.head(5).to_string())
    else:
        print("No successful backtest results were obtained.")