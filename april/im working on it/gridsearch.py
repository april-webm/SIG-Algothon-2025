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
import warnings

# Suppress warnings from the statsmodels OLS, which can be noisy
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
# SECTION 2: HELPER FUNCTIONS (OU ESTIMATOR)
# ==============================================================================
IDEAL_PAIRS = [
    (2, 6), (8, 34), (11, 23), (12, 49), (18, 29), (20, 22), 
    (26, 45), (33, 35), (37, 41), (40, 45), (47, 49), (48, 49)
]

def estimate_ou_mu(spread_series: np.ndarray) -> float:
    """
    Estimates the long-term mean (mu) of the Ornstein-Uhlenbeck process.
    """
    spread_lagged = spread_series[:-1]
    spread_diff = spread_series[1:] - spread_lagged
    
    x_reg = sm.add_constant(spread_lagged)
    model = sm.OLS(spread_diff, x_reg).fit()
    alpha, beta = model.params
    
    # mu = alpha / -beta
    # If beta (and thus theta) is close to zero, the mean is unstable; use simple mean
    if abs(beta) < 1e-6:
        return np.mean(spread_series)
    else:
        return alpha / -beta

# ==============================================================================
# SECTION 3: THE TRADING STRATEGY CLASS (UPGRADED TO OU)
# ==============================================================================
class OU_Strategy:
    """
    Uses an expanding window for hedge ratios and an Ornstein-Uhlenbeck model
    for signal generation.
    """
    def __init__(self, params: dict):
        self.params = params
        self.pair_position_state = {str(p): 0 for p in IDEAL_PAIRS}

    def getMyPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        final_positions = np.zeros(50)
        current_day = prcSoFar.shape[1]
        
        lookback = self.params['OU_LOOKBACK']
        trading_start_day = self.params['TRADING_START_DAY']
        entry_threshold = self.params['ENTRY_THRESHOLD']
        stop_loss_threshold = self.params['STOP_LOSS_THRESHOLD']
        position_size = self.params['POSITION_SIZE']

        if current_day < trading_start_day or current_day < lookback:
            return final_positions

        for pair in IDEAL_PAIRS:
            asset1_idx, asset2_idx = pair

            # Hedge Ratio: Expanding window OLS
            y = prcSoFar[asset1_idx, :]
            x = prcSoFar[asset2_idx, :]
            model = sm.OLS(y, sm.add_constant(x)).fit()
            hedge_ratio = model.params[1]

            # --- NEW OU-BASED SIGNAL GENERATION ---
            ou_window = prcSoFar[:, -lookback:]
            spread_series = ou_window[asset1_idx, :] - hedge_ratio * ou_window[asset2_idx, :]
            
            spread_std = np.std(spread_series)
            if spread_std < 1e-6: continue

            try:
                # Estimate the theoretical mean using OU
                mu = estimate_ou_mu(spread_series)
                
                # Create a normalized signal score based on deviation from the OU mean
                current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
                signal_score = (current_spread - mu) / spread_std

                # --- Stateful Trading Logic ---
                pair_key = str(pair)
                current_pos = self.pair_position_state.get(pair_key, 0)
                new_pos = current_pos

                if current_pos == 0:
                    if signal_score < -entry_threshold: new_pos = 1
                    elif signal_score > entry_threshold: new_pos = -1
                elif current_pos == 1:
                    if signal_score >= 0 or signal_score < -stop_loss_threshold: new_pos = 0
                elif current_pos == -1:
                    if signal_score <= 0 or signal_score > stop_loss_threshold: new_pos = 0
                
                self.pair_position_state[pair_key] = new_pos

                if new_pos != 0:
                    pos1 = new_pos * (position_size / prcSoFar[asset1_idx, -1])
                    final_positions[asset1_idx] += pos1
                    final_positions[asset2_idx] += - (pos1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]
            
            except Exception:
                # If OU estimation fails, do not trade the pair for that day
                continue
                
        return final_positions.astype(int)

# ==============================================================================
# SECTION 4: PARALLEL BACKTESTING WORKER
# ==============================================================================
def run_backtest_worker(params, prices_path):
    try:
        strategy = OU_Strategy(params=params)
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
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    PRICES_PATH = os.path.join(parent_dir, "prices.txt")

    # Grid for the OU-based strategy
    grid = {
        "TRADING_START_DAY": (50, 601, 50), # Start after a longer warm-up for OU
        "OU_LOOKBACK": (50, 251, 25), # Lookback for OU parameter estimation
        "ENTRY_THRESHOLD": (0.25, 2.51, 0.25),
        "STOP_LOSS_THRESHOLD": (2.5, 4.01, 0.5),
        "POSITION_SIZE": [10000],
    }
    
    processed_grid = {k: np.arange(*v).tolist() if isinstance(v, tuple) else v for k, v in grid.items()}
    param_combinations = list(product(*processed_grid.values()))
    param_names = list(processed_grid.keys())
    
    print(f"--- Starting OU Strategy Grid Search: {len(param_combinations)} combinations ---")
    results_list = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_backtest_worker, dict(zip(param_names, comb)), PRICES_PATH)
                   for comb in param_combinations]
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
        
        filename = "final_OU_strategy_results.csv"
        best_results_df.to_csv(filename, index=False)
        
        print("\n\n--- Grid Search Complete ---")
        print(f"\nSUCCESS: All results saved to '{filename}'")
        print("\n--- Top 5 Best Overall Parameter Sets (OU Strategy) ---")
        print(best_results_df.head(5).to_string())
    else:
        print("No successful backtest results were obtained.")