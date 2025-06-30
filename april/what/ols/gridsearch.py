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
# SECTION 2: INITIAL DATA ANALYSIS
# ==============================================================================
IDEAL_PAIRS = [
    (2, 6), (8, 34), (11, 23), (12, 49), (18, 29), (20, 22), 
    (26, 45), (33, 35), (37, 41), (40, 45), (47, 49), (48, 49)
]

def create_initial_static_ratios(price_df: pd.DataFrame, static_calculation_period: int):
    """Calculates static hedge ratios using OLS regression."""
    print(f"--- Calculating initial static hedge ratios using first {static_calculation_period} days (OLS) ---")
    static_period_df = price_df.iloc[:static_calculation_period]
    initial_ratios = []
    for pair in IDEAL_PAIRS:
        # For statsmodels, y can be 1D, x needs a constant
        y = static_period_df.iloc[:, pair[0]].values
        x = static_period_df.iloc[:, pair[1]].values
        model = sm.OLS(y, sm.add_constant(x)).fit()
        initial_ratios.append({'pair': pair, 'static_hedge_ratio': model.params[1]})
    print("--- Static ratio calculation complete ---")
    return initial_ratios

# ==============================================================================
# SECTION 3: THE FINAL TRADING STRATEGY CLASS
# ==============================================================================
class PairsTradingStrategy:
    """
    Final strategy class that accepts distinct parameters for static and adaptive periods.
    """
    def __init__(self, params: dict, static_ratios: list):
        self.params = params
        self.static_ratios_map = {str(info['pair']): info['static_hedge_ratio'] for info in static_ratios}
        self.pair_position_state = {pair_key: 0 for pair_key in self.static_ratios_map.keys()}

    def getMyPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        final_positions = np.zeros(50)
        current_day = prcSoFar.shape[1]
        
        active_params = self.params['STATIC_PARAMS'] if current_day < self.params['ADAPTIVE_START_DAY'] else self.params['ADAPTIVE_PARAMS']
        
        if current_day < active_params['ZSCORE_LOOKBACK']: return final_positions

        for pair_key, static_hr in self.static_ratios_map.items():
            pair = eval(pair_key)
            asset1_idx, asset2_idx = pair

            if current_day < self.params['ADAPTIVE_START_DAY']:
                hedge_ratio = static_hr
            else:
                # Use expanding window with OLS
                y = prcSoFar[asset1_idx, :]
                x = prcSoFar[asset2_idx, :]
                model = sm.OLS(y, sm.add_constant(x)).fit()
                hedge_ratio = model.params[1]

            zscore_window = prcSoFar[:, -active_params['ZSCORE_LOOKBACK']:]
            spread = zscore_window[asset1_idx, :] - hedge_ratio * zscore_window[asset2_idx, :]
            if np.std(spread) < 1e-6: continue
            
            current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
            z_score = (current_spread - np.mean(spread)) / np.std(spread)

            current_pos = self.pair_position_state.get(pair_key, 0)
            new_pos = current_pos

            if current_pos == 0:
                if z_score < -active_params['ENTRY_THRESHOLD']: new_pos = 1
                elif z_score > active_params['ENTRY_THRESHOLD']: new_pos = -1
            elif current_pos == 1:
                if z_score >= 0 or z_score < -active_params['STOP_LOSS_THRESHOLD']: new_pos = 0
            elif current_pos == -1:
                if z_score <= 0 or z_score > active_params['STOP_LOSS_THRESHOLD']: new_pos = 0
            
            self.pair_position_state[pair_key] = new_pos

            if new_pos != 0:
                pos1 = new_pos * (self.params['POSITION_SIZE'] / prcSoFar[asset1_idx, -1])
                final_positions[asset1_idx] += pos1
                final_positions[asset2_idx] += - (pos1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]
                
        return final_positions.astype(int)

# ==============================================================================
# SECTION 4: PARALLEL BACKTESTING WORKER
# ==============================================================================
def run_backtest_worker(params, static_ratios, prices_path):
    try:
        strategy = PairsTradingStrategy(params=params, static_ratios=static_ratios)
        bt_params = Params(strategy_function=strategy.getMyPosition, start_day=1, end_day=750, prices_filepath=prices_path)
        backtester = Backtester(bt_params)
        results = backtester.run(start_day=bt_params.start_day, end_day=bt_params.end_day)
        
        # Score on the entire active trading period for a holistic result
        start_idx = 151 # Max lookback from static grid is 150
        target_pnl = results['daily_pnl'][start_idx:]
        
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
    
    # --- SETUP ---
    PRICES_PATH = os.path.join(parent_dir, "prices.txt")
    prices_df = pd.read_csv(PRICES_PATH, header=None, sep=r'\s+')
    initial_static_ratios = create_initial_static_ratios(prices_df, static_calculation_period=500)

    # --- Step 1: Define the grid for ALL parameters ---
    static_grid = {
        "ZSCORE_LOOKBACK": np.arange(60, 131, 30),
        "ENTRY_THRESHOLD": np.arange(1.0, 3.00, 0.5),
        "STOP_LOSS_THRESHOLD": [2.5, 3.0, 3.5, 4.0],
    }
    adaptive_grid = {
        "ZSCORE_LOOKBACK": np.arange(60, 131, 30),
        "ENTRY_THRESHOLD": np.arange(1.0, 2.00, 0.5),
        "STOP_LOSS_THRESHOLD": [2.5, 3.0, 3.5, 4.0],
    }
    adaptive_start_day_grid = [301, 401, 501, 601]

    # --- Step 2: Create all combinations ---
    static_combos = [dict(zip(static_grid.keys(), v)) for v in product(*static_grid.values())]
    adaptive_combos = [dict(zip(adaptive_grid.keys(), v)) for v in product(*adaptive_grid.values())]
    
    final_params_list = []
    for start_day in adaptive_start_day_grid:
        for static_p in static_combos:
            for adaptive_p in adaptive_combos:
                final_params_list.append({
                    "ADAPTIVE_START_DAY": start_day,
                    "POSITION_SIZE": 10000,
                    "STATIC_PARAMS": static_p,
                    "ADAPTIVE_PARAMS": adaptive_p
                })

    print(f"--- Starting Ultimate Grid Search: {len(final_params_list)} combinations ---")
    results_list = []

    # --- Step 3: Run grid search in parallel ---
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_backtest_worker, params, initial_static_ratios, PRICES_PATH)
                   for params in final_params_list]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results_list.append(result)
            print(f"Completed {i+1}/{len(final_params_list)} -> Score: {result['score']:.2f}")

    # --- Step 4: Find and print the best results ---
    if results_list:
        results_df = pd.DataFrame(results_list)
        flat_params = pd.json_normalize(results_df['params'])
        scores_df = results_df[['score']]
        full_results_df = pd.concat([flat_params, scores_df], axis=1)
        best_results_df = full_results_df.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        filename = "ultimate_grid_search_results_OLS.csv"
        best_results_df.to_csv(filename, index=False)
        
        print("\n\n--- Grid Search Complete ---")
        print(f"\nSUCCESS: All results saved to '{filename}'")
        print("\n--- Top 5 Best Overall Parameter Sets ---")
        print(best_results_df.head(5).to_string())
    else:
        print("No successful backtest results were obtained.")