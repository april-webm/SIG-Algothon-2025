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
from statsmodels.tsa.stattools import coint

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
    print(f"--- Calculating initial static hedge ratios using first {static_calculation_period} days ---")
    static_period_df = price_df.iloc[:static_calculation_period]
    initial_ratios = []
    for pair in IDEAL_PAIRS:
        y = static_period_df.iloc[:, pair[0]]
        x = static_period_df.iloc[:, pair[1]]
        model = sm.OLS(y, sm.add_constant(x)).fit()
        hedge_ratio = model.params.iloc[1]
        initial_ratios.append({'pair': pair, 'static_hedge_ratio': hedge_ratio})
    print("--- Static ratio calculation complete ---")
    return initial_ratios

# ==============================================================================
# SECTION 3: THE TRADING STRATEGY CLASS
# ==============================================================================
class PairsTradingStrategy:
    def __init__(self, params: dict, static_ratios: list):
        self.params = params
        self.static_ratios = static_ratios
        self.pair_position_state = {str(info['pair']): 0 for info in static_ratios}

    def getMyPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        final_positions = np.zeros(50)
        current_day = prcSoFar.shape[1]

        zscore_lookback = self.params['ZSCORE_LOOKBACK']
        adaptive_start_day = self.params['ADAPTIVE_START_DAY']
        entry_threshold = self.params['ENTRY_THRESHOLD']
        stop_loss_threshold = self.params['STOP_LOSS_THRESHOLD']
        position_size = self.params['POSITION_SIZE']

        if current_day < zscore_lookback:
            return final_positions

        for pair_info in self.static_ratios:
            pair = pair_info['pair']
            asset1_idx, asset2_idx = pair

            if current_day < adaptive_start_day:
                hedge_ratio = pair_info['static_hedge_ratio']
            else:
                regression_window = prcSoFar[:, :current_day]
                y, x = regression_window[asset1_idx, :], regression_window[asset2_idx, :]
                model = sm.OLS(y, sm.add_constant(x)).fit()
                hedge_ratio = model.params[1]

            zscore_window = prcSoFar[:, -zscore_lookback:]
            spread_series = zscore_window[asset1_idx, :] - hedge_ratio * zscore_window[asset2_idx, :]
            if np.std(spread_series) < 1e-6: continue
            
            current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
            z_score = (current_spread - np.mean(spread_series)) / np.std(spread_series)

            pair_key = str(pair)
            current_pos_state = self.pair_position_state.get(pair_key, 0)
            new_pos_state = current_pos_state

            if current_pos_state == 0:
                if z_score < -entry_threshold: new_pos_state = 1
                elif z_score > entry_threshold: new_pos_state = -1
            elif current_pos_state == 1:
                if z_score >= 0 or z_score < -stop_loss_threshold: new_pos_state = 0
            elif current_pos_state == -1:
                if z_score <= 0 or z_score > stop_loss_threshold: new_pos_state = 0
            
            self.pair_position_state[pair_key] = new_pos_state

            if new_pos_state != 0:
                pos_asset1 = new_pos_state * (position_size / prcSoFar[asset1_idx, -1])
                final_positions[asset1_idx] += pos_asset1
                pos_asset2 = - (pos_asset1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]
                final_positions[asset2_idx] += pos_asset2
                
        return final_positions.astype(int)

# ==============================================================================
# SECTION 4: PARALLEL BACKTESTING WORKER FUNCTION
# ==============================================================================
def run_backtest_worker(params, static_ratios, prices_path):
    """
    This function is what each parallel process will execute.
    It takes one set of parameters, runs a full backtest, and returns the result.
    """
    try:
        strategy = PairsTradingStrategy(params=params, static_ratios=static_ratios)
        bt_params = Params(
            strategy_function=strategy.getMyPosition,
            start_day=1,
            end_day=750,
            prices_filepath=prices_path
        )
        backtester = Backtester(bt_params)
        results = backtester.run(start_day=bt_params.start_day, end_day=bt_params.end_day)
        
        start_index_for_test = 500
        test_period_pnl = results['daily_pnl'][start_index_for_test:]
        
        if len(test_period_pnl) > 0 and np.std(test_period_pnl) > 0:
            score = np.mean(test_period_pnl) - 0.1 * np.std(test_period_pnl)
        else:
            score = -np.inf
            
        return {'params': params, 'score': score}
    except Exception as e:
        # print(f"Error with params {params}: {e}") # Uncomment for deep debugging
        return {'params': params, 'score': -np.inf}

# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    # freeze_support() is necessary for creating frozen executables
    multiprocessing.freeze_support()
    
    # --- SETUP ---
    try:
        SRC_DIR = os.path.dirname(os.path.abspath(__file__))
        PARENT_DIR = os.path.dirname(SRC_DIR)
    except NameError:
        SRC_DIR, PARENT_DIR = ".", ".."

    PRICES_PATH = os.path.join(PARENT_DIR, "prices.txt")
    prices_df = pd.read_csv(PRICES_PATH, header=None, sep=r'\s+')
    
    # --- Step 1: Run the one-time analysis ---
    initial_static_ratios = create_initial_static_ratios(prices_df, static_calculation_period=500)

    # --- Step 2: Define the grid ---
    grid = {
        "ADAPTIVE_START_DAY": [501],
        "ZSCORE_LOOKBACK": (20, 151, 10),
        "ENTRY_THRESHOLD": (0.1, 2.51, 0.1),
        "STOP_LOSS_THRESHOLD": (0.25, 4.01, 0.25),
        "POSITION_SIZE": [10000]
    }
    
    processed_grid = {}
    for key, value in grid.items():
        if isinstance(value, tuple) and len(value) == 3:
            processed_grid[key] = np.arange(value[0], value[1], value[2]).tolist()
        else:
            processed_grid[key] = value

    param_combinations = list(product(*processed_grid.values()))
    param_names = list(processed_grid.keys())
    
    print(f"--- Starting Parallel Grid Search: {len(param_combinations)} combinations ---")
    print(f"--- Optimizing for performance ONLY in days 501-750 ---")
    results_list = []

    # --- Step 3: Run grid search in parallel ---
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Create a list of jobs to run
        futures = [executor.submit(run_backtest_worker, dict(zip(param_names, comb)), initial_static_ratios, PRICES_PATH)
                   for comb in param_combinations]
        
        # Process results as they are completed
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results_list.append(result)
            print(f"Completed {i+1}/{len(param_combinations)} -> Score: {result['score']:.2f} | Params: {result['params']}")

    # --- Step 4: Find and print the best results ---
    if results_list:
        results_df = pd.DataFrame(results_list)
        # We need to handle the 'params' column which is a dictionary
        params_df = pd.DataFrame(results_df['params'].tolist())
        scores_df = results_df[['score']]
        full_results_df = pd.concat([params_df, scores_df], axis=1)

        best_results_df = full_results_df.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        results_filename = "grid_search_results.csv"
        best_results_df.to_csv(results_filename, index=False)
        
        print("\n\n--- Grid Search Complete ---")
        print(f"\nSUCCESS: All {len(best_results_df)} results have been saved to '{results_filename}'")
        print("\n--- Top 5 Best Parameter Sets for the Adaptive Period (Days 501-750) ---")
        print(best_results_df.head(5).to_string())
    else:
        print("No successful backtest results were obtained.")