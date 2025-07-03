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
# --- MODIFICATION: Swapped Kalman Filter for Lasso ---
from sklearn.linear_model import Ridge, BayesianRidge, Lasso

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
# SECTION 2: HEDGE RATIO ESTIMATORS (MODIFIED)
# ==============================================================================
def get_hedge_ratio(y, x, estimator: str):
    """A centralized function to calculate hedge ratio using different models."""
    x_with_const = sm.add_constant(x)
    
    if estimator == 'ols':
        return sm.OLS(y, x_with_const).fit().params[1]
    
    elif estimator == 'ridge':
        model = Ridge(alpha=1.0)
        model.fit(x, y.ravel())
        return model.coef_[0]
        
    elif estimator == 'bayesian':
        model = BayesianRidge()
        model.fit(x, y.ravel())
        return model.coef_[0]

    # --- MODIFICATION: Replaced 'kalman' with 'lasso' ---
    elif estimator == 'lasso':
        model = Lasso(alpha=0.1) # Lasso requires a small alpha to not zero-out the coefficient
        model.fit(x, y.ravel())
        return model.coef_[0]
        
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

# ==============================================================================
# SECTION 3: INITIAL DATA ANALYSIS
# ==============================================================================
IDEAL_PAIRS = [
    (2, 6), (8, 34), (11, 23), (12, 49), (18, 29), (20, 22), 
    (26, 45), (33, 35), (37, 41), (40, 45), (47, 49), (48, 49)
]

def create_initial_static_ratios(price_df: pd.DataFrame, static_calculation_period: int, estimator: str):
    """Calculates static hedge ratios using a specified estimator."""
    print(f"--- Calculating static ratios using estimator: {estimator} ---")
    static_period_df = price_df.iloc[:static_calculation_period]
    initial_ratios = []
    for pair in IDEAL_PAIRS:
        y = static_period_df.iloc[:, pair[0]].values
        x = static_period_df.iloc[:, pair[1]].values.reshape(-1, 1)
        hedge_ratio = get_hedge_ratio(y, x, estimator)
        initial_ratios.append({'pair': pair, 'static_hedge_ratio': hedge_ratio})
    return initial_ratios

# ==============================================================================
# SECTION 4: THE TRADING STRATEGY CLASS
# ==============================================================================
class PairsTradingStrategy:
    def __init__(self, params: dict, static_ratios: list):
        self.params = params
        self.static_ratios_map = {str(info['pair']): info['static_hedge_ratio'] for info in static_ratios}
        self.pair_position_state = {pair_key: 0 for pair_key in self.static_ratios_map.keys()}

    def getMyPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        final_positions = np.zeros(50)
        current_day = prcSoFar.shape[1]
        
        is_static_period = current_day < self.params['ADAPTIVE_START_DAY']
        active_params = self.params['STATIC_PARAMS'] if is_static_period else self.params['ADAPTIVE_PARAMS']
        
        zscore_lookback = active_params['ZSCORE_LOOKBACK']
        entry_threshold = active_params['ENTRY_THRESHOLD']
        stop_loss_threshold = active_params['STOP_LOSS_THRESHOLD']

        if current_day < zscore_lookback: return final_positions

        for pair_key, static_hr in self.static_ratios_map.items():
            pair = eval(pair_key)
            asset1_idx, asset2_idx = pair

            if is_static_period:
                hedge_ratio = static_hr
            else:
                y = prcSoFar[asset1_idx, :].reshape(-1, 1)
                x = prcSoFar[asset2_idx, :].reshape(-1, 1)
                hedge_ratio = get_hedge_ratio(y, x, self.params['HEDGE_ESTIMATOR'])

            zscore_window = prcSoFar[:, -zscore_lookback:]
            spread = zscore_window[asset1_idx, :] - hedge_ratio * zscore_window[asset2_idx, :]
            if np.std(spread) < 1e-6: continue
            
            current_spread = prcSoFar[asset1_idx, -1] - hedge_ratio * prcSoFar[asset2_idx, -1]
            z_score = (current_spread - np.mean(spread)) / np.std(spread)

            current_pos = self.pair_position_state.get(pair_key, 0)
            new_pos = current_pos

            if current_pos == 0:
                if z_score < -entry_threshold: new_pos = 1
                elif z_score > entry_threshold: new_pos = -1
            elif current_pos == 1:
                if z_score >= 0 or z_score < -stop_loss_threshold: new_pos = 0
            elif current_pos == -1:
                if z_score <= 0 or z_score > stop_loss_threshold: new_pos = 0
            
            self.pair_position_state[pair_key] = new_pos

            if new_pos != 0:
                pos1 = new_pos * (self.params['POSITION_SIZE'] / prcSoFar[asset1_idx, -1])
                final_positions[asset1_idx] += pos1
                final_positions[asset2_idx] += - (pos1 * hedge_ratio * prcSoFar[asset1_idx, -1]) / prcSoFar[asset2_idx, -1]
                
        return final_positions.astype(int)

# ==============================================================================
# SECTION 5: PARALLEL BACKTESTING WORKER
# ==============================================================================
def run_backtest_worker(params, prices_path):
    try:
        prices_df = pd.read_csv(prices_path, header=None, sep=r'\s+')
        static_ratios = create_initial_static_ratios(prices_df, 500, params['HEDGE_ESTIMATOR'])
        
        strategy = PairsTradingStrategy(params=params, static_ratios=static_ratios)
        bt_params = Params(strategy_function=strategy.getMyPosition, start_day=1, end_day=750, prices_filepath=prices_path)
        backtester = Backtester(bt_params)
        results = backtester.run(start_day=bt_params.start_day, end_day=bt_params.end_day)
        
        optimization_target = params['OPTIMIZATION_TARGET']
        pnl = results['daily_pnl']
        
        if optimization_target == 'STATIC':
            start_idx, end_idx = 151, 500
        else: # 'ADAPTIVE'
            start_idx, end_idx = 501, 750
        
        target_pnl = pnl[start_idx:end_idx]
        
        if len(target_pnl) > 0 and np.std(target_pnl) > 0:
            score = np.mean(target_pnl) - 0.1 * np.std(target_pnl)
        else:
            score = -np.inf
            
        return {'params': params, 'score': score}
    except Exception:
        return {'params': params, 'score': -np.inf}

# ==============================================================================
# SECTION 6: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # +++ CHOOSE YOUR CAMPAIGN: 'STATIC' or 'ADAPTIVE' +++
    OPTIMIZATION_TARGET = 'STATIC' 
    
    PRICES_PATH = os.path.join(parent_dir, "prices.txt")

    # --- UPDATED GRID DEFINITION ---
    grid = {
        "HEDGE_ESTIMATOR": ['ols', 'ridge', 'bayesian', 'lasso'], # Replaced 'kalman' with 'lasso'
        "ZSCORE_LOOKBACK": (30, 151, 10),
        "ENTRY_THRESHOLD": (0.25, 2.51, 0.25),
        "STOP_LOSS_THRESHOLD": (1.0, 4.01, 0.5),
        "POSITION_SIZE": [10000],
        "ADAPTIVE_START_DAY": [501],
    }
    
    processed_grid = {k: np.arange(*v).tolist() if isinstance(v, tuple) else v for k, v in grid.items()}
    param_combinations = list(product(*processed_grid.values()))
    param_names = list(processed_grid.keys())
    
    final_params_list = []
    for comb in param_combinations:
        p_dict = dict(zip(param_names, comb))
        final_params_list.append({
            "HEDGE_ESTIMATOR": p_dict['HEDGE_ESTIMATOR'],
            "ADAPTIVE_START_DAY": p_dict['ADAPTIVE_START_DAY'],
            "POSITION_SIZE": p_dict['POSITION_SIZE'],
            "OPTIMIZATION_TARGET": OPTIMIZATION_TARGET,
            "STATIC_PARAMS": p_dict,
            "ADAPTIVE_PARAMS": p_dict,
        })
    
    print(f"--- Starting Mega Grid Search ---")
    print(f"--- Optimizing for: {OPTIMIZATION_TARGET} PERIOD ---")
    print(f"--- Total combinations: {len(final_params_list)} ---")
    
    results_list = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_backtest_worker, params, PRICES_PATH) for params in final_params_list]
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results_list.append(result)
            print(f"  ({i+1}/{len(final_params_list)}) Score: {result['score']:.2f} | Estimator: {result['params']['HEDGE_ESTIMATOR']}, ZL: {result['params']['STATIC_PARAMS']['ZSCORE_LOOKBACK']}")

    # --- ANALYZE RESULTS ---
    if results_list:
        results_df = pd.DataFrame(results_list)
        flat_params = pd.json_normalize(results_df['params'])
        scores_df = results_df[['score']]
        full_results_df = pd.concat([flat_params, scores_df], axis=1)
        
        best_results_df = full_results_df.sort_values(by='score', ascending=False).reset_index(drop=True)
        
        filename = f"mega_grid_search_results_{OPTIMIZATION_TARGET.lower()}.csv"
        best_results_df.to_csv(filename, index=False)
        
        print("\n\n--- Grid Search Complete ---")
        print(f"\nSUCCESS: All results saved to '{filename}'")
        print(f"\n--- Top 5 Best Parameter Sets for the {OPTIMIZATION_TARGET} Period ---")
        print(best_results_df.head(5).to_string())
    else:
        print("No successful backtest results were obtained.")