# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================
import os
import sys
import numpy as np
import pandas as pd
import warnings
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- ROBUST PATH SETUP ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
except NameError:
    parent_dir = ".."
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

from backtester import Backtester, Params

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 2: THE ADAPTIVE MEAN-REVERSION STRATEGY CLASS
# ==============================================================================

class AdaptiveEMA:
    """
    Encapsulates an adaptive EMA strategy designed to trade mean-reversion,
    filtered by the Hurst exponent.
    """
    def __init__(self, params: dict):
        self.params = params
        self.last_optimization_day = 0
        self.optimal_ema_params = {}
        self.max_position_size = 10000
        self.ema_grid = {
            "SHORT_WINDOWS": [5, 10, 15, 20, 40, 60, 80],
            "LONG_WINDOWS": [25, 30, 40, 50, 60, 80, 100, 120],
        }

    def _calculate_ema(self, prices: np.ndarray, window: int) -> np.ndarray:
        return pd.Series(prices).ewm(span=window, adjust=False).mean().to_numpy()

    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """
        Calculates the Hurst exponent on log prices in a numerically stable way.
        """
        # --- DEFINITIVE FIX: Dynamically set max_lag and handle edge cases ---
        max_lag = max(2, len(prices) // 2) # Ensure max_lag is reasonable for the series length
        lags = range(2, max_lag)

        if len(prices) < max_lag: return 0.5

        log_prices = np.log(prices)

        tau = []
        valid_lags = []
        for lag in lags:
            if len(log_prices) > lag:
                std_dev = np.std(np.subtract(log_prices[lag:], log_prices[:-lag]))
                if std_dev > 1e-9:
                    tau.append(std_dev)
                    valid_lags.append(lag)

        if len(tau) < 2: return 0.5

        log_tau = np.log(tau)
        log_lags = np.log(valid_lags)

        if not (np.all(np.isfinite(log_lags)) and np.all(np.isfinite(log_tau))): return 0.5

        poly = np.polyfit(log_lags, log_tau, 1)
        return poly[0]

    def _calculate_volatility_adjusted_size(self, asset_prices: np.ndarray) -> float:
        """Calculates position size based on inverse volatility."""
        vol_lookback = self.params["VOLATILITY_LOOKBACK"]
        if len(asset_prices) < vol_lookback: return 0

        returns = np.diff(asset_prices[-vol_lookback:]) / (asset_prices[-vol_lookback:-1] + 1e-9)
        realized_vol = np.std(returns) * np.sqrt(252)

        if realized_vol < 1e-9: return 0

        scaling_factor = self.params["TARGET_ANNUAL_VOL"] / realized_vol
        return min(scaling_factor * self.max_position_size, self.max_position_size)

    def _find_optimal_emas(self, prcSoFar: np.ndarray):
        """Finds the best EMA parameters for mean-reverting assets."""
        optimization_window = prcSoFar[:, -self.params["OPTIMIZATION_LOOKBACK"]:]
        new_optimal_params = {}


        hurst_values_log = []
        for asset_idx in range(prcSoFar.shape[0]):
            asset_prices = optimization_window[asset_idx]

            hurst = self._calculate_hurst_exponent(asset_prices)
            hurst_values_log.append(f"Asset {asset_idx}: H={hurst:.2f}")

            if hurst < self.params["HURST_THRESHOLD"]:
                continue

            best_score, best_params = -np.inf, None
            for short_w, long_w in product(self.ema_grid["SHORT_WINDOWS"], self.ema_grid["LONG_WINDOWS"]):
                if short_w >= long_w: continue

                short_ema = self._calculate_ema(asset_prices, short_w)
                long_ema = self._calculate_ema(asset_prices, long_w)

                signals = np.where(short_ema > long_ema, -1, 1)
                positions = signals[:-1]
                price_changes = np.diff(asset_prices)

                pnl = positions * price_changes

                if len(pnl) > 1 and np.std(pnl) > 0:
                    score = np.mean(pnl) - 0.1 * np.std(pnl)
                    if score > best_score:
                        best_score = score
                        best_params = {'short': short_w, 'long': long_w}

            if best_params:
                new_optimal_params[asset_idx] = best_params
        self.optimal_ema_params = new_optimal_params

    def get_my_positions(self, prcSoFar: np.ndarray) -> np.ndarray:
        """The main strategy logic, now encapsulated as a class method."""
        current_day = prcSoFar.shape[1]

        trading_start_day = self.params["OPTIMIZATION_LOOKBACK"]
        if current_day < trading_start_day:
            return np.zeros(50)

        if self.last_optimization_day == 0 or current_day >= self.last_optimization_day + self.params["RECALCULATION_FREQUENCY"]:
            self._find_optimal_emas(prcSoFar)
            self.last_optimization_day = current_day

        if not self.optimal_ema_params:
            return np.zeros(50)

        final_positions = np.zeros(50)
        for asset_idx, params in self.optimal_ema_params.items():
            asset_prices = prcSoFar[asset_idx]
            short_ema = self._calculate_ema(asset_prices, params['short'])
            long_ema = self._calculate_ema(asset_prices, params['long'])

            current_signal = 0
            if not np.isnan(short_ema[-1]) and not np.isnan(long_ema[-1]):
                if short_ema[-1] > long_ema[-1]: current_signal = -1
                elif short_ema[-1] < long_ema[-1]: current_signal = 1

            if current_signal != 0:
                adjusted_size = self._calculate_volatility_adjusted_size(asset_prices)
                final_positions[asset_idx] = current_signal * (adjusted_size / (asset_prices[-1] + 1e-9))
            else:
                final_positions[asset_idx] = 0

        return final_positions.astype(int)

# ==============================================================================
# SECTION 3: PARALLEL BACKTESTING WORKER
# ==============================================================================
def run_backtest_worker(params, prices_path):
    """This function is what each parallel process will execute."""
    try:
        strategy = AdaptiveEMA(params=params)

        bt_params = Params(
            strategy_function=strategy.get_my_positions,
            start_day=1,
            prices_filepath=prices_path
        )
        backtester = Backtester(bt_params)
        results = backtester.run(start_day=1, end_day=750)

        target_pnl = results['daily_pnl']

        if len(target_pnl) > 1 and np.std(target_pnl) > 0:
            score = np.mean(target_pnl) - 0.1 * np.std(target_pnl)
        else:
            score = -np.inf

        return {'params': params, 'score': score}
    except Exception as e:
        return {'params': params, 'score': -np.inf, 'error': str(e)}

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()

    PRICES_PATH = os.path.join(parent_dir, "prices.txt")

    grid = {
        "OPTIMIZATION_LOOKBACK": (50, 501, 50),
        "RECALCULATION_FREQUENCY": (25, 251, 25),
        "VOLATILITY_LOOKBACK": (20, 121, 20),
        "TARGET_ANNUAL_VOL": np.arange(0.1, 0.51, 0.1).tolist(),
        "HURST_THRESHOLD": [0.4,0.5,0.6],
    }

    param_combinations = [dict(zip(grid.keys(), combo)) for combo in product(*[np.arange(*v).round(2).tolist() if isinstance(v, tuple) else v for k, v in grid.items()])]

    print(f"--- Starting Grid Search for Adaptive Strategy: {len(param_combinations)} combinations ---")
    results_list = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(run_backtest_worker, combo, PRICES_PATH) for combo in param_combinations]

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results_list.append(result)
            error_msg = result.get('error')
            if error_msg:
                print(f"Completed {i+1}/{len(param_combinations)} -> FAILED with error: {error_msg}")
            else:
                print(f"Completed {i+1}/{len(param_combinations)} -> Score: {result.get('score', 'FAIL'):.2f}")

    if results_list:
        valid_results = [r for r in results_list if r['score'] > -np.inf]
        if valid_results:
            results_df = pd.DataFrame(valid_results).sort_values(by='score', ascending=False).reset_index(drop=True)
            params_df = pd.json_normalize(results_df['params'])
            final_display_df = pd.concat([results_df[['score']], params_df], axis=1)

            results_filename = "adaptive_EMA_grid_search_results.csv"
            final_display_df.to_csv(results_filename, index=False)

            print("\n\n--- Grid Search Complete ---")
            print(f"\nSUCCESS: All results saved to '{results_filename}'")
            print("\n--- Top 5 Best Overall Parameter Sets ---")
            print(final_display_df.head(5).to_string())
        else:
            print("No successful backtest results were obtained.")
    else:
        print("No results were generated by the grid search.")