# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================
import numpy as np
import pandas as pd
import warnings
from itertools import product

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 2: STRATEGY PARAMETERS & STATE MANAGEMENT
# ==============================================================================

# --- OPTIMAL PARAMETERS (from Grid Search) ---
STRATEGY_PARAMS = {
    "OPTIMISATION_LOOKBACK": 500,
    "RECALCULATION_FREQUENCY": 250,
    "VOLATILITY_LOOKBACK": 60,
    "TARGET_ANNUAL_VOL": 0.2,
    "STOP_LOSS_MULTIPLE": 4.0,
}

# --- Fixed Parameters ---
EMA_GRID = {
    "SHORT_WINDOWS": [5, 10, 15, 20, 40, 60, 80],
    "LONG_WINDOWS": [25, 30, 40, 50, 60, 80, 100, 120],
}
TRADING_START_DAY = STRATEGY_PARAMS["OPTIMISATION_LOOKBACK"]
MAX_POSITION_SIZE = 10000

# --- Global State Management ---
LAST_OPTIMISATION_DAY = 0
OPTIMAL_EMA_PARAMS = {}
CURRENT_POSITIONS = np.zeros(50)
ENTRY_PRICES = np.zeros(50)


# ==============================================================================
# SECTION 3: HELPER & OPTIMISATION FUNCTIONS
# ==============================================================================

def _calculate_ema(prices: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(prices).ewm(span=window, adjust=False).mean().to_numpy()

def _calculate_volatility_adjusted_size(asset_prices: np.ndarray) -> float:
    """Calculates position size based on inverse volatility."""
    vol_lookback = STRATEGY_PARAMS["VOLATILITY_LOOKBACK"]
    if len(asset_prices) < vol_lookback: return 0

    returns = np.diff(asset_prices[-vol_lookback:]) / (asset_prices[-vol_lookback:-1] + 1e-9)
    realized_vol = np.std(returns) * np.sqrt(252)

    if realized_vol < 1e-9: return 0

    scaling_factor = STRATEGY_PARAMS["TARGET_ANNUAL_VOL"] / realized_vol
    return min(scaling_factor * MAX_POSITION_SIZE, MAX_POSITION_SIZE)

def _find_optimal_emas(prcSoFar: np.ndarray):
    """
    Performs a grid search for each asset to find the best-performing
    EMA crossover parameters for a trend-following strategy.
    """
    global OPTIMAL_EMA_PARAMS
    optimisation_window = prcSoFar[:, -STRATEGY_PARAMS["OPTIMISATION_LOOKBACK"]:]
    new_optimal_params = {}

    for asset_idx in range(prcSoFar.shape[0]):
        asset_prices = optimisation_window[asset_idx]

        best_score, best_params = -np.inf, None
        for short_w, long_w in product(EMA_GRID["SHORT_WINDOWS"], EMA_GRID["LONG_WINDOWS"]):
            if short_w >= long_w: continue

            short_ema = _calculate_ema(asset_prices, short_w)
            long_ema = _calculate_ema(asset_prices, long_w)

            signals = np.where(short_ema > long_ema, 1, -1)
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

    OPTIMAL_EMA_PARAMS = new_optimal_params
    print(f"--- Optimisation Complete: Found optimal parameters for {len(OPTIMAL_EMA_PARAMS)} assets. ---")

# ==============================================================================
# SECTION 4: MAIN STRATEGY FUNCTION
# ==============================================================================

def get_aprils_positions(prcSoFar: np.ndarray) -> np.ndarray:
    """
    An adaptive EMA Crossover strategy with volatility-based position sizing
    and a volatility-based stop-loss.
    """
    global LAST_OPTIMISATION_DAY, OPTIMAL_EMA_PARAMS, CURRENT_POSITIONS, ENTRY_PRICES

    current_day = prcSoFar.shape[1]

    if current_day < TRADING_START_DAY:
        return np.zeros(50)

    if LAST_OPTIMISATION_DAY == 0 or current_day >= LAST_OPTIMISATION_DAY + STRATEGY_PARAMS["RECALCULATION_FREQUENCY"]:
        _find_optimal_emas(prcSoFar)
        LAST_OPTIMISATION_DAY = current_day
        # Reset state after optimisation
        CURRENT_POSITIONS = np.zeros(50)
        ENTRY_PRICES = np.zeros(50)

    if not OPTIMAL_EMA_PARAMS:
        return np.zeros(50)

    new_positions = np.copy(CURRENT_POSITIONS)
    for asset_idx, params in OPTIMAL_EMA_PARAMS.items():
        asset_prices = prcSoFar[asset_idx]
        current_price = asset_prices[-1]

        # --- Check Stop Loss for Existing Positions ---
        if CURRENT_POSITIONS[asset_idx] != 0:
            entry_price = ENTRY_PRICES[asset_idx]

            if len(asset_prices) > 22:
                daily_returns = np.diff(asset_prices[-22:]) / (asset_prices[-22:-1] + 1e-9)
                daily_vol = np.std(daily_returns)
                stop_loss_amount = STRATEGY_PARAMS["STOP_LOSS_MULTIPLE"] * daily_vol * entry_price

                if CURRENT_POSITIONS[asset_idx] > 0: # Long position
                    if current_price < entry_price - stop_loss_amount:
                        new_positions[asset_idx] = 0 # Trigger stop loss
                        ENTRY_PRICES[asset_idx] = 0
                        continue
                elif CURRENT_POSITIONS[asset_idx] < 0: # Short position
                    if current_price > entry_price + stop_loss_amount:
                        new_positions[asset_idx] = 0 # Trigger stop loss
                        ENTRY_PRICES[asset_idx] = 0
                        continue

                        # --- EMA Crossover Signal Logic ---
        short_ema = _calculate_ema(asset_prices, params['short'])
        long_ema = _calculate_ema(asset_prices, params['long'])

        current_signal = 0
        if not np.isnan(short_ema[-1]) and not np.isnan(long_ema[-1]):
            if short_ema[-1] > long_ema[-1]:
                current_signal = 1
            elif short_ema[-1] < long_ema[-1]:
                current_signal = -1

        previous_signal = np.sign(CURRENT_POSITIONS[asset_idx])

        # --- Trade Execution if Signal Changes ---
        if current_signal != previous_signal:
            if current_signal != 0:
                # New entry or flip
                adjusted_size = _calculate_volatility_adjusted_size(asset_prices)
                new_positions[asset_idx] = current_signal * (adjusted_size / (current_price + 1e-9))
                ENTRY_PRICES[asset_idx] = current_price
            else:
                # Exit signal
                new_positions[asset_idx] = 0
                ENTRY_PRICES[asset_idx] = 0

    CURRENT_POSITIONS = new_positions

    return CURRENT_POSITIONS.astype(int)