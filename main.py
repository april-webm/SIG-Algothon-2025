# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================
import numpy as np
from numpy import ndarray
import pandas as pd
import warnings

# --- STRATEGY IMPORTS ---
# Ensure your strategy files are accessible
try:
    from april.strategy import get_aprils_positions
    from john.strategy import get_johns_positions
    from william.strategy import get_williams_positions
except ImportError as e:
    print(f"Could not import strategy files. Please check folder structure.")
    # Define dummy functions if real ones are not found, to allow the script to load
    def get_aprils_positions(prcSoFar: ndarray) -> ndarray: return np.zeros(50)
    def get_johns_positions(prcSoFar: ndarray) -> ndarray: return np.zeros(50)
    def get_williams_positions(prcSoFar: ndarray) -> ndarray: return np.zeros(50)

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 2: ROLLING OPTIMIZATION PARAMETERS & STATE
# ==============================================================================

OPTIMIZATION_PARAMS = {
    "LOOKBACK_DAYS": 250,      # How many past days to evaluate performance on
    "FREQUENCY": 50,           # How often (in days) to re-run the optimization
}

# --- GLOBAL STATE MANAGEMENT ---
# These variables will be updated dynamically by the rolling optimizer.
LAST_OPTIMIZATION_DAY = 0
APRILS_ASSETS = []
JOHNS_ASSETS = []
WILLIAMS_ASSETS = []


# ==============================================================================
# SECTION 3: ROLLING OPTIMIZER LOGIC
# ==============================================================================

def run_rolling_optimization(prcSoFar: ndarray):
    """
    Evaluates all strategies over the lookback period, calculates the optimal
    asset allocation, and updates the global state.
    """
    global APRILS_ASSETS, JOHNS_ASSETS, WILLIAMS_ASSETS, LAST_OPTIMIZATION_DAY

    current_day = prcSoFar.shape[1]
    print(f"--- Day {current_day}: Running rolling optimization ---")

    # 1. Define the historical window for this optimization run
    lookback = OPTIMIZATION_PARAMS['LOOKBACK_DAYS']
    if current_day < lookback:
        print("Not enough data to run optimization. Waiting...")
        return

    optimization_window = prcSoFar[:, -lookback:]

    strategies = {
        "April": get_aprils_positions,
        "John": get_johns_positions,
        "William": get_williams_positions
    }

    strategy_scores = {}

    # 2. Run a mini-backtest for each strategy to get its performance on each asset
    for name, strategy_func in strategies.items():
        asset_pnl = np.zeros((50, lookback - 1))
        positions = np.zeros(50)

        for t in range(lookback - 1):
            # Get positions based on the history up to day t
            daily_prc_slice = optimization_window[:, :t+1]
            new_positions = strategy_func(daily_prc_slice)

            # Calculate PnL for day t+1 based on positions held at t
            price_change = optimization_window[:, t+1] - optimization_window[:, t]
            asset_pnl[:, t] = positions * price_change

            positions = new_positions

        # 3. Calculate the score for each asset for the current strategy
        mean_pnl = np.mean(asset_pnl, axis=1)
        std_pnl = np.std(asset_pnl, axis=1)
        strategy_scores[name] = mean_pnl - 0.1 * std_pnl

    # 4. Determine the best strategy for each asset
    results_df = pd.DataFrame(strategy_scores)
    best_strategy_per_asset = results_df.idxmax(axis=1)

    # 5. Update the global allocation lists
    allocations = {name: [] for name in strategies.keys()}
    for asset_idx, best_strategy_name in best_strategy_per_asset.items():
        allocations[best_strategy_name].append(asset_idx)

    APRILS_ASSETS = sorted(allocations.get("April", []))
    JOHNS_ASSETS = sorted(allocations.get("John", []))
    WILLIAMS_ASSETS = sorted(allocations.get("William", []))

    LAST_OPTIMIZATION_DAY = current_day

    print("--- Optimization Complete. New Allocations: ---")
    print(f"April: {len(APRILS_ASSETS)} assets, John: {len(JOHNS_ASSETS)} assets, William: {len(WILLIAMS_ASSETS)} assets")


# ==============================================================================
# SECTION 4: MAIN STRATEGY FUNCTION
# ==============================================================================

def getMyPosition(prcSoFar: ndarray) -> ndarray:
    """
    Dynamically allocates assets to the best-performing strategy based on a
    rolling lookback optimization.
    """
    global LAST_OPTIMIZATION_DAY

    current_day = prcSoFar.shape[1]

    # Trigger the first optimization run after enough data has accumulated
    if LAST_OPTIMIZATION_DAY == 0 and current_day >= OPTIMIZATION_PARAMS['LOOKBACK_DAYS']:
        run_rolling_optimization(prcSoFar)
    # Trigger subsequent re-optimizations based on the frequency
    elif current_day >= LAST_OPTIMIZATION_DAY + OPTIMIZATION_PARAMS['FREQUENCY']:
        run_rolling_optimization(prcSoFar)

    # Run all three strategies to get their proposed trades for all assets
    aprils_trades = get_aprils_positions(prcSoFar)
    johns_trades = get_johns_positions(prcSoFar)
    williams_positions = get_williams_positions(prcSoFar)

    # Initialize our final positions array
    final_positions = np.zeros(50)

    # Construct the portfolio using the dynamically updated global allocation lists
    np.put(final_positions, APRILS_ASSETS, aprils_trades.take(APRILS_ASSETS))
    np.put(final_positions, JOHNS_ASSETS, johns_trades.take(JOHNS_ASSETS))
    np.put(final_positions, WILLIAMS_ASSETS, williams_positions.take(WILLIAMS_ASSETS))

    return final_positions.astype(int)