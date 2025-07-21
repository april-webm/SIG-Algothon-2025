# ==============================================================================
# SECTION 1: IMPORTS & SETUP
# ==============================================================================
import numpy as np
from numpy import ndarray
import pandas as pd
import warnings

# --- STRATEGY IMPORTS ---
try:
	from april.strategy import get_aprils_positions
	from john.strategy import get_johns_positions
	from william.strategy import get_williams_positions
except ImportError as e:
	print(f"Could not import strategy files. Please check folder structure.")
	def get_aprils_positions(prcSoFar: ndarray) -> ndarray: return np.zeros(50)
	def get_johns_positions(prcSoFar: ndarray) -> ndarray: return np.zeros(50)
	def get_williams_positions(prcSoFar: ndarray) -> ndarray: return np.zeros(50)


warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 2: ROLLING OPTIMISATION PARAMETERS & STATE
# ==============================================================================

OPTIMISATION_PARAMS = {
	"LOOKBACK_DAYS": 50,
	"FREQUENCY": 125,
}

# --- GLOBAL STATE MANAGEMENT ---
LAST_OPTIMISATION_DAY = 0
APRILS_ASSETS = []
JOHNS_ASSETS = list(range(50))
WILLIAMS_ASSETS = []


# ==============================================================================
# SECTION 3: ROLLING OPTIMISER LOGIC (FINAL BENCHING)
# ==============================================================================

def calculate_scores(price_window: ndarray, strategies: dict) -> pd.DataFrame:
	"""Helper function to calculate scores for a given price window."""
	strategy_scores = {}
	price_changes = np.diff(price_window, axis=1).T
	lookback = price_window.shape[1]

	for name, strategy_func in strategies.items():
		positions_over_time = np.array([strategy_func(price_window[:, :t+1]) for t in range(lookback)])
		pnl_by_position = positions_over_time[:-1] * price_changes
		mean_pnl = np.mean(pnl_by_position, axis=0)
		std_pnl = np.std(pnl_by_position, axis=0)
		strategy_scores[name] = mean_pnl - 0.1 * std_pnl

	return pd.DataFrame(strategy_scores)


def run_rolling_optimisation(prcSoFar: ndarray):
	"""
	Evaluates strategies and allocates each asset to the best performer.
	"""
	global APRILS_ASSETS, JOHNS_ASSETS, WILLIAMS_ASSETS, LAST_OPTIMISATION_DAY

	current_day = prcSoFar.shape[1]
	print(f"--- Day {current_day}: Running rolling optimisation ---")

	lookback = OPTIMISATION_PARAMS['LOOKBACK_DAYS']
	if current_day < lookback:
		return

	optimisation_window = prcSoFar[:, -lookback:]

	strategies = {
		"April": get_aprils_positions, "John": get_johns_positions, "William": get_williams_positions
	}

	scores = calculate_scores(optimisation_window, strategies)

	john_scores = scores['John'].values
	april_scores = scores['April'].values
	william_scores = scores['William'].values

	# An asset is benched if its best possible score is NOT strictly positive.
	max_scores = np.maximum.reduce([april_scores, john_scores, william_scores])
	is_benched = max_scores <= 0

	# 2. Winner-Takes-All Allocation (for non-benched assets)
	# Find the strategy with the highest score for each asset.
	is_april_best = ~is_benched & (april_scores >= john_scores) & (april_scores >= william_scores)
	is_william_best = ~is_benched & (william_scores > april_scores) & (william_scores > john_scores)

	# 3. Final Allocation
	all_assets = np.arange(50)
	APRILS_ASSETS = all_assets[is_april_best]
	WILLIAMS_ASSETS = all_assets[is_william_best]

	is_allocated_to_others = is_april_best | is_william_best
	is_johns = ~is_benched & ~is_allocated_to_others
	JOHNS_ASSETS = all_assets[is_johns]

	LAST_OPTIMISATION_DAY = current_day

	benched_count = np.sum(is_benched)
	print("--- Optimisation Complete. New Allocations: ---")
	print(f"April: {len(APRILS_ASSETS)} assets, John: {len(JOHNS_ASSETS)} assets, William: {len(WILLIAMS_ASSETS)} assets, Benched: {benched_count} assets")


# ==============================================================================
# SECTION 4: MAIN STRATEGY FUNCTION
# ==============================================================================

def getMyPosition(prcSoFar: ndarray) -> ndarray:
	"""
	Dynamically allocates assets to the best-performing strategy based on a
	robust rolling lookback optimisation.
	"""
	global LAST_OPTIMISATION_DAY

	current_day = prcSoFar.shape[1]

	min_start_day = OPTIMISATION_PARAMS['LOOKBACK_DAYS']
	if current_day < min_start_day:
		return np.zeros(50)

	if LAST_OPTIMISATION_DAY == 0 or current_day >= LAST_OPTIMISATION_DAY + OPTIMISATION_PARAMS['FREQUENCY']:
		run_rolling_optimisation(prcSoFar)

	aprils_trades = get_aprils_positions(prcSoFar)
	johns_trades = get_johns_positions(prcSoFar)
	williams_positions = get_williams_positions(prcSoFar)

	final_positions = np.zeros(50)

	if len(APRILS_ASSETS) > 0:
		np.put(final_positions, APRILS_ASSETS, aprils_trades.take(APRILS_ASSETS))
	if len(JOHNS_ASSETS) > 0:
		np.put(final_positions, JOHNS_ASSETS, johns_trades.take(JOHNS_ASSETS))
	if len(WILLIAMS_ASSETS) > 0:
		np.put(final_positions, WILLIAMS_ASSETS, williams_positions.take(WILLIAMS_ASSETS))

	return final_positions.astype(int)