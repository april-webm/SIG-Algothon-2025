import pandas as pd
import numpy as np

from numpy import ndarray
from typing import List, Dict, Tuple
from pandas import Series, DataFrame

# CONSTANTS AND CONFIG ############################################################################
positions_limit: int = 10000
allocated_instruments: List[int] = list(range(0, 50))

last_signals: Dict[int, float] = {
	instrument_no: 0.0 for instrument_no in allocated_instruments
}

last_positions: Dict[int, int] = {
	instrument_no: 0 for instrument_no in allocated_instruments
}

last_trade_adjustment_params: Dict[int, Dict[str, float]] = {
	instrument_no: {
		"long_entry_p": np.nan,
		"short_entry_p": np.nan,
		"last_long": np.nan,
		"last_short": np.nan
	} for instrument_no in allocated_instruments
}

config: Dict[int, Dict[str, int | float | str]] = {
	0: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 65
	},
	1: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 15
	},
	2: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 89
	},
	3: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 15
	},
	4: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 35
	},
	5: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 50
	},
	6: {
		"strategy": "donchian breakout",
		"db type": "all trades",
		"db lookback": 151
	},
	7: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 21
	},
	8: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 14
	},
	9: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 13
	},
	10: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 37
	},
	11: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 12
	},
	12: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 35
	},
	13: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 20
	},
	14: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 12
	},
	15: {
		"strategy": "donchian breakout",
		"db type": "all trades",
		"db lookback": 45
	},
	16: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 18
	},
	17: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 24
	},
	18: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 24
	},
	19: {
		"strategy": "donchian breakout",
		"db type": "all trades",
		"db lookback": 167
	},
	20: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 39
	},
	21: {
		"strategy": "donchian breakout",
		"db type": "all trades",
		"db lookback": 84
	},
	22: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 34
	},
	23: {
		"strategy": "donchian breakout",
		"db type": "all trades",
		"db lookback": 109
	},
	24: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 12
	},
	25: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 77
	},
	26: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 16
	},
	27: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 13
	},
	28: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 24
	},
	29: {
		"strategy": "donchian breakout",
		"db type": "all trades",
		"db lookback": 153
	},
	30: {
		"strategy": "donchian breakout",
		"db type": "all trades",
		"db lookback": 133
	},
	31: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 39
	},
	32: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 16
	},
	33: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 12
	},
	34: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 42
	},
	35: {
		"strategy": "donchian breakout",
		"db type": "all trades",
		"db lookback": 136
	},
	36: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 12
	},
	37: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 26
	},
	38: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 18
	},
	39: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 56
	},
	40: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 21
	},
	41: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 18
	},
	42: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 65
	},
	43: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 31
	},
	44: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 12
	},
	45: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 108
	},
	46: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 39
	},
	47: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 13
	},
	48: {
		"strategy": "donchian breakout",
		"db type": "last loser",
		"db lookback": 139
	},
	49: {
		"strategy": "donchian breakout",
		"db type": "last winner",
		"db lookback": 12
	},
}


# STRATEGY FUNCTION ##############################################################################
def donchian_breakout(prices_so_far: ndarray, instrument_no: int) -> int:
	# Get Price History
	instrument_price_history: Series = pd.Series(prices_so_far[instrument_no])

	# Get Lookback and DB Type from config
	lookback = config[instrument_no]["db lookback"]
	db_type = config[instrument_no]["db type"]

	# If the number of trading days is less than the lookback, don't trade
	if len(prices_so_far[instrument_no]) < lookback: return 0

	# Get upper and lower band
	upper: float = instrument_price_history.rolling(lookback - 1).max().shift(1).iloc[-1]
	lower: float = instrument_price_history.rolling(lookback - 1).min().shift(1).iloc[-1]

	# Compare current price to upper and lower band and generate an initial signal
	current_price: float = instrument_price_history.iloc[-1]
	last_signal: float = last_signals[instrument_no]
	signal: float = 0.0

	if current_price > upper:
		signal = 1.0
	elif current_price < lower:
		signal = -1.0

	# If our model found a signal and its different to our last one
	if signal != 0.0 and signal != last_signal:
		# First check for trade dependence
		if db_type == "all trades":
			last_signals[instrument_no] = signal
			last_positions[instrument_no] = int(signal) * int(positions_limit // current_price)
			return last_positions[instrument_no]
		else:
			last_type: float = 1.0 if db_type == "last winner" else -1.0
			modified_signal: float = 0.0

			if signal == 1.0 and last_signal != 1.0:
				last_trade_adjustment_params[instrument_no]["long_entry_p"] = current_price
				if not np.isnan(last_trade_adjustment_params[instrument_no]["short_entry_p"]):
					last_trade_adjustment_params[instrument_no]["last_short"] = np.sign(
						last_trade_adjustment_params[instrument_no]["short_entry_p"] -
						current_price)
					last_trade_adjustment_params[instrument_no]["short_entry_p"] = np.nan

			if signal == -1.0 and last_signal != -1.0:
				last_trade_adjustment_params[instrument_no]["short_entry_p"] = current_price
				if not np.isnan(last_trade_adjustment_params[instrument_no]["long_entry_p"]):
					last_trade_adjustment_params[instrument_no]["last_long"] = np.sign(
						current_price - last_trade_adjustment_params[instrument_no]["long_entry_p"])
					last_trade_adjustment_params[instrument_no]["long_entry_p"] = np.nan

			last_signals[instrument_no] = signal

			if (signal == 1.0 and last_trade_adjustment_params[instrument_no]["last_short"] ==
				last_type):
				modified_signal = 1.0
			if (signal == -1.0 and last_trade_adjustment_params[instrument_no]["last_long"] ==
				last_type):
				modified_signal = -1.0

			last_positions[instrument_no] = int(modified_signal) * int(positions_limit //
																	   current_price)
			return last_positions[instrument_no]
	else:
		return last_positions[instrument_no]

# GRID SEARCHER ##################################################################################
def last_trade_adj_signal(strategy_output: Dict[int, DataFrame], signals: Dict[int, ndarray],
	last_winner: bool = False) -> Dict[int, ndarray]:
	modified_signals: Dict[int, ndarray] = {}

	for instrument_no in allocated_instruments:
		last_type: int = 1 if last_winner else -1

		prices: ndarray = strategy_output[instrument_no]["price"].to_numpy()
		modified_signal: ndarray = np.zeros(len(signals[instrument_no]))

		long_entry_p: ndarray = np.nan
		short_entry_p: ndarray = np.nan
		last_long: ndarray = np.nan
		last_short: ndarray = np.nan

		last_signal = 0
		for i in range(len(prices)):
			if signals[instrument_no][i] == 1 and last_signal != 1: # Long Entry
				long_entry_p = prices[i]
				if not np.isnan(short_entry_p):
					last_short = np.sign(short_entry_p - prices[i])
					short_entry_p = np.nan

			if signals[instrument_no][i] == -1 and last_signal != -1: # Short Entry
				short_entry_p = prices[i]
				if not np.isnan(long_entry_p):
					last_long = np.sign(prices[i] - long_entry_p)
					long_entry_p = np.nan

			last_signal = signals[instrument_no][i]

			if signals[instrument_no][i] == 1 and last_short == last_type:
				modified_signal[i] = 1
			if signals[instrument_no][i] == -1 and last_long == last_type:
				modified_signal[i] = -1

		modified_signals[instrument_no] = modified_signal

	return modified_signals

def donchian_breakout_for_grid_search(price_history: Dict[int, DataFrame], lookbacks: Dict[int,
	int]) -> Dict[int, DataFrame]:
	for instrument_no in allocated_instruments:
		# Assign upper and lower band
		price_history[instrument_no]["upper"] = price_history[instrument_no]["price"].rolling(
			lookbacks[instrument_no] - 1).max().shift(1)
		price_history[instrument_no]["lower"] = price_history[instrument_no]["price"].rolling(
			lookbacks[instrument_no] - 1).min().shift(1)

		# Assign signals
		price_history[instrument_no]["signal"] = np.nan
		price_history[instrument_no].loc[price_history[instrument_no]["price"] > price_history[
			instrument_no]["upper"], 'signal']	= 1
		price_history[instrument_no].loc[price_history[instrument_no]["price"] < price_history[
			instrument_no]["lower"], 'signal']	= -1

		price_history[instrument_no]["signal"] = price_history[instrument_no]["signal"].ffill()

	return price_history

def grid_search(price_history: ndarray, grid_search_lookback: int) -> None:
	best_params: Dict[int, Dict[str, str | int | float]] = {
		instrument_no: {
			"best_profit_factor": -1000000.0
		} for instrument_no in allocated_instruments
	}

	for lookback in range(12, 169):
		# Setup price window dictionary
		data: Dict[int, DataFrame] = {
			instrument_no: pd.DataFrame(columns=["price"]) for instrument_no in allocated_instruments
		}

		# Implement price and returns
		for instrument_no in allocated_instruments:
			data[instrument_no]["price"] = price_history[instrument_no][
				len(price_history) - grid_search_lookback:
			]
			data[instrument_no]["returns"] = (np.log(data[instrument_no]["price"]).diff().shift(-1))

		# Get Strategy Output for normal Donchian Breakout
		lookbacks: Dict[int, int] = {instrument_no: lookback for instrument_no in
									 allocated_instruments}
		strategy_output: Dict[int, DataFrame] = donchian_breakout_for_grid_search(data,
			lookbacks)

		# Get strategy output with trade dependence
		signals: Dict[int, ndarray] = {instrument_no: strategy_output[instrument_no]["signal"].to_numpy()
									   for instrument_no in allocated_instruments}
		last_lose_signals: Dict[int, ndarray] = last_trade_adj_signal(strategy_output, signals, False)
		last_win_signals: Dict[int, ndarray] = last_trade_adj_signal(strategy_output, signals, True)

		for instrument_no in allocated_instruments:
			strategy_output[instrument_no]["last_lose"] = last_lose_signals[instrument_no]
			strategy_output[instrument_no]["last_win"] = last_win_signals[instrument_no]

		# Get returns for each strategy type
		original: Dict[int, Series] = {
			instrument_no: (strategy_output[instrument_no]["returns"] *
							strategy_output[instrument_no]["signal"]) for instrument_no in allocated_instruments
		}
		lose: Dict[int, Series] = {
			instrument_no: (strategy_output[instrument_no]["returns"] *
							strategy_output[instrument_no]["last_lose"]) for instrument_no in
			allocated_instruments
		}
		win: Dict[int, Series] = {
			instrument_no: (strategy_output[instrument_no]["returns"] *
							strategy_output[instrument_no]["last_win"]) for instrument_no in allocated_instruments
		}

		# Get sharpe ratio and adjust best params
		for instrument_no in allocated_instruments:
			original_profit_factor: float
			lose_profit_factor: float
			win_profit_factor: float

			original_profit_factor_denom: float = original[instrument_no][original[instrument_no]
																		  < 0].abs().sum()
			lose_profit_factor_denom: float = (lose[instrument_no][lose[instrument_no] < 0].abs()
											   .sum())
			win_profit_factor_denom: float = (win[instrument_no][win[instrument_no] < 0].abs()
											  .sum())
			if original_profit_factor_denom == 0:
				original_profit_factor = np.nan
			else:
				original_profit_factor: float = (original[instrument_no][original[instrument_no]
																		 > 0].sum() / original_profit_factor_denom)
			if lose_profit_factor_denom == 0:
				lose_profit_factor = np.nan
			else:
				lose_profit_factor: float = (lose[instrument_no][lose[instrument_no] > 0].sum() /
											 lose_profit_factor_denom)

			if win_profit_factor_denom == 0:
				win_profit_factor = np.nan
			else:
				win_profit_factor: float = (win[instrument_no][win[instrument_no] > 0].sum() /
											win_profit_factor_denom)

			if original_profit_factor > best_params[instrument_no]["best_profit_factor"]:
				best_params[instrument_no]["best_profit_factor"] = original_profit_factor
				best_params[instrument_no]["lookback"] = lookback
				best_params[instrument_no]["db_type"] = "all trades"

			if lose_profit_factor > best_params[instrument_no]["best_profit_factor"]:
				best_params[instrument_no]["best_profit_factor"] = lose_profit_factor
				best_params[instrument_no]["lookback"] = lookback
				best_params[instrument_no]["db_type"] = "last loser"

			if win_profit_factor > best_params[instrument_no]["best_profit_factor"]:
				best_params[instrument_no]["best_profit_factor"] = win_profit_factor
				best_params[instrument_no]["lookback"] = lookback
				best_params[instrument_no]["db_type"] = "last winner"

	# Adjust config based on best params
	for instrument_no in allocated_instruments:
		config[instrument_no]["strategy"] = "donchian breakout"
		config[instrument_no]["db lookback"] = best_params[instrument_no]["lookback"]
		config[instrument_no]["db type"] = best_params[instrument_no]["db_type"]

# GET POSITIONS FUNCTION ######################################################################
positions: ndarray = np.zeros(50)

def get_johns_positions(prices_so_far: ndarray) -> ndarray:
	# Grid search every 50 days past 400 days
	if len(prices_so_far[0]) > 400 and len(prices_so_far[0]) % 50 == 0:
		print(f"Grid searching on day {len(prices_so_far[0])}")
		grid_search(prices_so_far, 100)

	for instrument_no in allocated_instruments:
		if instrument_no not in config.keys(): continue

		if config[instrument_no]["strategy"] == "donchian breakout":
			positions[instrument_no] = donchian_breakout(prices_so_far, instrument_no)
		else:
			positions[instrument_no] = 0

	return positions