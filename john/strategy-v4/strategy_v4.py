import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from pandas import Series
from typing import List, Dict
from numpy import ndarray
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregressionResults
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

instrument_nos: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19,
							 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 36, 38, 39, 40,
							 41, 42, 43, 44, 45,
							 46, 47, 48, 49]
positions: ndarray = np.zeros(50)
last_signal: ndarray = np.zeros(50)
positions_limit: int = 10000

is_zero_uptrend: Dict[int, bool] = {
	0: True,
	1: True,
	2: True,
	3: False,
	4: True,
	5: True,
	6: True,
	7: False,
	8: True,
	9: True,
	10: False,
	11: False,
	12: False,
	13: False,
	14: False,
	15: False,
	16: True,
	17: False,
	19: True,
	20: True,
	21: False,
	22: False,
	23: False,
	24: False,
	25: True,
	27: True,
	28: False,
	29: False,
	30:True,
	31: False,
	32: False,
	34: False,
	35: True,
	36: False,
	38: False,
	39: False,
	40: False,
	41: False,
	42: False,
	43: True,
	44: False,
	45: False,
	46: True,
	47: True,
	48: False,
	49: False,
}


def apply_model(instrument_no: int, prices_so_far: ndarray) -> ndarray:
	# Load results
	results: MarkovAutoregressionResults
	with open(f"john/strategy-v4/models/instrument_{instrument_no}_model.pkl", "rb") as model_file:
		results = pickle.load(model_file)

	# Get Returns
	returns: ndarray = np.diff(prices_so_far[instrument_no])

	# Load model
	model: MarkovAutoregression = MarkovAutoregression(
		returns,
		k_regimes=results.model.k_regimes,
		order=results.model.order,
		switching_ar=results.model.switching_ar,
		switching_variance=results.model.switching_variance
	)

	filt_res = model.filter(results.params)
	probs = filt_res.filtered_marginal_probabilities
	predicted_regimes = np.argmax(probs, axis=1)

	return predicted_regimes

def get_signal(predicted_regimes: ndarray, instrument_no: int, prices_so_far: ndarray) -> (
	int):
	# Return a signal that corresponds with the classified regime
	if is_zero_uptrend[instrument_no] and predicted_regimes[-1] == 0:
		return 1
	elif is_zero_uptrend[instrument_no] and predicted_regimes[-1] == 1:
		return -1
	elif not is_zero_uptrend[instrument_no] and predicted_regimes[-1] == 0:
		return -1
	else:
		return 1


def get_johns_positions(prices_so_far: ndarray) -> ndarray:
	print(f"Day {len(prices_so_far[0])}")
	if len(prices_so_far[0]) < 3: return positions

	if len(prices_so_far[0]) == 550:
		return np.zeros(50)

	for instrument_no in instrument_nos:
		predicted_regimes: ndarray = apply_model(instrument_no, prices_so_far)
		signal: int = get_signal(predicted_regimes,instrument_no, prices_so_far)

		# Generate positions - if new signal is different to last signal, create new trade.
		# Otherwise just leave it
		current_price: float = prices_so_far[instrument_no][-1]
		if signal != last_signal[instrument_no]:
			positions[instrument_no] = signal * int(positions_limit / current_price)
			last_signal[instrument_no] = signal

	return positions
