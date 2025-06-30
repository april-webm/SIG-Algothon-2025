from typing import Dict, List
from pandas import DataFrame, Series
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from numpy import ndarray
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import joblib
import json
from math import prod

instrument_nos: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ,17, 19,
							 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 36, 38, 39,
							 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
RESPONSE_EMA_LOOKBACK: int = 7
CUTOFF: int = 60

# =================================================================================================
# FEATURES
# =================================================================================================

# Perform an ADF test on each instrument and returns a dictionary of each non-stationary
# instrument with a dataframe for extension
def get_non_stationary_instruments() -> Dict[int,
	DataFrame]:
	raw_prices: DataFrame = pd.read_csv("../../prices.txt", sep=r"\s+", header=None, index_col=None)
	price_history: ndarray = raw_prices.values[:550][:].T

	data: Dict[int, DataFrame] = {}

	for instrument_no in instrument_nos:
		data[instrument_no] = pd.DataFrame(columns=["price"])
		data[instrument_no]["price"] = price_history[instrument_no]

	return data

# This time: if EMA Future > EMA now, 1. Else 0
def implement_response_variable(data: Dict[int, DataFrame], params: Dict[int, Dict[str,
	float | int]]) -> Dict[int, DataFrame]:
	for instrument_no in data:
		price_history: Series = data[instrument_no]["price"]
		ema_now: Series = price_history.ewm(span=RESPONSE_EMA_LOOKBACK).mean()
		ema_future: Series = (price_history.shift(-params[instrument_no]["response_lookforward"])
							  .ewm(span=RESPONSE_EMA_LOOKBACK).mean())
		response: Series = (ema_future > ema_now).astype(int)
		data[instrument_no]["response"] = response.iloc[CUTOFF:]

	return data

def implement_rsi(data: Dict[int, DataFrame], params: Dict[int, Dict[str, float | int]]) -> (
	Dict)[int, DataFrame]:
	for instrument_no in data:
		delta_prices: Series = data[instrument_no]["price"].diff()

		gains: Series= delta_prices.clip(lower=0)
		losses: Series = -delta_prices.clip(upper=0)

		avg_gain: Series = gains.rolling(window=params[instrument_no]["rsi_lookback"]).mean()
		avg_loss: Series = losses.rolling(window=params[instrument_no]["rsi_lookback"]).mean()

		relative_strength: Series = avg_gain / avg_loss
		relative_strength_index: Series = 100 - (100 / (1 + relative_strength))
		relative_strength_index.iloc[:params[instrument_no]["rsi_lookback"]] = 0.0
		data[instrument_no]["rsi"] = relative_strength_index

	return data

def implement_volatility(data: Dict[int, DataFrame], params: Dict[int, Dict[str, float | int]]) -> (
	Dict)[int, DataFrame]:
	for instrument_no in data:
		prices_window: Series= data[instrument_no]["price"]
		returns: Series = prices_window.pct_change()
		volatility: Series = returns.rolling(window=params[instrument_no]["vol_lookback"]).std()
		data[instrument_no]["volatility"] = volatility.iloc[CUTOFF:]

	return data

# Positive: Slow EMA > Fast EMA - Bearish
# Negative: Slow EMA < Fast EMA - Bullish
def implement_ema_crossover(data: Dict[int, DataFrame], params: Dict[int, Dict[str, float | int]]
) -> Dict[int, DataFrame]:
	for instrument_no in data:
		prices_window: Series = data[instrument_no]["price"]
		slow_lookback: int = params[instrument_no]["slow_ema_lookback"]
		fast_lookback: int = params[instrument_no]["fast_ema_lookback"]

		slow_ema: Series = prices_window.ewm(span=slow_lookback, adjust=False).mean()
		fast_ema: Series = prices_window.ewm(span=fast_lookback, adjust=False).mean()

		crossover_gap: Series = slow_ema - fast_ema
		crossover_gap.iloc[:slow_lookback] = 0.0
		data[instrument_no]["ema_crossover"] = crossover_gap

	return data

def implement_donchian_breakout(data: Dict[int, DataFrame], params: Dict[int, Dict[str, float | int
]]) -> Dict[int, DataFrame]:
	for instrument_no in data:
		lookback: int = params[instrument_no]["db_lookback"]
		# Assign upper and lower band
		upper: Series = data[instrument_no]["price"].rolling(window=lookback- 1).max().shift(1)
		lower: Series = data[instrument_no]["price"].rolling(window=lookback-1).min().shift(1)

		# Assign signals
		data[instrument_no]["db_signal"] = 0
		data[instrument_no].loc[data[instrument_no]["price"] > upper, 'db_signal']	= 1
		data[instrument_no].loc[data[instrument_no]["price"] < lower, 'db_signal']	= -1
		data[instrument_no]["db_signal"] = data[instrument_no]["db_signal"].ffill()

	return data


# =================================================================================================
# MODEL FIT + GRID SEARCHER
# =================================================================================================
def setup_data(params: Dict[int, Dict[str, float | int]]) -> Dict[int, DataFrame]:
	data: Dict[int, DataFrame] = get_non_stationary_instruments()
	data = implement_response_variable(data, params)
	data = implement_rsi(data, params)
	data = implement_volatility(data, params)
	data = implement_ema_crossover(data, params)
	data = implement_donchian_breakout(data, params)

	for instrument_no in data: data[instrument_no] = (data[instrument_no]
													  .iloc[CUTOFF:].reset_index(drop=True))
	return data


def fit_lgr(params: Dict[int, Dict[str, float | int]]) -> Dict[int, Dict[str, any]]:
	# Extract predictors and response
	data: Dict[int, DataFrame] = setup_data(params)
	results: Dict[int, Dict[str, float | int]] = {
		instrument_no : {} for instrument_no in data
	}

	for instrument_no in data:
		predictors = data[instrument_no].drop("response", axis=1)
		predictors = predictors.drop("price", axis=1)
		response = data[instrument_no]["response"]

		predictors_values = predictors.values
		response_values = response.values

		tscv = TimeSeriesSplit(n_splits=5)
		accuracy_scores = []
		precision_scores = []
		recall_scores = []

		# Cross Validate
		for train_idx, test_idx in tscv.split(predictors_values):
			if len(np.unique(response_values[train_idx])) < 2: continue

			# Scale Values as classes are unbalanced
			scaler = StandardScaler().fit(predictors_values[train_idx])
			predictors_train_scaled = scaler.transform(predictors_values[train_idx])
			predictors_test_scaled = scaler.transform(predictors_values[test_idx])

			# Fit model
			model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
			model.fit(predictors_train_scaled, response_values[train_idx])

			# Predict
			response_prediction_probability = model.predict_proba(predictors_test_scaled)[:,1]
			predictions = (response_prediction_probability >= 0.5).astype(int)

			# Evaluate
			accuracy_scores.append(accuracy_score(response_values[test_idx], predictions))
			precision_scores.append(precision_score(response_values[test_idx], predictions,
				zero_division=1))
			recall_scores.append(recall_score(response_values[test_idx], predictions,
				zero_division=1))

		results[instrument_no]["accuracy_score"] = np.mean(accuracy_scores)
		results[instrument_no]["precision_score"] = np.mean(precision_scores)
		results[instrument_no]["recall_score"] = np.mean(recall_scores)

	return results

import itertools

param_grid = {
	"response_lookforward": [50],
	"rsi_lookback": [14],
	"vol_lookback": [10],
	"fast_ema_lookback": [5],
	"slow_ema_lookback": [50],
	"db_lookback": [36]
}

optimise_for: str = "accuracy_score"

def grid_search_lgr() -> Dict[int, Dict[str, any]]:
	keys, value_lists = zip(*param_grid.items())
	instruments_to_trade: List[int] = list(get_non_stationary_instruments().keys())

	best_params: Dict[int, Dict[str, any]] = {
		inst: {
			"accuracy_score": -10000.0,
			"precision_score": -10000.0,
			"recall_score": -10000.0
		} for inst in instruments_to_trade
	}

	combinations = list(itertools.product(*value_lists))
	print(f"Grid searching {len(combinations)} combinations…")
	config_num = 1
	for combination in combinations:
		# Progress tracker
		print(f"Running combination {config_num} out of {len(combinations)}")
		params_dict = dict(zip(keys, combination))
		params: Dict[int, any] = {
			instrument_no: params_dict for instrument_no in instruments_to_trade
		}
		results: Dict[int, Dict[str, any]] = fit_lgr(params)

		for instrument_no in instruments_to_trade:
			if best_params[instrument_no][optimise_for] < results[instrument_no][optimise_for]:
				best_params[instrument_no]["accuracy_score"] = results[instrument_no]["accuracy_score"]
				best_params[instrument_no]["precision_score"] = (
					results)[instrument_no]["precision_score"]
				best_params[instrument_no]["recall_score"] = results[instrument_no]["recall_score"]
				best_params[instrument_no]["params"] = params[instrument_no]
		config_num += 1

	# Transform into a table that's easy to read
	grid_search_results_dict: Dict[str, List[float | int]] = {}
	grid_search_results_dict["Instrument No."] = instruments_to_trade

	accuracy_score_list: List[float] = []
	precision_score_list: List[float] = []
	recall_score_list: List[float] = []
	rsi_lookback_list: List[int] = []
	vol_lookback_list: List[int] = []
	fast_ema_lookback_list: List[int] = []
	slow_ema_lookback_list: List[int] = []
	db_lookback_list: List[int] = []

	for instrument_no in instruments_to_trade:
		accuracy_score_list.append(best_params[instrument_no]["accuracy_score"])
		precision_score_list.append(best_params[instrument_no]["precision_score"])
		recall_score_list.append(best_params[instrument_no]["recall_score"])
		rsi_lookback_list.append(best_params[instrument_no]["params"]["rsi_lookback"])
		vol_lookback_list.append(best_params[instrument_no]["params"]["vol_lookback"])
		fast_ema_lookback_list.append(best_params[instrument_no]["params"]["fast_ema_lookback"])
		slow_ema_lookback_list.append(best_params[instrument_no]["params"]["slow_ema_lookback"])
		db_lookback_list.append(best_params[instrument_no]["params"]["db_lookback"])

	grid_search_results_dict["Accuracy Score"] = accuracy_score_list
	grid_search_results_dict["Precision Score"] = precision_score_list
	grid_search_results_dict["Recall Score"] = recall_score_list
	grid_search_results_dict["RSI Lookback"] = rsi_lookback_list
	grid_search_results_dict["Vol Lookback"] = vol_lookback_list
	grid_search_results_dict["Fast EMA Lookback"] = fast_ema_lookback_list
	grid_search_results_dict["Slow EMA Lookback"] = slow_ema_lookback_list
	grid_search_results_dict["DB Lookback"] = db_lookback_list

	grid_search_results_df = pd.DataFrame(grid_search_results_dict)
	print(grid_search_results_df.to_string(index=False))
	return best_params

# =================================================================================================
# SAVING THE MODEL
# =================================================================================================
def save_lgr(instrument_no: int, params: Dict[int, Dict[str, float | int]]) -> None:
	os.makedirs("models", exist_ok=True)
	# Prepare Data
	data = setup_data(params)

	predictors = data[instrument_no].drop(columns=['response', 'price']).values
	response = data[instrument_no]["response"].values.astype(int)

	# Retrain on all data
	scaler_full = StandardScaler().fit(predictors)
	predictors_full = scaler_full.transform(predictors)
	model_full = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
	model_full.fit(predictors_full, response)

	# Save the model + scaler
	model_path = os.path.join("models", f"logistic_regression_inst_{instrument_no}.pkl")
	joblib.dump({ "scaler": scaler_full, "model": model_full}, model_path)

def save_all_models() -> None:
	print("Running Grid Search...")
	grid_search_results: Dict[int, Dict[str, any]] = grid_search_lgr()

	# Save the config
	with open("config.json", "w") as file:
		json.dump(grid_search_results, file)
		print("Config saved.")


	instruments: List[int] = list(grid_search_results.keys())
	for instrument in instruments:
		params: Dict[int, Dict[str, float | int]] = {
			instrument_no: {
				"response_lookforward": grid_search_results[instrument]["params"][
					"response_lookforward"],
				"rsi_lookback": grid_search_results[instrument]["params"]["rsi_lookback"],
				"vol_lookback": grid_search_results[instrument]["params"]["vol_lookback"],
				"fast_ema_lookback": grid_search_results[instrument]["params"]["fast_ema_lookback"],
				"slow_ema_lookback": grid_search_results[instrument]["params"]["slow_ema_lookback"],
				"db_lookback": grid_search_results[instrument]["params"]["db_lookback"]
			} for instrument_no in instruments
		}
		save_lgr(instrument, params)

	print("Done Saving all models")

# =================================================================================================
# MAIN EXECUTION
# =================================================================================================
save_all_models()
