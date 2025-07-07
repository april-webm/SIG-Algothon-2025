from typing import List, Dict

from pandas import Series
from numpy import ndarray
import numpy as np
import pandas as pd
from backtester import Backtester, BacktesterResults, Params
import json

positions_limit: int = 10000
allocated_instruments: List[int] =[0, 1, 2, 3, 4, 5, 6, 7, 22, 23, 24, 26, 27]
positions: ndarray = np.zeros(50)

# HELPER FUNCTIONS ################################################################################
def get_ema(instrument_price_history: Series, lookback: int) -> float:
    if len(instrument_price_history) < lookback: return 0

    return instrument_price_history.ewm(span=lookback, adjust=False).mean().iloc[-1]

def get_instrument_positions(instrument_no: int, instrument_price_history: Series,
         short_ema_lookback: int, long_ema_lookback: int, current_position: int) -> int:

    short_ema: float = get_ema(instrument_price_history, short_ema_lookback)
    long_ema: float = get_ema(instrument_price_history, long_ema_lookback)

    if short_ema > long_ema and positions[instrument_no] <= 0:
        return positions_limit // instrument_price_history.iloc[-1]
    elif short_ema < long_ema and positions[instrument_no] >= 0:
        return -(positions_limit // instrument_price_history.iloc[-1])

    return current_position

# MAIN STRATEGY FUNCTION ##########################################################################
def get_johns_position(prices_so_far: ndarray, config: Dict[int, Dict[str, Dict[str, float
    ]]] | None = None, instruments_to_test: List[int] | None = None) -> ndarray:
    # FOR GRID SEARCH
    if config is not None:
        for instrument_no in instruments_to_test:
            instrument_price_data: Series = pd.Series(prices_so_far[instrument_no])
            positions[instrument_no] = get_instrument_positions(
                instrument_no,
                instrument_price_data,
                int(config[instrument_no]["short_ema_lookback"]["current_value"]),
                int(config[instrument_no]["long_ema_lookback"]["current_value"]),
                positions[instrument_no]
            )

    # FOR PRODUCTION
    else:
        # FOR PRODUCTION
        with open("./john/strategy-v1/strategy_v1_config.json", "r") as config_file:
            config: Dict[int, Dict[str, Dict[str, float]]] = json.load(config_file)

            for instrument_no in allocated_instruments:
                instrument_price_data: Series = pd.Series(prices_so_far[instrument_no])
                positions[instrument_no] = get_instrument_positions(
                    instrument_no,
                    instrument_price_data,
                    int(config[str(instrument_no)]["short_ema_lookback"]["best_value"]),
                    int(config[str(instrument_no)]["long_ema_lookback"]["best_value"]),
                    positions[instrument_no])

    return positions

# GRID SEARCHER ###################################################################################
def grid_search() -> None:
    # Open Config File
    with open("strategy_v1_config.json", "r+") as config_file:
        # Create a dictionary with the instrument numbers as keys, and their parameters as the value
        config: Dict[int, Dict[str, Dict[str, float]]] = {}

        for instrument_no in allocated_instruments:
            instrument_parameters: Dict[str, Dict[str, float]] = {}
            instrument_parameters["short_ema_lookback"]: Dict[str, float] = {}
            instrument_parameters["short_ema_lookback"]["window_start"] = 5.0
            instrument_parameters["short_ema_lookback"]["window_end"] = 10.0
            instrument_parameters["short_ema_lookback"]["current_value"] = 5.0

            instrument_parameters["long_ema_lookback"]: Dict[str, float] = {}
            instrument_parameters["long_ema_lookback"]["window_start"] = 50.0
            instrument_parameters["long_ema_lookback"]["window_end"] = 200.0
            instrument_parameters["long_ema_lookback"]["current_value"] = 50.0

            config[instrument_no] = instrument_parameters

        # Dump initial configuration onto the config file
        config_file.seek(0)
        config_file.truncate()
        json.dump(config, config_file)

        # Iterate through each instrument and perform a backtest on each parameter configuration
        backtester_params: Params = Params(
            strategy_filepath="strategy_v1.py",
            strategy_function_name="get_johns_position",
            start_day=1,
            end_day=500,
            enable_commission=False,
            prices_filepath="../../prices.txt"
        )
        backtester: Backtester = Backtester(backtester_params)

        for instrument_no in allocated_instruments:

            instruments_to_test: List[int] = [instrument_no]
            best_score: float = -100000000.0
            for short_ema_lookback in range(
                int(config[instrument_no]["short_ema_lookback"]["window_start"]),
                int(config[instrument_no]["short_ema_lookback"]["window_end"]) + 1
            ):
                config[instrument_no]["short_ema_lookback"]["current_value"] = short_ema_lookback
                for long_ema_lookback in range(
                    int(config[instrument_no]["long_ema_lookback"]["window_start"]),
                    int(config[instrument_no]["long_ema_lookback"]["window_end"]) + 1
                ):
                    config[instrument_no]["long_ema_lookback"]["current_value"] = long_ema_lookback
                    results: BacktesterResults = backtester.run(1, 500, config, instruments_to_test)
                    score: float = results["daily_pnl"].mean() - 0.1 * results["daily_pnl"].std()
                    if score > best_score:
                        best_score = score
                        config[instrument_no]["short_ema_lookback"]["best_value"] = short_ema_lookback
                        config[instrument_no]["long_ema_lookback"]["best_value"] = long_ema_lookback

            print(f"Grid Search on instrument {instrument_no} finished")

        config_file.seek(0)
        config_file.truncate(0)
        json.dump(config, config_file)



# MAIN EXECUTION ##################################################################################
if __name__ == "__main__":
    grid_search()

