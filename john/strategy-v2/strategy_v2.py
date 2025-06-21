"""
STRATEGY V2

Features adopted from old strategies:
- EMA Crossover (with grid-searched lookbacks)

Features to add and test:
- Volatility based position sizing
- Stop Loss / Profit Target Rules
- Signal Filtering & Ensemble
"""

from typing import List, Dict, TypedDict
from numpy import ndarray
from pandas import Series
import pandas as pd
import numpy as np
import json
from backtester import Backtester, BacktesterResults, Params

positions_limit: int = 10000
allocated_instruments: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 22, 23, 24, 26, 27]
positions: ndarray = np.zeros(50)


# HELPER FUNCTIONS ################################################################################
def get_ema(instrument_price_history: Series, lookback: int) -> float:
    if len(instrument_price_history) < lookback: return 0

    return instrument_price_history.ewm(span=lookback, adjust=False).mean().iloc[-1]

def get_annualised_volatility(instrument_price_history: Series, lookback: int) -> float:
    if len(instrument_price_history) < lookback + 1: return 0.0

    prices_window: Series= instrument_price_history[-(lookback + 1):]
    returns: Series = prices_window.pct_change().dropna()
    vol: float = returns.std(ddof=1)
    return float(vol * np.sqrt(252))

class InstrumentPositionParams(TypedDict):
    instrument_no: int
    instrument_price_history: Series
    short_ema_lookback: int
    long_ema_lookback: int
    current_position: int
    volatility_target: float
    volatility_lookback: int


def get_instrument_positions(params: InstrumentPositionParams) -> int:
    # Unpack params
    instrument_no: int = params["instrument_no"]
    instrument_price_history: Series = params["instrument_price_history"]
    short_ema_lookback: int = params["short_ema_lookback"]
    long_ema_lookback: int = params["long_ema_lookback"]
    current_position: int = params["current_position"]
    volatility_target: float = params["volatility_target"]
    volatility_lookback: int = params["volatility_lookback"]

    # Get short and long EMAs
    short_ema: float = get_ema(instrument_price_history, short_ema_lookback)
    long_ema: float = get_ema(instrument_price_history, long_ema_lookback)

    # Get Rolling volatility of instrument over the last n days
    vol: float = get_annualised_volatility(instrument_price_history, volatility_lookback)

    # Get Notional Capital
    capital: float = float(positions_limit)
    if volatility_target < vol:
        capital *= volatility_target / vol

    if short_ema > long_ema and positions[instrument_no] <= 0:
        return int(capital // instrument_price_history.iloc[-1])
    elif short_ema < long_ema and positions[instrument_no] >= 0:
        return int(-(capital // instrument_price_history.iloc[-1]))

    return current_position


def strategy_function(
    prices_so_far: ndarray,
    config: Dict[int, Dict[str, Dict[str, float]]] | None = None,
    instruments_to_test: List[int] | None = None
) -> ndarray:
    # FOR GRID SEARCH
    if config is not None:
        for instrument_no in instruments_to_test:
            instrument_price_data: Series = pd.Series(prices_so_far[instrument_no])
            params: InstrumentPositionParams = InstrumentPositionParams()
            params["instrument_no"] = instrument_no
            params["instrument_price_history"] = instrument_price_data
            params["short_ema_lookback"] = int(
                config[str(instrument_no)]["short_ema_lookback"]["best_value"]
            )
            params["long_ema_lookback"] = int(
                config[str(instrument_no)]["long_ema_lookback"]["best_value"]
            )
            params["current_position"] = positions[instrument_no]
            params["volatility_target"] = config[str(instrument_no)]["volatility_target"][
                "current_value"]
            params["volatility_lookback"] = config[str(instrument_no)]["volatility_lookback"][
                "current_value"]
            if len(instrument_price_data) < params["long_ema_lookback"]: continue

            positions[instrument_no] = get_instrument_positions(params)

    # FOR PRODUCTION
    else:
        # FOR PRODUCTION
        with open("./john/strategy-v2/strategy_v2_config.json", "r") as config_file:
            config: Dict[int, Dict[str, Dict[str, float]]] = json.load(config_file)

            for instrument_no in allocated_instruments:
                instrument_price_data: Series = pd.Series(prices_so_far[instrument_no])
                params: InstrumentPositionParams = InstrumentPositionParams()
                params["instrument_no"] = instrument_no
                params["instrument_price_history"] = instrument_price_data
                params["short_ema_lookback"] = int(
                    config[str(instrument_no)]["short_ema_lookback"]["best_value"]
                )
                params["long_ema_lookback"] = int(
                    config[str(instrument_no)]["long_ema_lookback"]["best_value"]
                )
                params["current_position"] = positions[instrument_no]
                params["volatility_target"] = 1.00
                params["volatility_lookback"] = 14

                if len(instrument_price_data) < params["long_ema_lookback"]: continue

                positions[instrument_no] = get_instrument_positions(params)

    return positions

# GRID SEARCHER ###################################################################################
def grid_search() -> None:
    volatility_target_window: List[float] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    volatility_lookback_window: List[float] = [5, 10, 14, 21, 30, 50, 75, 100, 125, 252]

    # Open Config File
    with open("strategy_v2_config.json", "r+") as config_file:
        # Grab config
        config: Dict[str, Dict[str, Dict[str, float | List[float]]]] = json.load(config_file)

        # Set initial configuration for each instrument
        for instrument_no in allocated_instruments:
            instrument_no: str = str(instrument_no)
            # Set Volatility Target parameter
            config[instrument_no]["volatility_target"]: Dict[str, float] = {}
            config[instrument_no]["volatility_lookback"]: Dict[str, float] = {}

        # Dump initial configuration onto the config file
        json.dump(config, config_file)

        # Iterate through each instrument and perform a backtest on each parameter configuration
        backtester_params: Params = Params(
            strategy_filepath="strategy_v2.py",
            strategy_function_name="strategy_function",
            start_day=1,
            end_day=500,
            enable_commission=True,
            prices_filepath="../../prices.txt"
        )
        backtester: Backtester = Backtester(backtester_params)

        for instrument_no in allocated_instruments:
            instrument_no: str = str(instrument_no)
            instruments_to_test: List[int] = [int(instrument_no)]
            best_score: float = -100000000.0

            for volatility_target in volatility_target_window:
                for volatility_lookback in volatility_lookback_window:
                    config[instrument_no]["volatility_target"]["current_value"] = volatility_target
                    config[instrument_no]["volatility_lookback"]["current_value"] = volatility_lookback
                    results: BacktesterResults = backtester.run(1, 500, config, instruments_to_test)

                    score: float = results["daily_pnl"].mean() - 0.1 * results["daily_pnl"].std()

                    if score > best_score:
                        best_score = score
                        config[instrument_no]["volatility_target"]["best_value"] = volatility_target
                        config[instrument_no]["volatility_lookback"]["best_value"] = volatility_lookback

            print(f"Grid Search on instrument {instrument_no} finished")

        config_file.seek(0)
        config_file.truncate(0)
        json.dump(config, config_file)



# MAIN EXECUTION ##################################################################################
if __name__ == "__main__":
    grid_search()