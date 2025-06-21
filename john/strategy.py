from typing import List,Dict
from pandas import Series
from numpy import ndarray
import pandas as pd
import json
import numpy as np

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
def get_johns_positions(prices_so_far: ndarray) -> ndarray:
    # FOR PRODUCTION
    with open("./john/config.json", "r") as config_file:
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