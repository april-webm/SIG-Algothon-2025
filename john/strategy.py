from typing import List
from pandas import DataFrame, Series
from numpy import ndarray
import numpy as np
import pandas as pd

positions_limit: int = 10000
allocated_instruments: List[int] = [0]
positions: ndarray = np.zeros(50)

def get_ema(instrument_price_history: Series, lookback: int) -> float:
    if len(instrument_price_history) < lookback: return 0

    return instrument_price_history.ewm(span=lookback, adjust=False).mean().iloc[-1]

def getMyPosition(prices_so_far: ndarray):
    short_ema_lookback: int = 10
    long_ema_lookback: int = 50

    if len(prices_so_far[0]) < long_ema_lookback:
        return positions

    for instrument_no in allocated_instruments:
        instrument_price_data: Series = pd.Series(prices_so_far[instrument_no])

        short_ema: float = get_ema(instrument_price_data, short_ema_lookback)
        long_ema: float = get_ema(instrument_price_data, long_ema_lookback)

        if short_ema > long_ema and positions[instrument_no] <= 0:
            positions[instrument_no] = positions_limit // instrument_price_data.iloc[-1]
        elif short_ema < long_ema and positions[instrument_no] >= 0:
            positions[instrument_no] = -(positions_limit // instrument_price_data.iloc[-1])

    return positions