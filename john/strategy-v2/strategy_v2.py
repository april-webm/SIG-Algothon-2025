from typing import List, Dict
from numpy import ndarray
import numpy as np
import pandas as pd
from pandas import Series

positions_limit: int = 10000
allocated_instruments: List[int] =[4,5,20,21,23,30,47]
positions: ndarray = np.zeros(50)

# CONFIG ##########################################################################################
config: Dict[int, Dict[str, int | float]] = {
    4: {
        "strategy": "donchian breakout",
        "db_lookback": 51
    },
    5: {
        "strategy": "donchian breakout",
        "db_lookback": 50
    },
    20: {
        "strategy": "donchian breakout",
        "db_lookback": 157
    },
    21: {
        "strategy": "donchian breakout",
        "db_lookback":59
    },
    23: {
        "strategy": "donchian breakout",
        "db_lookback": 85
    },
    30: {
        "strategy": "donchian breakout",
        "db_lookback": 26
    },
    47: {
        "strategy": "donchian breakout",
        "db_lookback": 147
    },

}

last_signals: Dict[int, int] = {
    instrument_no: 0 for instrument_no in allocated_instruments
}

last_positions: Dict[int, int] = {
    instrument_no: 0 for instrument_no in allocated_instruments
}

# HELPER FUNCTIONS ################################################################################
def donchian_breakout(prices_so_far: ndarray, instrument_no: int) -> int:
    instrument_price_history: Series = pd.Series(prices_so_far[instrument_no])

    # Get lookback from config
    lookback: int = config[instrument_no]["db_lookback"]

    if len(prices_so_far[instrument_no]) < lookback: return 0

    # Get upper and lower band
    upper: float = instrument_price_history.rolling(lookback - 1).max().shift(1).iloc[-1]
    lower: float = instrument_price_history.rolling(lookback - 1).min().shift(1).iloc[-1]

    # Compare current price to upper and lower band and return positions
    current_price: float = instrument_price_history.iloc[-1]

    # Get last signal, and change if regime changes
    last_signal: int = last_signals[instrument_no]
    signal: int = 0

    if current_price > upper:
        signal = 1
        last_signals[instrument_no] = 1
    elif current_price < lower:
        signal = -1
        last_signals[instrument_no] = -1

    if signal != 0 and signal == last_signal:
        return last_positions[instrument_no]
    elif signal != 0 and signal != last_signal:
        last_positions[instrument_no] = signal * int(positions_limit // current_price)
        return last_positions[instrument_no]
    else:
        return last_positions[instrument_no]

def ema_crossover(prices_so_far: ndarray, instrument_no: int) -> int:
    instrument_price_history: Series = pd.Series(prices_so_far[instrument_no])

    # Get lookback, and return 0 if current day is less than lookback
    fast_ema_lookback: int = config[instrument_no]["fast_ema_lookback"]
    slow_ema_lookback: int = config[instrument_no]["slow_ema_lookback"]

    if len(instrument_price_history) < slow_ema_lookback: return 0

    # Get fast MA and slow MA
    fast_ema: float = instrument_price_history.ewm(span=fast_ema_lookback, adjust=False).mean(
        ).iloc[-1]
    slow_ema: float = instrument_price_history.ewm(span=slow_ema_lookback, adjust=False).mean(
        ).iloc[-1]

    # Get a signal
    last_signal: int = last_signals[instrument_no]
    signal: int = 0

    if fast_ema > slow_ema:
        signal = 1
        last_signals[instrument_no] = 1
    elif fast_ema < slow_ema:
        signal = -1
        last_signals[instrument_no] = -1

    current_price: float = instrument_price_history.iloc[-1]

    if signal != 0 and signal == last_signal:
        return last_positions[instrument_no]
    elif signal != 0 and signal != last_signal:
        last_positions[instrument_no] = signal * int(positions_limit // current_price)
        return last_positions[instrument_no]
    else:
        return last_positions[instrument_no]


# MAIN STRATEGY FUNCTION ##########################################################################
def get_johns_positions(prices_so_far: ndarray) -> ndarray:
    for instrument_no in allocated_instruments:
        if config[instrument_no]["strategy"] == "donchian breakout":
            positions[instrument_no] = donchian_breakout(prices_so_far, instrument_no)
        elif config[instrument_no]["strategy"] == "ema crossover":
            positions[instrument_no] = ema_crossover(prices_so_far, instrument_no)

    return positions

