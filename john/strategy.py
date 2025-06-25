import pandas as pd
import numpy as np

from numpy import ndarray
from typing import List, Dict
from pandas import Series

# CONSTANTS AND CONFIG ############################################################################
positions_limit: int = 10000
allocated_instruments: List[int] = [6, 10, 14, 16, 17, 20, 28, 32, 36, 39]

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
    6: {
        "strategy": "donchian breakout",
        "db type": "last loser",
        "db lookback": 54
    },
    10: {
        "strategy": "donchian breakout",
        "db type": "last loser",
        "db lookback": 99
    },
    14: {
        "strategy": "donchian breakout",
        "db type": "last winner",
        "db lookback": 146
    },
    16: {
        "strategy": "donchian breakout",
        "db type": "last winner",
        "db lookback": 24
    },
    17: {
        "strategy": "donchian breakout",
        "db type": "last winner",
        "db lookback": 12
    },
    20: {
        "strategy": "donchian breakout",
        "db type": "last loser",
        "db lookback": 123
    },
    28: {
        "strategy": "donchian breakout",
        "db type": "last winner",
        "db lookback": 23
    },
    32: {
        "strategy": "donchian breakout",
        "db type": "last loser",
        "db lookback": 16
    },
    36: {
        "strategy": "donchian breakout",
        "db type": "last winner",
        "db lookback": 12
    },
    39: {
        "strategy": "donchian breakout",
        "db type": "last winner",
        "db lookback": 13
    },
}

# HELPER FUNCTIONS ################################################################################
def donchian_breakout(prices_so_far: ndarray, instrument_no: int) -> int:
    # Get Price History
    instrument_price_history: Series = pd.Series(prices_so_far[instrument_no])

    # Get Lookback and DB Type from config
    lookback: int = config[instrument_no]["db lookback"]
    db_type: str = config[instrument_no]["db type"]

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


# STRATEGY FUNCTION ###############################################################################
positions: ndarray = np.zeros(50)

def get_johns_positions(prices_so_far: ndarray) -> ndarray:
    for instrument_no in allocated_instruments:
        if config[instrument_no]["strategy"] == "donchian breakout":
            positions[instrument_no] = donchian_breakout(prices_so_far, instrument_no)

    return positions