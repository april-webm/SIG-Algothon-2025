from typing import List, Dict
from numpy import ndarray
import numpy as np
import pandas as pd
from pandas import Series

positions_limit: int = 10000
allocated_instruments: List[int] =[23]
positions: ndarray = np.zeros(50)

# HELPER FUNCTIONS ################################################################################

# MAIN STRATEGY FUNCTION ##########################################################################
def get_johns_position(prices_so_far: ndarray, config: Dict[int, Dict[str, Dict[str, float
]]] | None = None, instruments_to_test: List[int] | None = None) -> ndarray:
    lookback: int = 120
    if len(prices_so_far[0]) < lookback: return np.zeros(50)

    for instrument_no in allocated_instruments:
        price_series: Series = pd.Series(prices_so_far[instrument_no])

        # Calculate upper and lower Band
        upper: float = price_series.rolling(lookback-1).max().shift(1).iloc[-1]
        lower: float = price_series.rolling(lookback-1).min().shift(1).iloc[-1]

        # General positions
        current_price: float = price_series.iloc[-1]
        if current_price > upper:
            positions[instrument_no] = positions_limit // price_series.iloc[-1]
        elif current_price < lower:
            positions[instrument_no] = positions_limit // price_series.iloc[-1]

    return positions

# GRID SEARCHER ###################################################################################
def grid_search() -> None:
   pass


# MAIN EXECUTION ##################################################################################
if __name__ == "__main__":
    grid_search()

