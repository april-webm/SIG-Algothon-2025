from typing import List, TypedDict, Literal
import sys

import pandas as pd
from pandas import DataFrame, Series

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

# CONSTANT VARIABLES #############################################################################
START_DATE: int = 0
END_DATE: int = 300
NUMBER_OF_INSTRUMENTS: int = 50

# Standard Deviation Lookbacks
SHORT_TERM_STDDEV_LOOKBACK: int = 10
MONTHLY_STDDEV_LOOKBACK: int = 21
QUARTERLY_STDDEV_LOOKBACK: int = 60
YEARLY_STDDEV_LOOKBACK: int = 252

# Auto correlation default lag limit
default_lag_limit: int = 20

# Usage Error message
usage_error_message: str = """
    Usage: analysis.py [instrument_no] [start_day] [end_day] [OPTION]
    
    OPTIONS:
    --price - plot price chart
        --sma [lookback] - plot simple moving average with the price plot (must specify --price)
        --ema [lookback] - plot exponential moving average with the price plot (must specify 
        --price)
    --acf [lag_limit] - plot auto-correlation with lags 1 to lag_limit
    --volatility [lookback] - plot volatility
    --returns - plot returns distribution
    
    Examples:
        analysis.py 5 0 500 -all
        analysis.py 20 100 150 --price 
        analysis.py 1 200 400 --price --sma 10
    
    NOTES:
    - You can only specify a max of 1 option
    - start_day must be greater than or equal to 0
    - end_day must be less than or equal to 500
    - start_day must be less than end_day
"""

# UTILITY SCHEMAS ###############################################################################
class PricePlotOptions(TypedDict):
    moving_average: Literal['none', 'simple', 'exponential']
    moving_average_periods: int

# CLASSES ########################################################################################
class MarketData:
    def __init__(self, start_day: int, end_day: int, prices_filepath: str) -> None:
        # Initialise the market data dataframe
        self.market_data_df: DataFrame | None = None
        self.initialise_market_data_dataframe(start_day, end_day, prices_filepath)

    def get_instrument_data(self, instrument_no: int, start_day: int, end_day: int) -> DataFrame:
        # Filter rows matching the instrument and day range and sort by day
        instrument_data: DataFrame = self.market_data_df[
            (self.market_data_df["instrument-no"] == instrument_no) &
            (self.market_data_df["day"] >= start_day) & (self.market_data_df["day"] <= end_day)
            ].sort_values(by="day")

        return instrument_data

    """
    loads the prices.txt file and initialises the market data dataframe with the following 
    columns:

    - day: day number
    - instrument-no: the instrument number
    - open-price: the price at market open for that instrument on that day
    - return: the percentage change for that instrument on that day based on the previous day
    """
    def initialise_market_data_dataframe(self, start_day, end_day, prices_filepath: str) -> None:
        # Read the raw prices dataset file
        raw_prices_df: DataFrame = pd.read_csv(prices_filepath, sep=r"\s+", header=None)

        # Initialise a new empty market data dataframe
        market_data_df: DataFrame = pd.DataFrame(columns=["day", "instrument-no", "open-price",
                                                          "change-pct"])

        for day in range(start_day, end_day + 1):
            for instrument_no in range(0,NUMBER_OF_INSTRUMENTS):
                new_entry: List[int, int, float, float] = []
                new_entry.append(day)
                new_entry.append(instrument_no)
                new_entry.append(raw_prices_df.iloc[day, instrument_no])

                # If it has at least been 1 day, add returns
                if day > 0:
                    change_pct: float = raw_prices_df.iloc[day, instrument_no] / raw_prices_df.iloc[
                        day - 1, instrument_no]
                    new_entry.append(change_pct)
                else:
                    new_entry.append(0)

                # Add the new entry to the dataframe
                market_data_df.loc[day*NUMBER_OF_INSTRUMENTS + instrument_no] = new_entry

        self.market_data_df = market_data_df

    """
    plots an instrument's price data over a specified timeline. You can also supply an
    'PricePlotOptions' dictionary to provide extra specifications when plotting  (i.e. embedding
    a simple/exponential moving average) 
    """
    def plot_instrument_price_data(self, instrument_no: int, start_day: int, end_day: int,
                                   options: PricePlotOptions = None) -> None:
        instrument_data: DataFrame = self.get_instrument_data(instrument_no, start_day, end_day)

        days: Series = instrument_data["day"]
        prices: Series = instrument_data["open-price"]

        # If a moving average was specified
        moving_average: Series
        if options is not None and options["moving_average"] != "none":
            # if it's a simple moving average
            if options["moving_average"] == "simple":
                moving_average= prices.rolling(window=options["moving_average_periods"]).mean()
            else:
                moving_average = prices.ewm(span=options["moving_average_periods"],
                                            adjust=False).mean()


        # Plot
        plt.figure(figsize=(10,5))
        plt.plot(days, prices, linestyle="-")
        if options is not None and options["moving_average"] != "none":
            plt.plot(days, moving_average, color="orange")
        plt.xlabel("Day")
        plt.ylabel("Open Price")
        plt.title(f"Instrument {instrument_no}: Price from Day {start_day} to {end_day}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    """
    plots an instrument's volatility over a specified timeline and a given lookback. Function
    assumes that start_day >= lookback
    """
    def plot_instrument_volatility(self, instrument_no: int, start_day: int, end_day: int,lookback:
    int) -> None:
        instrument_data: DataFrame = self.get_instrument_data(instrument_no, start_day, end_day)

        # Compute returns from price series
        prices: Series = instrument_data["open-price"]
        returns: Series = prices.pct_change().fillna(0)

        # Compute rolling standard deviation over lookback window
        rolling_std: Series = returns.rolling(window=lookback).std()

        # Restrict only the days >= start_day
        rolling_std = rolling_std[instrument_data["day"] >= start_day]
        days: ndarray = np.arange(start_day, end_day)

        # Plot volatility
        plt.figure(figsize=(10,5))
        plt.plot(days, rolling_std, linestyle="-")
        plt.xlabel("Day")
        plt.ylabel("Volatility")
        plt.title(f"Instrument {instrument_no}: Volatility from Day"
                  f" {start_day} to {end_day} with lookback of {lookback} days")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    """
    plots as instrument's autocorrelation over a specified timeline with lags 1 to a specified
    max lag limit
    """
    def plot_instrument_autocorrelation(self, instrument_no: int, start_day: int, end_day: int,
                                        lag_limit: int) -> None:
        instrument_data: DataFrame = self.get_instrument_data(instrument_no, start_day, end_day)

        mean: float = instrument_data["open-price"].mean()

        # Form demeaned series
        prices: ndarray = instrument_data["open-price"]
        demeaned = prices - mean

        # Compute denominator
        denominator: float = np.dot(demeaned,demeaned)

        # Compute lag-k autocorrelations
        autocorrelations: ndarray = np.array([np.dot(demeaned[k:], demeaned[:-k]) / denominator
                                              for k in range (1, lag_limit+1)])

        # Set up series of lags from 1 to lag_limit
        lags: ndarray = np.arange(1, lag_limit + 1)

        # Plot
        plt.figure(figsize=(10,5))
        plt.plot(lags, autocorrelations, linestyle="-")
        plt.xlabel("K")
        plt.ylabel("Auto-Correlation")
        plt.title(f"Instrument {instrument_no}: Auto-correlation from day {start_day} to "
                  f"{end_day} with lags 1 to {lag_limit}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    """
    Plots an instruments returns distribution over a specified timeline
    """
    def plot_instrument_returns_distribution(self, instrument_no: int, start_day: int,
                                             end_day: int) -> None:
        instrument_data: DataFrame = self.get_instrument_data(instrument_no, start_day, end_day)

        # Get arithmetic returns
        prices: ndarray = instrument_data["open-price"]
        returns: ndarray = np.diff(prices) / prices[:-1]

        # Plot returns distribution
        plt.figure(figsize=(10,5))
        plt.hist(returns, bins=20, density=True, alpha=0.7, edgecolor="black")
        plt.xlabel("Daily Return")
        plt.ylabel("Probability Density")
        plt.title(f"Instrument {instrument_no}: Distribution of daily returns from day "
                  f"{start_day} to {end_day}")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

# MAIN EXECUTION #################################################################################
def main() -> None:
    # Get command line arguments
    total_args: int = len(sys.argv)

    if total_args < 5:
        print(usage_error_message)
    else:
        instrument_no: int = int(sys.argv[1])
        start_day: int = int(sys.argv[2])
        end_day: int = int(sys.argv[3])

        market_data = MarketData(start_day, end_day)
        option: str = sys.argv[4]

        if option == "--price":
            if total_args >= 6:
                moving_average_arg = sys.argv[5]
                if moving_average_arg == "--sma" or moving_average_arg == "--ema":
                    if 7 < total_args:
                        print(usage_error_message)
                        return
                    else:
                        lookback: int = int(sys.argv[6])
                        options: PricePlotOptions = PricePlotOptions()
                        options["moving_average_periods"] = lookback
                        options["moving_average"] = "simple" if moving_average_arg == "sma" else (
                            "exponential")
                        market_data.plot_instrument_price_data(instrument_no, start_day,
                                                               end_day, options)
                else:
                    print(usage_error_message)
                    return
            else:
                market_data.plot_instrument_price_data(instrument_no, start_day, end_day)
        elif option == "--acf":
            if total_args < 6:
                lag_limit: int = int(sys.argv[5])
                market_data.plot_instrument_autocorrelation(instrument_no, start_day, end_day,
                                                            lag_limit)
            else:
                print(usage_error_message)
                return
        elif option == "--volatility":
            if total_args < 6:
                lookback: int = int(sys.argv[5])
                market_data.plot_instrument_volatility(instrument_no, start_day, end_day,
                                                       lookback)
            else:
                print(usage_error_message)
                return
        elif option == "--returns":
            market_data.plot_instrument_returns_distribution(instrument_no, start_day,
                                                             end_day)
        else:
            print(usage_error_message)
            return

main()