import numpy as np
import pandas as pd

# ######### Parameters #########
PRICES_FILE = "prices.txt"            # Path to historical price data
TRAIN_DAYS = 520                       # Number of days for initial model calibration
DLR_POS_LIMIT = 10_000                 # Maximum dollar exposure per asset
COMM_RATE = 0.0005                     # Commission rate (0.05% per round-trip)

# ######### Strategy Lookback & Thresholds #########
TF_LOOKBACK = 20                       # SMA window for Trend-Follow signals
BOLL_LOOKBACK = 20                     # Window for Bollinger Band statistics
BOLL_WIDTH = 2                         # Number of standard deviations for band width
RSI_PERIOD = 14                        # Lookback period for RSI calculation
RSI_LOWER = 30                         # RSI lower bound to signal oversold
RSI_UPPER = 70                         # RSI upper bound to signal overbought
MOM_LOOKBACK = 5                       # Lookback period for momentum strategy

# ######### Asset Groups #########
MEANREV_ASSETS = {43, 8, 13, 41, 31, 21, 24, 32, 7, 23, 3, 49, 35, 25, 40, 5, 11, 44, 14, 48, 39, 9, 19, 38, 12, 28, 15, 42, 33}
POSITIVE_TREND = {26, 6, 29, 16, 4, 30, 36, 17, 18}
ALL_ASSETS = set(range(50))
GOOD_BOLL = {2, 27, 46, 34, 20, 47, 10, 22, 1}
XOVER_ASSETS = {0}
RSI_ASSETS = {45}
SIMPLE_ASSETS = {37, 35}
MOM_ASSETS = ALL_ASSETS - MEANREV_ASSETS - POSITIVE_TREND - GOOD_BOLL - XOVER_ASSETS - RSI_ASSETS - SIMPLE_ASSETS

# ######### Data Loading & Cleaning #########
_raw = pd.read_csv(PRICES_FILE, delim_whitespace=True, header=None)
_raw = _raw[_raw.apply(lambda r: r.count() == 50, axis=1)]       # Drop rows missing data
prices = _raw.values.T                                            # Transpose: assets x days
nInst, nt = prices.shape
assert nInst == 50, f"Expected 50 assets, got {nInst}"

# ######### Calibration Phase #########
train_df = pd.DataFrame(prices[:, :TRAIN_DAYS])                   # Training window
mu = train_df.mean(axis=1)                                        # Per-asset mean
sigma = train_df.std(axis=1, ddof=0)                              # Per-asset stddev
rets_train = train_df.pct_change(axis=1).shift(-1, axis=1).iloc[:, :-1]

# ######### Mean-Reversion Threshold Optimization #########
thresholds = {}
for a in MEANREV_ASSETS:
    z_scores = (train_df.loc[a] - mu[a]) / sigma[a]
    best_sharpe, best_thresh = -np.inf, 1.0
    for t in np.arange(0.5, 3.01, 0.1):
        signals = pd.Series(0, index=z_scores.index)
        signals[z_scores > t] = -1
        signals[z_scores < -t] = 1
        pnl = (signals.shift(1) * rets_train.loc[a]).dropna()
        m, s = pnl.mean(), pnl.std(ddof=0)
        sharpe = (m/s) * np.sqrt(252) if s > 0 else 0
        if sharpe > best_sharpe:
            best_sharpe, best_thresh = sharpe, t
    thresholds[a] = best_thresh

# ######### Position Function #########
def getMyPosition(prcHistSoFar: np.ndarray) -> np.ndarray:
    """
    Compute positions for each asset under various strategies.
    """
    df = pd.DataFrame(prcHistSoFar)
    today_prices = df.iloc[:, -1]
    sig = pd.Series(0, index=df.index, dtype=int)

    # ### Mean-Reversion Signals ###
    z_today = (today_prices - mu) / sigma
    for a in MEANREV_ASSETS:
        t = thresholds[a]
        if z_today[a] > t:
            sig[a] = -1
        elif z_today[a] < -t:
            sig[a] = 1

    hist = df.T

    # ### Trend-Follow Signals ###
    sma20 = hist.rolling(TF_LOOKBACK).mean().iloc[-1]
    for a in POSITIVE_TREND:
        sig[a] = 1 if today_prices[a] > sma20[a] else -1

    # ### Bollinger Bands Signals ###
    roll_mu = hist.rolling(BOLL_LOOKBACK).mean().iloc[-1]
    roll_std = hist.rolling(BOLL_LOOKBACK).std(ddof=0).iloc[-1]
    for a in GOOD_BOLL:
        if today_prices[a] > roll_mu[a] + BOLL_WIDTH * roll_std[a]:
            sig[a] = -1
        elif today_prices[a] < roll_mu[a] - BOLL_WIDTH * roll_std[a]:
            sig[a] = 1

    # ### MA Crossover Signal for Asset 0 ###
    short_ma, long_ma = hist[0].rolling(10).mean().iloc[-1], hist[0].rolling(50).mean().iloc[-1]
    sig[0] = 1 if short_ma > long_ma else -1

    # ### RSI Signals for Asset 45 ###
    delta = hist[45].diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = -delta.clip(upper=0).rolling(RSI_PERIOD).mean()
    rsi_val = (100 - (100/(1 + (gain/loss)))).iloc[-1]
    if rsi_val < RSI_LOWER:
        sig[45] = 1
    elif rsi_val > RSI_UPPER:
        sig[45] = -1

    # ### Buy-and-Hold for Simple Assets ###
    for a in SIMPLE_ASSETS:
        sig[a] = 1

    # ### Momentum Signals ###
    momentum = hist.diff(MOM_LOOKBACK).iloc[-1]
    for a in MOM_ASSETS:
        sig[a] = 1 if momentum[a] > 0 else -1

    # ######### Position Sizing #########
    positions = (DLR_POS_LIMIT / today_prices) * sig
    return positions.values

# ######### Williams %R Placeholder #########
def get_williams_positions(prices_so_far: np.ndarray) -> np.ndarray:
    return np.zeros(50)
