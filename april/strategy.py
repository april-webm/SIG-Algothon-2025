import numpy as np
import pandas as pd


# All of these pairs and ratios were calculated in a notebook using OLS.
# Currently, these are not BLUE, meaning the estimates are not optimal.
# This is because of heteroskedasticity and serial correlation being present 
# for majority of the assets.
# I am working on a fix.
STATIC_PAIRS = [
    {'pair': (48, 49), 'hedge_ratio': 0.861531},
    {'pair': (8, 34), 'hedge_ratio': -0.478832},
    {'pair': (11, 23), 'hedge_ratio': 0.488072},
    {'pair': (2, 22), 'hedge_ratio': -0.701530},
    {'pair': (18, 30), 'hedge_ratio': 0.393543},
    {'pair': (29, 42), 'hedge_ratio': 0.965483},
    {'pair': (9, 31), 'hedge_ratio': 1.878606},
    {'pair': (7, 13), 'hedge_ratio': 1.295080},
    {'pair': (25, 37), 'hedge_ratio': 0.825191},
    {'pair': (3, 44), 'hedge_ratio': 0.376606},
    {'pair': (26, 45), 'hedge_ratio': 1.568608},
    {'pair': (0, 40), 'hedge_ratio': 0.295217},
    {'pair': (20, 35), 'hedge_ratio': 0.920088},
    {'pair': (15, 27), 'hedge_ratio': 0.333482},
    {'pair': (19, 33), 'hedge_ratio': 0.702839},
    {'pair': (41, 46), 'hedge_ratio': 0.286714}
]

# CONSTANTS
LOOKBACK_DAYS = 90    
Z_ENTRY_THRESHOLD = 0.17 # Z-score level to enter a trade
POSITION_SIZE = 10000

def get_aprils_positions(prcSoFar, johns_trades):
    """
    Calculates the desired position for each instrument based on a rolling pairs trading strategy.
    
    Args:
        prcSoFar (numpy.ndarray): A 2D array of prices, where rows are instruments (50)
                                  and columns are days.
    
    Returns:
        numpy.ndarray: A 1D array of length 50 with the desired integer positions.
    """
    # Initialize our desired positions to zero for all 50 instruments
    final_positions = np.zeros(50)

    current_day_index = prcSoFar.shape[1]
    
    if current_day_index < LOOKBACK_DAYS:
        return final_positions
    for pair_info in STATIC_PAIRS:
        asset1_idx, asset2_idx = pair_info['pair']
        hedge_ratio = pair_info['hedge_ratio']
        
        # Get the price history for the pair over the lookback window
        asset1_prices = prcSoFar[int(asset1_idx), -LOOKBACK_DAYS:]
        asset2_prices = prcSoFar[int(asset2_idx), -LOOKBACK_DAYS:]

        # Calculate the spread over the lookback window
        spread_series = asset1_prices - hedge_ratio * asset2_prices
        
        if spread_series.std() > 1e-6: # Avoid division by zero
            rolling_mean = spread_series.mean()
            rolling_std = spread_series.std()
            current_spread_value = spread_series[-1]
            z_score = (current_spread_value - rolling_mean) / rolling_std
            
            # Get the most recent prices for position sizing
            current_price_asset1 = prcSoFar[int(asset1_idx), -1]

            # Position sizing to get integer number of shares
            position_asset1 = POSITION_SIZE / current_price_asset1

            # Readjust for john's trades
            if johns_trades[asset1_idx] != 0: position_asset1 = johns_trades[asset1_idx]
            
            # If z-score is high, we want a short position on the spread
            if z_score > Z_ENTRY_THRESHOLD:
                final_positions[int(asset1_idx)] = -position_asset1
                final_positions[int(asset2_idx)] = position_asset1 * hedge_ratio

                # Readjust for john's trades
                if johns_trades[asset2_idx] != 0: final_positions[asset2_idx] = 0
            
            # If the z-score is low, we want a long position on the spread
            elif z_score < -Z_ENTRY_THRESHOLD:
                final_positions[int(asset1_idx)] = position_asset1
                final_positions[int(asset2_idx)] = -position_asset1 * hedge_ratio

                # Readjust for john's trades
                if johns_trades[asset2_idx] != 0: final_positions[asset2_idx] = 0
                
    return final_positions.astype(int)
