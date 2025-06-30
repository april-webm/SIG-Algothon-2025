from numpy import ndarray
import numpy as np


from april.strategy import get_aprils_positions
from john.strategy import get_johns_positions
from william.strategy import get_williams_positions

try:
    from strategy_allocations import APRILS_ASSETS, JOHNS_ASSETS, WILLIAMS_ASSETS
except ImportError:
    print("FATAL ERROR: 'strategy_allocations.py' not found.")
    print("Please run 'create_allocations.py' first to generate the allocations.")
    # Define empty lists as a fallback to prevent crashing
    APRILS_ASSETS, JOHNS_ASSETS, WILLIAMS_ASSETS = [], [], []

# GET POSITIONS FUNCTION #########################################################################
def getMyPosition(prcSoFar: ndarray) -> ndarray:
    """
    Constructs the final portfolio by intelligently combining trades from three
    different strategies, using a pre-optimized allocation map to assign each
    asset to its best-performing model.
    """
    # Run all three strategies to get their proposed trades for all assets
    aprils_trades: ndarray = get_aprils_positions(prcSoFar)
    johns_trades: ndarray = get_johns_positions(prcSoFar)
    williams_positions: ndarray = get_williams_positions(prcSoFar)
    
    # Initialize our final positions array
    final_positions = np.zeros(50)
    
    np.put(final_positions, APRILS_ASSETS, aprils_trades.take(APRILS_ASSETS))
    np.put(final_positions, JOHNS_ASSETS, johns_trades.take(JOHNS_ASSETS))
    np.put(final_positions, WILLIAMS_ASSETS, williams_positions.take(WILLIAMS_ASSETS))
    
    return final_positions.astype(int)
