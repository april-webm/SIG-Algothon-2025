from numpy import ndarray
import numpy as np

from april.strategy import get_aprils_positions
from john.strategy import get_johns_positions
from william.strategy import get_williams_positions

# CONSTANTS ######################################################################################

nInst = 50
currentPos = np.zeros(nInst)

# GET POSITIONS FUNCTION #########################################################################
def getMyPosition(prcSoFar: ndarray) -> ndarray:
    johns_trades: ndarray = get_johns_positions(prcSoFar)
    aprils_trades: ndarray = get_aprils_positions(prcSoFar)
    williams_positions: ndarray = get_williams_positions(prcSoFar)
    all_trades: ndarray = aprils_trades + johns_trades + williams_positions
    return all_trades


