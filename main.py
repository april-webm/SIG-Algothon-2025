from pandas import DataFrame
from typing import List
from numpy import ndarray
import numpy as np

from april.strategy import get_aprils_positions
from john.strategy import get_johns_positions
from aaryan.strategy import get_aaryans_positions
from inuka.strategy import get_inukas_positions

# CONSTANTS ######################################################################################

nInst = 50
currentPos = np.zeros(nInst)

# GET POSITIONS FUNCTION #########################################################################
def getMyPosition(prcSoFar: ndarray) -> ndarray:
    johns_trades: ndarray = get_johns_positions(prcSoFar)
    aprils_trades: ndarray = get_aprils_positions(prcSoFar, johns_trades)
    inukas_trades: ndarray = get_inukas_positions(prcSoFar)
    aaryans_trades: ndarray = get_aaryans_positions(prcSoFar)
    all_trades: ndarray = aprils_trades + johns_trades + inukas_trades + aaryans_trades
    return all_trades


