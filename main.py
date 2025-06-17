from pandas import DataFrame
from typing import List
from numpy import ndarray
import numpy as np

from april.strategy import get_aprils_positions
from john.strategy import get_johns_positions
from aaryan.strategy import get_aaryans_positions
from inuka.strategy import get_inukas_positions

# CONSTANTS ######################################################################################
# Instrument Allocations
APRILS_INSTRUMENTS: List[int] = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
JOHNS_INSTRUMENTS: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 22, 23, 24, 26, 27]
AARYANS_INSTRUMENTS: List[int] = [8,25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
INUKAS_INSTRUMENTS: List[int] = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]


nInst = 50
currentPos = np.zeros(nInst)

# GET POSITIONS FUNCTION #########################################################################
def getMyPosition(prcSoFar: DataFrame) -> ndarray:
    print(prcSoFar)
    aprils_trades: ndarray = get_aprils_positions(prcSoFar, APRILS_INSTRUMENTS)
    johns_trades: ndarray = get_johns_positions(prcSoFar, JOHNS_INSTRUMENTS)
    inukas_trades: ndarray = get_inukas_positions(prcSoFar, INUKAS_INSTRUMENTS)
    aaryans_trades: ndarray = get_aaryans_positions(prcSoFar, AARYANS_INSTRUMENTS)
    all_trades: ndarray = aprils_trades + johns_trades + inukas_trades + aaryans_trades
    return all_trades;


