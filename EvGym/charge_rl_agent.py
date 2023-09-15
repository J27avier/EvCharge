import numpy as np
import cvxpy as cp # type: ignore
import pandas as pd
from . import config
from typing import TYPE_CHECKING, Any
import numpy.typing as npt

np.set_printoptions(linewidth=np.nan) # type: ignore

class agentPPO():
    def __init__(self, max_cars: int = config.max_cars):
        self.max_cars = max_cars

    def get_action(self, df_state: pd.DataFrame, t: int = 0):
        return 0
