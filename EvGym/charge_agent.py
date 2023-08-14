import numpy as np
import cvxpy as cp
import pandas as pd
from . import config
from typing import TYPE_CHECKING, Any
import numpy.typing as npt

class agentASAP():
    def __init__(self, max_cars: int = config.max_cars):
        self.max_cars = max_cars

    def get_action(self, df_state: pd.DataFrame, t: int = 0) -> npt.NDArray[Any]:
        """
        Charge action ASAP
        df_state: State of the parking lot
        t: Unused (left for consistency with other classes)
        """
        action = np.zeros(self.max_cars)

        for i, (_, car) in enumerate(df_state.iterrows()):
            if car.idSess >= 0: 
                if car.soc_t < config.FINAL_SOC:
                    action[i] = min(config.alpha_c/config.B, (config.FINAL_SOC-car.soc_t)/config.eta_c)

        return action

class agentOptim():
    def __init__(self, df_price, max_cars: int = config.max_cars):
        self.max_cars = max_cars
        self.df_price = df_price

    def _get_prediction(self, t, n):
        l_idx_t0 = self.df_price.index[self.df_price["ts"] == t].to_list()
        assert len(l_idx_t0) == 1, "Timestep for prediction not unique or not existent"
        idx_t0 = l_idx_t0[0]
        idx_tend = min(idx_t0+n, self.df_price.index.max()+1)
        pred_price = self.df_price.iloc[idx_t0:idx_tend]["price_im"].values
        return pred_price

    def get_action(self, df_state: pd.DataFrame, t: int) -> npt.NDArray[Any]:
        action = np.zeros(self.max_cars)
        occ_spots = df_state["idSess"] != -1 # Occupied spots
        num_cars = occ_spots.sum()
        if num_cars > 0:
            t_end = df_state[occ_spots]["t_dep"].max()
            n = t_end - t + 1
            pred_price = self._get_prediction(t, n)
            Y = cp.Variable((num_cars, n), nonneg=True)
            SOC = cp.Variable((num_cars, n), nonneg=True)
        
            
            # Charge asap
            for i, (_, car) in enumerate(df_state.iterrows()):
                if car.idSess >= 0: 
                    if car.soc_t < config.FINAL_SOC:
                        action[i] = min(config.alpha_c/config.B, (config.FINAL_SOC-car.soc_t)/config.eta_c)
        return action
