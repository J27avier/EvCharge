import numpy as np
import pandas as pd
from . import config
from typing import TYPE_CHECKING, Any
import numpy.typing as npt

class agentASAP():
    def __init__(self, max_cars: int = config.max_cars):
        self.max_cars = max_cars

    def get_action(self, df_state: pd.DataFrame) -> npt.NDArray[Any]:
        action = np.zeros(self.max_cars)

        for i, (_, car) in enumerate(df_state.iterrows()):
            if car.idSess >= 0: 
                if car.soc_t < config.FINAL_SOC:
                    action[i] = min(config.alpha_c/config.B, (config.FINAL_SOC-car.soc_t)/config.eta_c)

        return action

