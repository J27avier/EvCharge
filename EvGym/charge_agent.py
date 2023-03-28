import numpy as np
import pandas as pd
from . import config


class agentASAP():
    def __init__(self, max_cars = 25):
        self.max_cars = max_cars

    def get_action(self, df_state):
        action = np.zeros(self.max_cars)

        for i, (_, car) in enumerate(df_state.iterrows()):
            if car.idSess >= 0: 
                if car.soc_t < 1:
                    action[i] = min(config.alpha_c, 1-car.soc_t)

        return action

