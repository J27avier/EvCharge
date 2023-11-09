import numpy as np
import pandas as pd
import sys
import os

from . import config

def bounds_from_obs(obs):
    batch_size = obs.shape[0]
    np_lower = np.zeros((batch_size, config.max_cars))
    np_upper = np.zeros((batch_size, config.max_cars))
    np_obs = obs.cpu().numpy()

    for i in range(batch_size):
        data_state = np_obs[i, :config.max_cars*4]
        df_state = pd.DataFrame(data = data_state.reshape((config.max_cars, 4)),
                                columns = ["soc_t", "t_rem", "soc_dis", "t_dis"])

        occ_spots = df_state["t_rem"] > 0 # Occupied spots
        cont_spots = df_state["t_dis"] > 0 # Spots with contracts

        hat_y_low = config.FINAL_SOC-df_state["soc_t"] - config.alpha_c*config.eta_c*(df_state["t_rem"] - 1)/config.B
        y_low = np.maximum(hat_y_low * config.eta_c, hat_y_low / config.eta_d)
        y_low[~occ_spots] = 0

        dis_lim = np.zeros(config.max_cars)
        dis_lim[cont_spots] += -df_state[cont_spots]["soc_dis"]*config.eta_d

        lower = np.maximum(y_low, np.maximum(-config.alpha_d/config.B, dis_lim))
        lower[~occ_spots] = 0

        upper_soc = (config.FINAL_SOC - df_state["soc_t"]) / config.eta_c
        upper = np.minimum(upper_soc,  config.alpha_c / config.B)
        upper[~occ_spots] = 0
        
        np_lower[i] = lower
        np_upper[i] = upper

    return np_lower, np_upper

# Cvxpylayers has some warnings, but ok
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
