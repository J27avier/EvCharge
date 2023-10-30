import numpy as np
import pandas as pd

from . import config


def bounds_from_obs(obs):
    np_obs = obs.cpu().numpy()
    data_state = np_obs[0, :config.max_cars*4]
    df_state = pd.DataFrame(data = data_state.reshape((config.max_cars, 4)), columns = ["soc_t", "t_rem", "soc_dis", "t_dis"])

    occ_spots = df_state["t_rem"] > 0 # Occupied spots

    hat_y_low = config.FINAL_SOC-df_state["soc_t"] - config.alpha_c*config.eta_c*(df_state["t_rem"] - 1)/config.B
    y_low = np.minimum(hat_y_low * config.eta_c, hat_y_low / copnfig.eta_d)
    y_low[~occ_spots] = 0

    lower = np.maximum(y_low, np.maximum(-config.alpha_d/config.B,
                                              -df_state["soc_dis"]))
    lower[~occ_spots] = 0

    upper_soc = (config.FINAL_SOC - df_state["soc_t"]) / config.eta_c
    upper = np.minimum(upper_soc,  config.alpha_c / config.B)
    upper[~occ_spots] = 0
    return lower, upper
