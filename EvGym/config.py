import sys
from typing import Union
import numpy as np
import pandas as pd

# Dummy imports for stubs
import yaml
import scipy # type: ignore
import cvxpy as cp # type: ignore
import matplotlib.pyplot as plt # type: ignore
from scipy.stats import truncnorm # type: ignore

# 
Number = Union[int, float, np.number]
maxint = sys.maxsize
car_columns_full = ["idSess",
                     "B",
                     "t_arr",
                     "soc_arr",
                     "E_arr",
                     "t_dep",
                     "E_rem",
                     "soc_rem",
                     "E_t",
                     "soc_t",
                     "t_rem",
                     ]

car_columns_proc = ["t_dis", "soc_dis", "lax", "type_w", "type_l"] # Columns that are processed on arrival
car_columns_proc_default = [0, 0, 0, -1, -1] # Default values

car_columns_full_lag = ["idSess",
                     "B",
                     "t_arr",
                     "soc_arr",
                     "E_arr",
                     "t_dep",
                     "E_rem",
                     "soc_rem",
                     "E_t",
                     "soc_t",
                     "soc_lag",
                     "t_rem",
                     ]

car_columns_simple = ["idSess", "t_rem", "soc_rem"]

# Elaad Preprocessing
elaad_rename = {"TransactionId": "session",
                  "UTCTransactionStart": "starttime_parking",
                  "UTCTransactionStop": "endtime_parking",
                  "TotalEnergy": "total_energy",
                  "ConnectedTime": "connected_time_float",
                  "ChargeTime": "charged_time_float",
                  "MaxPower": "max_power",
                }

timestep = 60*60 # in seconds, this is 1 hour

# EV Session Characteristics
B = 80           # Battery capacity in kWh
alpha_c = 11     # Max Charging rate in kW
alpha_d = 11     # Max Discharging rate in kW
FINAL_SOC = 0.97 # Final SOC, for preprocessing elaad
MIN_SOC = 0      # Minimum SOC
eta_c = 0.98     # Charging efficiency
eta_d = 0.98     # Discharging efficiency
psi = (alpha_c * eta_c * eta_d)/(alpha_d + alpha_c * eta_c * eta_d) # Constant for maximum energy
max_cars = 25    # Max cars in parking lot... Important that it is a constant number for RL purposes
starttime_min = pd.to_datetime("2000-01-01 00:00:00") # Startime for absolute timestamps
elec_retail = 0.14 # Retail price of electricity
tol = 0.000001 # Tolerance for checking limits

# Paths
data_path = "data/"
results_path = "ExpLogs/"


# Colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
