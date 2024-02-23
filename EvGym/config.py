import sys
from typing import Union
import numpy as np
import pandas as pd

# Dummy imports for stubs
import cvxpy as cp # type: ignore
import matplotlib.pyplot as plt # type: ignore

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

car_columns_proc = ["lax", "soc_dis", "t_dis", "idx_theta_w", "idx_theta_l"] # Columns that are processed on arrival
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
#eta_c = 1        # Charging efficiency
#eta_d = 1        # Discharging efficieny
eta_c = 0.98     # Charging efficiency
eta_d = 0.98     # Discharging efficiency
psi = (alpha_c * eta_c * eta_d)/(alpha_d + alpha_c * eta_c * eta_d) # Constant for maximum energy
#max_cars = 25    # Max cars in parking lot... Important that it is a constant number for RL purposes
max_cars = 30    # Max cars in parking lot... Important that it is a constant number for RL purposes
starttime_min = pd.to_datetime("2000-01-01 00:00:00") # Startime for absolute timestamps
#elec_retail = 0.14 # Retail price of electricity
elec_retail = 0.064 # Retail price of electricity
tol = 0.000001 # Tolerance for checking limits

# Paths
data_path = "data/"
results_path = "ExpLogs/"
agents_path = "Agents/"

# Contract parameters // This could be read later from YAML, but good enough for now
#thetas_i = [1/1.25, 1/1, 1/0.75]
#thetas_j = [1/1.25, 1/1, 1/0.75]
#c1 = 0.01
#c2 = 0.1
#kappa1 = 0.1
#kappa2 = 0.5
thetas_i = [0.75,1,1.25]
thetas_j = [0.75,1,1.25]

c1 = 0.01
c2 = 0.05
kappa1 = 0.4
kappa2 = 0.6
integer = True

# For synthetic data
sdg_pot = 1.8
sdg_norm = 0.28

# SAC
LOG_STD_MAX = 2
LOG_STD_MIN = -5
action_space_high = 1
action_space_low = 0

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

