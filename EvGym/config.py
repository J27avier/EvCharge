import sys
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
psi = (alpha_c * eta_c * eta_d)/(alpha_d + alpha_c * eta_c * eta_d)

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
