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
eta_c = 0.98     # Charging efficiency
eta_d = 0.98     # Discharging efficiency
psi = (alpha_c * eta_c * eta_d)/(alpha_d + alpha_c * eta_c * eta_d)
