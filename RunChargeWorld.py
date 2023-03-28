# Modules
import pandas as pd
import numpy as np
from tabulate import tabulate
from IPython.display import display
import os
import pyfiglet
from colorama import init, Back, Fore

# User defined modules
from EvGym.charge_world import ChargeWorldEnv
from EvGym.charge_agent import agentASAP
from EvGym import config

# ['session', 'ChargePoint', 'Connector', 'starttime_parking', 'endtime_parking', 'StartCard', 
#  'connected_time_float', 'charged_time_float', 'total_energy', 'max_power', 'start_hour',
# 'day_no', 'energy_supplied', 'initial_soc', 'charged_time', 'connected_time', 'ts_arr',
# 'ts_dep', 'ts_soj', 'laxity', 'depart_hour', 'xi'],
def main():
    os.system("clear")
    #print(config.bcolors.OKCYAN)
    print(Fore.BLUE)
    print(pyfiglet.figlet_format("Welcome to Ev Charge World"))
    #print(config.bcolors.ENDC)
    print(Fore.RESET)
    df_sessions = pd.read_csv("data/df_elaad_preproc.csv", parse_dates = ["starttime_parking", "endtime_parking"])
    print(df_sessions.head())
    display(df_sessions.head())
    print(df_sessions.columns)
    world = ChargeWorldEnv(df_sessions)
    df_state = world.reset()
    agent = agentASAP()
    print("Press Enter to begin...")
    input()
    os.system("clear")

    for i in range(df_sessions.shape[0]):
        action = agent.get_action(df_state)
        df_state, reward, done, info = world.step(action)
        world.print(-1, clear = True)









if __name__ == "__main__":
    main()
