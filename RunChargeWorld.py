# Modules
import pandas as pd
import numpy as np
from tabulate import tabulate
import os
import pyfiglet # type: ignore
from colorama import init, Back, Fore
import argparse
from tqdm import tqdm

# User defined modules
from EvGym.charge_world import ChargeWorldEnv
from EvGym.charge_agent import agentASAP
from EvGym import config

# ['session', 'ChargePoint', 'Connector', 'starttime_parking', 'endtime_parking', 'StartCard', 
#  'connected_time_float', 'charged_time_float', 'total_energy', 'max_power', 'start_hour',
# 'day_no', 'energy_supplied', 'initial_soc', 'charged_time', 'connected_time', 'ts_arr',
# 'ts_dep', 'ts_soj', 'laxity', 'depart_hour', 'xi'],
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-R", "--print-dash", help = "Print dashboard", action="store_true")
    return parser.parse_args()


def main():
    title = "EvWorld"
    args = parse_args()

    df_sessions = pd.read_csv("data/df_elaad_preproc.csv", parse_dates = ["starttime_parking", "endtime_parking"])
    ts_max = df_sessions["ts_dep"].max()
    ts_min = df_sessions["ts_arr"].min()

    world = ChargeWorldEnv(df_sessions)
    df_state = world.reset()
    agent = agentASAP()

    if args.print_dash:
        os.system("clear")
        print(Fore.BLUE, pyfiglet.figlet_format("Welcome to Ev Charge World"), Fore.RESET)
        print(df_sessions.describe())
        print("Press Enter to begin...")
        input()
        os.system("clear")

    for _ in tqdm(range(int(ts_max-ts_min)), desc = f"{title}: "):
        action = agent.get_action(df_state)
        df_state, reward, done, info = world.step(action)

        if args.print_dash:
            world.print(-1, clear = True)

    world.tracker.save_log()

if __name__ == "__main__":
    main()
