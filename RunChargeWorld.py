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
from EvGym.charge_agent import agentASAP, agentOptim
from EvGym import config

# ['session', 'ChargePoint', 'Connector', 'starttime_parking', 'endtime_parking', 'StartCard', 
#  'connected_time_float', 'charged_time_float', 'total_energy', 'max_power', 'start_hour',
# 'day_no', 'energy_supplied', 'initial_soc', 'charged_time', 'connected_time', 'ts_arr',
# 'ts_dep', 'ts_soj', 'laxity', 'depart_hour', 'xi'],
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-R", "--print-dash", help = "Print dashboard", action="store_true")
    parser.add_argument("-I", "--file_price", help = "Name of imbalance price dataframe", 
                        type=str, default= "df_price_2019.csv")
    parser.add_argument("-S", "--no-save", help="Does not save results csv", action="store_true")
    parser.add_argument("-A", "--agent", help="Type of agent", type=str, required=True)
    parser.add_argument("-D", "--desc", help="Description of the expereiment, starting with \"_\"", type=str, default="")
    return parser.parse_args()

def main():
    args = parse_args()
    title = f"EvWorld {args.agent}"

    # Load datasets
    df_sessions = pd.read_csv(f"{config.data_path}df_elaad_preproc.csv", parse_dates = ["starttime_parking", "endtime_parking"])
    ts_max = df_sessions["ts_dep"].max()
    ts_min = df_sessions["ts_arr"].min()

    df_price = pd.read_csv(f"{config.data_path}{args.file_price}", parse_dates=["date"])

    # Initialize objects
    world = ChargeWorldEnv(df_sessions, df_price)
    df_state = world.reset()

    if args.agent == "ASAP":
        agent = agentASAP()
    elif args.agent == "No-V2G":
        agent = agentOptim(df_price)
    else:
        raise Exception(f"Agent name not recognized")

    # Print welcome screen
    if args.print_dash:
        os.system("clear")
        print(Fore.BLUE, pyfiglet.figlet_format("Welcome to Ev Charge World"), Fore.RESET)
        print("df_sessions:")
        print(df_sessions.describe())
        print("="*80)
        print("df_price:")
        print(df_price.describe())
        print("Press Enter to begin...")
        skips = 0
        input()
        os.system("clear")

    # Environment loop
    #for _ in tqdm(range(int(ts_max-ts_min)), desc = f"{title}: "):
    for t in tqdm(range(int(ts_min)-1, int(ts_max)), desc = f"{title}: "):
        action = agent.get_action(df_state, t)
        df_state, reward, done, info = world.step(action)
        assert t+1 == info['t'], "Main time and env time out of sync"

        if args.print_dash:
            if skips > 0: # Logic to jump forward
                skips -= 1
            else:
                usr_in = world.print(-1, clear = True)
            if usr_in.isnumeric():
                skips = int(usr_in)
                usr_in = ""
            print("Tracker: ts, chg_e_req, imbalance_bill, n_cars, avg_lax")
            print(world.tracker.imbalance_bill)

    if not args.no_save:
        world.tracker.save_log(args, path=config.results_path)
        world.tracker.save_desc(args, {"title": title}, path=config.results_path)

if __name__ == "__main__":
    main()
