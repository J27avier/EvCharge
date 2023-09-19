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
from EvGym.charge_agent import agentASAP, agentOptim, agentNoV2G, agentOracle
from EvGym import config

# Contracts
from ContractDesign.time_contracts import general_contracts

# ['session', 'ChargePoint', 'Connector', 'starttime_parking', 'endtime_parking', 'StartCard', 
#  'connected_time_float', 'charged_time_float', 'total_energy', 'max_power', 'start_hour',
# 'day_no', 'energy_supplied', 'initial_soc', 'charged_time', 'connected_time', 'ts_arr',
# 'ts_dep', 'ts_soj', 'laxity', 'depart_hour', 'xi'],
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-R", "--print-dash", help = "Print dashboard", action="store_true")
    parser.add_argument("-S", "--no-save", help="Does not save results csv", action="store_true")
    parser.add_argument("-C", "--save-contracts", help="Saves the contracts accepted to each car", action="store_true")
    parser.add_argument("-A", "--agent", help="Type of agent", type=str, required=True)
    parser.add_argument("-D", "--desc", help="Description of the expereiment, starting with \"_\"", type=str, default="")
    parser.add_argument("-E", "--seed", help="Seed to use for the rng", type=int, default=42)

    # Files
    parser.add_argument("-I", "--file-price", help = "Name of imbalance price dataframe", 
                        type=str, default= "df_price_2019.csv")
    parser.add_argument("-O", "--file-contracts", help = "CSV of contracts offered", 
                        type=str, default= "ExpLogs/2023-09-13-15:25:05_Contracts_ev_world_Optim.csv")
    parser.add_argument("-N", "--file-sessions", help = "CSV of charging sessions",
                        type=str, default= "df_elaad_preproc.csv")

    return parser.parse_args()

def print_welcome(df_sessions, df_price, contract_info):
    G, W, L = contract_info["G"], contract_info["W"], contract_info["L"]
    os.system("clear")
    print(Fore.BLUE, pyfiglet.figlet_format("Welcome to Ev Charge World"), Fore.RESET)
    print("df_sessions:")
    print(df_sessions.describe())
    print("="*80)
    print("df_price:")
    print(df_price.describe())
    print("Press Enter to begin...")
    print("="*80)
    print("Contracts")
    print("G ", G.shape)
    print(G)
    print("\nW", W.shape)
    print(W)
    print("\nL", L.shape)
    print(L)
    input()
    os.system("clear")

def main():
    args = parse_args()
    title = f"EvWorld {args.agent}{args.desc}"

    # Random number generator, same throught the program for reproducibility
    rng = np.random.default_rng(args.seed)

    # Load datasets
    df_sessions = pd.read_csv(f"{config.data_path}{args.file_sessions}", parse_dates = ["starttime_parking", "endtime_parking"])
    ts_min = df_sessions["ts_arr"].min()
    ts_max = df_sessions["ts_dep"].max()

    df_price = pd.read_csv(f"{config.data_path}{args.file_price}", parse_dates=["date"])

    # Calculate contracts
    G, W, L_cont = general_contracts(thetas_i = config.thetas_i,
                                     thetas_j = config.thetas_j,
                                     c1 = config.c1,
                                     c2 = config.c2,
                                     kappa1 = config.kappa1,
                                     kappa2 = config.kappa2,
                                     alpha_d = config.alpha_d,
                                     psi = config.psi,
                                     IR = "fst", IC = "ort_l", monotonicity=False) # Tractable formulation

    L = np.round(L_cont,0) # L_cont →  L continuous
    contract_info = {"G": G, "W": W, "L": L}

    # Some agents are not allowed to discharge energy
    skip_contracts = True if args.agent in ["ASAP", "NoV2G"] else False

    # Initialize objects
    world = ChargeWorldEnv(df_sessions, df_price, contract_info, rng, skip_contracts = skip_contracts)
    df_state = world.reset()

    # Declare agent
    if args.agent == "ASAP":
        agent = agentASAP()
    elif args.agent == "NoV2G":
        agent = agentNoV2G(df_price, myprint = True)
    elif args.agent == "Optim":
        agent = agentOptim(df_price, myprint = True)
    elif args.agent == "Oracle":
        df_contracts = pd.read_csv(f"{args.file_contracts}")
        agent = agentOracle(df_price, df_sessions, df_contracts, lookahead = 24, myprint = False)
    else:
        raise Exception(f"Agent name not recognized")

    # Print welcome screen
    if args.print_dash:
        print_welcome(df_sessions, df_price, contract_info)
        skips = 0

    # Environment loop
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
            # print("Tracker: ts, chg_e_req, imbalance_bill, n_cars, avg_lax")
            # print(world.tracker.chg_bill)

    if not args.no_save:
        world.tracker.save_log(args, path=config.results_path)
        world.tracker.save_desc(args, {"title": title}, path=config.results_path)

    if args.save_contracts:
        world.tracker.save_contracts(args, path=config.results_path)


if __name__ == "__main__":
    main()
