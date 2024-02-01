# Modules
import pandas as pd
import numpy as np
from tabulate import tabulate
import os
import argparse
from tqdm import tqdm

# User defined modules
from EvGym.charge_world import ChargeWorldEnv
from EvGym.charge_agent import agentASAP, agentOptim, agentNoV2G, agentOracle
from EvGym import config
from EvGym.charge_utils import print_welcome, parse_args # type: ignore

# Contracts
from ContractDesign.time_contracts import general_contracts

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
    
    # Try this
    sigma = args.price_noise*(df_price["price_im"].quantile(0.75) - df_price["price_im"].quantile(0.25))
    df_price["price_im"] = df_price["price_im"] + rng.normal(0, sigma, len(df_price))

    # Calculate contracts
    thetas_i, thetas_j = eval(args.thetas_i), eval(args.thetas_j)
    G, W, L_cont = general_contracts(thetas_i = thetas_i,
                                     thetas_j = thetas_j,
                                     c1 = args.c1,
                                     c2 = args.c2,
                                     kappa1 = args.kappa1,
                                     kappa2 = args.kappa2,
                                     alpha_d = config.alpha_d,
                                     psi = config.psi,
                                     IR = "fst", IC = "ort_l",
                                     integer = args.integer,
                                     monotonicity=False) # Tractable formulation

    L = np.round(L_cont,0) # L_cont →  L continuous
    contract_info = {"G": G, "W": W, "L": L, "thetas_i": thetas_i, "thetas_j": thetas_j, "c1": args.c1, "c2": args.c2}

    # Some agents are not allowed to discharge energy
    skip_contracts = True if args.agent in ["ASAP", "NoV2G"] else False

    # Initialize objects
    world = ChargeWorldEnv(df_sessions, df_price, contract_info, rng, skip_contracts = skip_contracts)
    df_state = world.reset()

    # Declare agent
    if args.agent == "ASAP":
        agent = agentASAP()
    elif args.agent == "NoV2G":
        agent = agentNoV2G(df_price, args, myprint = False)
    elif args.agent == "Optim":
        agent = agentOptim(df_price, args, myprint = False)
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
    for t in tqdm(range(int(ts_min)-1, int(ts_max)), desc = f"{title}: ", smoothing=0.01):
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
