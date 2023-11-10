import numpy as np
import pandas as pd
import sys
import os
import argparse
from distutils.util import strtobool

from . import config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-R", "--print-dash", help = "Print dashboard", action="store_true")
    parser.add_argument("-S", "--no-save", help="Does not save results csv", action="store_true")
    parser.add_argument("-C", "--save-contracts", help="Saves the contracts accepted to each car", action="store_true")
    parser.add_argument("-A", "--agent", help="Type of agent", type=str, required=True)
    parser.add_argument("-D", "--desc", help="Description of the expereiment, starting with \"_\"", type=str, default="")
    parser.add_argument("-E", "--seed", help="Seed to use for the rng", type=int, default=42)
    parser.add_argument("-G", "--save-agent", help="Saves the agent", action="store_true")
    parser.add_argument("--save-name", help="Name to save experiment", type=str, default="")
    parser.add_argument("-Y", "--years", help="Number of years to run the simulation for", type=int)

    # Files
    parser.add_argument("-I", "--file-price", help = "Name of imbalance price dataframe", 
                        type=str, default= "df_price_2019.csv")
    parser.add_argument("-O", "--file-contracts", help = "CSV of contracts offered", 
                        type=str, default= "ExpLogs/2023-09-13-15:25:05_Contracts_ev_world_Optim.csv")
    parser.add_argument("-N", "--file-sessions", help = "CSV of charging sessions",
                        type=str, default= "df_elaad_preproc.csv")

    # Torch
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?",
            const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    # Train tuning
    parser.add_argument("--reward-coef", type=float, default=1)
    parser.add_argument("--proj-coef", type=float, default=0)
    parser.add_argument("--lax-coef", type=float, default=0)
    parser.add_argument("--logstd", type=float, default=-2)
    parser.add_argument("--n-state", type=int, default = 38)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--relu", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--no-safety", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--norm-state", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--without-perc", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--norm-reward", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--reset-std", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--grad-std", type= lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--df-imit", type=str, default="")

    # Algorithm specific
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-steps", type=int, default = 24, #default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4, # default 32, 
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0, # 0.1?
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")


    args = parser.parse_args()
    args.batch_size = int(1 * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

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


def bounds_from_obs(obs):
    batch_size = obs.shape[0]
    np_lower = np.zeros((batch_size, config.max_cars))
    np_upper = np.zeros((batch_size, config.max_cars))
    np_obs = obs.cpu().numpy()

    for i in range(batch_size):
        data_state = np_obs[i, :config.max_cars*4]
        df_state = pd.DataFrame(data = data_state.reshape((config.max_cars, 4)),
                                columns = ["soc_t", "t_rem", "soc_dis", "t_dis"])

        occ_spots = df_state["t_rem"] > 0 # Occupied spots
        cont_spots = df_state["t_dis"] > 0 # Spots with contracts

        hat_y_low = config.FINAL_SOC-df_state["soc_t"] - config.alpha_c*config.eta_c*(df_state["t_rem"] - 1)/config.B
        y_low = np.maximum(hat_y_low * config.eta_c, hat_y_low / config.eta_d)
        y_low[~occ_spots] = 0

        dis_lim = np.zeros(config.max_cars)
        dis_lim[cont_spots] += -df_state[cont_spots]["soc_dis"]*config.eta_d

        lower = np.maximum(y_low, np.maximum(-config.alpha_d/config.B, dis_lim))
        lower[~occ_spots] = 0

        upper_soc = (config.FINAL_SOC - df_state["soc_t"]) / config.eta_c
        upper = np.minimum(upper_soc,  config.alpha_c / config.B)
        upper[~occ_spots] = 0
        
        np_lower[i] = lower
        np_upper[i] = upper

    return np_lower, np_upper

# Cvxpylayers has some warnings, but ok
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
