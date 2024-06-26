import numpy as np
import pandas as pd
import sys
import os
import argparse
from distutils.util import strtobool
from colorama import init, Back, Fore
import pyfiglet # type: ignore

from . import config

def parse_sac_args():
    parser = argparse.ArgumentParser()
    # Javier args
    parser.add_argument("-R", "--print-dash", help = "Print dashboard", action="store_true")
    parser.add_argument("-S", "--no-save", help="Does not save results csv", action="store_true")
    parser.add_argument("-C", "--save-contracts", help="Saves the contracts accepted to each car", action="store_true")
    parser.add_argument("-A", "--agent", help="Type of agent", type=str, default="SAC-sagg", required=True)
    parser.add_argument("-D", "--desc", help="Description of the expereiment, starting with \"_\"", type=str, default="")
    parser.add_argument("-E", "--seed", help="Seed to use for the rng", type=int, default=42)
    parser.add_argument("--save-agent", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--save-name", help="Name to save experiment", type=str, default="")
    parser.add_argument("-Y", "--years", help="Number of years to run the simulation for", type=int)
    parser.add_argument("--summary", type= lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument("--general", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--month", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)

    # Files
    #parser.add_argument("-I", "--file-price", help = "Name of imbalance price dataframe", 
                        #type=str, default= "df_price_2019.csv")
    parser.add_argument("-I", "--file-price", help = "Name of imbalance price dataframe", 
                        type=str, default= "df_prices_c.csv")
    parser.add_argument("-O", "--file-contracts", help = "CSV of contracts offered", 
                        type=str, default= "ExpLogs/2023-09-13-15:25:05_Contracts_ev_world_Optim.csv")
    parser.add_argument("-N", "--file-sessions", help = "CSV of charging sessions",
                        type=str, default= "df_elaad_preproc.csv")

    # Train tuning, some here might not be needed
    parser.add_argument("--reward-coef", type=float, default=1)
    parser.add_argument("--proj-coef", type=float, default=0)
    parser.add_argument("--lax-coef", type=float, default=0)
    parser.add_argument("--logstd", type=float, default=-2)
    parser.add_argument("--n-state", type=int, default = 59)
    parser.add_argument("--n-action", type=int, default = 1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--norm-reward", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--state-rep", type=str, default="nothmd")
    parser.add_argument("--disagg", type=str, default="P")
    parser.add_argument("--test", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--rng-test", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--price-noise", type=float, default=0)
    parser.add_argument("--pred-noise", type=float, default=0)
    

    # Clean RL arguments
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    #parser.add_argument("--seed", type=int, default=1,
    #    help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    #parser.add_argument("--env-id", type=str, default="Hopper-v4",
    #    help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=24 , #default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=False,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-R", "--print-dash", help = "Print dashboard", action="store_true")
    parser.add_argument("-S", "--no-save", help="Does not save results csv", action="store_true")
    parser.add_argument("-C", "--save-contracts", help="Saves the contracts accepted to each car", action="store_true")
    parser.add_argument("-A", "--agent", help="Type of agent", type=str, required=True)
    parser.add_argument("-D", "--desc", help="Description of the expereiment, starting with \"_\"", type=str, default="")
    parser.add_argument("-E", "--seed", help="Seed to use for the rng", type=int, default=42)
    parser.add_argument("--save-name", help="Name to save experiment", type=str, default="")
    parser.add_argument("--summary", type= lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)

    # Files
    #parser.add_argument("-I", "--file-price", help = "Name of imbalance price dataframe", 
    #                    type=str, default= "df_price_2019.csv")
    parser.add_argument("-I", "--file-price", help = "Name of imbalance price dataframe", 
                        type=str, default= "df_prices_c.csv")
    parser.add_argument("-O", "--file-contracts", help = "CSV of contracts offered", 
                        type=str, default= "ExpLogs/2023-09-13-15:25:05_Contracts_ev_world_Optim.csv")
    parser.add_argument("-N", "--file-sessions", help = "CSV of charging sessions",
                        type=str, default= "df_elaad_preproc.csv")

    # Contract arguments
    parser.add_argument("--thetas_i", type=str, default=f"{config.thetas_i}")
    parser.add_argument("--thetas_j", type=str, default=f"{config.thetas_j}")
    parser.add_argument("--c1", type=float, default=config.c1)
    parser.add_argument("--c2", type=float, default=config.c2)
    parser.add_argument("--kappa1", type=float, default=config.kappa1)
    parser.add_argument("--kappa2", type=float, default=config.kappa2)
    parser.add_argument("--integer", type= lambda x: bool(strtobool(x)), default=config.integer)

    parser.add_argument("--pred-noise", type=float, default=0)

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

def zipf(N, s):
    x = np.arange(N)
    H = (1 / (x+1)**s).sum()
    return 1/H * 1/(x+1)**s

# Cvxpylayers has some warnings, but ok
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
# 
def parse_rl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-R", "--print-dash", help = "Print dashboard", action="store_true")
    parser.add_argument("-S", "--no-save", help="Does not save results csv", action="store_true")
    parser.add_argument("-C", "--save-contracts", help="Saves the contracts accepted to each car", action="store_true")
    parser.add_argument("-A", "--agent", help="Type of agent", type=str, required=True)
    parser.add_argument("-D", "--desc", help="Description of the expereiment, starting with \"_\"", type=str, default="")
    parser.add_argument("-E", "--seed", help="Seed to use for the rng", type=int, default=42)
    parser.add_argument("--save-agent", type= lambda x: bool(strtobool(x)), default=False, nargs='?', const=False)
    parser.add_argument("--save-name", help="Name to save experiment", type=str, default="")
    parser.add_argument("-Y", "--years", help="Number of years to run the simulation for", type=int)
    parser.add_argument("--summary", type= lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)

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
