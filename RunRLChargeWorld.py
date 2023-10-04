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
#from EvGym.charge_agent import agentASAP, agentOptim, agentNoV2G, agentOracle
from EvGym.charge_rl_agent import agentPPO
from EvGym import config

# Contracts
from ContractDesign.time_contracts import general_contracts

# For RL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import time
from distutils.util import strtobool
import random
#import pdb; pdb.set_trace()


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

    # Torch
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?",
            const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

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
    parser.add_argument("--ent-coef", type=float, default=0.0,
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

def main():
    args = parse_args()
    title = f"EvWorld-{args.agent}{args.desc}"

    # Writer
    writer = SummaryWriter(f"runs/{title}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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


    # Torch options
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Agent info
    pred_price_n = 8 # Could be moved to argument
    envs = {"single_observation_space": config.max_cars*4 + pred_price_n + 1, # max_cars*(soc_t, t_rem, t_dis, soc_dis), 24*(p_im), hr day
            "single_action_space": config.max_cars,
            }

    # Agents
    if args.agent == "PPO":
        agent = agentPPO(envs, df_price, device, pred_price_n=pred_price_n).to(device)
        optimizer = optim.Adam(agent.parameters(), lr = args.learning_rate, eps = 1e-5)
    else:
        raise Exception(f"Agent name not recognized")


    obs      = torch.zeros((args.num_steps, 1, envs["single_observation_space"]) ).to(device) # Manual concat
    actions  = torch.zeros((args.num_steps, 1, envs["single_action_space"])).to(device) # Manual concat
    logrpobs = torch.zeros((args.num_steps, 1)).to(device)
    rewards  = torch.zeros((args.num_steps, 1,  envs["single_action_space"])).to(device) # Manual concat
    dones    = torch.zeros((args.num_steps, 1)).to(device)
    values   = torch.zeros((args.num_steps, 1)).to(device)

    # Initialize objects
    world = ChargeWorldEnv(df_sessions, df_price, contract_info, rng, skip_contracts = skip_contracts)
    df_state = world.reset()

    next_obs = agent.df_to_state(df_state, ts_min) # should be ts_min -1 , but only matters for this timestep
    #next_obs = torch.randn(1,125).to(device)
    next_done = torch.zeros(1).to(device)
    total_timesteps = len(list(range(int(ts_min)-1, int(ts_max))))
    num_updates =  total_timesteps // args.batch_size

    # Print welcome screen
    if args.print_dash:
        print_welcome(df_sessions, df_price, contract_info)
        skips = 0

    # Environment loop
    #for t in tqdm(range(int(ts_min)-1, int(ts_max)), desc = f"{title}: "):
    t = int(ts_min - 1)
    start_wallTime = time.time()

    while t in range(int(ts_min)-1, int(ts_max)):
        #update = t % num_updates - ((ts_min - 1) % num_updates) + 1
        for update in range(1, num_updates+1):
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                t += 1
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                #-------
                df_state, reward, done, info = world.step(action)
                assert t+1 == info['t'], "Main time and env time out of sync"
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = agent.df_to_state(df_state, t)
                next_done = torch.Tensor(done).to(device)

                if args.print_dash:
                    if skips > 0: # Logic to jump forward
                        skips -= 1
                    else:
                        usr_in = world.print(-1, clear = True)
                    if usr_in.isnumeric():
                        skips = int(usr_in)
                        usr_in = ""
                else:
                    print(f"{t=}/{ts_max}, {t/ts_max*100}%")
                # print("Tracker: ts, chg_e_req, imbalance_bill, n_cars, avg_lax")
                # print(world.tracker.chg_bill)
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1,-1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for i in reversed(range(args.num_steps)):
                    if  i == args.num_steps -1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[i + 1]
                        nextvalues = values[i + 1]
                    delta = rewards[i] + args.gamma * nextvalues * nextnonterminal - values[i]
                    advantages[i] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # Flatten the batch
            b_obs = obs.reshape((-1, envs["single_observation_space"]))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, envs["single_action_space"]))
            b_advantages = advantages.reshape(-1)
            b_values = values.reshape(-1)


            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoc in range(args.update_epocs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds] 
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # Calculate approx_kl
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_indsk]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 -args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -args.clip_coef,
                                args.clip_coef
                                )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) **2 
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], t)
            writer.add_scalar("losses/value_loss", v_loss.item(), t)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), t)
            writer.add_scalar("losses/entropy", entropy_loss.item(), t)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), t)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), t)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), t)
            writer.add_scalar("losses/explained_variance", explained_var, t)
            print("SPS:", int(t / (time.time() - start_wallTime)))
            writer.add_scalar("charts/SPS", int(t / (time.time() - start_wallTime)), t)
            
            

    if not args.no_save:
        world.tracker.save_log(args, path=config.results_path)
        world.tracker.save_desc(args, {"title": title}, path=config.results_path)

    if args.save_contracts:
        world.tracker.save_contracts(args, path=config.results_path)

    writer.close()


if __name__ == "__main__":
    main()

# ACKNOWLEDGMENTS
# Parts of this code are adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
