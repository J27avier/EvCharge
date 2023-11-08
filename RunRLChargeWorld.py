import pandas as pd
import numpy as np
from tabulate import tabulate
import os
import argparse
from tqdm import tqdm
from icecream import ic # type: ignore

# User defined modules
from EvGym.charge_world import ChargeWorldEnv
#from EvGym.charge_agent import agentASAP, agentOptim, agentNoV2G, agentOracle
from EvGym.charge_rl_agent import agentPPO_sep, agentPPO_lay, agentPPO_agg
from EvGym.charge_utils import parse_args, print_welcome
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
torch.set_num_threads(8)

def runSim(args = None):
    if args is None:
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
    ic(device)

    # Agent info
    pred_price_n = 8 # Could be moved to argument
    envs = {"single_observation_space": config.max_cars*4 + pred_price_n + 1, # max_cars*(soc_t, t_rem, t_dis, soc_dis), 24*(p_im), hr day
            "single_action_space": config.max_cars,
            }

    # Agents
    if args.agent == "PPO-sep":
        agent = agentPPO_sep(envs, df_price, device, pred_price_n=pred_price_n, myprint = False).to(device)

    elif args.agent == "PPO-lay":
        agent = agentPPO_lay(envs, df_price, device, pred_price_n=pred_price_n, myprint = False).to(device)

    elif args.agent == "PPO-agg":
        envs["single_observation_space"] = 37
        agent = agentPPO_agg(envs, df_price, device, pred_price_n=pred_price_n, myprint = False).to(device)

    else:
        try:
            print(f"Attempting to load: {args.agent}")
            if "agg" in args.agent:
                envs["single_observation_space"] = 37
            agent = torch.load(f"{config.agents_path}{args.agent}.pt")
            print(f"Loaded {args.agent}")
        except Exception as e:
            print(e)
            print(f"Agent name not recognized")
            exit(1)

    reward_coef = args.reward_coef
    proj_coef = args.proj_coef
    #ic(reward_coef, type(reward_coef))
    #ic(proj_coef, type(proj_coef))

    optimizer = optim.Adam(agent.parameters(), lr = args.learning_rate, eps = 1e-5)

    obs      = torch.zeros((args.num_steps, 1, envs["single_observation_space"]) ).to(device) # Manual concat
    actions  = torch.zeros((args.num_steps, 1, envs["single_action_space"])).to(device) # Manual concat
    logprobs = torch.zeros((args.num_steps, 1)).to(device)
    #rewards  = torch.zeros((args.num_steps, 1,  envs["single_action_space"])).to(device) # Manual concat
    rewards  = torch.zeros((args.num_steps, 1)).to(device) # Manual concat
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

    pbar = tqdm(desc=args.save_name, total=int(ts_max-ts_min), smoothing=0)
    ts_max = int(ts_min + 24 * 31)
    while t in range(int(ts_min)-1, int(ts_max)):
        #update = t % num_updates - ((ts_min - 1) % num_updates) + 1
        for update in range(1, num_updates+1): # TODO: Find a smarter way to do this 
            if t > ts_max: break
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                t += 1
                pbar.update(1)
                #ic(t, len(df_state))
                #ic(len(agent._get_prediction(t, agent.pred_price_n)))
                #print("Calc state dim", len(df_state)*4 + len(agent._get_prediction(t, agent.pred_price_n)) + 1)
                #print("Const state dim", envs["single_observation_space"])
                #print("", end="", flush=True)

                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                #-------
                #df_state, reward, done, info = world.step(action.cpu().numpy().squeeze()) # previous
                df_state, reward, done, info = world.step(agent.action_to_env(action))
                # Reward tuning
                #ic(reward, type(reward))
                reward = reward_coef * reward + proj_coef * agent.proj_loss
                #print(f"{reward_coef=}, {type(reward_coef)=}")
                #print(f"{reward=}, {type(reward)=}")
                #print(f"{proj_coef=}, {type(proj_coef)=}")
                #print(f"{agent.proj_loss=}, {type(agent.proj_loss)=}")

                done = np.array([done])
                assert t == info['t'], "Main time and env time out of sync"
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
                    pass
                    #print(f"t={(t-ts_min)}/{(ts_max-ts_min)}, {(t-ts_min)/(ts_max-ts_min):.2%}%")
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
            b_advantages = advantages.reshape(-1) # Normailze
            b_advantages = (b_advantages - b_advantages.mean()) - (b_advantages.std() + 1e-8)
            b_returns = returns.reshape(-1) #  Normailze"
            b_returns = (b_returns - b_returns.mean()) - (b_returns.std() + 1e-8)
            b_values = values.reshape(-1) #Normailze 
            b_values = (b_values - b_values.mean()) - (b_values.std() + 1e-8)
            # deactivate norm in options

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds] 
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        pass
                        #mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
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
            #print("SPS:", int(t / (time.time() - start_wallTime)))
            writer.add_scalar("charts/SPS", int(t / (time.time() - start_wallTime)), t)
            

    if not args.no_save:
        world.tracker.save_log(args, path=config.results_path)
        world.tracker.save_desc(args, {"title": title}, path=config.results_path)

    if args.save_contracts:
        world.tracker.save_contracts(args, path=config.results_path)

    # Save agent
    if args.save_agent:
        if args.save_name != "":
            torch.save(agent, f"{config.agents_path}{args.save_name}.pt")
        else:
            torch.save(agent, f"{config.agents_path}{world.tracker.timestamp}_{args.agent.split('.')[0]}{args.desc}.pt")

    writer.close()
    pbar.close()


if __name__ == "__main__":
    args = parse_args()

    if args.years is None:
        runSim(args)
    else:
        og_save_name = args.save_name
        for i in range(args.years):
            if og_save_name != "":
                args.save_name = og_save_name + f"_{i}"
                if i > 0:
                    args.agent = og_save_name + f"_{i-1}"
                runSim(args)
            else:
                raise Exception("You must specify save name to run multiple years")


# ACKNOWLEDGMENTS
# Parts of this code are adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
