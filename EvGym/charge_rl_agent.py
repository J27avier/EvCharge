import numpy as np
import cvxpy as cp # type: ignore
import pandas as pd
from . import config
from typing import TYPE_CHECKING, Any
from cvxpylayers.torch import CvxpyLayer # type: ignore

from icecream import ic # type:ignore

np.set_printoptions(linewidth=np.nan) # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(linewidth=1000, sci_mode=False)

from .safety_layer import SafetyLayer, SafetyLayerAgg
from .charge_utils import bounds_from_obs

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    #torch.nn.init.normal_(layer.weight, 1, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Safe_Actor_Mean_Agg(nn.Module):
    def __init__(self, envs, args,  device):
        super(Safe_Actor_Mean_Agg, self).__init__()
        self.no_safety = args.no_safety
        hidden = args.hidden
        self.linear1 = layer_init(nn.Linear(envs["single_observation_space"], hidden))
        if args.relu:
            self.activation1 = nn.ReLU()
        else:
            self.activation1 = nn.Tanh()
        self.linear2 = layer_init(nn.Linear(hidden, hidden))
        if args.relu:
            self.activation2 = nn.Tanh()
        else:
            self.activation2 = nn.ReLU()
        self.linear3 = layer_init(nn.Linear(hidden, 1), std=0.01)
        self.safetyL = SafetyLayerAgg(1, device)

    def forward(self, x):
        obs = x.detach().clone()
        x = self.linear1.forward(x)
        x = self.activation1(x)
        x = self.linear2.forward(x)
        x = self.activation2(x)
        x = self.linear3.forward(x)
        if self.no_safety:
            return x, torch.tensor(0)
        else:
            x_safe = self.safetyL.forward(x, obs)
            with torch.no_grad():
                proj_loss = torch.norm(x - x_safe)
            return x_safe, proj_loss 

class agentPPO_agg(nn.Module):
    def __init__(self, envs, df_price, device, args, pred_price_n=8, max_cars: int = config.max_cars, myprint = False):
        super().__init__()
        self.args = args
        self.critic = nn.Sequential(
                layer_init(nn.Linear(envs["single_observation_space"], 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64,1), std=1.0),
                )
        self.actor_mean = Safe_Actor_Mean_Agg(envs, args, device)
        #self.actor_logstd = nn.Parameter(torch.zeros(1,1))
        self.actor_logstd = nn.Parameter(args.logstd*torch.ones(1,1)) 

        # Ev parameters
        self.max_cars = max_cars
        self.df_price = df_price
        self.device = device
        self.envs = envs
        self.pred_price_n = pred_price_n
        self.myprint = myprint
        self.proj_loss = 0

    def get_value(self, x):
        return self.critic(x)

    def _get_prediction(self, t, n):
        l_idx_t0 = self.df_price.index[self.df_price["ts"]== t].to_list()
        assert len(l_idx_t0) == 1, "Timestep for prediction not unique or non existent"
        idx_t0 = l_idx_t0[0]
        #idx_tend = min(idx_t0+n, self.df_price.index.max()+1)
        idx_tend = idx_t0+n
        pred_price = self.df_price.iloc[idx_t0:idx_tend]["price_im"].values
        return pred_price

    def df_to_state(self, df_state, t):
        # Aggregate
        # Get occ spots (Nt)
        occ_spots = df_state["t_rem"] > 0 # Occupied spots
        num_cars = occ_spots.sum()

        # Get (ND) 
        cont_spots = df_state["t_dis"] > 0 # Spots with contracts
        num_cars_dis = (cont_spots).sum() 

        # Sum Soc
        sum_soc = df_state[occ_spots]["soc_t"].sum()

        # Sum SOC_rem
        sum_diff_soc = num_cars * config.FINAL_SOC - sum_soc

        # Sum soc_dis
        sum_soc_dis = df_state[cont_spots]["soc_dis"].sum()

        # Sum Y_min
        hat_y_low = config.FINAL_SOC-df_state["soc_t"] - config.alpha_c*config.eta_c*(df_state["t_rem"] - 1)/config.B
        y_low = np.maximum(hat_y_low * config.eta_c, hat_y_low / config.eta_d)
        y_low[~occ_spots] = 0

        sum_y_low = y_low.sum() 

        # Lax
        lax = df_state["t_rem"] - (config.FINAL_SOC - df_state["soc_t"])*config.B / (config.alpha_c*config.eta_c)
        lax[~occ_spots] = 0
        sum_lax = lax.sum()

        # p25, p50, p75, max, of soc_t, t_rem, soc_dis, t_dis
        p_soc_t = df_state[occ_spots]["soc_t"].quantile([0, 0.25, 0.5, 0.75, 1])
        p_t_rem = df_state[occ_spots]["t_rem"].quantile([0, 0.25, 0.5, 0.75, 1])
        p_soc_dis = df_state[cont_spots]["soc_dis"].quantile([0, 0.25, 0.5, 0.75, 1])
        p_t_dis = df_state[cont_spots]["t_dis"].quantile([0, 0.25, 0.5, 0.75, 1])

        # Bounds
        dis_lim = np.zeros(config.max_cars)
        dis_lim[cont_spots] += -df_state[cont_spots]["soc_dis"]*config.eta_d

        self.lower = np.maximum(y_low, np.maximum(-config.alpha_d/config.B, dis_lim))
        self.lower[~occ_spots] = 0

        upper_soc = (config.FINAL_SOC - df_state["soc_t"]) / config.eta_c
        self.upper = np.minimum(upper_soc,  config.alpha_c / config.B)
        self.upper[~occ_spots] = 0

        self.sum_lower = self.lower.sum() + 0.0001
        self.sum_upper = self.upper.sum()

        if self.args.norm_state and num_cars > 0:
            sum_soc /= num_cars
            sum_diff_soc /= num_cars
            sum_y_low /= num_cars
            sum_lax /= num_cars

        if self.args.without_perc:
            state_cars = np.concatenate(([self.sum_lower],
                                         [self.sum_upper],
                                         [num_cars],
                                         [num_cars_dis],
                                         [sum_soc],
                                         [sum_diff_soc],
                                         [sum_soc_dis],
                                         [sum_y_low],
                                         [sum_lax],)) 
        else:
            state_cars = np.concatenate(([self.sum_lower],
                                         [self.sum_upper],
                                         [num_cars],
                                         [num_cars_dis],
                                         [sum_soc],
                                         [sum_diff_soc],
                                         [sum_soc_dis],
                                         [sum_y_low],
                                         [sum_lax],
                                         p_soc_t,
                                         p_t_rem,
                                         p_soc_dis,
                                         p_t_dis)) 
        # Note, all the "sums"  can be normalized
        state_cars = np.nan_to_num(state_cars)
        

        pred_price = self._get_prediction(t, self.pred_price_n)
        hour = np.array([t % 24])
        np_x = np.concatenate((state_cars, pred_price, hour)).astype(float)
        #ic(np_x, np_x.shape, type(np_x))
        x = torch.tensor(np_x).to(self.device).float()#reshape(1, self.envs["single_observation_space"])
        #print(f"{x=}, {x.shape=}")
        self.t = t
        return x

    def state_to_df(self, obs):
        raise Exception("Not valid in aggregate")
        np_obs = obs.cpu().numpy()
        data_state = np_obs[0, :config.max_cars*4]
        df_state_simpl = pd.DataFrame(data = data_state.reshape((config.max_cars, 4)), columns = ["soc_t", "t_rem", "soc_dis", "t_dis"])
        pred_price = np_obs[config.max_cars*4:config.max_cars*4+self.pred_price_n]
        hour = np_obs[-1]
        return df_state_simpl, pred_price, hour

    def _get_action_and_value(self, x,  action=None):
        if x.ndim == 1:
            action_mean, proj_loss = self.actor_mean(x)
            action_mean = action_mean.unsqueeze(1)
        else:
            action_mean, proj_loss = self.actor_mean(x)
        #action_mean, proj_loss = self.actor_mean(x)
        self.proj_loss = proj_loss.cpu().numpy().squeeze()
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd) #/ 10
        probs = Normal(action_mean, action_std)
        if action is None:
            #probs = Normal(action_mean, (self.sum_upper-self.sum_lower)+0.0001)
            #print(f"{action_mean=}, {action_mean.shape=}, {type(action_mean)=}")
            action_t = probs.sample()
            # Double safety
            action = self._clamp_bounds(action_t) # before, also needed x

        value = self.critic(x)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value 

    def _clamp_bounds(self, action_t):
        # Aggregate clamp
        Tlower = torch.tensor(self.sum_lower).to(self.device)
        Tupper = torch.tensor(self.sum_upper).to(self.device)

        action = torch.clamp(action_t, Tlower, Tupper)
        return action

    def get_action_and_value(self, x,  action=None):
        # Right now it doesn't do anything, but left for consistency with the other method
        action, logprob, entropy, value = self._get_action_and_value(x, action)
        return action, logprob, entropy, value 

    def action_to_env(self, action_agg):
        # Dissagregation
        Y_tot = action_agg.cpu().numpy().squeeze()
        action = self.lower.copy()

        range_y  = self.upper - self.lower
        if range_y.sum() > 0:
            n_range_y = range_y  / range_y.sum()
            action  += (Y_tot - self.lower.sum()) * n_range_y

        #print("-- double clip --")
        #print(f"{self.lower=}, {self.lower.shape=}, {type(self.lower)=}")
        #print(f"{self.upper=}, {self.upper.shape=}, {type(self.upper)=}")
        #input()
        #print(f"{action_agg=}")
        #print(f"{action=}, {action.shape=}, {type(action)=}")
        return action

class Safe_Actor_Mean(nn.Module):
    def __init__(self, envs, device):
        super(Safe_Actor_Mean, self).__init__()
        self.linear1 = layer_init(nn.Linear(envs["single_observation_space"], 64))
        self.activation1 = nn.Tanh()
        self.linear2 = layer_init(nn.Linear(64, 64))
        self.activation2 = nn.Tanh()
        self.linear3 = layer_init(nn.Linear(64, envs["single_action_space"]), std=0.01)
        self.safetyL = SafetyLayer(envs["single_action_space"], device)

    def forward(self, x):
        obs = x.detach().clone()
        x = self.linear1.forward(x)
        x = self.activation1(x)
        x = self.linear2.forward(x)
        x = self.activation2(x)
        x = self.linear3.forward(x)
        x = x * config.alpha_c / config.B
        #print(f"pre_action_mean: {x=}")
        x_safe = self.safetyL.forward(x, obs)
        with torch.no_grad():
            proj_loss = torch.norm(x - x_safe)
        return x_safe, proj_loss


class agentPPO_lay(nn.Module):
    def __init__(self, envs, df_price, device, pred_price_n=8, max_cars: int = config.max_cars, myprint = False):
        super().__init__()
        self.critic = nn.Sequential(
                layer_init(nn.Linear(envs["single_observation_space"], 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64,1), std=1.0)
                )
        self.actor_mean = Safe_Actor_Mean(envs, device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, envs["single_action_space"]))

        # Ev parameters
        self.max_cars = max_cars
        self.df_price = df_price
        self.device = device
        self.envs = envs
        self.pred_price_n = pred_price_n
        self.myprint = myprint
        self.proj_loss = 0
        

    def get_value(self, x):
        return self.critic(x)

    def _get_prediction(self, t, n):
        l_idx_t0 = self.df_price.index[self.df_price["ts"]== t].to_list()
        assert len(l_idx_t0) == 1, "Timestep for prediction not unique or non existent"
        idx_t0 = l_idx_t0[0]
        #idx_tend = min(idx_t0+n, self.df_price.index.max()+1)
        idx_tend = idx_t0+n
        pred_price = self.df_price.iloc[idx_t0:idx_tend]["price_im"].values
        return pred_price

    def df_to_state(self, df_state, t):
        state_cars = df_state[["soc_t", "t_rem", "soc_dis", "t_dis"]].values.flatten().astype(np.float64)
        pred_price = self._get_prediction(t, self.pred_price_n)
        hour = np.array([t % 24])
        np_x = np.concatenate((state_cars, pred_price, hour))
        x = torch.tensor(np_x).to(self.device).float().reshape(1, self.envs["single_observation_space"])
        self.t = t
        return x

    def state_to_df(self, obs):
        np_obs = obs.cpu().numpy()
        data_state = np_obs[0, :config.max_cars*4]
        df_state_simpl = pd.DataFrame(data = data_state.reshape((config.max_cars, 4)), columns = ["soc_t", "t_rem", "soc_dis", "t_dis"])
        pred_price = np_obs[config.max_cars*4:config.max_cars*4+self.pred_price_n]
        hour = np_obs[-1]
        return df_state_simpl, pred_price, hour

    def _get_action_and_value(self, x,  action=None):
        action_mean, proj_loss = self.actor_mean(x)
        self.proj_loss = proj_loss.cpu().numpy().squeeze()
        action_logstd = self.actor_logstd.expand_as(action_mean) # / 10
        action_std = torch.exp(action_logstd) 
        probs = Normal(action_mean, action_std)
        if action is None:
            #ic(action_mean, action_mean.shape, type(action_mean))
            action_t = probs.sample()
            # Double safety
            action = self._clamp_bounds(x, action_t)

        value = self.critic(x)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value 

    def _clamp_bounds(self, x, action_t):
        lower, upper = bounds_from_obs(x)

        Tlower = torch.tensor(lower).to(self.device)
        Tupper = torch.tensor(upper).to(self.device)

        action = torch.clamp(action_t, Tlower, Tupper)
        ##action[0, idx_empty] = 0
        #print("--- Double clip ---")
        #ic(action, action.shape, action.ndim)

        return action

    def get_action_and_value(self, x,  action=None):
        # Right now it doesn't do anything, but left for consistency with the other method
        action, logprob, entropy, value = self._get_action_and_value(x, action)
        return action, logprob, entropy, value 

    def action_to_env(self, action):
        #input()
        return action.cpu().numpy().squeeze()


class agentPPO_sagg(nn.Module):
    def __init__(self, envs, df_price, device, pred_price_n=8, max_cars: int = config.max_cars, myprint = False):
        super().__init__()
        self.critic = nn.Sequential(
                layer_init(nn.Linear(envs["single_observation_space"], 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64,1), std=1.0),
                )
        self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(envs["single_observation_space"], 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=0.01),
                nn.Sigmoid(),
            )
        self.actor_logstd = nn.Parameter(torch.zeros(1,1))

        # Ev parameters
        self.max_cars = max_cars
        self.df_price = df_price
        self.device = device
        self.envs = envs
        self.pred_price_n = pred_price_n
        self.myprint = myprint
        self.proj_loss = 0

    def get_value(self, x):
        return self.critic(x)

    def _get_prediction(self, t, n):
        l_idx_t0 = self.df_price.index[self.df_price["ts"]== t].to_list()
        assert len(l_idx_t0) == 1, "Timestep for prediction not unique or non existent"
        idx_t0 = l_idx_t0[0]
        #idx_tend = min(idx_t0+n, self.df_price.index.max()+1)
        idx_tend = idx_t0+n
        pred_price = self.df_price.iloc[idx_t0:idx_tend]["price_im"].values
        return pred_price

    def df_to_state(self, df_state, t):
        # Aggregate
        # Get occ spots (Nt)
        occ_spots = df_state["t_rem"] > 0 # Occupied spots
        num_cars = occ_spots.sum()

        # Get (ND) 
        cont_spots = df_state["t_dis"] > 0 # Spots with contracts
        num_cars_dis = (cont_spots).sum() 

        # Sum Soc
        sum_soc = df_state[occ_spots]["soc_t"].sum()

        # Sum SOC_rem
        sum_diff_soc = num_cars * config.FINAL_SOC - sum_soc

        # Sum soc_dis
        sum_soc_dis = df_state[cont_spots]["soc_dis"].sum()

        # Sum Y_min
        hat_y_low = config.FINAL_SOC-df_state["soc_t"] - config.alpha_c*config.eta_c*(df_state["t_rem"] - 1)/config.B
        y_low = np.maximum(hat_y_low * config.eta_c, hat_y_low / config.eta_d)
        y_low[~occ_spots] = 0

        sum_y_low = y_low.sum()

        # Lax
        lax = df_state["t_rem"] - (config.FINAL_SOC - df_state["soc_t"])*config.B / (config.alpha_c*config.eta_c)
        lax[~occ_spots] = 0
        sum_lax = lax.sum()

        # p25, p50, p75, max, of soc_t, t_rem, soc_dis, t_dis
        p_soc_t = df_state[occ_spots]["soc_t"].quantile([0, 0.25, 0.5, 0.75, 1])
        p_t_rem = df_state[occ_spots]["t_rem"].quantile([0, 0.25, 0.5, 0.75, 1])
        p_soc_dis = df_state[occ_spots]["soc_dis"].quantile([0, 0.25, 0.5, 0.75, 1])
        p_t_dis = df_state[occ_spots]["t_dis"].quantile([0, 0.25, 0.5, 0.75, 1])

        # Bounds
        dis_lim = np.zeros(config.max_cars)
        dis_lim[cont_spots] += -df_state[cont_spots]["soc_dis"]*config.eta_d

        self.lower = np.maximum(y_low, np.maximum(-config.alpha_d/config.B, dis_lim))
        self.lower[~occ_spots] = 0

        upper_soc = (config.FINAL_SOC - df_state["soc_t"]) / config.eta_c
        self.upper = np.minimum(upper_soc,  config.alpha_c / config.B)
        self.upper[~occ_spots] = 0

        self.sum_lower = self.lower.sum()
        self.sum_upper = self.upper.sum()

        state_cars = np.concatenate(([self.sum_lower],
                                     [self.sum_upper],
                                     [num_cars],
                                     [num_cars_dis],
                                     [sum_soc],
                                     [sum_soc_dis],
                                     [sum_y_low],
                                     [sum_lax],
                                     p_soc_t,
                                     p_t_rem,
                                     p_soc_dis,
                                     p_t_dis)) 
        # Note, all the "sums"  can be normalized
        state_cars = np.nan_to_num(state_cars)
        

        pred_price = self._get_prediction(t, self.pred_price_n)
        hour = np.array([t % 24])
        np_x = np.concatenate((state_cars, pred_price, hour)).astype(float)
        #ic(np_x, np_x.shape, type(np_x))
        x = torch.tensor(np_x).to(self.device).float()#reshape(1, self.envs["single_observation_space"])
        #print(f"{x=}, {x.shape=}")
        self.t = t
        return x

    def state_to_df(self, obs):
        raise Exception("Not valid in aggregate")
        np_obs = obs.cpu().numpy()
        data_state = np_obs[0, :config.max_cars*4]
        df_state_simpl = pd.DataFrame(data = data_state.reshape((config.max_cars, 4)), columns = ["soc_t", "t_rem", "soc_dis", "t_dis"])
        pred_price = np_obs[config.max_cars*4:config.max_cars*4+self.pred_price_n]
        hour = np_obs[-1]
        return df_state_simpl, pred_price, hour

    def _get_action_and_value(self, x,  action=None):
        #print(f"-- Agent step --")
        #print(f"{x.shape=}")
        if x.ndim == 1:
            action_mean = self.actor_mean(x).unsqueeze(1)
        else:
            action_mean = self.actor_mean(x)
        #ic(action_mean)
        #ic(action_mean.shape)
        #ic(self.actor_logstd)
        #ic(self.actor_logstd.shape)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)  / 10
        probs = Normal(action_mean, action_std)
        if action is None:
            #print(f"{action_mean=}, {action_mean.shape=}, {type(action_mean)=}")
            
            action_n = probs.sample()
            # Double safety
            action_t = action_n * (self.sum_upper) + (1- action_n)*(self.sum_lower)
            action = self._clamp_bounds(action_t) # before, also needed x

        value = self.critic(x)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value 

    def _clamp_bounds(self, action_t):
        # Aggregate clamp
        Tlower = torch.tensor(self.sum_lower).to(self.device)
        Tupper = torch.tensor(self.sum_upper).to(self.device)

        action = torch.clamp(action_t, Tlower, Tupper)
        return action

    def get_action_and_value(self, x,  action=None):
        # Right now it doesn't do anything, but left for consistency with the other method
        action, logprob, entropy, value = self._get_action_and_value(x, action)
        return action, logprob, entropy, value 

    def action_to_env(self, action_agg):
        # Dissagregation
        Y_tot = action_agg.cpu().numpy().squeeze()
        action = self.lower.copy()

        range_y  = self.upper - self.lower
        if range_y.sum() > 0:
            n_range_y = range_y  / range_y.sum()
            action  += (Y_tot - self.lower.sum()) * n_range_y

        #print("-- double clip --")
        #print(f"{self.lower=}, {self.lower.shape=}, {type(self.lower)=}")
        #print(f"{self.upper=}, {self.upper.shape=}, {type(self.upper)=}")
        #input()
        #print(f"{action_agg=}")
        #print(f"{action=}, {action.shape=}, {type(action)=}")
        return action

class agentPPO_sep(nn.Module):
    """
    WARNING: This is proof of concept, and constraint enforcement is separate from model
    """
    def __init__(self, envs, df_price, device, pred_price_n=8, max_cars: int = config.max_cars, myprint = False):
        super().__init__()
        self.critic = nn.Sequential(
                layer_init(nn.Linear(envs["single_observation_space"], 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64,1), std=1.0)
                )
        self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(envs["single_observation_space"], 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64,64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs["single_action_space"]), std=0.01),
                )
        self.actor_logstd = nn.Parameter(torch.zeros(1, envs["single_action_space"]))

        # Ev parameters
        self.max_cars = max_cars
        self.df_price = df_price
        self.device = device
        self.envs = envs
        self.pred_price_n = pred_price_n
        self.myprint = myprint
        self.proj_loss = 0
        self.x = None
        

    def get_value(self, x):
        return self.critic(x)

    def _get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            #action = action*config.alpha_c / config.B
        value = self.critic(x)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value 

    def _get_prediction(self, t, n):
        l_idx_t0 = self.df_price.index[self.df_price["ts"]== t].to_list()
        assert len(l_idx_t0) == 1, "Timestep for prediction not unique or non existent"
        idx_t0 = l_idx_t0[0]
        #idx_tend = min(idx_t0+n, self.df_price.index.max()+1)
        idx_tend = idx_t0+n
        pred_price = self.df_price.iloc[idx_t0:idx_tend]["price_im"].values
        return pred_price

    def df_to_state(self, df_state, t):
        state_cars = df_state[["soc_t", "t_rem", "soc_dis", "t_dis"]].values.flatten().astype(np.float64)
        pred_price = self._get_prediction(t, self.pred_price_n)
        hour = np.array([t % 24])
        np_x = np.concatenate((state_cars, pred_price, hour))
        x = torch.tensor(np_x).to(self.device).float().reshape(1, self.envs["single_observation_space"])
        self.t = t
        return x

    def state_to_df(self, obs, t):
        np_obs = obs.cpu().numpy()
        data_state = np_obs[:config.max_cars*4]
        df_state_simpl = pd.DataFrame(data = data_state.reshape((config.max_cars, 4)), columns = ["soc_t", "t_rem", "soc_dis", "t_dis"])
        pred_price = np_obs[config.max_cars*4:config.max_cars*4+self.pred_price_n]
        hour = np_obs[-1]
        return df_state_simpl, pred_price, hour

    def _enforce_single_safety(self, action_t, x, t):
        raise Exception("Deprecated for now")
        df_state, price_pred, hour = self.state_to_df(x, t)
        action = np.zeros((config.max_cars,1))

        if self.myprint:
            str_price_pred = np.array2string(price_pred, separator=", ")
            print(f"{str_price_pred}, t= {t}")
            ic(action_t, action_t.shape, type(action_t))

        if any(df_state["t_rem"] > 0):
            # We only need lax constraint
            constraints = []
            AC =  cp.Variable((config.max_cars, 1))
            AD = cp.Variable((config.max_cars, 1))
            Y = cp.Variable((config.max_cars, 1))
            SOC = cp.Variable((config.max_cars, 2), nonneg=True)
            LAX = cp.Variable((config.max_cars), nonneg = True)

            constraints += [SOC  >= 0]
            constraints += [SOC <= config.FINAL_SOC]

            constraints += [SOC[:,0] ==  df_state["soc_t"]]
            
            #Charging limits
            constraints += [AC >= 0]
            constraints += [AC <= config.alpha_c / config.B]

            # Discharging limits
            constraints += [AD <= 0]
            constraints += [AD >= -config.alpha_d / config.B]

            # Discharging ammount
            constraints += [ - cp.sum(AD, axis=1)/ config.eta_d <= df_state["soc_dis"]]

            for i, car in enumerate(df_state.itertuples()):
                if car.t_rem == 0:
                    constraints += [AD[i,:] == 0]
                    constraints += [AC[i,:] == 0]
                    constraints += [LAX[i] == 0]
                else:
                    constraints += [SOC[i,1] == SOC[i,0] + AC[i,0] * config.eta_c + AD[i,0] / config.eta_d]
                    constraints += [LAX[i] == (car.t_rem-1) - ((config.FINAL_SOC - SOC[i,1])*config.B) / 
                                    (config.alpha_c * config.eta_c)] 
                    constraints += [LAX[i] >= 0]

                    if car.t_dis <= 0:
                        constraints += [AD[i,0] == 0]

            constraints += [Y == AC + AD]

            objective = cp.Minimize(cp.sum_squares(Y[:,0] - action_t))
            #objective = cp.Minimize(cp.sum(cp.abs(Y[:,0] - action_t))) # Works well
            prob = cp.Problem(objective, constraints)
            #prob.solve(solver=cp.MOSEK, verbose=False)
            prob.solve(solver=cp.ECOS, verbose=False,  max_iters = 10_000_000)
            #prob.solve(solver=cp.GUROBI, verbose=False)
            #prob.solve(solver=cp.MOSEK, mosek_params = {'MSK_IPAR_NUM_THREADS': 8, 'MSK_IPAR_BI_MAX_ITERATIONS': 2_000_000, "MSK_IPAR_INTPNT_MAX_ITERATIONS": 800}, verbose=False)  

            if prob.status not in  ['optimal', 'optimal_inaccurate']:
                ic(prob.status)
                raise Exception("Optimal schedule not found")

            best_cost = prob.value
            action = Y.value 
            action[np.abs(action) < config.tol] = 0


            if self.myprint:
                ic(prob.status)
                ic(action[:,0], action.shape, type(action))
                ic(AC.value[:,0])
                ic(AD.value[:,0])

        return action

    def _enforce_safety(self, action_t, x, t):
        raise Exception("Deprecated for now")
        action_t_np = action_t.cpu().numpy()
        l_actions = []

        if False:
            print(f"""---action_t---
                    {action_t}
                    {type(action_t)}
                    {action_t.shape}, {action_t.shape[0]}, {type(action_t.shape[0])},
                    {action_t.ndim}
                    {'-'*6}""")
        
        # Account for batches
        if action_t.ndim == 2:
            loops = action_t.shape[0]
            for i in range(loops):
                action_i = self._enforce_single_safety(action_t_np[i], x[i], t)
                l_actions.append(action_i)
            action = np.array(l_actions)[0].T
        else:
            action = self._enforce_single_safety(action_t_np, x, t)

        return action

    def get_action_and_value(self, x, action=None):
        self.x = x.detach().clone()
        action_t, logprob, entropy, value = self._get_action_and_value(x, action)

        # Clamping will be done at the VERY LAST STEP NOW
        #action_np = self._enforce_safety(action_t, x, self.t )
        #action = torch.tensor(action_np).to(self.device).float()

        #return action, logprob, entropy, value 
        return action_t, logprob, entropy, value 

    def action_to_env(self, action_t):
        # Clipping
        lower, upper = bounds_from_obs(self.x)

        Tlower = torch.tensor(lower).to(self.device)
        Tupper = torch.tensor(upper).to(self.device)

        action = torch.clamp(action_t, Tlower, Tupper)

        return action.cpu().numpy().squeeze()
