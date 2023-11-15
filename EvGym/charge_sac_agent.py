import numpy as np
import cvxpy as cp # type: ignore
import pandas as pd
from . import config
from cvxpylayers.torch import CvxpyLayer # type: ignore
from math import isclose
from icecream import ic # type: ignore

np.set_printoptions(linewidth=np.nan) # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(linewidth=1000, sci_mode=False)

from .safety_layer import SafetyLayer, SafetyLayerAgg
from .charge_utils import bounds_from_obs

class SoftQNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        #self.fc1 = nn.Linear(np.array(args.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc1 = nn.Linear(args.n_state + args.n_action, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class agentSAC_sagg(nn.Module):
    def __init__(self, df_price, args, device, pred_price_n = 8):
        super().__init__()
        self.df_price = df_price
        self.args = args
        self.device = device
        self.pred_price_n = pred_price_n

        self.fc1 = nn.Linear(args.n_state, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, args.n_action)
        self.fc_logstd = nn.Linear(256, args.n_action)

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((config.action_space_high - config.action_space_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((config.action_space_high + config.action_space_low) / 2.0, dtype=torch.float32)
        )
        


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = config.LOG_STD_MIN + 0.5 * (config.LOG_STD_MAX - config.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

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
        self.lax = lax
        self.t_rem = df_state["t_rem"]
        self.t_dis = df_state["t_dis"]

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

        self.sum_lower = self.lower.sum() + 0.001
        self.sum_upper = self.upper.sum()
        
        if num_cars > 0:
            sum_lower = self.sum_lower / num_cars
            sum_upper = self.sum_upper / num_cars
        else:
            sum_lower = self.sum_lower
            sum_upper = self.sum_upper

        state_cars = np.concatenate(([sum_lower],
                                     [sum_upper],
                                     #[num_cars],
                                     #[num_cars_dis],
                                     #[sum_soc],
                                     #[sum_soc_dis],
                                     #[sum_y_low],
                                     #[sum_lax],
                                     #p_soc_t,
                                     #p_t_rem,
                                     #p_soc_dis,
                                     #p_t_dis)
                                     ))

        # Note, all the "sums"  can be normalized
        state_cars = np.nan_to_num(state_cars)

        pred_price = self._get_prediction(t, self.pred_price_n)
        pred_price = (pred_price - pred_price.mean()) / (pred_price.std() + 1e-3)
        hour = np.array([t % 24])
        day = np.array([t % (24*7)])
        pred_price_m = np.array([np.diff(pred_price).mean()])
        np_x = np.concatenate((state_cars, pred_price, pred_price_m, hour, day)).astype(float)
        #x = torch.tensor(np_x).to(self.device).float()#reshape(1, self.envs["single_observation_space"])
        self.t = t
        return np_x

    def _get_prediction(self, t, n):
        l_idx_t0 = self.df_price.index[self.df_price["ts"]== t].to_list()
        assert len(l_idx_t0) == 1, "Timestep for prediction not unique or non existent"
        idx_t0 = l_idx_t0[0]
        #idx_tend = min(idx_t0+n, self.df_price.index.max()+1)
        idx_tend = idx_t0+n
        pred_price = self.df_price.iloc[idx_t0:idx_tend]["price_im"].values
        return pred_price

    def action_to_env(self, action_agg):

        # Dissagregation
        alpha = action_agg[0] #.cpu().numpy().squeeze()
        alpha = np.clip(alpha, 0, 1)
        Y_tot = alpha*(self.sum_upper) + (1-alpha)*(self.sum_lower)
        occ_spots = self.t_rem > 0
        cont_spots = self.t_dis > 0
        range_y = self.upper - self.lower
        action = self.lower.copy()

        if range_y.sum() > 0:
            priority_list = np.argsort(self.lax) # Least laxity first
            #time_val = np.array([-t_d if t_d > 0 else t_r for t_d, t_r in zip(self.t_dis, self.t_rem)])
            #priority_list = np.argsort(-time_val)
            Y_temp = Y_tot - self.lower.sum()
            for i in priority_list:
                if i in occ_spots:
                    y_i = np.min([Y_temp, range_y[i]])
                    action[i] += y_i
                    Y_temp -= y_i
                if isclose(Y_temp, 0):
                    break

        action[~occ_spots] = 0
        #action[~cont_spots] = np.maximum(action[~cont_spots], 0)
        return action



