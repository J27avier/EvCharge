import numpy as np
import cvxpy as cp # type: ignore
import pandas as pd
from . import config
from typing import TYPE_CHECKING, Any
from cvxpylayers.torch import CvxpyLayer # type: ignore
from .charge_utils import bounds_from_obs 

from icecream import ic # type:ignore

np.set_printoptions(linewidth=np.nan) # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

class SafetyLayerAgg(torch.nn.Module):
    def __init__(self, D, device):
        super(SafetyLayerAgg, self).__init__()
        self.device = device

        # Variables
        Y = cp.Variable(1)

        # Parameters
        Y_hat = cp.Parameter(1)
        Y_lower = cp.Parameter(1)
        Y_upper = cp.Parameter(1)

        # Constraints
        constraints = []

        # SOC
        constraints += [Y >= Y_lower]
        constraints += [Y <= Y_upper]
        
        objective = cp.Minimize(cp.square(Y - Y_hat))
        prob = cp.Problem(objective, constraints)

        self.layer = CvxpyLayer(prob, [Y_hat, Y_lower, Y_upper], [Y])

        #self.solver_args = {"solve_method": "ECOS", "max_iters": 100_000_000} 
        self.solver_args = {"solve_method": "SCS", "max_iters": 1_000} 

    def forward(self, Y_hat, obs):

        if Y_hat.ndim == 2:
            batch_size = Y_hat.shape[0]
            Y_lower = obs[:,0].unsqueeze(1)
            Y_upper = obs[:,1].unsqueeze(1)
            #print(f"-- Layer --")
            #print(f"{Y_hat=}, {Y_hat.shape=}")
            #print(f"{Y_lower}, {Y_lower.shape=}")
            #print(f"{Y_upper}, {Y_upper.shape=}")
            #print(f"{obs.shape=}")

            action = self.layer(Y_hat, Y_lower, Y_upper, solver_args = self.solver_args)[0] 
            #print(f"{action=}, {action.shape=}")
            return action
        else:
            Y_lower = obs[0].unsqueeze(0)
            Y_upper = obs[1].unsqueeze(0)

            #print(f"{Y_hat=}, {Y_hat.shape=}")
            #print(f"{Y_lower=}, {Y_lower.shape=}")
            #print(f"{Y_upper=}, {Y_upper.shape=}")
            action = self.layer(Y_hat, Y_lower, Y_upper, solver_args = self.solver_args)[0]
            action = action.unsqueeze(dim=0)
            #print(f"{action=}, {action.shape=}")
            return action

class SafetyLayer(torch.nn.Module):
    def __init__(self, D, device):
        super(SafetyLayer, self).__init__()
        self.device = device

        # Optimization program definition
        Y = cp.Variable(D)

        # Parameters
        x = cp.Parameter(D)
        lower = cp.Parameter(D)
        upper = cp.Parameter(D)

        # Constraints
        constraints = []

        constraints += [Y >= lower]
        constraints += [Y <= upper]

        objective = cp.Minimize(cp.sum_squares(Y - x))
        prob = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(prob, [x, lower, upper], [Y])

        #self.solver_args = {"solve_method": "ECOS", "max_iters": 100_000_000} 
        self.solver_args = {"solve_method": "SCS", "max_iters": 1_000} 

    def forward(self, x, obs):
        np_lower, np_upper = bounds_from_obs(obs)
        lower = torch.tensor(np_lower).to(self.device).float()
        upper = torch.tensor(np_upper).to(self.device).float()
        #x = x + 1

        if x.ndim == 2:
            #print("--- Layer prep ---")
            #print(f"{obs=}, {obs.shape=}, {obs.ndim=}")
            #print(f"{lower=}, {lower.shape=}, {lower.ndim=}")
            #print(f"{upper=}, {upper.shape=}, {upper.ndim=}")
            #print("-- Layer --")
            #print(f"{x=}, {x.shape=}, {x.ndim=}")
            action = self.layer(x, lower, upper, solver_args = self.solver_args)[0] 
            #print(f"{action=}, {action.shape=}, {action.ndim=}")
            #return action.squeeze(dim=2)
            return action
        else:
            raise Exception("Expected x.ndim == 2")

class SafetyLayerDep(torch.nn.Module):
    def __init__(self, D, device):
        super(SafetyLayerDep, self).__init__()
        raise Exception("Deprecated!")
        self.device = device
        
        # Data
        self.np_t_rem = np.zeros(D)
        self.np_t_dis = np.zeros(D)

        # Optimization program definition
        # Variables
        AC = cp.Variable((D, 1))
        AD = cp.Variable((D, 1))
        Y = cp.Variable((D, 1))
        SOC = cp.Variable((D, 2), nonneg=True)
        LAX = cp.Variable((D), nonneg = True)

        # Parameters
        x = cp.Parameter(D)
        t_rem   = cp.Parameter(D)
        soc_t   = cp.Parameter(D)
        t_dis   = cp.Parameter(D)
        soc_dis = cp.Parameter(D)

        # Constraints
        constraints = []

        # SOC
        constraints += [SOC >= 0]
        constraints += [SOC <= config.FINAL_SOC]
        constraints += [SOC[:,0] == soc_t]

        # Charging rate limits
        constraints += [AC >= 0]
        constraints += [AC <= config.alpha_c / config.B]

        # Discharging rate limits
        constraints += [AD <= 0]
        constraints += [AD >= -config.alpha_d / config.B]

        # Discharging ammount
        constraints += [ -cp.sum(AD, axis = 1) / config.eta_d <= soc_dis ]

        for i in range(config.max_cars):
            # # If t_rem is 0, don't charge or discharge
            constraints += [AD[i,:] >= -1000*t_rem[i]]
            constraints += [AC[i,:] <=  1000*t_rem[i]]
            # # If t_dis is 0, don't discharge
            constraints += [AD[i,:] >= -1000*t_dis[i]]

            if self.np_t_rem[i] == 0:
            #if t_rem[i].value == 0:
                constraints += [AD[i,:] == 0]
                constraints += [AC[i,:] == 0]
                constraints += [LAX[i] == 0]
            else:
                constraints += [SOC[i,1] == SOC[i, 0] + AC[i,0] * config.eta_c + AD[i,0] / config.eta_d]
                constraints += [LAX[i] == (t_rem[i]-1) - ((config.FINAL_SOC - SOC[i,1]) * config.B) /
                                (config.alpha_c * config.eta_c)]
                if self.np_t_dis[i] == 0:
                #if t_dis[i].value == 0:
                    constraints += [AD[i,:] == 0]

            
        constraints += [LAX >= 0]

        constraints += [ Y == AC + AD]

        objective = cp.Minimize(cp.sum_squares(Y[:,0] - x))
        prob = cp.Problem(objective, constraints)
        #print("Params", prob.param_dict)
        #for key in prob.param_dict:
        #    print(prob.param_dict[key]._name)
        self.layer = CvxpyLayer(prob, [x, t_rem, soc_t, t_dis, soc_dis], [AC, AD, Y, SOC, LAX])

        #self.solver_args = {"solve_method": "ECOS", "max_iters": 100_000_000} 
        self.solver_args = {"solve_method": "SCS", "max_iters": 1_000} 

    def forward(self, x, obs):
        #np_obs = obs.detach().numpy()
        np_obs = obs.detach().cpu().numpy()
        data_state = np_obs[0, :config.max_cars*4]
        df_state = pd.DataFrame(data = data_state.reshape((config.max_cars, 4)), columns = ["soc_t", "t_rem", "soc_dis", "t_dis"])

        self.np_t_rem = df_state["t_rem"].values
        self.np_t_dis = df_state["t_dis"].values

        #t_rem   = torch.tensor(df_state["t_rem"].values).to(self.device).float()
        #soc_t   = torch.tensor(df_state["soc_t"].values).to(self.device).float()
        #t_dis   = torch.tensor(df_state["t_dis"].values).to(self.device).float()
        #soc_dis = torch.tensor(df_state["soc_dis"].values).to(self.device).float()

        #
        batch_size = x.shape[0]
        np_t_rem = np.zeros((batch_size, config.max_cars))
        np_soc_t = np.zeros((batch_size, config.max_cars))
        np_t_dis = np.zeros((batch_size, config.max_cars))
        np_soc_dis = np.zeros((batch_size, config.max_cars))

        for i in range(batch_size):
            data_state = np_obs[i, :config.max_cars*4]
            df_state = pd.DataFrame(data = data_state.reshape((config.max_cars, 4)), columns = ["soc_t", "t_rem", "soc_dis", "t_dis"])
            np_t_rem[i] = df_state["t_rem"]
            np_soc_t[i] = df_state["soc_t"]
            np_t_dis[i] = df_state["t_dis"]
            np_soc_dis[i] = df_state["soc_dis"]

        t_rem   = torch.tensor(np_t_rem).to(self.device).float()
        soc_t   = torch.tensor(np_soc_t).to(self.device).float()
        t_dis   = torch.tensor(np_t_dis).to(self.device).float()
        soc_dis = torch.tensor(np_soc_dis).to(self.device).float()
        x = x + 1


        if x.ndim == 2:
            batch_size = x.shape[0]
            print(f"{x=}, {x.shape=}, {x.ndim=}")
            print(f"{np_obs.shape=}")
            print(f"time {np_obs[:,-1]}")
            print(f"{t_rem=}, {t_rem.shape=}, {t_rem.ndim=}")

            # [0] is AC, [1] is AD, [2] is Y ... (in order of declaration)
            action = self.layer(x, t_rem, soc_t, t_dis, soc_dis, solver_args = self.solver_args)[2] 

            #action = self.layer(x, t_rem.repeat(batch_size, 1),
            #                     soc_t.repeat(batch_size, 1),
            #                     t_dis.repeat(batch_size, 1),
            #                     soc_dis.repeat(batch_size, 1),
            #                     solver_args = self.solver_args)[2] # [0] is AC, [1] is AD, [2] is Y ... (in order of declaration)

            print(f"{action}")
            return action.squeeze(dim=2)
        else:
            raise Exception("Expected x.ndim == 2")
        
        #    return self.layer(x, t_rem, soc_t, t_dis, soc_dis, solver_args = self.solver_args)[2]
