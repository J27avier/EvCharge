import numpy as np
import cvxpy as cp # type: ignore
import pandas as pd
from . import config
from typing import TYPE_CHECKING, Any
import numpy.typing as npt

np.set_printoptions(linewidth=np.nan) # type: ignore

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

class SafetyLayer(torch.nn.Module):
    def __init__(self, D, device):
        super(SafetyLayer, self).__init__()
        self.device = device
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
        constraints += [ -cp.sum(AD, axis = 1) / coinfig.eta_d <= soc_dis ]

        for i in range(config.max_cars):
            if t_rem[i] == 0:
                constraints += [AD[i,:] == 0]
                constraints += [AC[i,:] == 0]
                constraints += [LAX[i] == 0]
            else:
                constraints += [SOC[i,1] == SOC[i, 0] + AC[i,0] * config.eta_c + AD[i,0] / config.eta_d]
                constraints += [LAX[i] == (t_rem[i]-1) - ((config.FINAL_SOC - SOC[i,1]) * config.B) /
                                (config.alpha_c * config.eta_c)]
                constraints += [LAX[i] >= 0]

                if t_dis[i] <= 0:
                    constraints += [AD[i,0] == 0]
        constraints += [ Y == AC + AD]

        objective = cp.Minimize(cp.sum_squares(Y[:,0] - x))
        prob = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(prob, [x, t_rem, soc_t, t_dis, soc_dis], [AC, AD, Y, SOC, LAX])
        self.solver_args = {"solve_method": "ECOS", "max_iters": 10_000_000} 

    def forward(self, x, df_state):
        self.t_rem   = torch.nn.Parameter(torch.tensor(df_state["t_rem"].values).to(self.device).float()) 
        self.soc_t   = torch.nn.Parameter(torch.tensor(df_state["soc_t"].values).to(self.device).float())
        self.t_dis   = torch.nn.Parameter(torch.tensor(df_state["t_dis"].values).to(self.device).float())
        self.soc_dis = torch.nn.Parameter(torch.tensor(df_state["soc_dis"].values).to(self.device).float())

        #[x, t_rem, soc_t, t_dis, soc_dis]
        if x.ndim == 2:
            batch_size = x.shape[0]
            return self.layer(x, self.t_rem.repeat(batch_size, 1),
                                 self.soc_t.repeat(batch_size, 1),
                                 self.t_dis.repeat(batch_size, 1),
                                 self.soc_dis.repeat(batch_size, 1),
                                 solver_args = self.solver_args)[0]
        else:
            return self.layer(x, t_rem, soc_t, t_dis, soc_dis, solver_args = self.solver_args)[0]
 
# Cvxpylayers has some warnings, but ok
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

