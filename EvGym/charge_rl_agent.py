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

def layer_init(layer, std=np.sqrt(2), bias_const =0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class agentPPO_sepCvx(nn.Module):
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
        

    def get_value(self, x):
        return self.critic(x)

    def _get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            action = action*config.alpha_c / config.B
        value = self.critic(x)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value 

    def _get_prediction(self, t, n):
        l_idx_t0 = self.df_price.index[self.df_price["ts"]== t].to_list()
        assert len(l_idx_t0) == 1, "Timestep for prediction not unique or non existent"
        idx_t0 = l_idx_t0[0]
        idx_tend = min(idx_t0+n, self.df_price.index.max()+1)
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
        df_state, price_pred, hour = self.state_to_df(x, t)
        action = np.zeros((config.max_cars,1))

        if self.myprint:
            str_price_pred = np.array2string(price_pred, separator=", ")
            print(f"{str_price_pred=}, {t=}")
            print(f"{action_t=}, {action_t.shape=}, {type(action_t)}")

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
                    if car.t_dis <= 0:
                        constraints += [AD[i,0] == 0]

                    constraints += [LAX[i] >= 0]
            constraints += [Y == AC + AD]

            objective = cp.Minimize(cp.sum_squares(Y[:,0] - action_t))
            #objective = cp.Minimize(cp.sum(cp.abs(Y[:,0] - action_t))) # Works well
            prob = cp.Problem(objective, constraints)
            #prob.solve(solver=cp.MOSEK, verbose=False)
            prob.solve(solver=cp.ECOS, verbose=False,  max_iters = 1_000_000)
            #prob.solve(solver=cp.GUROBI, verbose=False)
            #prob.solve(solver=cp.MOSEK, mosek_params = {'MSK_IPAR_NUM_THREADS': 8, 'MSK_IPAR_BI_MAX_ITERATIONS': 2_000_000, "MSK_IPAR_INTPNT_MAX_ITERATIONS": 800}, verbose=False)  

            if prob.status != 'optimal':
                print(f"{prob.status} Optimal solution not found")
                raise Exception("Optimal schedule not found")

            best_cost = prob.value
            action = Y.value 
            action[np.abs(action) < config.tol] = 0


            if self.myprint:
                print(f"{prob.status=}")
                print(f"{action[:,0]=}, {action.shape=}, {type(action)=}")
                print(f"{AC.value[:,0]=}")
                print(f"{AD.value[:,0]=}")

        return action

    def _enforce_safety(self, action_t, x, t):
        action_t_np = action_t.cpu().numpy()
        l_actions = []

        if False:
            print(f"""---action_t---
                    {action_t=}
                    {type(action_t)=}
                    {action_t.shape=}, {action_t.shape[0]=}, {type(action_t.shape[0])=},
                    {action_t.ndim=}
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
        #x = self.construct_state(df_state, t) # Gets performed twice (also in main), can streamline later
        action_t, logprob, entropy, value = self._get_action_and_value(x, action)
        action_np = self._enforce_safety(action_t, x, self.t )
        action = torch.tensor(action_np).to(self.device).float()
        #x = torch.tensor(np_x).to(self.device).float().reshape(1, self.envs["single_observation_space"])
        return action, logprob, entropy, value 

class SafetyLayer(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(SafetyLayer, self).__init__()
        # Optimization prohram definition

    def forward(self, x, obs):
        pass
 
# Cvxpylayers has some warnings, but ok
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        


class agentPPO(nn.Module):
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
        

    def get_value(self, x):
        return self.critic(x)

    def _get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            action = action*config.alpha_c / config.B
        value = self.critic(x)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value 

    def _get_prediction(self, t, n):
        l_idx_t0 = self.df_price.index[self.df_price["ts"]== t].to_list()
        assert len(l_idx_t0) == 1, "Timestep for prediction not unique or non existent"
        idx_t0 = l_idx_t0[0]
        idx_tend = min(idx_t0+n, self.df_price.index.max()+1)
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
        df_state, price_pred, hour = self.state_to_df(x, t)
        action = np.zeros((config.max_cars,1))

        if self.myprint:
            str_price_pred = np.array2string(price_pred, separator=", ")
            print(f"{str_price_pred=}, {t=}")
            print(f"{action_t=}, {action_t.shape=}, {type(action_t)}")

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
                    if car.t_dis <= 0:
                        constraints += [AD[i,0] == 0]

                    constraints += [LAX[i] >= 0]
            constraints += [Y == AC + AD]

            #objective = cp.Minimize(cp.sum_squares(Y[:,0] - action_t)/config.max_cars)
            objective = cp.Minimize(cp.sum(cp.abs(Y[:,0] - action_t))/config.max_cars)
            prob = cp.Problem(objective, constraints)
            #prob.solve(solver=cp.MOSEK, verbose=False)
            prob.solve(solver=cp.GUROBI, verbose=False)
            #prob.solve(solver=cp.MOSEK, mosek_params = {'MSK_IPAR_NUM_THREADS': 8, 'MSK_IPAR_BI_MAX_ITERATIONS': 2_000_000, "MSK_IPAR_INTPNT_MAX_ITERATIONS": 800}, verbose=False)  

            if prob.status != 'optimal':
                print(f"{prob.status} Optimal solution not found")
                raise Exception("Optimal schedule not found")

            best_cost = prob.value
            action = Y.value


            if self.myprint:
                print(f"{action[:,0]=}, {action.shape=}, {type(action)=}")
                print(f"{AC.value[:,0]=}")
                print(f"{AD.value[:,0]=}")

        return action

    def _enforce_safety(self, action_t, x, t):
        action_t_np = action_t.cpu().numpy()
        l_actions = []

        if False:
            print(f"""---action_t---
                    {action_t=}
                    {type(action_t)=}
                    {action_t.shape=}, {action_t.shape[0]=}, {type(action_t.shape[0])=},
                    {action_t.ndim=}
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
        #x = self.construct_state(df_state, t) # Gets performed twice (also in main), can streamline later
        action_t, logprob, entropy, value = self._get_action_and_value(x, action)
        action_np = self._enforce_safety(action_t, x, self.t )
        action = torch.tensor(action_np).to(self.device).float()
        #x = torch.tensor(np_x).to(self.device).float().reshape(1, self.envs["single_observation_space"])
        return action, logprob, entropy, value 
