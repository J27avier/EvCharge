import numpy as np
import cvxpy as cp # type: ignore
import pandas as pd
from . import config
from typing import TYPE_CHECKING, Any
import numpy.typing as npt

np.set_printoptions(linewidth=np.nan) # type: ignore

class agentASAP():
    def __init__(self, max_cars: int = config.max_cars):
        self.max_cars = max_cars

    def get_action(self, df_state: pd.DataFrame, t: int = 0) -> npt.NDArray[Any]:
        """
        Charge action ASAP
        df_state: State of the parking lot
        t: Unused (left for consistency with other classes)
        """
        action = np.zeros(self.max_cars)

        for i, (_, car) in enumerate(df_state.iterrows()):
            if car.idSess >= 0: 
                if car.soc_t < config.FINAL_SOC:
                    action[i] = min(config.alpha_c/config.B, (config.FINAL_SOC-car.soc_t)/config.eta_c)

        return action

class agentOptim():
    def __init__(self, df_price, max_cars: int = config.max_cars, myprint = False):
        self.max_cars = max_cars
        self.df_price = df_price
        self.myprint = myprint

    def _get_prediction(self, t, n):
        l_idx_t0 = self.df_price.index[self.df_price["ts"] == t].to_list()
        assert len(l_idx_t0) == 1, "Timestep for prediction not unique or not existent"
        idx_t0 = l_idx_t0[0]
        idx_tend = min(idx_t0+n, self.df_price.index.max()+1)
        pred_price = self.df_price.iloc[idx_t0:idx_tend]["price_im"].values
        return pred_price

    def get_action(self, df_state: pd.DataFrame, t: int) -> npt.NDArray[Any]:
        lambda_lax = 0
        action = np.zeros(self.max_cars)
        occ_spots = df_state["idSess"] != -1 # Occupied spots
        num_cars = occ_spots.sum()
        if num_cars > 0:
            t_end = df_state[occ_spots]["t_dep"].max()
            n = int(t_end - t ) 
            pred_price = self._get_prediction(t, n)
            
            constraints = []
            AC = cp.Variable((num_cars, n)) # Charging action
            AD = cp.Variable((num_cars, n)) # Discharging action
            Y = cp.Variable((num_cars, n)) # Charging + discharging action
            SOC = cp.Variable((num_cars, n+1 ), nonneg=True) # SOC, need one more since action does not affect first col
            LAX = cp.Variable((num_cars), nonneg=True) # LAX, or cp Variable?

            # SOC limits
            constraints += [SOC >= 0]
            constraints += [SOC <= config.FINAL_SOC]

            # Define the first column of SOC_t0
            constraints += [SOC[:,0] == df_state[occ_spots]["soc_t"]]

            # Charging limits
            constraints += [AC >= 0]
            constraints += [AC <= config.alpha_c / config.B]

            # Discharging limits
            constraints += [AD <= 0]
            constraints += [AD >= -config.alpha_d / config.B]

            # Discharging ammount
            constraints += [ - cp.sum(AD, axis=1) / config.eta_d <= df_state[occ_spots]["soc_dis"]]

            for i, car in enumerate(df_state[occ_spots].itertuples()):
                # End charge
                j_end = int(min(car.t_dep - t, n)) # Only if n is in terms of t_dep and not t_dis
                constraints += [SOC[i, j_end:] == config.FINAL_SOC]
                # Discharging time
                j_dis = int(min(car.t_dis, n - 1)) # Only if n is in terms of t_dep and not t_dis
                constraints +=[AD[i, j_dis:] == 0]

                for j in range(n):
                    # Charge rule
                    constraints += [SOC[i, j+1] == SOC[i,j] + AC[i,j] * config.eta_c + AD[i,j] / config.eta_d] 

                if n > 0:
                    constraints += [LAX[i] == (car.t_dep - (t+1) ) - ((config.FINAL_SOC - SOC[i,1]) * config.B) / (config.alpha_c * config.eta_c)]

            constraints += [LAX >= 0]
            constraints += [Y == AC + AD]

            if self.myprint: 
                print(f"{num_cars=}, {n=}")
                print("pred_price=", end=' ')
                print(np.array2string(pred_price, separator=", "))

            objective = cp.Minimize(cp.sum(cp.multiply(np.asmatrix(pred_price), Y))) #  -lambda_lax*cp.sum(LAX)) # Laxity regularization
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.MOSEK, verbose=False)
            if prob.status != 'optimal':
                raise Exception("Optimal schedule not found")
                print("!!! Optimal solution not found")
            best_cost = prob.value
            Y_val = Y.value

            if self.myprint:
                print("AC=", end=' ')
                print(np.array2string(AC.value, separator=", "))
                print("AD=", end=' ')
                print(np.array2string(AD.value, separator=", "))
                print("LAX=", end=' ')
                print(np.array2string(LAX.value, separator=", "))
                print("SOC=", end=' ')
                print(np.array2string(SOC.value, separator=", "))
                print("best_cost=", end=' ')
                print(best_cost)

            j = 0
            for i, car in enumerate(df_state.itertuples()):
                if car.idSess != -1:
                    action[i] = Y_val[j,0] # MPC
                    j += 1
            
        return action

class agentNoV2G():
    def __init__(self, df_price, max_cars: int = config.max_cars, myprint = False):
        self.max_cars = max_cars
        self.df_price = df_price
        self.myprint = myprint

    def _get_prediction(self, t, n):
        l_idx_t0 = self.df_price.index[self.df_price["ts"] == t].to_list()
        assert len(l_idx_t0) == 1, "Timestep for prediction not unique or not existent"
        idx_t0 = l_idx_t0[0]
        idx_tend = min(idx_t0+n, self.df_price.index.max()+1)
        pred_price = self.df_price.iloc[idx_t0:idx_tend]["price_im"].values
        return pred_price

    def get_action(self, df_state: pd.DataFrame, t: int) -> npt.NDArray[Any]:
        lambda_lax = 0.001
        action = np.zeros(self.max_cars)
        occ_spots = df_state["idSess"] != -1 # Occupied spots
        num_cars = occ_spots.sum()
        if num_cars > 0:
            t_end = df_state[occ_spots]["t_dep"].max()
            n = int(t_end - t ) 
            pred_price = self._get_prediction(t, n)
            
            constraints = []
            AC = cp.Variable((num_cars, n), nonneg=True) # Action
            SOC = cp.Variable((num_cars, n+1 ), nonneg=True) # SOC, need one more since action does not affect first col
            LAX = cp.Variable((num_cars), nonneg=True) # LAX, or cp Variable?


            # SOC limits
            constraints += [SOC >= 0]
            constraints += [SOC <= config.FINAL_SOC]

            # Define the first column of SOC_t0
            constraints += [SOC[:,0] == df_state[occ_spots]["soc_t"]]

            # Charging limits
            constraints += [AC >= 0]
            constraints += [AC <= config.alpha_c / config.B]

            for i, car in enumerate(df_state[occ_spots].itertuples()):
                # End charge
                j_end = int(min(car.t_dep - t, n))
                constraints += [SOC[i, j_end:] == config.FINAL_SOC]

                for j in range(n):
                    # Charge rule
                    constraints += [SOC[i, j+1] == SOC[i,j] + AC[i,j] * config.eta_c] # Missing discharging

                if n > 0:
                    constraints += [LAX[i] == (car.t_dep - (t+1) ) - ((config.FINAL_SOC - SOC[i,1]) * config.B) / (config.alpha_c * config.eta_c)]

            constraints += [LAX >= 0]

            if self.myprint: 
                print(f"{num_cars=}, {n=}")
                print("pred_price=", end=' ')
                print(np.array2string(pred_price, separator=", "))

            objective = cp.Minimize(cp.sum(cp.multiply(np.asmatrix(pred_price), AC))) #  -lambda_lax*cp.sum(LAX)) # Laxity regularization
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.MOSEK, verbose=False)
            if prob.status != 'optimal':
                raise Exception("Optimal schedule not found")
                print("!!! Optimal solution not found")
            best_cost = prob.value
            AC_val = AC.value

            if self.myprint:
                print("AC=", end=' ')
                print(np.array2string(AC.value, separator=", "))
                print("LAX=", end=' ')
                print(np.array2string(LAX.value, separator=", "))
                print("SOC=", end=' ')
                print(np.array2string(SOC.value, separator=", "))
                print("best_cost=", end=' ')
                print(best_cost)

            j = 0
            for i, car in enumerate(df_state.itertuples()):
                if car.idSess != -1:
                    action[i] = AC_val[j,0] # MPC
                    j += 1
            
        return action
