import numpy as np
import pandas as pd
from . import config
from tabulate import tabulate
from colorama import init, Back, Fore
from typing import TYPE_CHECKING, Any, Optional, Dict, Tuple, Union, List
import time
import os
from math import isclose

# User defined
from .exp_tracker import ExpTracker

# From folder above
import sys
sys.path.append("..")
from ContractDesign.time_contracts import general_contracts, u_ev_general

class ChargeWorldEnv():
    def __init__(self, df_sessions: pd.DataFrame, df_price: pd.DataFrame, contract_info, rng,
                 max_cars: int = config.max_cars, skip_contracts = False, render_mode: Optional[str]  = None, lax_coef = 0, norm_reward = False):
        # Not implemented, for pygame
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Rng
        self.rng = rng

        # Data frames
        self.df_sessions = df_sessions
        self.df_price = df_price.set_index("ts")

        # Constants
        self.max_cars = max_cars
        self.tinit = self.df_sessions["ts_arr"].min() - 1
        self.t_max = self.df_sessions['ts_dep'].max()
        self.alpha_c = config.alpha_c
        self.alpha_d = config.alpha_d
        self.max_soc = config.FINAL_SOC
        self.min_soc = config.MIN_SOC

        # Charging vars
        self.t = self.tinit
        self.df_park = self._init_park()
        self.df_park_lag = self.df_park.copy()
        self.action = np.zeros(self.max_cars)

        # Money tracking vars
        self.tracker = ExpTracker(self.tinit, self.t_max)

        # Contract info
        self.G = contract_info["G"]
        self.W = contract_info["W"]
        self.L = contract_info["L"]
        self.count_I = len(self.W)
        self.count_J = len(self.L)
        self.skip_contracts = skip_contracts

        # Environment args
        self.lax_coef = lax_coef
        self.norm_reward = norm_reward

    def _init_park(self):
        df_park = pd.DataFrame(columns=config.car_columns_full)
        for i in range(self.max_cars):
            blank_session = Session()
            df_park = pd.concat((df_park,
                                 pd.DataFrame([blank_session.flatten_full()],
                                 columns = config.car_columns_full)))

        # Initialize columns that are used in simulation
        assert len(config.car_columns_proc) == len(config.car_columns_proc_default), "Lists config.car_columns_proc and config.car_columns_proc_default must have the same number of items"
        for col_d, val_d in zip(config.car_columns_proc, config.car_columns_proc_default):
            df_park[col_d] = val_d

        return df_park.reset_index(drop=True)

    def reset(self):
        """ Resets the parking lot to be empty """

        self.tinit = self.df_sessions["ts_arr"].min() - 1
        self.t = self.tinit
        self.df_park = self._init_park()
        self.df_park_lag = self.df_park.copy()
        return self.df_park.copy()

    def step(self, action) -> Tuple[pd.DataFrame, config.Number, bool, dict]:
        """ Applies the charging action, cars depart and then new cars arrive """

        self.df_park_lag = self.df_park.copy()

        # Apply action
        self.action = action # For future reference
        self._cars_charge(action)

        # Tick clock
        self.t += 1

        # Calculate reward
        reward = self._reward()
        # reward = self.calc_reward(self.state)

        self.df_depart = self.df_park[self.df_park["t_dep"] == self.t]

        # Check if done
        done = True if self.t > self.t_max else False

        # If not done update state
        if not done:
            self._cars_depart()
            self._cars_arrive()

        # Create placeholder info
        info = {"t": self.t}

        #self.df_park = self.df_park.sort_values(by=["t_arr"], ascending=False).reset_index(drop=True)
        return self.df_park.copy(), reward, done, info

    def _reward(self):
        occ_spots = self.df_park["idSess"] != -1 # Occupied spots
        n_cars = occ_spots.sum()
        reward = -self.imb_transf
        if self.norm_reward and n_cars > 0:
            reward /= n_cars

        return reward + self.lax_coef*self.df_park["lax"].sum()


    def _cars_depart(self):
        blank_session = Session()
        #Check that cars have finished charging
        df_unfinished_chrg = self.df_depart[~(np.isclose(self.df_depart["soc_t"], config.FINAL_SOC))]
        if len(df_unfinished_chrg) > 0:
            print(df_unfinished_chrg)
            raise Exception(f"Some cars ({len(df_unfinished_chrg)}) have not finished charging at time {self.t}")

        # Calculate payoff with a little bit of pandas magic
        if len(self.df_depart) > 0:
            payoff = self.df_park.loc[self.df_depart.index].apply(lambda dep_car: self.G[dep_car.idx_theta_w, dep_car.idx_theta_l]\
                                                                  if dep_car.idx_theta_w >= 0 and dep_car.idx_theta_l >= 0 else 0, axis = 1).sum()
        else:
            payoff = 0

        self.df_park.loc[self.df_depart.index, config.car_columns_full + config.car_columns_proc] = blank_session.flatten_full() + config.car_columns_proc_default

        self.tracker.dep_bill.append([self.t, payoff]) 

    def _cars_arrive(self):
        df_arrivals = self.df_sessions[self.df_sessions["ts_arr"] == self.t] # This could be sped up a lot
        idx_empty = self.df_park[self.df_park["idSess"] == -1].index
        arr_e_req = 0
        
        assigned_type = np.zeros((self.count_I, self.count_J))
        realized_type = np.zeros((self.count_I, self.count_J))
        fail_time = 0
        fail_energy1 = 0
        fail_energy2 = 0
        fail_energy_both = 0
        fail_IR = 0
      
        if len(df_arrivals) > len(idx_empty):
            raise Exception(f"Not enough {len(df_arrivals)} empty spots {len(idx_empty)} at timestep {self.t}!!")
        
        # Might be able to be a one liner with 
        #for i, (_, arr_car) in enumerate(df_arrivals.iterrows()):
        for i, arr_car in enumerate(df_arrivals.itertuples()):
            sess = Session(idSess = arr_car.session,
                           B = config.B,
                           t_arr = arr_car.ts_arr,
                           soc_arr = arr_car.soc_arr,
                           t_dep = arr_car.ts_dep,
                          )
            
            # Contracts
            if not self.skip_contracts:
                idx_theta_w = self.rng.choice(list(range(self.count_I)))
                idx_theta_l = self.rng.choice(list(range(self.count_J)))
                assigned_type[idx_theta_w, idx_theta_l] += 1
                lax = sess.t_soj - ((config.FINAL_SOC - sess.soc_arr) * config.B) / (config.alpha_c * config.eta_c)
                xi_max = lax * config.psi * config.alpha_d
                
                flag_fail = False
                
                while True:
                    # Contract parameters
                    w = self.W[idx_theta_w] 
                    soc_dis = w / config.B
                    t_dis = self.L[idx_theta_l]
                    theta_w, theta_l = config.thetas_i[idx_theta_w], config.thetas_j[idx_theta_l]
                    
                    # Entry checks
                    check_time = sess.t_soj >= t_dis
                    check_energy1 = sess.soc_arr >= soc_dis
                    check_energy2 = xi_max >= w
                    check_IR = u_ev_general(self.G[idx_theta_w, idx_theta_l], w, t_dis, theta_w, theta_l, c1 = config.c1, c2 = config.c2) >= 0
                    
                    # Contract is accepted 
                    if check_time and check_energy1 and check_energy2 and check_IR: break
                    
                    # Otherwise
                    # Slide back in time
                    if (not check_time) or (not check_IR): idx_theta_l -= 1
                    
                    # Slide back in energy
                    if (not check_energy1) or (not check_energy2) or (not check_IR): idx_theta_w -= 1 
                    
                    # Fail reason
                    if idx_theta_l < 0: 
                        fail_time += 1
                        flag_fail = True
                    
                    if idx_theta_w < 0:
                        fail_energy_both += 1
                        flag_fail = True
                        if not check_energy1: fail_energy1 += 1
                        if not check_energy2: fail_energy2 += 1
                    
                    # Exit with no other options
                    if flag_fail:
                        if not check_IR: fail_IR += 1
                        soc_dis, t_dis = 0, 0
                        break
                    
                if not flag_fail:
                    realized_type[idx_theta_w, idx_theta_l] += 1
            else:
                idx_theta_w, idx_theta_l = -1, -1
                lax, soc_dis, t_dis = 0, 0, 0
            
            self.df_park.loc[idx_empty[i], config.car_columns_full] = sess.flatten_full()
            self.df_park.loc[idx_empty[i], config.car_columns_proc] = [lax, soc_dis, t_dis, idx_theta_w, idx_theta_l]
            self. tracker.contract_log.append([sess.idSess, soc_dis, t_dis, self.G[idx_theta_w, idx_theta_l],
                                               idx_theta_w, idx_theta_l])
            
            arr_e_req += sess.E_req # Accumulate arrival demand
        
        if self.skip_contracts: self._update_lax()
        
        # Experiment tracking
        self.tracker.arr_bill.append([self.t, arr_e_req, arr_e_req * config.elec_retail, assigned_type, realized_type,
                                      fail_time, fail_energy1, fail_energy2, fail_energy_both, fail_IR])


    def _cars_charge(self, action):
        assert len(action) == self.df_park.shape[0], "There must be as many actions as parking spots"
        self.power_t = np.zeros(self.max_cars)
        occ_spots = self.df_park["idSess"] != -1 # Occupied spots
        if (action[~occ_spots] != 0).any():
            raise Exception(f"Agent is trying to charge empty spot. Spots {action[~occ_spots]} at time {self.t}")

        action_clip = np.clip(action, -self.alpha_d / config.B, self.alpha_c / config.B)
        total_action = np.sum(action_clip)
        if self.tinit < self.t <= self.t_max:
            price_im = self.df_price.loc[self.t].price_im
        else:
            price_im = 0
        self.imb_transf = total_action * price_im * config.B # Transfer to imbalance market
        # Multiplied by config.B because action is in SOC units, not in kWh

        # Charging and discharging inefficiencies
        soc_t = self.df_park["soc_t"].to_numpy()
        soc_temp = np.zeros(soc_t.shape[0])
        idx_dis = []
        for i, act_c in enumerate(action_clip):
            if act_c >= 0:
                soc_temp[i] = soc_t[i] + act_c * config.eta_c
            else:
                soc_temp[i] = soc_t[i] + act_c / config.eta_d
                if act_c < -config.tol: idx_dis.append(i)

            # Check soc bounds
            if soc_temp[i] < self.min_soc - config.tol or self.max_soc + config.tol < soc_temp[i]:
                print(f"Warning: Car {i} at time {self.t} would charge to {soc_temp[i]}")

        # Check time of contract
        idx_zero_t_dis = self.df_park[self.df_park["t_dis"] == 0].index
        viol_t_dis = list(set(idx_zero_t_dis).intersection(set(idx_dis)))
        if len(viol_t_dis) > 0:
            print(self.df_park.iloc[viol_t_dis])
            print(action_clip[viol_t_dis])
            raise Exception("Contract time exceeded")

        # Check energy of contract
        action_dis = np.minimum(action_clip, 0)
        if any(self.df_park["soc_dis"] + action_dis / config.eta_d < -config.tol):
            print(self.df_park[self.df_park["soc_dis"] < 0])
            raise Exception("Contract discharge energy exceeded")

        # Update contract parameters
        contract_spots = self.df_park["t_dis"] > 0
        self.df_park.loc[contract_spots, "t_dis"] = self.df_park[contract_spots]["t_dis"] - 1
        self.df_park["soc_dis"] = self.df_park["soc_dis"] + action_dis / config.eta_d

        soc_t_lag = soc_t

        # Update df_park
        soc_temp_clip = np.clip(soc_temp, self.min_soc, self.max_soc)
        self.df_park["soc_t"] = soc_temp_clip
        self.df_park.loc[occ_spots, "E_t"]  = self.df_park.loc[occ_spots]["soc_t"]  *  self.df_park.loc[occ_spots]["B"]
        self.df_park.loc[occ_spots, "soc_rem"] = config.FINAL_SOC - self.df_park.loc[occ_spots]["soc_t"]
        self.df_park.loc[occ_spots, "E_rem"] = self.df_park.loc[occ_spots]["soc_rem"] * self.df_park.loc[occ_spots]["B"]
        self.df_park.loc[occ_spots, "t_rem"] = self.df_park.loc[occ_spots]["t_dep"] - self.t - 1
        self.power_t = soc_temp_clip - soc_t_lag
        self._update_lax() # Updates the laxity of the cars
        avg_lax = self.df_park["lax"].mean()
        self.tracker.chg_bill.append([self.t, total_action, self.imb_transf, occ_spots.sum(), avg_lax])


    def print(self, wait=0, clear = True):
        # Get the difference of parking lot now and one step before
        delta_park = self.df_park["idSess"] != self.df_park_lag["idSess"]
        colors = []
        #for i, row in self.df_park.iterrows(): # Get color for each row
        for row in self.df_park.itertuples(): # Get color for each row
            row_print = []
            color = ""
            # Choose color
            if delta_park[row.Index]: # previously if delta_park[i]
                if row.idSess == -1:
                    # if a car left color it red
                    color  = Back.RED
                else:
                    # else color it blue
                    color = Back.BLUE
            if row.t_rem == 1:
                color = Back.YELLOW
            colors.append(color)
        lst_print = self._make_park_lst_colors(self.df_park, colors)

        # Print state
        print(f"t = {self.t:,.0f}")
        print(tabulate(lst_print, tablefmt = "grid",
                                  headers=config.car_columns_full_lag + config.car_columns_proc + ["A"]))
        print("----")
        print("Departed")

        # Print departed cars
        colors = []
        #for _, row in self.df_depart.iterrows(): # Get color for each row
        for row in self.df_depart.itertuples(): # Get color for each row
            row_print = []
            color = ""
            if isclose(row.soc_t, config.FINAL_SOC, abs_tol = config.tol):
                color = Back.GREEN
            else:
                color = Back.MAGENTA
            colors.append(color)

        lst_print = self._make_park_lst_colors(self.df_depart, colors)
        print(tabulate(lst_print, tablefmt = "grid", headers=config.car_columns_full_lag + config.car_columns_proc + ["A"] ))

        usr_in = ""
        # -1 means wait for user input
        if wait == -1:
            usr_in = input()
        elif wait > 0:
            time.sleep(wait)

        # Clear the screen or print a divider
        if clear:
            os.system("clear")
        else:
            print("="*80)

        return usr_in


    def _make_park_lst_colors(self, df_data: pd.DataFrame, colors: list) -> list:
        lst_print = []
        for i, (_, row) in enumerate(df_data.iterrows()):
            row_print = []
            color = colors[i]

            row_print.append(color + f"{row[config.car_columns_full_lag[0]]}")
            found_lag = self.df_park_lag[self.df_park_lag["idSess"] == row["idSess"]]
            for col in config.car_columns_full_lag[1:] + config.car_columns_proc:
                if col in ["t_arr", "t_dep", "t_rem", "t_dis"]: # Time is printed as int
                    row_print.append(f"{row[col]:,.0f}")
                elif col in ["soc_lag"]:
                    if len(found_lag > 0):
                        row_lag = found_lag.iloc[0]
                        row_print.append(f"{row_lag['soc_t']:,.2f}")
                    else:
                        row_print.append(f"-")
                elif type(row[col]) in [float, np.float64]:
                    row_print.append(f"{row[col]:,.2f}")
                else:
                    row_print.append(row[col])
            if len(found_lag) > 0 :  
                row_print.append(f"{self.action[i]:,.2f}")
            else:
                row_print.append(f"-")

            row_print[-1] += Back.RESET

            lst_print.append(row_print)
        return lst_print

    def _update_lax(self):
        occ_spots = self.df_park["idSess"] != -1 # Occupied spots
        self.df_park.loc[occ_spots, "lax"] = self.df_park.loc[occ_spots]["t_rem"] \
                                             - ((config.FINAL_SOC - self.df_park.loc[occ_spots]["soc_t"])*config.B) \
                                             /(config.alpha_c * config.eta_c)
        if any(self.df_park[occ_spots]["lax"] < - config.tol): # Some tolerance because f numerical error in division
            print(self.df_park[self.df_park["lax"] < 0])
            raise Exception("Negative laxity detected")

# car_columns_full = ["idSess", "B", "t_arr", "soc_arr", "E_arr", "t_dep", "E_rem", "soc_rem", "self.E_t", "soc_t", "t_rem", ]
class Session():
    #def __init__(self, idSess=-1, C=80,  t_arr=0, soc_arr=0, t_dep=maxint):
    def __init__(self, idSess=-1, B=0, t_arr=0, soc_arr=0, t_dep=0):
        self.idSess = idSess
        self.t_arr = t_arr
        self.t = self.t_arr
        self.soc_arr = soc_arr
        self.soc_t = soc_arr
        self.t_dep = t_dep
        self.B = B # kWh, Similar to Model 3
        self.soc_dep = config.FINAL_SOC
        self._calc_dependent()

    def _calc_dependent(self):
        self.t_soj = self.t_dep - self.t_arr
        self.E_arr = self.soc_arr * self.B
        self.E_t = self.soc_t * self.B
        self.soc_req = self.soc_dep - self.soc_t
        self.E_req = self.B * self.soc_req
        self.t_rem = self.t_dep - self.t
        self.soc_rem = self.soc_dep - self.soc_t
        self.E_rem = self.soc_rem * self.B

    def flatten_full(self):
        self._calc_dependent()
        return [self.idSess, self.B, self.t_arr, self.soc_arr, self.E_arr, self.t_dep, self.E_rem, self.soc_rem, self.E_t, self.soc_t, self.t_rem]
