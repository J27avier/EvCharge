import numpy as np 
import pandas as pd 
from . import config
from tabulate import tabulate
from colorama import init, Back, Fore
import time
import os


class ChargeWorldEnv():                       
    def __init__(self, df_sessions, max_cars=25, render_mode = None, state_mode = "full"):
        # Not implemented, for pygame
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Constants
        self.max_cars = max_cars
        self.df_sessions = df_sessions
        self.tinit = self.df_sessions["ts_arr"].min()
        self.t_max = self.df_sessions['ts_dep'].max() 
        self.state_mode = state_mode
        self.alpha_c = config.alpha_c
        self.alpha_d = config.alpha_d
        self.max_soc = config.FINAL_SOC
        self.min_soc = config.MIN_SOC

        # Charging vars
        self.t = self.tinit
        self.df_park = self.init_park()
        self.df_park_lag = self.df_park.copy()

    def init_park(self):
        if self.state_mode == "full":
            df_park = pd.DataFrame(columns=config.car_columns_full)
            for i in range(self.max_cars):
                blank_session = Session()
                df_park = pd.concat((df_park,
                                     pd.DataFrame([blank_session.flatten_full()],
                                     columns = config.car_columns_full)))
        elif self.mode == "simple":
            df_park = pd.DataFrame(columns=config.car_columns_simple)
            for i in range(self.max_cars):
                blank_session = Session()
                df_park = pd.concat((df_park, pd.DataFrame([blank_session.flatten_simple()], columns = config.car_columns_simple)))
            raise NotImplementedError

        return df_park.reset_index(drop=True)

    def reset(self):
        self.tinit = self.df_sessions["ts_arr"].min()
        self.t = self.tinit
        self.df_park = self.init_park()
        self.df_park_lag = self.df_park.copy()
        return self.df_park.copy()

    def step(self, action):
        self.df_park_lag = self.df_park.copy()

        # Apply action
        # self.state = self.apply_action(action)

        self.cars_charge(action)

        # Tick clock
        self.t += 1

        # Calculate reward
        self.df_depart = self.df_park[self.df_park["t_dep"] == self.t]
        # reward = self.calc_reward(self.state)
        reward = 0

        # Check if done
        done = True if self.t > self.t_max else False

        # If not done update state
        if not done:
            #self.cars_depart()
            self.cars_arrive()

        # Create placeholder info
        info = {}
        
        #self.df_park = self.df_park.sort_values(by=["t_arr"], ascending=False).reset_index(drop=True)
        return self.df_park.copy(), reward, done, info

    def cars_depart(self):
        blank_session = Session()
        self.df_park.loc[self.df_depart.index, config.car_columns_simple] = blank_session.flatten_full()
        
        # Mask is too slow
        #self.df_park = self.df_park.mask(self.df_park["t_dep"] == self.t, blank_session.flatten_simple())
        # Funny edge case where sessions depart and arrive in the same timestep


    def cars_arrive(self):
        df_arrivals = self.df_sessions[self.df_sessions["ts_arr"] == self.t] # This could be sped up a lot
        idx_empty = self.df_park[self.df_park["idSess"] == -1].index

        if len(df_arrivals) > len(idx_empty):
            raise Exception(f"Not enough {len(df_arrivals)} empty spots {len(idx_empty)} at timestep {self.t}!!")
        
        # Might be able to be a one liner with 
        for i, (_, arr_car) in enumerate(df_arrivals.iterrows()):
            sess = Session(idSess = arr_car.session,
                           B = config.B,
                           t_arr = arr_car.ts_arr,
                           soc_arr = arr_car.soc_arr,
                           t_dep = arr_car.ts_dep,
                          )
            self.df_park.loc[idx_empty[i], config.car_columns_full] = sess.flatten_full()

    def cars_charge(self, action):
        assert len(action) == self.df_park.shape[0], "There must be as many actions as parking spots"
        self.power_t = np.zeros(self.max_cars)
        for i, (_, car) in enumerate(self.df_park.iterrows()):
            if car.idSess == -1 and action[i] > 0:
                raise Exception(f"Agent is trying to charge empty spot. Spot {i} at time {self.t}")

            # Clip action between min and max charging
            action_clip = max(min(self.alpha_c, action[i]), -self.alpha_d)
            if action_clip >= 0:
                soc_temp = car.soc_t + action_clip * config.eta_c
            else:
                soc_temp = car.soc_t - action_clip / config.eta_d

            # Check soc bounds
            if self.min_soc > soc_temp or soc_temp > self.max_soc:
                print(f"Warning: Car {i} at time {self.t} would charge to {soc_temp}")
            soc_t_lag = car.soc_t

            # Update soc
            car.soc_t = max(self.min_soc, min(self.max_soc, soc_temp))
            self.power_t[i] = car.soc_t - soc_t_lag

    def print(self, wait=0, clear = True):
        delta_park = self.df_park["idSess"] != self.df_park_lag["idSess"]
        lst_print = []
        for i, row in self.df_park.iterrows():
            row_print = []
            if delta_park[i]:
                if row.idSess == -1:
                    row_print.append(Back.RED + str(row.idSess) + Back.RESET)
                else: 
                    row_print.append(Back.BLUE + str(row.idSess) + Back.RESET)
            else:
                row_print.append(str(row.idSess))
            for col in config.car_columns_full[1:]:
                if type(row[col]) == float:
                    row_print.append(f"{row[col]:.2f}")
                else:
                    row_print.append(row[col])
            lst_print.append(row_print)

        print(f"t = {self.t:.0f}")
        print(tabulate(lst_print, tablefmt = "grid", headers=config.car_columns_full))

        if wait == -1:
            input()
        elif wait > 0:
            time.sleep(wait)
        if clear:
            os.system("clear")
        else:
            print("============================================================")


# car_columns_full = ["idSess", "B", "t_arr", "soc_arr", "E_arr", "t_dep", "E_rem", "soc_rem", "self.E_t", "soc_t", "t_rem", ]
# car_columns_simple = ["idSess", "t_rem", "soc_rem"]
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
        self.E_req = self.B - self.E_t
        self.t_rem = self.t_dep - self.t
        self.soc_rem = self.soc_dep - self.soc_t
        self.E_rem = self.soc_rem * self.B

    def flatten_full(self):
        self._calc_dependent()
        return [self.idSess, self.B, self.t_arr, self.soc_arr, self.E_arr, self.t_dep, self.E_rem, self.soc_rem, self.E_t, self.soc_t, self.t_rem]

    def flatten_simple(self):
        # In simple case we dont calculate dependent
        return [self.idSess, self.t_arr, self.soc_arr, self.t_dep, self.soc_t]

