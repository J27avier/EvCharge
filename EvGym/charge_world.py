import numpy as np
import pandas as pd
from dataclasses import dataclass
#import pygame;

maxint = 9223372036854775807
car_columns_fully_obs = ["idSess", "C", "t_arr", "soc_arr", "E_arr", "t_dep", "E_req", "t_soj", "E_t", "soc_t", "t_rem"]

class ChargeWorldEnv():                       
    def __init__(self, data, tinit = 0, max_cars=25, minP=0, maxP=10, render_mode = None):
        self.max_cars = max_cars
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.t = tinit
        self.data = data
        self.df_active = self.init_active()

    def init_active(self):
        df_active = pd.DataFrame(columns=car_columns_fully_obs)
        for i in range(self.max_cars):
            blank_session = Session()
            df_active = pd.concat((df_active, pd.DataFrame([blank_session.flatten_fully_obs()], columns = car_columns_fully_obs)))
        return df_active.reset_index(drop=True)

    def step(self):
        # Apply action
        # self.state = self.apply_action(action)

        # Tick clock
        # self.t += 1

        # Calculate reward
        # reward = self.calc_reward(self.state)

        # Check if done
        # if self.t >= self.max_t: done = True; else: False

        # If not done
        # self.state = self.update_state()

        # Create placeholder info
        # info = {}
        self.cars_depart()
        self.cars_arrive()
        self.df_active = self.df_active.sort_values(by=["t_arr"], ascending=False).reset_index(drop=True)
        self.t += 1
        # return self.state, reward, done, info

    def cars_depart(self):
        df_departures = self.data[self.data["TransactionStopTS"] == self.t]
        
        for _, dep_car in df_departures.iterrows():
            blank_session = Session()
            df_occupied = self.df_active[self.df_active["idSess"] != -1]
            self.df_occupied = df_occupied
            if len(df_occupied) == 0:
                raise Exception("No cars inside world")
            else:
                idx_leaving = df_occupied[df_occupied["idSess"] == dep_car.TransactionId].index[0]
                self.df_active.loc[idx_leaving, car_columns_fully_obs] = blank_session.flatten_fully_obs()
    # Funny edge case where sessions depart and arrive in the same timestep


    def cars_arrive(self):
        df_arrivals = self.data[self.data["TransactionStartTS"] == self.t]
        
        for _, arr_car in df_arrivals.iterrows():
            sess = Session(idSess = arr_car.TransactionId,
                           C = arr_car.BatteryCapacity,
                           t_arr = arr_car.TransactionStartTS,
                           soc_arr = arr_car.SOC_arr,
                           t_dep = arr_car.TransactionStopTS,
                          )
            df_empty = self.df_active[self.df_active["idSess"] == -1]
            if len(df_empty) == 0:
                raise Exception("No empty slots for arrivals!")
            else:
                idx_empty = df_empty.index[0]
                self.df_active.loc[idx_empty, car_columns_fully_obs] = sess.flatten_fully_obs()

    
    def reset(self):
        raise NotImplementedError



# car_columns_fully_obs = ["idSess", "C", "t_arr", "soc_arr", "E_arr", "t_dep", "E_req", "t_soj", "E_t", "soc_t", "t_rem"]
class Session():
    #def __init__(self, idSess=-1, C=80,  t_arr=0, soc_arr=0, t_dep=maxint):
    def __init__(self, idSess=-1, C=0, t_arr=0, soc_arr=0, t_dep=0):
        self.idSess = idSess
        self.t_arr = t_arr 
        self.t = self.t_arr
        self.soc_arr = soc_arr
        self.soc_t = soc_arr
        self.t_dep = t_dep
        self.C = C # kWh, Similar to Model 3
        self._calc_dependent()
    def _calc_dependent(self):
        self.t_soj = self.t_dep - self.t_arr
        self.E_arr = self.soc_arr * self.C
        self.E_t = self.soc_t * self.C
        self.E_req = self.C - self.E_t
        self.t_rem = self.t_dep - self.t

    def flatten_fully_obs(self):
        self._calc_dependent()
        return [self.idSess, self.C, self.t_arr, self.soc_arr, self.E_arr, self.t_dep, self.E_req, self.t_soj, self.E_t, self.soc_t, self.t_rem]

