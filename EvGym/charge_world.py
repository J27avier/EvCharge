import numpy as np 
import pandas as pd 
from . import config


class ChargeWorldEnv():                       
    def __init__(self, df_data, tinit = 0, max_cars=25, render_mode = None, state_mode = "full"):
        # Set action and obs spaces
        self.max_cars = max_cars
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.t = tinit
        self.df_data = df_data
        self.t_max = df_data['ts_arr'].max() 
        self.state_mode = state_mode
        self.df_park = self.init_park()

        # Charging vars
        self.alpha_c = config.alpha_c
        self.alpha_d = config.alpha_d
        self.max_soc = config.FINAL_SOC
        self.min_soc = 0

    def init_park(self):
        if self.state_mode == "full":
            df_park = pd.DataFrame(columns=config.car_columns_full)
            for i in range(self.max_cars):
                blank_session = Session()
                df_park = pd.concat((df_park,
                                     pd.DataFrame([blank_dession.flatten_full()],
                                     columns = config.car_columns_full)))
        elif self.mode == "simple":
            df_park = pd.DataFrame(columns=config.car_columns_simple)
            for i in range(self.max_cars):
                blank_session = Session()
                df_park = pd.concat((df_park, pd.DataFrame([blank_session.flatten_simple()], columns = config.car_columns_simple)))
            raise NotImplementedError

        return df_park.reset_index(drop=True)

    def reset(self):
        self.t = 0
        self.df_park = self.init_park()
        return self.df_park

    def step(self, action):
        # Apply action
        # self.state = self.apply_action(action)
        self.cars_charge(action)

        # Tick clock
        # self.t += 1
        self.t += 1

        # Calculate reward
        self.df_depart = self.df_park[self.df_park["t_dep"] == self.t]
        # reward = self.calc_reward(self.state)
        reward = 0

        # Check if done
        done = True if self.t > self.t_max else False

        # If not done update state
        if not done:
            self.cars_depart()
            self.cars_arrive()

        # Create placeholder info
        info = {}
        
        #self.df_park = self.df_park.sort_values(by=["t_arr"], ascending=False).reset_index(drop=True)
        return self.df_park.copy(), reward, done, info

    def cars_depart(self):
        blank_session = Session()
        self.df_park.loc[self.df_depart.index, config.car_columns_simple] = blank_session.flatten_simple()
        
        # Mask is too slow
        #self.df_park = self.df_park.mask(self.df_park["t_dep"] == self.t, blank_session.flatten_simple())
        # Funny edge case where sessions depart and arrive in the same timestep


    def cars_arrive(self):
        df_arrivals = self.df_data[self.df_data["TransactionStartTS"] == self.t] # This could be sped up a lot
        idx_empty = self.df_park[self.df_park["idSess"] == -1].index

        if len(df_arrivals) > len(idx_empty):
            raise Exception(f"Not enough {len(df_arrivals)} empty spots {len(idx_empty)} at timestep {self.t}!!")
        
        # Might be able to be a one liner with 
        for i, (_, arr_car) in enumerate(df_arrivals.iterrows()):
            sess = Session(idSess = arr_car.TransactionId,
                           C = arr_car.BatteryCapacity,
                           t_arr = arr_car.TransactionStartTS,
                           soc_arr = arr_car.SOC_arr,
                           t_dep = arr_car.TransactionStopTS,
                          )
            self.df_park.loc[idx_empty[i], config.car_columns_simple] = sess.flatten_simple()

    def cars_charge(self, action):
        assert len(action) == self.df_park.shape[0], "There must be as many actions as parking spots"
        self.power_t = np.zeros(self.max_cars)
        for i, (_, car) in enumerate(self.df_park.iterrows()):
            if car.idSess == -1 and action[i] > 0:
                raise Exception(f"Agent is trying to charge empty spot. Spot {i} at time {self.t}")

            soc_temp = car.soc_t + action[i]
            if self.min_soc > soc_temp or soc_temp > self.max_soc:
                print(f"Warning: Car {i} at time {self.t} would charge to {soc_temp}")
            soc_tminus1 = car.soc_t
            car.soc_t = max(self.min_soc, min(self.max_soc, car.soc_t + action[i]))
            self.power_t[i] = car.soc_t - soc_tminus1




# car_columns_full = ["idSess", "t_arr", "soc_arr", "E_arr", "t_dep", "E_rem", "soc_rem", "soc_t", "t_rem", ]
# car_columns_simple = ["idSess", "t_rem", "soc_rem"]
class Session():
    #def __init__(self, idSess=-1, C=80,  t_arr=0, soc_arr=0, t_dep=maxint):
    def __init__(self, idSess=-1, C=0, t_arr=0, soc_arr=0, t_dep=0):
        self.idSess = idSess
        self.t_arr = t_arr 
        self.t = self.t_arr
        self.soc_arr = soc_arr
        self.soc_t = soc_arr
        self.t_dep = t_dep
        self.B = config.B # kWh, Similar to Model 3
        soc_dep = config.FINAL_SOC
        self._calc_dependent()
    def _calc_dependent(self):
        self.t_soj = self.t_dep - self.t_arr
        self.E_arr = self.soc_arr * self.B
        self.E_t = self.soc_t * self.B
        self.E_req = self.B - self.E_t
        self.t_rem = self.t_dep - self.t

    def flatten_fully_obs(self):
        self._calc_dependent()
        return [self.idSess, self.B, self.t_arr, self.soc_arr, self.E_arr, self.t_dep, self.E_req, self.t_soj, self.E_t, self.soc_t, self.t_rem]

    def flatten_simple(self):
        # In simple case we dont calculate dependent
        return [self.idSess, self.t_arr, self.soc_arr, self.t_dep, self.soc_t]

