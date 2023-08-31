import numpy as np
import cvxpy as cp #type: ignore
import pandas as pd
from tqdm import tqdm
import argparse


import sys
sys.path.append("..")
from EvGym import config #type: ignore
from ContractDesign.time_contracts import general_contracts #type: ignore

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--file_price", help = "Name of imbalance price dataframe", 
                        type=str, default= "df_price_2019.csv")
    return parser.parse_args()

def main():
    df_sessions = pd.read_csv(f"../{config.data_path}df_elaad_preproc.csv", parse_dates = ["starttime_parking", "endtime_parking"])
    ts_max = df_sessions["ts_dep"].max()
    ts_min = df_sessions["ts_arr"].min() 
    args = parse_args()
    df_price = pd.read_csv(f"../{config.data_path}{args.file_price}", parse_dates=["date"])

    n = int(ts_max - ts_min)
    num_cars = df_sessions.shape[0]

    price_im = df_price["price_im"].to_numpy()


    constraints = []
    AC = cp.Variable((num_cars, n)) # Charging action
    AD = cp.Variable((num_cars, n)) # Discharging action
    Y = cp.Variable((num_cars, n)) # Charging + discharging action
    SOC = cp.Variable((num_cars, n+1 ), nonneg=True) # SOC, need one more since action does not affect first col

    for i, car in enumerate(df_sessions.itertuples()):
        t_arr = int(car.ts_arr - ts_min)
        t_dep = int(car.ts_dep - ts_min)
        if t_arr > 0: constraints += [SOC[i,:t_arr] == 0]
        constraints += [SOC[i, t_dep] == config.FINAL_SOC]
        constraints += [SOC[i, t_dep+1:] == 0]
        



if __name__ == "__main__":
    main()
