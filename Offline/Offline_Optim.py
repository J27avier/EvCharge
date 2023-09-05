import numpy as np
import cvxpy as cp #type: ignore
import pandas as pd
from tqdm import tqdm
import argparse
import os
import pickle

import sys
sys.path.append("..")
from EvGym import config #type: ignore
from ContractDesign.time_contracts import general_contracts #type: ignore

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-R", "--print-dash", help = "Print dashboard", action="store_true")
    parser.add_argument("-I", "--file_price", help = "Name of imbalance price dataframe", 
                        type=str, default= "df_price_2019.csv")
    return parser.parse_args()

def print_welcome(df_sessions, df_price, info):
    os.system("clear")
    print("df_sessions")
    #print(df_sessions.describe())
    print("="*80)
    print("df_price")
    #print(df_price.describe())
    print("="*80)
    print("\tInfo")
    for key in info.keys():
        print(f"{key} = {info[key]}")

# Running for 90+ hours, no solution. Must break down in scrolling time windows
def main():
    df_sessions = pd.read_csv(f"../{config.data_path}df_elaad_preproc.csv", parse_dates = ["starttime_parking", "endtime_parking"])
    ts_min = df_sessions["ts_arr"].min() 
    ts_max = df_sessions["ts_dep"].max()
    args = parse_args()
    df_price = pd.read_csv(f"../{config.data_path}{args.file_price}", parse_dates=["date"])

    n = int(ts_max - ts_min)
    num_cars = df_sessions.shape[0]

    price_im = df_price["price_im"].to_numpy()[:n]

    info = {}
    info["ts_min"] = ts_min
    info["ts_max"] = ts_max
    info["n"] = n
    info["num_cars"] = num_cars
    info["shape(price_im)"] = price_im.shape

    if args.print_dash:
        print_welcome(df_sessions, df_price, info)
    
    constraints = []
    AC = cp.Variable((num_cars, n)) # Charging action
    AD = cp.Variable((num_cars, n)) # Discharging action
    Y = cp.Variable((num_cars, n)) # Charging + discharging action
    SOC = cp.Variable((num_cars, n+1 ), nonneg=True) # SOC, need one more since action does not affect first col

    # SOC limits
    constraints += [SOC >= 0]
    constraints += [SOC <= config.FINAL_SOC]

    # Charging limits
    constraints += [AC >= 0]
    constraints += [AC <= config.alpha_c / config.B]

    # Discharging
    constraints += [AD == 0] # For now, don't allow discharging

    print("Adding constraints")
    for i, car in tqdm(enumerate(df_sessions.itertuples()), total= num_cars, miniters=100):
        t_arr = int(car.ts_arr - ts_min)
        t_dep = int(car.ts_dep - ts_min)

        # Before arrival
        if t_arr > 0: constraints += [SOC[i,:t_arr] == 0]

        # Connected limits
        constraints += [SOC[i, t_arr] == car.soc_arr]
        constraints += [SOC[i, t_dep] == config.FINAL_SOC]

        # After departure
        if t_dep < n: constraints += [SOC[i, t_dep+1:] == 0]

        # Connected
        for j in range(t_arr, t_dep):
            constraints += [SOC[i,j+1] == SOC[i,j] + AC[i,j] * config.eta_c - AD[i,j] / config.eta_d]

        # We don't need laxity-lookahead since all charging deadlines are guaranteed by constraints 

    constraints += [Y == AC + AD]

    print("Solving problem")
    print(f"{Y.shape=}")
    print(f"{np.asmatrix(price_im).shape=}")
    print(f"{cp.multiply(np.asmatrix(price_im), Y).shape=}")
    print(f"{cp.sum(cp.multiply(np.asmatrix(price_im), Y)).shape=}")
    objective = cp.Minimize(cp.sum(cp.multiply(np.asmatrix(price_im), Y)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.MOSEK, verbose = True)
    if prob.status != "optimal":
        raise Exception("Optimal schedule not found")

    best_cost = prob.value
    print("SOLUTION")
    print(best_cost)

    result = {"best_cost": best_cost,
              "AC_val": AC.val,
              "AD_val": AD.val,
              "Y_val": Y.val,
              "SOC_val": SOC.val}

    with open('res/offline_nv2g.pkl', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
