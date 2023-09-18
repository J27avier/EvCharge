import pandas as pd

import sys
sys.path.append("..")
from EvGym import config # type: ignore

def load_res(name, path="../ExpLogs"):
    df_res = pd.read_csv(f"{path}/{name}.csv")
    with open(f"{path}/{name}.txt", "r") as f:
        lines = f.readlines()
    return df_res, lines

def summ_table(l_names, l_files, verbose = False):
    assert len(l_names) == len(l_files), "Names and files lists should have same length"
    sum_cols = ["exp", "transf", "client", "payoff"]

    l_sum = []

    for file, name in zip(l_files, l_names):
        df_exp, lines = load_res(file)
        if verbose: 
            for line in lines: print(line, end="")
        transf = df_exp["imbalance_bill"].sum()
        client = df_exp["client_bill"].sum()
        if name not in ["ASAP", "NoV2G"]:
            payoff = df_exp["payoff"].sum()
        else:
            payoff = 0

        l_sum.append([name, transf, client, payoff])
    df_sum = pd.DataFrame(l_sum, columns = ["name", "transf", "client", "payoff"])
    df_sum["total"] = df_sum["client"] - df_sum["transf"] - df_sum["payoff"]
    return df_sum


