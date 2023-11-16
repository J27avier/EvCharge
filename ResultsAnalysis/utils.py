import pandas as pd
import numpy as np

import sys
sys.path.append("..")
from EvGym import config # type: ignore

def load_res(name, path="../ExpLogs"):
    df_res = pd.read_csv(f"{path}/{name}.csv")

    try:
        with open(f"{path}/{name}.txt", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
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

def drawLearn(name, count, ax, label =""):
    l_sum = [f"{name}_{i}" for i in range(count)] 
    df_sum = summ_table(l_sum, l_sum) 
    if label == "":
        label = name.split('/')[-1]
    ax.plot(df_sum["transf"], label=label)
    return ax

def lat_val(l_names, l_counts):
    l_fnames = [f"{name}_{count-1}" for name, count in zip(l_names, l_counts)]
    df_sum = summ_table(l_fnames, l_fnames)
    return df_sum["transf"].values

def taguchi_l18(values):
    l_table = [[1,1,1,1,1,1,1],
               [1,1,2,2,2,2,2],
               [1,1,3,3,3,3,3],
               [1,2,1,1,2,3,3],
               [1,2,2,2,3,1,1],
               [1,2,3,3,1,2,2],
               [1,3,1,2,3,2,3],
               [1,3,2,3,1,3,1],
               [1,3,3,1,2,1,2],
               [2,1,1,3,2,2,1],
               [2,1,2,1,3,3,2],
               [2,1,3,2,1,1,3],
               [2,2,1,2,1,3,2],
               [2,2,2,3,2,1,3],
               [2,2,3,1,3,2,1],
               [2,3,1,3,3,1,2],
               [2,3,2,1,1,2,3],
               [2,3,3,2,2,3,1],]

    results = np.zeros((7,3))
    for i in range(18):
        for j in range(7):
            results[j][l_table[i][j]-1] += values[i]

    return results.T / 6
    
    
 
# logstd	gamma	gae_lambda	clip_coef	vf_coef	max_grad_norm	ent_coef
# -1.8  	0.99	0.95	        0.1	        0.3	0.3	        0
# -2.2  	0.95	0.9	        0.2	        0.5	0.5	        0.01
#       	0.9	0.85	        0.3	        0.7	0.7	        0.1
    
    
    
    
    
    
    
    
    
    
    
    
    
    

