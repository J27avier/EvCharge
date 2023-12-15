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
        #if name not in ["ASAP", "NoV2G"]:
        if not (("ASAP" in name) or ("No-V2G" in name) or ("NoV2G" in name)):
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

def sttrLearn(name, count, ax, label ="", x=0):
    l_sum = [f"{name}_{i}" for i in range(count)] 
    df_sum = summ_table(l_sum, l_sum) 
    if label == "":
        label = name.split('/')[-1]
    ax.scatter(np.arange(len(df_sum["transf"]))+x, df_sum["transf"], label=label)
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

def taguchi_l7(values):
    taguchi7 = [
    [0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 1, 1, 1, 1,],
    [0, 1, 1, 0, 0, 1, 1,],
    [0, 1, 1, 1, 1, 0, 0,],
    [1, 0, 1, 0, 1, 0, 1,],
    [1, 0, 1, 1, 0, 1, 0,],
    [1, 1, 0, 0, 1, 1, 0,],
    [1, 1, 0, 1, 0, 0, 1,],
    ]

    results = np.zeros(7,2)
    for i in range(8):
        for j in range(7):
            results[j][taguchi7[i]] += values[i]

    return values

# logstd	gamma	gae_lambda	clip_coef	vf_coef	max_grad_norm	ent_coef
# -1.8  	0.99	0.95	        0.1	        0.3	0.3	        0
# -2.2  	0.95	0.9	        0.2	        0.5	0.5	        0.01
#       	0.9	0.85	        0.3	        0.7	0.7	        0.1
 

def load_rl_gen(name, num, dir = "", month = False, norm_dict = None, val = True):
    train_str = [f"./../ExpLogs/{dir}summ_train_{name}_{i}.csv" for i in range(num)]
    if val: val_str   = [f"./../ExpLogs/{dir}summ_val_{name}_{i}.csv" for i in range(num)]
    test_str  = [f"./../ExpLogs/{dir}summ_test_{name}_{i}.csv" for i in range(num)]

    df_train = pd.concat([pd.read_csv(i) for i in train_str], axis=0).reset_index(drop=True)
    if val: df_val = pd.concat([pd.read_csv(i) for i in val_str], axis=0).reset_index(drop=True)
    df_test = pd.concat([pd.read_csv(i) for i in test_str], axis=0).reset_index(drop=True)

    if norm_dict is not None:
        df_train["transf"] /= norm_dict["train"]
        if val: df_val["transf"] /= norm_dict["val"]
        df_test["transf"] /= norm_dict["test"]

    if val:
        return df_train, df_val, df_test
    else:
        return df_train, df_test

def load_rl(name, num, dir = ""):
    load_str = [f"./../ExpLogs/{dir}summ_{name}_{i}.csv" for i in range(num)]
    df_res = pd.concat([pd.read_csv(i) for i in load_str], axis=0).reset_index(drop=True)
    return df_res

def draw_hlines(ax, asap, nov2g, optim, x_max = 100, color="k"):
    fontsize = 15
    x_min, x_max = ax.get_xlim()
    ax.hlines(asap.sum(),  0, x_max, color=color, ls=':')
    ax.text(x_max*0.1, asap.sum(), "ASAP", fontsize = fontsize-4)
    ax.hlines(nov2g.sum(),  0, x_max, color=color, ls=':')
    ax.text(x_max*0.1, nov2g.sum(), "NoV2G", fontsize = fontsize-4)
    ax.hlines(optim.sum(), 0, x_max, color=color, ls=':')
    ax.text(x_max*0.1, optim.sum(), "Optim", fontsize = fontsize-4)

    #ax.set_ylim(0, 1.1*asap.sum())
    ax.set_xlabel("Episode")

    return ax
    
    
def plot_rl_gen(ax1, ax2, ax3, df_train, df_val, df_test, label="", ls="-", val=True):
    ax1.plot(df_train["transf"], label=label, ls=ls)
    if val:
        ax2.plot(df_val["transf"], label=label, ls=ls)
        ax3.plot(df_test["transf"], label=label, ls=ls)
        return ax1, ax2, ax3
    else:
        ax2.plot(df_test["transf"], label=label, ls=ls)
        return ax1, ax2

def plot_rl_gen_stoc(ax1, ax2, dfs, label="", ls="", norm_dict = None, ax3=None):
    if norm_dict is None:
        ax1.scatter(dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_jan.csv"]["name"].apply(lambda x: int(x.split("_")[-1])), dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_jan.csv"]["transf"], label = "Jan")
        ax1.scatter(dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_feb.csv"]["name"].apply(lambda x: int(x.split("_")[-1])), dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_feb.csv"]["transf"], label = "Feb")
        ax1.scatter(dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_mar.csv"]["name"].apply(lambda x: int(x.split("_")[-1])), dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_mar.csv"]["transf"], label = "Mar")
    else:
        ax1.scatter(dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_jan.csv"]["name"].apply(lambda x: int(x.split("_")[-1])), dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_jan.csv"]["transf"]/norm_dict["jan"], label = "Jan")
        ax1.scatter(dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_feb.csv"]["name"].apply(lambda x: int(x.split("_")[-1])), dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_feb.csv"]["transf"]/norm_dict["feb"], label = "Feb")
        ax1.scatter(dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_mar.csv"]["name"].apply(lambda x: int(x.split("_")[-1])), dfs[0][dfs[0]["sessions"] == "df_elaad_preproc_mar.csv"]["transf"]/norm_dict["mar"], label = "Mar")

    if ax3 is None:
        ax2.plot(dfs[1]["transf"])
    else:
        ax2.plot(dfs[1]["transf"]/norm_dict["apr"])
        ax3.plot(dfs[1]["total"])
    
    
    
    
    
    
    
    
    

