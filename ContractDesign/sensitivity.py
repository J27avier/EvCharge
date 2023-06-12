import cvxpy as cp
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from math import floor, log10
from contracts import get_contract, l_transpose 
def sensitivity_experiment(kappas = [0.01, 0.05, 0.1, 0.5, 1], 
                           gammas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10], 
                           types = 3,
                           tau=1):
    l_igammas = []
    l_gammas = []
    l_ikappas = []
    l_kappas = []
    l_gws = []
    
    for i_gamma, gamma in enumerate(gammas):
        for i_kappa, kappa in enumerate(kappas):
            gs, ws = get_contract(types, TAU=tau, GAMMA= gamma, KAPPA=kappa)
            l_igammas.append(i_gamma)
            l_gammas.append(gamma)
            l_ikappas.append(i_kappa)
            l_kappas.append(kappa)
            l_gws.append(l_transpose([gs, ws]))
            
    data_T = [l_igammas, l_gammas, l_ikappas, l_kappas, l_gws] # List of lists
    data = l_transpose(data_T) 
    df_resContracts = pd.DataFrame(data = data, columns = ["idx_gamma", "gamma", "idx_kappa",  "kappa", "gw"])
    
    return df_resContracts

def expand_results(df_resContracts):
    df_res = df_resContracts.explode("gw")
    df_res['type'] = df_res.groupby(level=0).cumcount() + 1 
    sig=4
    df_res["g_round"] = df_res["gw"].apply(lambda x: round(x[0], sig-int(floor(log10(abs(x[0]))))-1)) # Round to "sig" significant digits
    df_res["g"] = df_res["gw"].apply(lambda x: round(x[0], 4))
    df_res["w"] = df_res["gw"].apply(lambda x: round(x[1], 4))
    df_res = df_res.drop(columns=["gw"])
    df_res = df_res.reset_index(drop=True)
    df_res = df_res.drop_duplicates(subset=["gamma", "kappa", "g_round"], keep="first")
    return df_res
    

def plot_sensitivity(df_res, colors = ["beige", "paleturquoise", "lightpink"]):
    kappas = df_res.sort_values(by = ["kappa"])["kappa"].unique()
    gammas = df_res.sort_values(by = ["gamma"])["gamma"].unique()
    types = df_res['type'].nunique()
    
    fig1, axs = plt.subplots(nrows=3, ncols=len(kappas), figsize =(18, 6))
    barwidth = 0.2
    
    for j, kappa in enumerate(kappas):
        for type_ev in range(1, types+1):
            df_ax = df_res[(df_res["kappa"] == kappa) & (df_res["type"] == type_ev)]
    
            x = df_ax["idx_gamma"] + 0.25 * (type_ev - 2) + 1
            y1 = df_ax["w"]
            y2 = df_ax["g"]
            y3 = df_ax["g"] / df_ax["w"]
            
            axs[0,j].bar(x,y1, width = barwidth, edgecolor = "black", color=colors[type_ev-1])
            axs[1,j].bar(x,y2, width = barwidth, edgecolor = "black", color=colors[type_ev-1])
            axs[2,j].bar(x,y3, width = barwidth, edgecolor = "black", color=colors[type_ev-1], label = f'EV owner type {type_ev}')
            
            axs[0,j].set_ylabel(r'$w_m$ (in kWh)')
            axs[1,j].set_ylabel(r'$g_m$ (in €)')
            axs[2,j].set_ylabel(r"$g_m$ per $w_m$" "\n" r"(in € / kWh)")
            
    for row in axs:
        for ax in row:
            ax.set_xticks(range(1,len(gammas)+1))
            ax.set_xticklabels(gammas, rotation = 45)
            ax.set_xlabel(r'$\gamma$ (in €/kWh)')
    
    
    handles, labels = axs[2,3].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', ncol = 6, bbox_to_anchor=(0.5, 1.05))
    fig1.tight_layout()
    return fig1
