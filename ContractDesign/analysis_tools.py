import numpy as np
import traceback
import pandas as pd

import sys
sys.path.append("..")
from utility import utility as util

def read_results_dict(taus, percs, kappa, gamma, res_type, str_fmt, directory = '../w_mc_results/contract_no_solar/',
                      columns = ["tau", "perc", "kappa", "gamma", "revenue", "participation", "type"] , pad = False):
    l_results = []
    for tau in taus:
        for perc in percs:
            temp_dict = {}
            file_str = str_fmt.replace('<tau>', str(tau)).replace('<perc>', str(perc))
            try: 
                temp_dict[str(perc)] = util.load_result(directory + file_str)
                revenue = np.sum(temp_dict[str(perc)][f"total_revenue_{perc}_run0_E"])
                participation = np.sum(temp_dict[str(perc)][f"num_v2g_evs_w_contract_lst_{perc}_run0_E"])
                l_results.append([tau, perc, kappa, gamma, revenue, participation, res_type])
            except Exception:
                if pad:
                    l_results.append([tau, perc, kappa, gamma, np.nan, np.nan, res_type])
                print(f"File {tau=} {perc=} \"{file_str}\" not found")
    df_sched = pd.DataFrame(data = l_results, columns = columns)
    df_sched = df_sched.sort_values(by = ["tau", "perc"]).reset_index(drop = True)
    return df_sched       

def read_results_exp_dict(taus, percs, kappa, gamma, res_type, str_fmt, directory = '../w_mc_results/contract_no_solar/',
                      columns = ["tau", "perc", "kappa", "gamma", "total_rev",
                                 "im_buy", "im_sell", "da_rev", "retail_rev", "owner_pay", "assigned_idx", "assigned_counts", "realized_idx", "realized_counts", "participation", "no_participation", "type"] , pad = False):
    l_results = []
    for tau in taus:
        for perc in percs:
            temp_dict = {}
            file_str = str_fmt.replace('<tau>', str(tau)).replace('<perc>', str(perc))
            try: 
                temp_dict[str(perc)] = util.load_result(directory + file_str)
                total_rev = np.sum(temp_dict[str(perc)][f"total_revenue_{perc}_run0_E"])
                im_buy = np.sum(temp_dict[str(perc)][f"im_buy_{perc}_run0_E"])
                im_sell = np.sum(temp_dict[str(perc)][f"im_sell_{perc}_run0_E"])
                da_rev = np.sum(temp_dict[str(perc)][f"da_revenue{perc}_run0_E"])
                retail_rev = np.sum(temp_dict[str(perc)][f"retail_revenue{perc}_run0_E"])
                owner_pay = np.sum(temp_dict[str(perc)][f"owner_pay{perc}_run0_E"])

                assigned_type_ll = temp_dict[str(perc)][f"assigned_type{perc}_run0_E"]
                assigned_type = np.array([item for assigned_type_l in assigned_type_ll for item in assigned_type_l])
                assigned_idx, assigned_counts = np.unique(assigned_type, return_counts = True)

                realized_type_ll = temp_dict[str(perc)][f"realized_type{perc}_run0_E"]
                realized_type = np.array([item for realized_type_l in realized_type_ll for item in realized_type_l])
                realized_idx, realized_counts = np.unique(realized_type, return_counts=True)               

                participation = np.sum(temp_dict[str(perc)][f"num_v2g_evs_w_contract_lst_{perc}_run0_E"])
                #num_v2g_evs_no_contract_lst_
                no_participation = np.sum(temp_dict[str(perc)][f"num_v2g_evs_no_contract_lst_{perc}_run0_E"])
                l_results.append([tau, perc, kappa, gamma, total_rev, im_buy, im_sell, da_rev, retail_rev, owner_pay, assigned_idx, assigned_counts, realized_idx, realized_counts, participation, no_participation, res_type])
            except Exception:
                print(traceback.format_exc())
                if pad:
                    l_results.append([tau, perc, kappa, gamma,
                                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, res_type])
                print(f"File {tau=} {perc=} \"{file_str}\" not found")
    df_sched = pd.DataFrame(data = l_results, columns = columns)
    df_sched = df_sched.sort_values(by = ["tau", "perc"]).reset_index(drop = True)
    return df_sched       

#total_rev = np.array(dict_result['total_revenue_25_run0_E']).sum()
#im_buy = np.array(dict_result['im_buy_25_run0_E']).sum()
#im_sell = np.array(dict_result['im_sell_25_run0_E']).sum()
#da_rev = np.array(dict_result['da_revenue25_run0_E']).sum()
#retail_rev = np.array(dict_result["retail_revenue25_run0_E"]).sum()
#owner_pay = np.array(dict_result["owner_pay25_run0_E"]).sum()
