import datetime
import pandas as pd
import numpy as np
from . import config
from typing import Dict, List

class ExpTracker():
    def __init__(self, tinit: int, t_max: int, name: str = "_ev_world", timestamp: bool = True):
        self.name = name
        self.arr_bill_columns = ["ts", "arr_e_req", "client_bill", "assigned_type", "realized_type", "fail_time", "fail_energy1", "fail_energy2", "fail_energy_both", "fail_IR"] # Per timestep
        self.arr_bill = [] # type: ignore

        self.chg_bill_columns = ["ts", "chg_e_req", "imbalance_bill", "n_cars", "avg_lax"] # Per timestep
        self.chg_bill = [] # type: ignore

        self.dep_bill_columns = ["ts", "payoff"]
        self.dep_bill = [] # type: ignore

        self.contract_log_cols = ["idSess", "soc_dis", "t_dis", "g", "idx_theta_w", "idx_theta_l"]
        self.contract_log = [] # type: ignore

        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.tinit = tinit
        self.t_max  = t_max

    def save_log(self, args, path = "Results/results_log/"):
        df_log = pd.DataFrame(np.arange(self.tinit, self.t_max+1), columns = ["ts"])

        bills = [self.arr_bill, self.chg_bill, self.dep_bill]
        bill_columns = [self.arr_bill_columns, self.chg_bill_columns, self.dep_bill_columns]

        for bill, bill_columns in zip(bills, bill_columns):
            df_bill = pd.DataFrame(bill, columns = bill_columns)
            df_log = pd.merge(df_log, df_bill, on = ["ts"], how = "outer")

        # TODO: fillna
        df_log.to_csv(f"{path}{self.timestamp}{self.name}_{args.agent}{args.desc}.csv", index = False)
        
    def save_contracts(self, args, path="Results/results_log/"):
        df_contract_log = pd.DataFrame(self.contract_log, columns = self.contract_log_cols)
        if args.save_name != "":
            df_contract_log.to_csv(f"{path}{args.save_name}.csv", index = False)
        else: 
            df_contract_log.to_csv(f"{path}{self.timestamp}_Contracts{self.name}_{args.agent.split('.')[0]}{args.desc}.csv", 
                                   index = False)

    def save_desc(self, args, info, path = "Results/results_log"):
        text = []
        text.append(f"timestamp: {self.timestamp}")

        text.append("")
        text.append("--Info")
        for key, value in info.items():
            text.append(f"{key}: {value}")

        text.append("")
        text.append("--Args")
        for key, value in vars(args).items():
            text.append(f"{key}: {value}")

        if args.save_name != "":
            with open(f"{path}{args.save_name}.txt", 'w') as f:
                for line in text:
                    f.write(line)
                    f.write('\n')
        else:
            with open(f"{path}{self.timestamp}{self.name}_{args.agent.split('.')[0]}{args.desc}.txt", 'w') as f:
                for line in text:
                    f.write(line)
                    f.write('\n')


#df_arr_bill = pd.DataFrame(self.arr_bill, columns = self.arr_bill_columns)
#df_chg_bill = pd.DataFrame(self.chg_bill, columns = self.chg_bill_columns)
#df_dep_bill = pd.DataFrame(self.dep_bill, columns = self.dep_bill_columns)
#df_log = pd.merge(df_log, df_arr_bill, on = ["ts"], how = "outer")
#df_log = pd.merge(df_log, df_chg_bill, on = ["ts"], how = "outer")
#df_log = pd.merge(df_log, df_dep_bill, on = ["ts"], how = "outer")
