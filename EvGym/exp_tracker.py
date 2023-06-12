import datetime
import pandas as pd
import numpy as np
from . import config
from typing import Dict, List

class ExpTracker():
    def __init__(self, tinit: int, t_max: int, name: str = "_ev_world", timestamp: bool = True):
        self.name = name
        self.client_bill_columns = ["ts", "arr_e_req", "client_bill"]
        self.client_bill: List[List[config.Number]]= [] 

        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.tinit = tinit
        self.t_max  = t_max

    def save_log(self, path = "ExpLogs/"):
        df_log = pd.DataFrame(np.arange(self.tinit, self.t_max+1), columns = ["ts"])
        df_client_bill = pd.DataFrame(self.client_bill, columns = self.client_bill_columns)
        df_log = pd.merge(df_log, df_client_bill, on = ["ts"], how = "outer")
        # TODO: fillna
        df_log.to_csv(f"{path}{self.timestamp}{self.name}.csv", index = False)
