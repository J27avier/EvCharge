from EvGym.charge_world import ChargeWorldEnv, Session
import pandas as pd
import numpy as np
import time
from tqdm.notebook import tqdm
from IPython.display import clear_output

df_data = pd.read_csv("data/prepared_elaad_transactions.csv", parse_dates=["TransactionStartDT", "TransactionStopDT"])
world = ChargeWorldEnv(df_data)
for i in range(df_data["TransactionStopTS"].max().astype(int)):
    world.step()
