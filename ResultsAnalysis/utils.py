import pandas as pd

def load_res(name, path="../ExpLogs"):
    df_res = pd.read_csv(f"{path}/{name}.csv")
    with open(f"{path}/{name}.txt", "r") as f:
        lines = f.readlines()
    return df_res, lines

