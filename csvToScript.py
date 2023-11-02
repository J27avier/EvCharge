import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-N", "--name", help="Name of file to read", type=str, required=True)
args = parser.parse_args()

filename = args.name
df_data = pd.read_csv(f"{filename}.csv")
with open(f"{filename}.sh", "w") as f:
    for row in df_data.itertuples():
        if row.do == "T":
            print_str = f"python3 RunRLChargeWorld.py --agent {row.agent} --save-name {row.save_name}"
            if row.save_agent == "T":
                print_str += f" --save-agent"
            print_str += f" --reward-coef {row.reward_coef} --proj-coef {row.proj_coef} --learning-rate {row.learning_rate}"
            print_str += f" --file-price {row.file_price} --years {row.years}"
            if row.seq == "s":
                print_str += " && "
            elif row.seq == "e":
                print_str += f" && echo \"Done {row.save_name}!\" &\n"
            f.write(print_str)

os.system(f"cat {filename}.sh")
