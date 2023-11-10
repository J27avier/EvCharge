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
            print_str += f" --num-steps {row.num_steps} --anneal-lr {row.anneal_lr}  --update-epochs {row.update_epochs} --num-minibatches {row.num_minibatches}"
            print_str += f" --lax-coef {row.lax_coef} --logstd {row.logstd} --n-state {row.n_state} --hidden {row.hidden}"
            print_str += f" --relu {row.relu} --no-safety {row.no_safety} --norm-state {row.norm_state}"
            print_str += f" --without-perc {row.without_perc} --norm-reward {row.norm_reward} --reset-std {row.reset_std} --optimizer {row.optimizer}"

            if row.seq == "s":
                print_str += " && "
            else:
                print_str += f" && echo \"Done {row.save_name}!\" &\n"
            f.write(print_str)

os.system(f"cat {filename}.sh")

# YOU'VE FOUND THE SECRET MESSAGE XD!
