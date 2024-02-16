# Create script to run Disaggregation experiments
#nohup python3 RunSACChargeWorld.py --agent train_6month_sac_pred_noise_1a_199 --save-name 6month_sac_disagg_1a_PF  --pred-noise 0.00 --years 1 --batch-size 512 --learning-starts 0 --alpha 0.02 --rng-test True --test True --policy-frequency 4 --target-network-frequency 2 --disagg PF --file-sessions df_elaad_preproc_l6months.csv &
from itertools import product

l_disaggs = [ "PF ", "LL ", "ML ", "LD ", "MD ", "HLL", "P  ",]
l_pred_noise = ["0.00", "0.01", "0.02", "0.04", "0.06"]
l_runs = [1,2,3,4,5]
l_names = ["a", "b", "c", "d", "e"]
i=0

lines = []

for ((names, pred_noise), runs, disaggs) in product(zip(l_names, l_pred_noise), l_runs, l_disaggs):
    lines.append(f"nohup python3 RunSACChargeWorld.py --agent train_6month_sac_pred_noise_{runs}{names}_199 --save-name 6month_sac_disagg_{runs}{names}_{disaggs} --seed {runs} --pred-noise {pred_noise} --years 1 --batch-size 512 --learning-starts 0 --alpha 0.02 --rng-test True --test True --policy-frequency 4 --target-network-frequency 2 --disagg {disaggs} --file-sessions df_elaad_preproc_l6months.csv &")

len_lines = len(lines)
with open('run_RLDisagg.sh', 'w') as f:
    for i, line in enumerate(lines):
        if ((i+1) % 5) != 0:
            f.write(f"{line}&\n")
        else:
            f.write(f"{line}\n")
