#nohup python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_a --years 1000 --norm-reward --save-agent --state-rep nothmd --autotune False --disagg P    --n-state 59 &
#nohup python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_b --years 1000 --norm-reward --save-agent --state-rep nothmd --autotune False --disagg LL   --n-state 59 &
#nohup python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_c --years 1000 --norm-reward --save-agent --state-rep nothmd --autotune False --disagg ML   --n-state 59 &
#nohup python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_d --years 1000 --norm-reward --save-agent --state-rep nothmd --autotune False --disagg LD   --n-state 59 &
#nohup python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_f --years 1000 --norm-reward --save-agent --state-rep nothmd --autotune False --disagg HLL  --n-state 59 &
#python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_e --years 100 --norm-reward --state-rep nothmd --autotune False --disagg MD   --n-state 59 &
#python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_g --years 100 --norm-reward --state-rep nothmd --autotune False --disagg HML  --n-state 59 &
#python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_h --years 100 --norm-reward --state-rep nothmd --autotune False --disagg HLD  --n-state 59 &
#python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_i --years 100 --norm-reward --state-rep nothmd --autotune False --disagg HMD  --n-state 59 &
#python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_j --years 100 --norm-reward --state-rep nothmd --autotune False --disagg SHLL --n-state 59 &
#python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_k --years 100 --norm-reward --state-rep nothmd --autotune False --disagg SHLD --n-state 59 &
#python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_l --years 100 --norm-reward --state-rep nothmd --autotune False --disagg HKL  --n-state 59 &
#python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VII_m --years 100 --norm-reward --state-rep nothmd --autotune False --disagg SLL --n-state 59 &


#python3 RunSACChargeWorld.py --agent train_6month_sac_gen_j_299 --save-name 6month_sac_disagg_a --years 1 --batch-size 512 --learning-starts 0 --alpha 0.02 --rng-test True --policy-frequency 4 --target-network-frequency 2 --disagg LL --file-sessions df_elaad_preproc_l6months.csv

python3 RunSACChargeWorld.py --agent SAC-sagg --save-name 6month_sac_disagg_a --years 2 --batch-size 512 --learning-starts 0 --alpha 0.02 --rng-test True --policy-frequency 4 --target-network-frequency 2 --disagg LL --general True --save-agent True
echo "+++++++++++++++++++++++++++++++++++++++++"
python3 RunSACChargeWorld.py --agent train_6month_sac_disagg_a_1 --save-name 6month_sac_disagg_reload --years 1 --batch-size 512 --learning-starts 0 --alpha 0.02 --rng-test True --test True --policy-frequency 4 --target-network-frequency 2 --disagg LL --file-sessions df_elaad_preproc_l6months.csv
