python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VI_a --years 50 --norm-reward --state-rep nothmd --autotune False --disagg P  --n-state 59 &
python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VI_b --years 50 --norm-reward --state-rep nothmd --autotune False --disagg LL --n-state 59 &
python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VI_c --years 50 --norm-reward --state-rep nothmd --autotune False --disagg ML --n-state 59 &
python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VI_d --years 50 --norm-reward --state-rep nothmd --autotune False --disagg LH --n-state 59 &
python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_VI_e --years 50 --norm-reward --state-rep nothmd --autotune False --disagg MH --n-state 59 &
