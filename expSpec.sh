python3 RunRLChargeWorld.py --agent PPO-lay --save-name month_ind_nomini_norm_rew_n --save-agent --reward-coef 1.0 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 24 --years 100 --anneal-lr True  --update-epochs 10 --num-minibatches 1 && echo "Done month_ind_norm_rew_n!" &
