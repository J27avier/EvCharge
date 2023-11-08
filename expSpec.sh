python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_agg_norma_n --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 24 --years 50 --anneal-lr True  --update-epochs 10 --num-minibatches 8  && echo "Done month_agg!" &

python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_agg_norma_moresteps_n --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 72 --years 50 --anneal-lr True  --update-epochs 10 --num-minibatches 8  && echo "Done month_agg!" &

python3 RunRLChargeWorld.py --agent PPO-lay --save-name month_ind_norma_n --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 24 --years 50 --anneal-lr True  --update-epochs 10 --num-minibatches 8  && echo "Done month_agg!" &

python3 RunRLChargeWorld.py --agent PPO-lay --save-name month_ind_norma_moresteps_n --save-agent --reward-coef  1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 72 --years 50 --anneal-lr True  --update-epochs 10 --num-minibatches 8  && echo "Done month_agg!" &
