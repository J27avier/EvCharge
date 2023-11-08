python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_agg_crit_n --save-agent --reward-coef 1.0 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 24 --years 50 --anneal-lr True  --update-epochs 10 --num-minibatches 1            && echo "Done month_agg_crit_1!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_agg_crit_moremini_n --save-agent --reward-coef 1.0 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 24 --years 50 --anneal-lr True  --update-epochs 10 --num-minibatches 12  && echo "Done month_agg_crit_2!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_agg_crit_proj1_n --save-agent --reward-coef 1.0 --proj-coef 1.0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 24 --years 50 --anneal-lr True  --update-epochs 10 --num-minibatches 1    && echo "Done month_agg_crit_3!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_agg_crit_proj01_n --save-agent --reward-coef 1.0 --proj-coef 0.1 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 24 --years 50 --anneal-lr True  --update-epochs 10 --num-minibatches 1   && echo "Done month_agg_crit_4!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_agg_crit_proj001_n --save-agent --reward-coef 1.0 --proj-coef 0.01 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 24 --years 50 --anneal-lr True  --update-epochs 10 --num-minibatches 1 && echo "Done month_agg_crit_5!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_agg_crit_n_less_steps --save-agent --reward-coef 1.0 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 8 --years 50 --anneal-lr True  --update-epochs 10 --num-minibatches 1  && echo "Done month_agg_crit_6!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_agg_crit_n_nalr --save-agent --reward-coef 1.0 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --num-steps 24 --years 50 --anneal-lr False  --update-epochs 10 --num-minibatches 1      && echo "Done month_agg_crit_7!" &

