python3 RunRLChargeWorld.py --agent PPO-agg --save-name mont_agg_nomini_dstd2 --save-agent --reward-coef 1.0 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 20 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 && echo "Done month_agg_nomini!" 
