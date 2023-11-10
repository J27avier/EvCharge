python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_a --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr False  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 1.0 --gae-lambda 0.95 --clip-coef 0.1 --vf-coef 0.3 --max-grad-norm 0.3 --ent-coef 0.0 --df-imit df_optim.csv && echo "Done month_aggt_a!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_b --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr False  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 1.0 --gae-lambda 0.9 --clip-coef 0.2 --vf-coef 0.5 --max-grad-norm 0.5 --ent-coef 0.01 --df-imit df_optim.csv && echo "Done month_aggt_b!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_c --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr False  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 1.0 --gae-lambda 0.85 --clip-coef 0.3 --vf-coef 0.7 --max-grad-norm 0.7 --ent-coef 0.1 --df-imit df_optim.csv && echo "Done month_aggt_c!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_d --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr False  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.99 --gae-lambda 0.95 --clip-coef 0.1 --vf-coef 0.5 --max-grad-norm 0.7 --ent-coef 0.1 --df-imit df_optim.csv && echo "Done month_aggt_d!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_e --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr False  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.99 --gae-lambda 0.9 --clip-coef 0.2 --vf-coef 0.7 --max-grad-norm 0.3 --ent-coef 0.0 --df-imit df_optim.csv && echo "Done month_aggt_e!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_f --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr False  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.99 --gae-lambda 0.85 --clip-coef 0.3 --vf-coef 0.3 --max-grad-norm 0.5 --ent-coef 0.01 --df-imit df_optim.csv && echo "Done month_aggt_f!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_g --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr False  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.8 --gae-lambda 0.95 --clip-coef 0.2 --vf-coef 0.7 --max-grad-norm 0.5 --ent-coef 0.1 --df-imit df_optim.csv && echo "Done month_aggt_g!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_h --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr False  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.99 --gae-lambda 0.9 --clip-coef 0.3 --vf-coef 0.3 --max-grad-norm 0.7 --ent-coef 0.0 --df-imit df_optim.csv && echo "Done month_aggt_h!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_i --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr False  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.8 --gae-lambda 0.85 --clip-coef 0.1 --vf-coef 0.5 --max-grad-norm 0.3 --ent-coef 0.01 --df-imit df_optim.csv && echo "Done month_aggt_i!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_j --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 1.0 --gae-lambda 0.95 --clip-coef 0.3 --vf-coef 0.5 --max-grad-norm 0.5 --ent-coef 0.0 --df-imit df_optim.csv && echo "Done month_aggt_j!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_k --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 1.0 --gae-lambda 0.9 --clip-coef 0.1 --vf-coef 0.7 --max-grad-norm 0.7 --ent-coef 0.01 --df-imit df_optim.csv && echo "Done month_aggt_k!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_l --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 1.0 --gae-lambda 0.85 --clip-coef 0.2 --vf-coef 0.3 --max-grad-norm 0.3 --ent-coef 0.1 --df-imit df_optim.csv && echo "Done month_aggt_l!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_m --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.99 --gae-lambda 0.95 --clip-coef 0.2 --vf-coef 0.3 --max-grad-norm 0.7 --ent-coef 0.01 --df-imit df_optim.csv && echo "Done month_aggt_m!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_n --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.99 --gae-lambda 0.9 --clip-coef 0.3 --vf-coef 0.5 --max-grad-norm 0.3 --ent-coef 0.1 --df-imit df_optim.csv && echo "Done month_aggt_n!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_o --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.99 --gae-lambda 0.85 --clip-coef 0.1 --vf-coef 0.7 --max-grad-norm 0.5 --ent-coef 0.0 --df-imit df_optim.csv && echo "Done month_aggt_o!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_p --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.8 --gae-lambda 0.95 --clip-coef 0.3 --vf-coef 0.7 --max-grad-norm 0.3 --ent-coef 0.01 --df-imit df_optim.csv && echo "Done month_aggt_p!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_q --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.8 --gae-lambda 0.9 --clip-coef 0.1 --vf-coef 0.3 --max-grad-norm 0.5 --ent-coef 0.1 --df-imit df_optim.csv && echo "Done month_aggt_q!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_r --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 0.99 --gae-lambda 0.85 --clip-coef 0.2 --vf-coef 0.5 --max-grad-norm 0.7 --ent-coef 0.0 --df-imit df_optim.csv && echo "Done month_aggt_r!" &
python3 RunRLChargeWorld.py --agent PPO-agg --save-name month_aggt_s --save-agent --reward-coef 1 --proj-coef 0 --learning-rate 0.0003 --file-price df_price_2019_pad.csv --years 50 --num-steps 24 --anneal-lr True  --update-epochs 10 --num-minibatches 1 --lax-coef 0 --logstd -1.8 --n-state 19 --hidden 64 --relu False --no-safety False --norm-state False --without-perc True --norm-reward False --reset-std False --optimizer Adam --grad-std True --gamma 1.0 --gae-lambda 0.85 --clip-coef 0.1 --vf-coef 0.3 --max-grad-norm 0.5 --ent-coef 0.0 --df-imit df_optim.csv && echo "Done month_aggt_s!" &
