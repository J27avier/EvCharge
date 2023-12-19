nohup python3 RunChargeWorld.py --agent NoV2G --save-name apr_NoV2G_pred_noise_a --pred-noise 0.000 --file-sessions df_elaad_preproc_marapr.csv &
nohup python3 RunChargeWorld.py --agent NoV2G --save-name apr_NoV2G_pred_noise_b --pred-noise 0.001 --file-sessions df_elaad_preproc_marapr.csv &
nohup python3 RunChargeWorld.py --agent NoV2G --save-name apr_NoV2G_pred_noise_c --pred-noise 0.025 --file-sessions df_elaad_preproc_marapr.csv &
nohup python3 RunChargeWorld.py --agent NoV2G --save-name apr_NoV2G_pred_noise_d --pred-noise 0.050 --file-sessions df_elaad_preproc_marapr.csv &
nohup python3 RunChargeWorld.py --agent NoV2G --save-name apr_NoV2G_pred_noise_e --pred-noise 0.100 --file-sessions df_elaad_preproc_marapr.csv &
nohup python3 RunChargeWorld.py --agent Optim --save-name apr_Optim_pred_noise_a --pred-noise 0.000 --file-sessions df_elaad_preproc_marapr.csv &
nohup python3 RunChargeWorld.py --agent Optim --save-name apr_Optim_pred_noise_b --pred-noise 0.001 --file-sessions df_elaad_preproc_marapr.csv &
nohup python3 RunChargeWorld.py --agent Optim --save-name apr_Optim_pred_noise_c --pred-noise 0.025 --file-sessions df_elaad_preproc_marapr.csv &
nohup python3 RunChargeWorld.py --agent Optim --save-name apr_Optim_pred_noise_d --pred-noise 0.050 --file-sessions df_elaad_preproc_marapr.csv &
nohup python3 RunChargeWorld.py --agent Optim --save-name apr_Optim_pred_noise_e --pred-noise 0.100 --file-sessions df_elaad_preproc_marapr.csv &
#nohup python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_pred_noise_a --pred-noise 0.000 --years 300 --batch-size 512 --learning-starts 0 --alpha 0.03 --rng-test True --policy-frequency 4 --target-network-frequency 2 --general True & 
#nohup python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_pred_noise_b --pred-noise 0.001 --years 300 --batch-size 512 --learning-starts 0 --alpha 0.03 --rng-test True --policy-frequency 4 --target-network-frequency 2 --general True & 
#nohup python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_pred_noise_c --pred-noise 0.025 --years 300 --batch-size 512 --learning-starts 0 --alpha 0.03 --rng-test True --policy-frequency 4 --target-network-frequency 2 --general True & 
#nohup python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_pred_noise_d --pred-noise 0.050 --years 300 --batch-size 512 --learning-starts 0 --alpha 0.03 --rng-test True --policy-frequency 4 --target-network-frequency 2 --general True & 
#nohup python3 RunSACChargeWorld.py --agent SAC-sagg --save-name month_sac_pred_noise_e --pred-noise 0.100 --years 300 --batch-size 512 --learning-starts 0 --alpha 0.03 --rng-test True --policy-frequency 4 --target-network-frequency 2 --general True & 
