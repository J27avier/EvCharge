#python3 RunChargeWorld.py --agent Optim --save-name c2_Optim  --file-price df_prices_2019_c.csv --kappa1 0.4 --kappa2 0.6 --c1 0.01 --c2 0.05 &
#python3 RunChargeWorld.py --agent ASAP  --save-name c_ASAP   --file-price df_prices_2019_c.csv &
#python3 RunChargeWorld.py --agent NoV2G --save-name c_NoV2G  --file-price df_prices_2019_c.csv &
#python3 RunChargeWorld.py --agent Optim --save-name c_Optim  --file-price df_prices_2019_c.csv &

#nohup python3 RunChargeWorld.py --agent ASAP  --save-name train_ASAP  --file-sessions df_synth_sessions_2014_2018.csv --file-price df_prices_c.csv --summary False &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name train_NoV2G --file-sessions df_synth_sessions_2014_2018.csv --file-price df_prices_c.csv --summary False &
#nohup python3 RunChargeWorld.py --agent Optim --save-name train_Optim --file-sessions df_synth_sessions_2014_2018.csv --file-price df_prices_c.csv --summary False &
#nohup python3 RunChargeWorld.py --agent ASAP  --save-name val_ASAP    --file-sessions df_synth_sessions_2019.csv      --file-price df_prices_c.csv --summary False &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name val_NoV2G   --file-sessions df_synth_sessions_2019.csv      --file-price df_prices_c.csv --summary False &
#nohup python3 RunChargeWorld.py --agent Optim --save-name val_Optim   --file-sessions df_synth_sessions_2019.csv      --file-price df_prices_c.csv --summary False &
#nohup python3 RunChargeWorld.py --agent ASAP  --save-name test_ASAP   --file-sessions df_elaad_preproc.csv            --file-price df_prices_c.csv --summary False &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name test_NoV2G  --file-sessions df_elaad_preproc.csv            --file-price df_prices_c.csv --summary False &
#nohup python3 RunChargeWorld.py --agent Optim --save-name test_Optim  --file-sessions df_elaad_preproc.csv            --file-price df_prices_c.csv --summary False &

#python3 RunChargeWorld.py --agent Optim --save-name nc2_Optim --summary False --file-price df_prices_c.csv --kappa1 0.4 --kappa2 0.6 --c1 0.01 --c2 0.05 &
#python3 RunChargeWorld.py --agent ASAP  --save-name nc_ASAP   --summary False --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent NoV2G --save-name nc_NoV2G  --summary False --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent Optim --save-name nc_Optim  --summary False --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent Optim --save-name nc2_Optim --summary True  --file-price df_prices_c.csv --kappa1 0.4 --kappa2 0.6 --c1 0.01 --c2 0.05 &
#python3 RunChargeWorld.py --agent ASAP  --save-name nc_ASAP   --summary True  --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent NoV2G --save-name nc_NoV2G  --summary True  --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent Optim --save-name nc_Optim  --summary True  --file-price df_prices_c.csv &

python3 RunChargeWorld.py --agent ASAP  --save-name jan_ASAP   --summary False --file-sessions df_elaad_preproc_jan.csv --file-price df_prices_c.csv && echo "jan_ASAP " >> notify.txt &
python3 RunChargeWorld.py --agent NoV2G --save-name jan_NoV2G  --summary False --file-sessions df_elaad_preproc_jan.csv --file-price df_prices_c.csv && echo "jan_NoV2G" >> notify.txt &
python3 RunChargeWorld.py --agent Optim --save-name jan_Optim  --summary False --file-sessions df_elaad_preproc_jan.csv --file-price df_prices_c.csv && echo "jan_Optim" >> notify.txt &
python3 RunChargeWorld.py --agent ASAP  --save-name feb_ASAP   --summary False --file-sessions df_elaad_preproc_feb.csv --file-price df_prices_c.csv && echo "feb_ASAP " >> notify.txt &
python3 RunChargeWorld.py --agent NoV2G --save-name feb_NoV2G  --summary False --file-sessions df_elaad_preproc_feb.csv --file-price df_prices_c.csv && echo "feb_NoV2G" >> notify.txt &
python3 RunChargeWorld.py --agent Optim --save-name feb_Optim  --summary False --file-sessions df_elaad_preproc_feb.csv --file-price df_prices_c.csv && echo "feb_Optim" >> notify.txt &
python3 RunChargeWorld.py --agent ASAP  --save-name mar_ASAP   --summary False --file-sessions df_elaad_preproc_mar.csv --file-price df_prices_c.csv && echo "mar_ASAP " >> notify.txt &
python3 RunChargeWorld.py --agent NoV2G --save-name mar_NoV2G  --summary False --file-sessions df_elaad_preproc_mar.csv --file-price df_prices_c.csv && echo "mar_NoV2G" >> notify.txt &
python3 RunChargeWorld.py --agent Optim --save-name mar_Optim  --summary False --file-sessions df_elaad_preproc_mar.csv --file-price df_prices_c.csv && echo "mar_Optim" >> notify.txt &

#nohup python3 RunChargeWorld.py --agent ASAP  --save-name a_ASAP   --file-price df_prices_2019_a.csv &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name a_NoV2G  --file-price df_prices_2019_a.csv &
#nohup python3 RunChargeWorld.py --agent Optim --save-name a_Optim  --file-price df_prices_2019_a.csv &
#nohup python3 RunChargeWorld.py --agent ASAP  --save-name b_ASAP   --file-price df_prices_2019_b.csv &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name b_NoV2G  --file-price df_prices_2019_b.csv &
#nohup python3 RunChargeWorld.py --agent Optim --save-name b_Optim  --file-price df_prices_2019_b.csv &
#nohup python3 RunChargeWorld.py --agent ASAP  --save-name c_ASAP   --file-price df_prices_2019_c.csv &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name c_NoV2G  --file-price df_prices_2019_c.csv &
#nohup python3 RunChargeWorld.py --agent Optim --save-name c_Optim  --file-price df_prices_2019_c.csv &
#nohup python3 RunChargeWorld.py --agent ASAP  --save-name d_ASAP   --file-price df_prices_2019_d.csv &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name d_NoV2G  --file-price df_prices_2019_d.csv &
#nohup python3 RunChargeWorld.py --agent Optim --save-name d_Optim  --file-price df_prices_2019_d.csv &
#nohup python3 RunChargeWorld.py --agent ASAP  --save-name as_ASAP   --file-price df_prices_2019_as.csv &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name as_NoV2G  --file-price df_prices_2019_as.csv &
#nohup python3 RunChargeWorld.py --agent Optim --save-name as_Optim  --file-price df_prices_2019_as.csv &
#nohup python3 RunChargeWorld.py --agent ASAP  --save-name bs_ASAP   --file-price df_prices_2019_bs.csv &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name bs_NoV2G  --file-price df_prices_2019_bs.csv &
#nohup python3 RunChargeWorld.py --agent Optim --save-name bs_Optim  --file-price df_prices_2019_bs.csv &
#nohup python3 RunChargeWorld.py --agent ASAP  --save-name cs_ASAP   --file-price df_prices_2019_cs.csv &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name cs_NoV2G  --file-price df_prices_2019_cs.csv &
#nohup python3 RunChargeWorld.py --agent Optim --save-name cs_Optim  --file-price df_prices_2019_cs.csv &
#nohup python3 RunChargeWorld.py --agent ASAP  --save-name ds_ASAP   --file-price df_prices_2019_ds.csv &
#nohup python3 RunChargeWorld.py --agent NoV2G --save-name ds_NoV2G  --file-price df_prices_2019_ds.csv &
#nohup python3 RunChargeWorld.py --agent Optim --save-name ds_Optim  --file-price df_prices_2019_ds.csv &
