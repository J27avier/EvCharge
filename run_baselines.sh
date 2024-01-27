#python3 RunChargeWorld.py --agent Optim --save-name c2_Optim  --file-price df_prices_2019_c.csv --kappa1 0.4 --kappa2 0.6 --c1 0.01 --c2 0.05 &
#python3 RunChargeWorld.py --agent ASAP  --save-name c_ASAP   --file-price df_prices_2019_c.csv &
#python3 RunChargeWorld.py --agent NoV2G --save-name c_NoV2G  --file-price df_prices_2019_c.csv &
#python3 RunChargeWorld.py --agent Optim --save-name c_Optim  --file-price df_prices_2019_c.csv &

nohup python3 RunChargeWorld.py --agent ASAP  --save-name f6_ASAP  --summary False --file-sessions df_elaad_preproc_f6months.csv &
nohup python3 RunChargeWorld.py --agent NoV2G --save-name f6_NoV2G --summary False --file-sessions df_elaad_preproc_f6months.csv &
nohup python3 RunChargeWorld.py --agent Optim --save-name f6_Optim --summary False --file-sessions df_elaad_preproc_f6months.csv &
nohup python3 RunChargeWorld.py --agent ASAP  --save-name l6_ASAP  --summary False --file-sessions df_elaad_preproc_l6months.csv &
nohup python3 RunChargeWorld.py --agent NoV2G --save-name l6_NoV2G --summary False --file-sessions df_elaad_preproc_l6months.csv &
nohup python3 RunChargeWorld.py --agent Optim --save-name l6_Optim --summary False --file-sessions df_elaad_preproc_l6months.csv &

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
#python3 RunChargeWorld.py --agent Optim --save-name nc3_Optim --summary True  --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent ASAP  --save-name nc_ASAP   --summary True  --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent NoV2G --save-name nc_NoV2G  --summary True  --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent Optim --save-name nc_Optim  --summary True  --file-price df_prices_c.csv &

#python3 RunChargeWorld.py --agent ASAP  --save-name jan_ASAP   --summary False --file-sessions df_elaad_preproc_jan.csv --file-price df_prices_c.csv && echo "jan_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name jan_NoV2G  --summary False --file-sessions df_elaad_preproc_jan.csv --file-price df_prices_c.csv && echo "jan_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name jan_Optim  --summary False --file-sessions df_elaad_preproc_jan.csv --file-price df_prices_c.csv && echo "jan_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name feb_ASAP   --summary False --file-sessions df_elaad_preproc_feb.csv --file-price df_prices_c.csv && echo "feb_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name feb_NoV2G  --summary False --file-sessions df_elaad_preproc_feb.csv --file-price df_prices_c.csv && echo "feb_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name feb_Optim  --summary False --file-sessions df_elaad_preproc_feb.csv --file-price df_prices_c.csv && echo "feb_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name mar_ASAP   --summary False --file-sessions df_elaad_preproc_mar.csv --file-price df_prices_c.csv && echo "mar_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name mar_NoV2G  --summary False --file-sessions df_elaad_preproc_mar.csv --file-price df_prices_c.csv && echo "mar_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name mar_Optim  --summary False --file-sessions df_elaad_preproc_mar.csv --file-price df_prices_c.csv && echo "mar_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name apr_ASAP   --summary False --file-sessions df_elaad_preproc_apr.csv --file-price df_prices_c.csv && echo "apr_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name apr_NoV2G  --summary False --file-sessions df_elaad_preproc_apr.csv --file-price df_prices_c.csv && echo "apr_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name apr_Optim  --summary False --file-sessions df_elaad_preproc_apr.csv --file-price df_prices_c.csv && echo "apr_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name may_ASAP   --summary False --file-sessions df_elaad_preproc_may.csv --file-price df_prices_c.csv && echo "may_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name may_NoV2G  --summary False --file-sessions df_elaad_preproc_may.csv --file-price df_prices_c.csv && echo "may_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name may_Optim  --summary False --file-sessions df_elaad_preproc_may.csv --file-price df_prices_c.csv && echo "may_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name jun_ASAP   --summary False --file-sessions df_elaad_preproc_jun.csv --file-price df_prices_c.csv && echo "jun_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name jun_NoV2G  --summary False --file-sessions df_elaad_preproc_jun.csv --file-price df_prices_c.csv && echo "jun_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name jun_Optim  --summary False --file-sessions df_elaad_preproc_jun.csv --file-price df_prices_c.csv && echo "jun_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name jul_ASAP   --summary False --file-sessions df_elaad_preproc_jul.csv --file-price df_prices_c.csv && echo "jul_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name jul_NoV2G  --summary False --file-sessions df_elaad_preproc_jul.csv --file-price df_prices_c.csv && echo "jul_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name jul_Optim  --summary False --file-sessions df_elaad_preproc_jul.csv --file-price df_prices_c.csv && echo "jul_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name aug_ASAP   --summary False --file-sessions df_elaad_preproc_aug.csv --file-price df_prices_c.csv && echo "aug_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name aug_NoV2G  --summary False --file-sessions df_elaad_preproc_aug.csv --file-price df_prices_c.csv && echo "aug_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name aug_Optim  --summary False --file-sessions df_elaad_preproc_aug.csv --file-price df_prices_c.csv && echo "aug_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name sep_ASAP   --summary False --file-sessions df_elaad_preproc_sep.csv --file-price df_prices_c.csv && echo "sep_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name sep_NoV2G  --summary False --file-sessions df_elaad_preproc_sep.csv --file-price df_prices_c.csv && echo "sep_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name sep_Optim  --summary False --file-sessions df_elaad_preproc_sep.csv --file-price df_prices_c.csv && echo "sep_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name oct_ASAP   --summary False --file-sessions df_elaad_preproc_oct.csv --file-price df_prices_c.csv && echo "oct_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name oct_NoV2G  --summary False --file-sessions df_elaad_preproc_oct.csv --file-price df_prices_c.csv && echo "oct_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name oct_Optim  --summary False --file-sessions df_elaad_preproc_oct.csv --file-price df_prices_c.csv && echo "oct_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name nov_ASAP   --summary False --file-sessions df_elaad_preproc_nov.csv --file-price df_prices_c.csv && echo "nov_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name nov_NoV2G  --summary False --file-sessions df_elaad_preproc_nov.csv --file-price df_prices_c.csv && echo "nov_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name nov_Optim  --summary False --file-sessions df_elaad_preproc_nov.csv --file-price df_prices_c.csv && echo "nov_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name dec_ASAP   --summary False --file-sessions df_elaad_preproc_dec.csv --file-price df_prices_c.csv && echo "dec_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name dec_NoV2G  --summary False --file-sessions df_elaad_preproc_dec.csv --file-price df_prices_c.csv && echo "dec_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name dec_Optim  --summary False --file-sessions df_elaad_preproc_dec.csv --file-price df_prices_c.csv && echo "dec_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name jan_ASAP   --summary True --file-sessions df_elaad_preproc_jan.csv --file-price df_prices_c.csv && echo "jan_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name jan_NoV2G  --summary True --file-sessions df_elaad_preproc_jan.csv --file-price df_prices_c.csv && echo "jan_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name jan_Optim  --summary True --file-sessions df_elaad_preproc_jan.csv --file-price df_prices_c.csv && echo "jan_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name feb_ASAP   --summary True --file-sessions df_elaad_preproc_feb.csv --file-price df_prices_c.csv && echo "feb_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name feb_NoV2G  --summary True --file-sessions df_elaad_preproc_feb.csv --file-price df_prices_c.csv && echo "feb_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name feb_Optim  --summary True --file-sessions df_elaad_preproc_feb.csv --file-price df_prices_c.csv && echo "feb_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name mar_ASAP   --summary True --file-sessions df_elaad_preproc_mar.csv --file-price df_prices_c.csv && echo "mar_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name mar_NoV2G  --summary True --file-sessions df_elaad_preproc_mar.csv --file-price df_prices_c.csv && echo "mar_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name mar_Optim  --summary True --file-sessions df_elaad_preproc_mar.csv --file-price df_prices_c.csv && echo "mar_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name apr_ASAP   --summary True --file-sessions df_elaad_preproc_apr.csv --file-price df_prices_c.csv && echo "apr_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name apr_NoV2G  --summary True --file-sessions df_elaad_preproc_apr.csv --file-price df_prices_c.csv && echo "apr_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name apr_Optim  --summary True --file-sessions df_elaad_preproc_apr.csv --file-price df_prices_c.csv && echo "apr_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name may_ASAP   --summary True --file-sessions df_elaad_preproc_may.csv --file-price df_prices_c.csv && echo "may_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name may_NoV2G  --summary True --file-sessions df_elaad_preproc_may.csv --file-price df_prices_c.csv && echo "may_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name may_Optim  --summary True --file-sessions df_elaad_preproc_may.csv --file-price df_prices_c.csv && echo "may_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name jun_ASAP   --summary True --file-sessions df_elaad_preproc_jun.csv --file-price df_prices_c.csv && echo "jun_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name jun_NoV2G  --summary True --file-sessions df_elaad_preproc_jun.csv --file-price df_prices_c.csv && echo "jun_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name jun_Optim  --summary True --file-sessions df_elaad_preproc_jun.csv --file-price df_prices_c.csv && echo "jun_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name jul_ASAP   --summary True --file-sessions df_elaad_preproc_jul.csv --file-price df_prices_c.csv && echo "jul_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name jul_NoV2G  --summary True --file-sessions df_elaad_preproc_jul.csv --file-price df_prices_c.csv && echo "jul_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name jul_Optim  --summary True --file-sessions df_elaad_preproc_jul.csv --file-price df_prices_c.csv && echo "jul_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name aug_ASAP   --summary True --file-sessions df_elaad_preproc_aug.csv --file-price df_prices_c.csv && echo "aug_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name aug_NoV2G  --summary True --file-sessions df_elaad_preproc_aug.csv --file-price df_prices_c.csv && echo "aug_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name aug_Optim  --summary True --file-sessions df_elaad_preproc_aug.csv --file-price df_prices_c.csv && echo "aug_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name sep_ASAP   --summary True --file-sessions df_elaad_preproc_sep.csv --file-price df_prices_c.csv && echo "sep_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name sep_NoV2G  --summary True --file-sessions df_elaad_preproc_sep.csv --file-price df_prices_c.csv && echo "sep_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name sep_Optim  --summary True --file-sessions df_elaad_preproc_sep.csv --file-price df_prices_c.csv && echo "sep_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name oct_ASAP   --summary True --file-sessions df_elaad_preproc_oct.csv --file-price df_prices_c.csv && echo "oct_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name oct_NoV2G  --summary True --file-sessions df_elaad_preproc_oct.csv --file-price df_prices_c.csv && echo "oct_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name oct_Optim  --summary True --file-sessions df_elaad_preproc_oct.csv --file-price df_prices_c.csv && echo "oct_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name nov_ASAP   --summary True --file-sessions df_elaad_preproc_nov.csv --file-price df_prices_c.csv && echo "nov_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name nov_NoV2G  --summary True --file-sessions df_elaad_preproc_nov.csv --file-price df_prices_c.csv && echo "nov_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name nov_Optim  --summary True --file-sessions df_elaad_preproc_nov.csv --file-price df_prices_c.csv && echo "nov_Optim" >> notify.txt &
#python3 RunChargeWorld.py --agent ASAP  --save-name dec_ASAP   --summary True --file-sessions df_elaad_preproc_dec.csv --file-price df_prices_c.csv && echo "dec_ASAP " >> notify.txt &
#python3 RunChargeWorld.py --agent NoV2G --save-name dec_NoV2G  --summary True --file-sessions df_elaad_preproc_dec.csv --file-price df_prices_c.csv && echo "dec_NoV2G" >> notify.txt &
#python3 RunChargeWorld.py --agent Optim --save-name dec_Optim  --summary True --file-sessions df_elaad_preproc_dec.csv --file-price df_prices_c.csv && echo "dec_Optim" >> notify.txt &

#python3 RunChargeWorld.py --agent ASAP  --save-name janfebmar_ASAP   --summary True  --file-sessions df_elaad_preproc_janfebmar.csv --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent NoV2G --save-name janfebmar_NoV2G  --summary True  --file-sessions df_elaad_preproc_janfebmar.csv --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent Optim --save-name janfebmar_Optim  --summary True  --file-sessions df_elaad_preproc_janfebmar.csv --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent ASAP  --save-name janfebmar_ASAP   --summary False --file-sessions df_elaad_preproc_janfebmar.csv --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent NoV2G --save-name janfebmar_NoV2G  --summary False --file-sessions df_elaad_preproc_janfebmar.csv --file-price df_prices_c.csv &
#python3 RunChargeWorld.py --agent Optim --save-name janfebmar_Optim  --summary False --file-sessions df_elaad_preproc_janfebmar.csv --file-price df_prices_c.csv &

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
