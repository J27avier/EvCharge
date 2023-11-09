import sys
sys.path.append("..")
from utils import *  # type: ignore

# Batch 1
l_month_agg_crit =             [f"month/batch_1/month_agg_crit_{i}" for i in range(30)]
l_month_agg_moresteps =        [f"month/batch_1/month_agg_moresteps_{i}" for i in range(32)]
l_month_agg_nomini =           [f"month/batch_1/month_agg_nomini_{i}" for i in range(99)]
l_month_agg_nomini_moresteps = [f"month/batch_1/month_agg_nomini_moresteps_{i}" for i in range(50)]
l_month_ind_crit =             [f"month/batch_1/month_ind_crit_{i}" for i in range(99)]
l_month_ind_moresteps =        [f"month/batch_1/month_ind_moresteps_{i}" for i in range(99)]
l_month_ind_nomini =           [f"month/batch_1/month_ind_nomini_{i}" for i in range(99)]
l_month_ind_nomini_moresteps = [f"month/batch_1/month_ind_nomini_moresteps_{i}" for i in range(99)]

# Batch 2
#l_month_agg_nomini_det =       [f"month/batch_2/month_agg_nomini_det_{i}" for i in range(73)]
#l_month_agg_nomini_relu =      [f"month/batch_2/month_agg_nomini_relu_{i}" for i in range(92)]
#l_month_agg_nomini_norm_rew_e= [f"month/batch_2/month_agg_nomini_norm_rew_e_{i}" for i in range(99)]
#l_month_agg_nomini_norm_rew_n= [f"month/batch_2/month_agg_nomini_norm_rew_n_{i}" for i in range(99)]
#l_month_ind_nomini_norm_rew_e= [f"month/batch_2/month_ind_nomini_norm_rew_e_{i}" for i in range(99)]
#l_month_ind_nomini_norm_rew_n= [f"month/batch_2/month_ind_nomini_norm_rew_n_{i}" for i in range(99)]
#
## Batch 3
#l_month_agg_nomini_norm_hstd_rew_n      = [f"month/batch_3/month_agg_nomini_norm_hstd_rew_n_{i}" for i in range(49)]
#l_month_agg_nomini_norm_nalr_rew_n      = [f"month/batch_3/month_agg_nomini_norm_nalr_rew_n_{i}" for i in range(49)]
#l_month_agg_nomini_norm_naLr_rew_n      = [f"month/batch_3/month_agg_nomini_norm_naLr_rew_n_{i}" for i in range(49)]
#l_month_agg_nomini_norm_NN_rew_n        = [f"month/batch_3/month_agg_nomini_norm_NN_rew_n_{i}" for i in range(49)]
#l_month_ind_nomini_norm_hstd_rew_n      = [f"month/batch_3/month_ind_nomini_norm_hstd_rew_n_{i}" for i in range(49)]
#l_month_ind_nomini_norm_nalr_rew_n      = [f"month/batch_3/month_ind_nomini_norm_nalr_rew_n_{i}" for i in range(49)]
#l_month_ind_nomini_norm_naLr_rew_n      = [f"month/batch_3/month_ind_nomini_norm_naLr_rew_n_{i}" for i in range(49)]
#l_month_ind_nomini_norm_narl_relu_rew_n = [f"month/batch_3/month_ind_nomini_norm_narl_relu_rew_n_{i}" for i in range(49)] # narl?
#l_month_ind_nomini_norm_NN_rew_n        = [f"month/batch_3/month_ind_nomini_norm_NN_rew_n_{i}" for i in range(49)] 
#l_month_ind_nomini_norm_relu_rew_n      = [f"month/batch_3/month_ind_nomini_norm_relu_rew_n_{i}" for i in range(49)]
#
## Batch 4
#l_month_agg_crit_moremini_n   = [f"month/batch_4/month_agg_crit_moremini_n_{i}" for i in range(49)]
#l_month_agg_crit_n            = [f"month/batch_4/month_agg_crit_n_{i}" for i in range(49)]
#l_month_agg_crit_n_less_steps = [f"month/batch_4/month_agg_crit_n_less_steps_{i}" for i in range(49)]
#l_month_agg_crit_n_nalr       = [f"month/batch_4/month_agg_crit_n_nalr_{i}" for i in range(49)]
#l_month_agg_crit_proj001_n    = [f"month/batch_4/month_agg_crit_proj001_n_{i}" for i in range(49)]
#l_month_agg_crit_proj01_n     = [f"month/batch_4/month_agg_crit_proj01_n_{i}" for i in range(49)]
#l_month_agg_crit_proj1_n      = [f"month/batch_4/month_agg_crit_proj1_n_{i}" for i in range(49)]
#l_month_agg_moremini_n        = [f"month/batch_4/month_agg_moremini_n_{i}" for i in range(49)]
#l_month_agg_n                 = [f"month/batch_4/month_agg_n_{i}" for i in range(49)]
#l_month_agg_n_less_steps      = [f"month/batch_4/month_agg_n_less_steps_{i}" for i in range(49)]
#l_month_agg_n_nalr            = [f"month/batch_4/month_agg_n_nalr_{i}" for i in range(49)]
#l_month_agg_proj001_n         = [f"month/batch_4/month_agg_proj001_n_{i}" for i in range(49)]
#l_month_agg_proj01_n          = [f"month/batch_4/month_agg_proj01_n_{i}" for i in range(49)]
#l_month_agg_proj1_n           = [f"month/batch_4/month_agg_proj1_n_{i}" for i in range(49)]
#
## Batch 5
#l_month_agg_relu_n            = [f"month/batch_5/month_agg_relu_n_{i}" for i in range(49)]
#l_month_agg_relu_n_less_steps = [f"month/batch_5/month_agg_relu_n_less_steps_{i}" for i in range(49)]
#l_month_agg_relu_proj001_n    = [f"month/batch_5/month_agg_relu_proj001_n_{i}" for i in range(49)]
#l_month_agg_relu_proj1_n      = [f"month/batch_5/month_agg_relu_proj1_n_{i}" for i in range(49)]
#
## Batch 6
#l_month_agg_mini_relu_n       = [f"month/batch_6/month_agg_mini_relu_n_{i}" for i in range(49)]
#l_month_agg_norma_moresteps_n = [f"month/batch_6/month_agg_norma_moresteps_n_{i}" for i in range(49)]
#l_month_agg_norma_n           = [f"month/batch_6/month_agg_norma_n_{i}" for i in range(49)]
#l_month_ind_norma_moresteps_n = [f"month/batch_6/month_ind_norma_moresteps_n_{i}" for i in range(49)] 
#l_month_ind_norma_n           = [f"month/batch_6/month_ind_norma_n_{i}" for i in range(49)]
#
# Batch 7
l_month_agg_norma           = [f"month/batch_7/month_agg_norma_{i}" for i in range(49)]
l_month_agg_norma_moresteps = [f"month/batch_7/month_agg_norma_moresteps_{i}" for i in range(49)]
l_month_ind_norma           = [f"month/batch_7/month_ind_norma_{i}" for i in range(49)]
l_month_ind_norma_moresteps = [f"month/batch_7/month_ind_norma_moresteps_{i}" for i in range(49)] 
lus_month_agg_norma         = [f"month/batch_7/lus_month_agg_norma_{i}" for i in range(10)]
lus_month_ind_norma         = [f"month/batch_7/lus_month_agg_norma_{i}" for i in range(10)]
l_month_agg_init_norm       = [f"month_agg_init_norm_{i}" for i in range(10)]
l_month_sep_init_norm       = [f"month_sep_init_norm_{i}" for i in range(10)]

# Batch 1
df_month_agg_crit              = summ_table( l_month_agg_crit , l_month_agg_crit )
df_month_agg_moresteps         = summ_table( l_month_agg_moresteps , l_month_agg_moresteps )
df_month_agg_nomini            = summ_table( l_month_agg_nomini , l_month_agg_nomini )
df_month_agg_nomini_moresteps  = summ_table( l_month_agg_nomini_moresteps , l_month_agg_nomini_moresteps )
df_month_ind_crit              = summ_table( l_month_ind_crit , l_month_ind_crit )
df_month_ind_moresteps         = summ_table( l_month_ind_moresteps , l_month_ind_moresteps )
df_month_ind_nomini            = summ_table( l_month_ind_nomini , l_month_ind_nomini )
df_month_ind_nomini_moresteps  = summ_table( l_month_ind_nomini_moresteps, l_month_ind_nomini_moresteps)

# Batch 2
#df_month_agg_nomini_det        = summ_table( l_month_agg_nomini_det, l_month_agg_nomini_det)
#df_month_agg_nomini_relu       = summ_table( l_month_agg_nomini_relu, l_month_agg_nomini_relu)
#df_month_agg_nomini_norm_rew_e = summ_table( l_month_agg_nomini_norm_rew_e, l_month_agg_nomini_norm_rew_e)
#df_month_agg_nomini_norm_rew_n = summ_table( l_month_agg_nomini_norm_rew_n, l_month_agg_nomini_norm_rew_n)
#df_month_ind_nomini_norm_rew_e = summ_table( l_month_ind_nomini_norm_rew_e, l_month_ind_nomini_norm_rew_e)
#df_month_ind_nomini_norm_rew_n = summ_table( l_month_ind_nomini_norm_rew_n, l_month_ind_nomini_norm_rew_n)
#
##Batch 3
#df_month_agg_nomini_norm_hstd_rew_n      = summ_table( l_month_agg_nomini_norm_hstd_rew_n, l_month_agg_nomini_norm_hstd_rew_n)
#df_month_agg_nomini_norm_nalr_rew_n      = summ_table( l_month_agg_nomini_norm_nalr_rew_n, l_month_agg_nomini_norm_nalr_rew_n)
#df_month_agg_nomini_norm_naLr_rew_n      = summ_table( l_month_agg_nomini_norm_naLr_rew_n, l_month_agg_nomini_norm_naLr_rew_n)
#df_month_agg_nomini_norm_NN_rew_n        = summ_table( l_month_agg_nomini_norm_NN_rew_n, l_month_agg_nomini_norm_NN_rew_n)
#df_month_ind_nomini_norm_hstd_rew_n      = summ_table( l_month_ind_nomini_norm_hstd_rew_n, l_month_ind_nomini_norm_hstd_rew_n)
#df_month_ind_nomini_norm_nalr_rew_n      = summ_table( l_month_ind_nomini_norm_nalr_rew_n, l_month_ind_nomini_norm_nalr_rew_n)
#df_month_ind_nomini_norm_naLr_rew_n      = summ_table( l_month_ind_nomini_norm_naLr_rew_n, l_month_ind_nomini_norm_naLr_rew_n)
#df_month_ind_nomini_norm_narl_relu_rew_n = summ_table( l_month_ind_nomini_norm_narl_relu_rew_n, l_month_ind_nomini_norm_narl_relu_rew_n)
#df_month_ind_nomini_norm_NN_rew_n        = summ_table( l_month_ind_nomini_norm_NN_rew_n, l_month_ind_nomini_norm_NN_rew_n)
#df_month_ind_nomini_norm_relu_rew_n      = summ_table( l_month_ind_nomini_norm_relu_rew_n, l_month_ind_nomini_norm_relu_rew_n)
#
##Batch 4
#df_month_agg_crit_moremini_n   = summ_table(l_month_agg_crit_moremini_n ,l_month_agg_crit_moremini_n )
#df_month_agg_crit_n            = summ_table(l_month_agg_crit_n ,l_month_agg_crit_n )
#df_month_agg_crit_n_less_steps = summ_table(l_month_agg_crit_n_less_steps ,l_month_agg_crit_n_less_steps )
#df_month_agg_crit_n_nalr       = summ_table(l_month_agg_crit_n_nalr ,l_month_agg_crit_n_nalr )
#df_month_agg_crit_proj001_n    = summ_table(l_month_agg_crit_proj001_n ,l_month_agg_crit_proj001_n )
#df_month_agg_crit_proj01_n     = summ_table(l_month_agg_crit_proj01_n ,l_month_agg_crit_proj01_n )
#df_month_agg_crit_proj1_n      = summ_table(l_month_agg_crit_proj1_n ,l_month_agg_crit_proj1_n )
#df_month_agg_moremini_n        = summ_table(l_month_agg_moremini_n ,l_month_agg_moremini_n )
#df_month_agg_n                 = summ_table(l_month_agg_n ,l_month_agg_n )
#df_month_agg_n_less_steps      = summ_table(l_month_agg_n_less_steps ,l_month_agg_n_less_steps )
#df_month_agg_n_nalr            = summ_table(l_month_agg_n_nalr ,l_month_agg_n_nalr )
#df_month_agg_proj001_n         = summ_table(l_month_agg_proj001_n ,l_month_agg_proj001_n )
#df_month_agg_proj01_n          = summ_table(l_month_agg_proj01_n ,l_month_agg_proj01_n )
#df_month_agg_proj1_n           = summ_table(l_month_agg_proj1_n ,l_month_agg_proj1_n )
#
## Batch 5
#df_month_agg_relu_n           = summ_table(l_month_agg_relu_n ,l_month_agg_relu_n )
#df_month_agg_relu_n_less_steps= summ_table(l_month_agg_relu_n_less_steps ,l_month_agg_relu_n_less_steps )
#df_month_agg_relu_proj001_n   = summ_table(l_month_agg_relu_proj001_n ,l_month_agg_relu_proj001_n )
#df_month_agg_relu_proj1_n     = summ_table(l_month_agg_relu_proj1_n ,l_month_agg_relu_proj1_n )
#
## Batch 6
#df_month_agg_mini_relu_n       = summ_table(l_month_agg_mini_relu_n ,l_month_agg_mini_relu_n )
#df_month_agg_norma_moresteps_n = summ_table(l_month_agg_norma_moresteps_n ,l_month_agg_norma_moresteps_n )
#df_month_agg_norma_n           = summ_table(l_month_agg_norma_n ,l_month_agg_norma_n )
#df_month_ind_norma_moresteps_n = summ_table(l_month_ind_norma_moresteps_n ,l_month_ind_norma_moresteps_n )
#df_month_ind_norma_n           = summ_table(l_month_ind_norma_n ,l_month_ind_norma_n )

# Batch 7
df_month_agg_norma           = summ_table(l_month_agg_norma ,l_month_agg_norma )
df_month_agg_norma_moresteps = summ_table(l_month_agg_norma_moresteps ,l_month_agg_norma_moresteps )
df_month_ind_norma           = summ_table(l_month_ind_norma ,l_month_ind_norma )
df_month_ind_norma_moresteps = summ_table(l_month_ind_norma_moresteps ,l_month_ind_norma_moresteps ) 
df_lus_month_agg_norma       = summ_table(lus_month_agg_norma, lus_month_agg_norma) 
df_lus_month_ind_norma       = summ_table(lus_month_agg_norma, lus_month_agg_norma) 
df_month_agg_init_norm       = summ_table( l_month_agg_init_norm, l_month_agg_init_norm)
df_month_sep_init_norm       = summ_table( l_month_sep_init_norm, l_month_sep_init_norm)
