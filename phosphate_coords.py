# idea is to check the correlation between plosphate coordinates and sugar pucker mode. 
# Rahul Sharma 6 March 2022

import sys, numpy as np
from joblib import Parallel, delayed
import pandas as pd, time, random
from Data_loader import merge_total_data
from classes import Opt, Data
from scipy.stats import pearsonr

opt = Opt()
data = Data()
opt.path='/work/lcvmm/'
opt.NA='palin.bscl.tip3p.jc'
opt.NA='rna.ol3.tip3p.jc'
opt.seq_id_list = [1,2,3,4,5,6,7,8,9,10,11,12]
opt.seq_id_list_val = [13,14,15,16]
val_number = 4
opt.run_nbr_list=np.arange(1,15)
opt.remove_ends = 3
opt.sparse=100
opt.what = 'sugar_atoms' #what = 'dihed',  #what = 'pucker'
opt.which_seq ='RRR'    ## YRY, RRR
opt.seq_specific = True
opt.nbp=24

dihed1, pucker1, _, _, ic1 = merge_total_data(opt,'Train')  ## the second argument is merely symbolic and tells which sequence id list one should load.. this will load --> opt.seq_id_list

opt.which_seq ='RYR'
dihed2, pucker2, _, _, ic2 = merge_total_data(opt,'Train') 

opt.which_seq ='YYR'
dihed3, pucker3, _, _, ic3 = merge_total_data(opt,'Train')

opt.which_seq ='YRY'
dihed4, pucker4, _, _, ic4 = merge_total_data(opt,'Train')


dihed  = pd.concat([dihed1 ,dihed2 ,dihed3, dihed4])
pucker = pd.concat([pucker1,pucker2,pucker3,pucker4])
ic     = pd.concat([ic1    ,ic2    ,ic3    ,ic4])
#pucker.loc[pucker['p']>180,'p'] = 360 - pucker.loc[pucker['p']>180,'p'] 

