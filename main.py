import sys, numpy as np
from joblib import Parallel, delayed
import pandas as pd, time, random 
from Data_loader import load_train_test_data
from pytorch_utilities import run_model, hyper_param, init_model, save_model
from analysis import compare_pucker_angles, compare_backbone_angles, compare_atomic_pos, compare_3D_atomic_pos
from utilities import sugar_atom_label, bond_length_bw, find_best_model
from classes import Opt, Data

hyper_param()
hyper =  pd.read_csv('hyperparam.txt',delimiter='\s+')
df,df_min = find_best_model('/work/lcvmm/rsharma/cgDNAplus_sugar_fitting/')
sys.exit()



opt = Opt()
data = Data()

opt.path='/work/lcvmm/'
opt.NA='palin.bscl.tip3p.jc'
opt.seq_id_list = [1,2,3,4,5,6,7,8,9,10,11,12]
opt.seq_id_list_val = [13,14,15,16]
val_number = 4
opt.run_nbr_list=np.arange(1,50)
opt.remove_ends = 3
opt.sparse=100
opt.what = 'sugar_atoms' #what = 'dihed',  #what = 'pucker'
opt.which_seq = str(sys.argv[1])    ## YRY, RRR
opt.seq_specific = True
opt.nbp=24

data.model_type = 'NN'  #'LM'
if opt.what == 'pucker':
	data.param = ['0v','1v','2v','3v','4v']
elif opt.what == 'dihed': 
	data.param = ['a','b','g','d','e','z','x']
elif opt.what == 'sugar_atoms':
	data.param = sugar_atom_label
data.out_dim = len(data.param)

opt, data = load_train_test_data(opt, data)
data.inp_dim = data.X_Train.shape[1]

def parallel_run(index):
	time.sleep(random.randint(0,80))
	data.hyper_value = hyper.iloc[int(index)]         
	data.save_path = opt.which_seq + '_model_hyper.' + str(index) + '_val_lot.' + str(val_number)
	data.model = init_model(data)
	data.model = run_model(data)
	save_model(data)
	return None

Parallel(n_jobs=20)(delayed(parallel_run)(index) for index in np.arange(261,720))



