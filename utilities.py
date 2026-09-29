import scipy.io as sio
import pandas as pd, sys, numpy as np
from Ideal_coords_2 import A,T,C,G,P

dihed_label = ['a','b','g','d','e','z','x']    ## 7 dihedrals per interior nt and 5 diheds per end nt. 
# a and b is missing for 5' end while e and z is missing for 3' end. 
pucker_label = ['0v','1v','2v','3v','4v','p']  ## 5 dihedrals and one pucker parameter
ideal_atoms = dict({'A':A, 'T':T, 'C':C, 'G':G, 'P':P})
timers_YR = ['RRR', 'RRY', 'RYR', 'YRR', 'YYR', 'YRY', 'RYY', 'YYY']

sugar_atom_label = [
                        'C5px', 'C5py', 'C5pz',
                        'C4px', 'C4py', 'C4pz',
                        'O4px', 'O4py', 'O4pz',
                        'C1px', 'C1py', 'C1pz',
                        'C3px', 'C3py', 'C3pz',
                        'C2px', 'C2py', 'C2pz'
                        ]
def atom_labels(atoms,ind=''):
	return [ind+i+j for i in atoms for j in ['x','y','z']] 

def np2df(data,col):
	return pd.DataFrame(data=data,columns=col)

def comp(base):
        complement = ''
        for i in base:
                if i=="C":
                        com = "G"
                elif i=="G":
                        com = "C"
                elif i=="A":
                        com = "T"
                elif i=="T":
                        com = "A"
                elif i=="Y":
                        com = "R"
                elif i=="R":
                        com = "Y"
                complement = com + complement
        return complement

def bond_length(a1,a2):
	return np.linalg.norm(a1-a2)

def seq2YR(seq):
	YR = ''
	for s in seq:
		if s in ['A','G']:
			YR = YR + 'R'
		elif s in ['C','T']:
			YR = YR + 'Y'
		else:
			print("unrecingnised seq --------")
			sys.exit()
	return YR

def rotate(D,R):
        for k in D.keys():
                try:
                        D[k] = np.matmul(R,D[k])
                except: 
                        D[k] = []
        return D

def shift(D,R,r):
        for k in D.keys():
                try:
                        D[k] = np.matmul(R,D[k]-r)
                except: 
                        D[k] = []
        return D

def seq_to_binary(seq):
        binary = np.zeros(len(seq))
        for enum,s in enumerate(seq):
                if s in ['A','G','R']:
                        binary[enum] = 1
        binara = {}
        binara['seq'] = binary
        return binara

def bond_length_bw(data,a1,a2,ind):
	b1 = data.loc[ind][atom_labels(a1)]
	b2 = data.loc[ind][atom_labels(a2)]
	print(b1,b2)
	b1,b2 = b1.to_numpy(), b2.to_numpy()
	return  bond_length(b1,b2)


def read_dihedrals(opt):
        total_path = opt.path + opt.NA + '/' + str(opt.seq_id) + '/Dihed/' + opt.NA + '_' + str(opt.seq_id) + '.' + str(opt.run_nbr) + '.dihed.out'
        dihed = pd.read_csv(total_path,engine='c',delimiter='\s+')
        dihed = dihed.iloc[::opt.sparse, :]
        dihed = dihed.reset_index(drop=True)
        return dihed

def read_pucker(opt):
        total_path = opt.path + opt.NA + '/' + str(opt.seq_id) + '/Pucker/' + opt.NA + '_' + str(opt.seq_id) + '.' + str(opt.run_nbr) + '.pucker.out'
        pucker = pd.read_csv(total_path,engine='c',delimiter='\s+')
        pucker = pucker.iloc[::opt.sparse, :]
        pucker = pucker.reset_index(drop=True)
        return pucker

def read_coord(opt):
        total_path1 = opt.path + opt.NA + '/' + str(opt.seq_id) + '/Coord_cgf/' + opt.NA + '_' + str(opt.seq_id) + '.' + str(opt.run_nbr) + '.coord'
        total_path2 = opt.path + opt.NA + '/' + str(opt.seq_id) + '/Coord/' + opt.NA + '_' + str(opt.seq_id) + '.' + str(opt.run_nbr) + '.coord'
        try:
                coord = pd.read_csv(total_path1,engine='c',delimiter='\s+',header=None)
        except:
                coord = pd.read_csv(total_path2,engine='c',delimiter='\s+',header=None)
        coord = coord.T
        coord = coord.iloc[::opt.sparse, :]
        coord = coord.reset_index(drop=True)
        return coord

def read_seq(opt):
        # seq.palin.bscl.tip3p.jc_1.txt
        total_path = opt.path + opt.NA + '/' + str(opt.seq_id) + '/seq.' + opt.NA + '_' + str(opt.seq_id) + '.txt'
        seq = pd.read_csv(total_path,header=None)
        seq = seq[0][0]
        return seq

def find_best_model(path):
	dfs = pd.DataFrame(columns=['seq','loss','mae','val_loss','val_mae','index'])
	count = 0
	for tri in timers_YR:
		for f_index in range(2000):
			try:
				df = pd.read_csv('Logs/'+tri+'_model_hyper.'+ str(f_index) +'_val_lot.4.csv',sep=';')
				s = df.shape[0]
				mean = df.loc[s-10:s].mean() 
				dfs.loc[count] = [tri]+list(mean.to_numpy()[1::]) + [f_index]
				count = count + 1
			except:
				None
	df_min = pd.DataFrame(columns=['seq','loss','mae','val_loss','val_mae','index'])
	for enum,tri in enumerate(timers_YR):
		idx = dfs[dfs['seq']==tri]['val_mae'].idxmin()
		df_min.loc[enum] = dfs.iloc[idx]
	return dfs, df_min

def read_hbond_filter_file(opt):
        path = opt.path + opt.NA+'/' + str(opt.seq_id) +'/anl/HB/par_hbonds.mat'
        par_file = sio.loadmat(path,squeeze_me=True)
        iall = par_file['iall']
        iall = iall[(opt.run_nbr-1)*5000:opt.run_nbr*5000]
        iall = iall[::opt.sparse]
        non_zero_iall_index = np.nonzero(iall)[0]
        opt.iall = non_zero_iall_index
        return opt

def pseudo_rot_angle_3DNA(df):
	# implemented in 3DNA
	#a = np.sin(36*np.pi/180) + np.sin(72*np.pi/180) = 1.5388417685876266 * 2  = 3.077683537175253
	num = df['4v'] + df['1v'] - df['3v'] - df['0v']
	den = 3.077683537175253*df['2v']
	f = num/den
	P = 180*np.arctan(f)/np.pi
	if df['2v'] < 0 :
		P = P + 180
	return P


def pseudo_rot_angle_CURVESP(df):
        #a = np.sin(36*np.pi/180) + np.sin(72*np.pi/180) = 1.5388417685876266 * 2  = 3.077683537175253
        # [v0, v1, v2, v3, v4] = [t4, t5, t1, t2, t3]
        # this definition has been implemented in CPPtraj and curves+
        num, den = 0, 0	
        for enum,v in enumerate(['2v', '3v', '4v', '0v', '1v']):
                num = num - df[v]*np.sin(4*np.pi*(enum)/5)
                den = den + df[v]*np.cos(4*np.pi*(enum)/5)
        P = 180*np.arctan(num/den)/np.pi
        if df['2v'] < 0 :
                P = P + 180
        return P
