import numpy as np, pandas as pd
import sys, copy, random, itertools, re
sys.path.append('./cgDNAplus_python/modules')
sys.path.append('./cgDNAplus_python/classes')
from E_transform import Etrans
from cgDNAUtils import frames
from joblib import Parallel, delayed
import MDAnalysis as mda
import RotationUtils as Rot
import scipy.io as sio
from utilities import seq2YR, shift, rotate, seq_to_binary, dihed_label, pucker_label, sugar_atom_label, atom_labels, np2df, comp, ideal_atoms
from utilities import read_hbond_filter_file, read_seq, read_dihedrals, read_pucker, read_coord
from Ideal_coords_2 import base_atoms, phos_atoms

def frame_loader(opt):  ## check the path for frame
	total_path = opt.path + opt.NA+'/' +str(opt.seq_id) + '/Frames_cgf/' + opt.NA + '_' + str(opt.seq_id) + '.' + str(opt.run_nbr)+'.fra'
	orig_data  = pd.read_csv(total_path,header=None,engine='c',delimiter='\s+')  ### Reading frames
	nbp = orig_data[1].max()
	nframe = orig_data.shape[0]//(2*nbp*opt.sparse)
	mid_frame = np.zeros((3*(2*nbp),4*nframe))  # remove 2 when use mid-frame
	for n in range(nbp):
		ind1w    =  [3*n   + x for x in range(3)]
		ind1c    =  [3*2*nbp-3*(n+1) + x for x in range(3)]  ## this way frame direction will be 5' to 3' in both cases
		data    = orig_data[orig_data[1]==n+1]  ## only reading nth frame .. to find nth base-pair frame
		w_frame = data[data[0]==1].reset_index(drop=True)
		c_frame = data[data[0]==2].reset_index(drop=True)
		w_frame = w_frame.iloc[::opt.sparse].reset_index(drop=True)
		c_frame = c_frame.iloc[::opt.sparse].reset_index(drop=True)
		for i in range(nframe):
			ind2  =  [4*i + x for x in range(4)]
			R1,r1 = w_frame.iloc[i][2:11].to_numpy().reshape(3,3), w_frame.iloc[i][11:14].to_numpy()
			R2,r2 = c_frame.iloc[i][2:11].to_numpy().reshape(3,3), c_frame.iloc[i][11:14].to_numpy()
			R2[:,[1,2]] = - R2[:,[1,2]] # flip the frame 
#			mid_frame[np.ix_(ind1,ind2[0:3])] = Rot.midFrame(R2,np.matmul(R2.T,R1))
#			mid_frame[ind1,ind2[3:4]]         = (r1+r2)/2

			mid_frame[np.ix_(ind1w,ind2[0:3])], mid_frame[ind1w,ind2[3:4]]  = R1, r1
			mid_frame[np.ix_(ind1c,ind2[0:3])], mid_frame[ind1c,ind2[3:4]]  = R2, r2
	return mid_frame

def reorient_sugar(data,frame):
	# in frame nbp+1 frame corresponds to first bp
	# in sugar nbp+1 frame corresponds to last bp -- so need to swap -- implemented above
	nb = 48
	for k in data.keys():
		for n in range(nb//2):
			ind1,ind2 = [3*n + x for x in range(3)],  [4*k + x for x in range(4)]
			R, r = frame[np.ix_(ind1,ind2[0:3])], frame[ind1,ind2[3:4]]
			data[k][n*6:(n+1)*6,:] =   np.matmul(data[k][n*6:(n+1)*6,:] - r,R.T)  # 6 X 3 
			# inverse of svd fit approach
		if k == 0:
			tmp = np.reshape(data[k],-1)[:,np.newaxis]
		else:
			tmp = np.concatenate([tmp,np.reshape(data[k],-1)[:,np.newaxis]],axis=1)

	df = np2df(tmp.T,np.arange(nb*6*3))
	return df


def read_sugar_atoms(opt):
	total_path = opt.path+opt.NA+'/' +str(opt.seq_id) + '/anl/Dry/' + opt.NA + '_' + str(opt.seq_id) + '.' + str(opt.run_nbr)+'.ions.pdb'
	u = mda.Universe(total_path)	
	nbp = len(u.residues.ix)//2
	data = {}
	for enum,ts in enumerate(u.trajectory[0::opt.sparse]):
		data[enum] = u.atoms.positions
	frame = frame_loader(opt)
	df = reorient_sugar(data,frame)
	return df

def sequence_specific_nt_list(opt):
	seq = read_seq(opt)
	seq = seq2YR(seq)
#	print(seq,opt.which_seq)
	seq = seq[opt.remove_ends::]
	if opt.seq_specific == True:
		nt_list = np.array([_.start() for _ in re.finditer(opt.which_seq, seq)]) + 2 + opt.remove_ends
	else:
		nt_list = np.arange(2 + opt.remove_ends, opt.nbp-opt.remove_ends)
#	print(nt_list)
	return nt_list

def check_seq_list(opt):
	tmp = []
	for seq_id in opt.seq_id_list:
		opt.seq_id = seq_id
		tmp_list = sequence_specific_nt_list(opt) 
		opt.seq_id = None
		if len(tmp_list) > 0:
			tmp.append(seq_id)
	opt.seq_id_list = tmp
	return opt

def extract_nth_data(opt,nt_list,data,data_type):

	out = {}
	if data_type == 'pucker':
		for j in nt_list:
			tmp_label = [p+str(j) for p in pucker_label]  # indexing in sugar starts from 1
			out[j] = data[tmp_label]

	elif data_type == 'dihed':
		for j in nt_list:
			tmp_label = [d+str(j) for d in dihed_label]
			out[j] = data[tmp_label]

	elif data_type == 'coord':
		for j in nt_list:
#			if j < nbp:
			tmp_label = np.arange(24*j-30-18,24*j+6)    #### extracting coordinares for trimer context
			out[j]    = data[tmp_label] 
#			elif j > nbp:
#				tmp_label = np.arange(24*(j-nbp)-30-18,24*(j-nbp)+6) # on complementary strand
#				out[j]    = data[tmp_label]	
#				None
	elif data_type == 'sugar_atoms':
		for j in nt_list:
			tmp_label = np.arange(18*(j-1),18*(j-1)+18)
			out[j] = data[tmp_label]
			out[j].columns = sugar_atom_label
	else:
		print("Incorrect data type provided for data extraction ----- \n")
		sys.exit()
	return out


#### to compute coordinate for bases and phos
# note that in rigid body transformation order matters, here first rotation is applied and then translation
# see svd eth article  # https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
# find the trimer and fit the frame 
# q = Rp + t  -->  p = R^T(q-t) 



def compute_and_orient_frame_vector(c):
	_, _, Rc, rc, Rw, rw, Rpw , rpw , Rpc, rpc = frames(c.to_numpy())
	R, r = Rw[1].T, rw[1]

	Rc  = rotate(Rc ,R)
	Rw  = rotate(Rw ,R)
	rc  = shift(rc ,R,r)
	rw  = shift(rw ,R,r)
	Rpc  = rotate(Rpc ,R)
	Rpw  = rotate(Rpw ,R)
	rpw  = shift(rpw ,R,r)	
	rpc  = shift(rpc ,R,r)
	return Rc, rc, Rw, rw, Rpw , rpw , Rpc, rpc


def fit_atomistic_coords(rot,trans,seq):
	c = {}
	for enum,k in enumerate(rot.keys()):
		try:
			c[k] = np.matmul(ideal_atoms[seq[enum]],rot[k].T) + trans[k] ## rotation followed by translation
		except:
			None   ### so now the 3' phos is ignored 
	return c

def atomistic_coords_to_vec(data):
	vec = np.array([])
	for D in data:
		for key in D.keys():		
			tmp = D[key].flatten('C')
			vec = np.concatenate((vec, tmp), axis=0)
	vec = vec.flatten('C')
	return vec


def find_atomistic_coords(seq,c):
	Rc, rc, Rw, rw, Rpw , rpw , Rpc, rpc = compute_and_orient_frame_vector(c)
	Bw = fit_atomistic_coords(Rw , rw , seq)
	Bc = fit_atomistic_coords(Rc , rc , comp(seq))
	Pw = fit_atomistic_coords(Rpw, rpw, 'PPP')
	Pc = fit_atomistic_coords(Rpc, rpc, 'PPP')
#	vec = atomistic_coords_to_vec([Bw,Bc,Pw,Pc])
	binary = seq_to_binary(seq)
	vec = atomistic_coords_to_vec([binary,Bw,Pw])
	return vec

def coord_to_atomistic_coords(opt,coord):
	dim_label = ['S1','S2','S3'] + atom_labels(base_atoms[opt.which_seq[0]],'1')+ atom_labels(base_atoms[opt.which_seq[1]],'2') +atom_labels(base_atoms[opt.which_seq[2]],'3')+ atom_labels(phos_atoms,'1')+atom_labels(phos_atoms,'2')
	dimension = len(dim_label) 
	vec_coord = np.array([])
	seq = read_seq(opt)
#	cseq = comp(seq)
	for nt in coord.keys():
#		if nt < 24:  # note it works for seq of length 24
#			tmp_seq = seq[nt-2:nt+1]
#		else:
#			tmp_seq = cseq[nt-2-24:nt+1-24]
		tmp_seq = seq[nt-2:nt+1]  ## only dealing with Watson strand  
		for ind in coord[nt].index.to_list():
			tmp = find_atomistic_coords(tmp_seq, coord[nt].loc[ind])
			vec_coord = np.concatenate((vec_coord, tmp), axis=None)			
	vec_coord = np.reshape(vec_coord, (-1,dimension)) 
	df = np2df(vec_coord, dim_label)	
	return df

def read_total_data(opt):

	coord  = read_coord(opt)
	dihed  = read_dihedrals(opt)
	pucker = read_pucker(opt)
	sugar  = read_sugar_atoms(opt)
	coord  = coord.iloc[opt.iall]
	dihed  = dihed.iloc[opt.iall]
	pucker = pucker.iloc[opt.iall]
	sugar  = sugar.iloc[opt.iall]
	nt_list = sequence_specific_nt_list(opt)
	d = extract_nth_data(opt,nt_list,dihed ,'dihed' )
	p = extract_nth_data(opt,nt_list,pucker,'pucker')
	internal_coord = extract_nth_data(opt,nt_list,coord ,'coord' )
	s = extract_nth_data(opt,nt_list,sugar ,'sugar_atoms' )
	vec_coord = coord_to_atomistic_coords(opt,internal_coord)
	return d, p, vec_coord, s, internal_coord


def merge_data(opt,run_nbr,seq_id):
	opt.run_nbr,opt.seq_id=run_nbr,seq_id
	print("Reading seq_id and run_nbr ....", opt.seq_id ,opt.run_nbr)
	opt = read_hbond_filter_file(opt)
	dt = pd.DataFrame(data=None, columns=dihed_label )#, index=d[2].index)
	pt = pd.DataFrame(data=None, columns=pucker_label)#, index=p[2].index)
	st = pd.DataFrame(data=None, columns=sugar_atom_label)#, index=p[2].index)
	ic = pd.DataFrame(data=None, columns=np.arange(54))
	if np.size(opt.iall) > 0:
		d, p, ct, s,internal_coord = read_total_data(opt)
		for k in d.keys():
			d[k].columns = dihed_label
			p[k].columns = pucker_label
			dt = dt.append(d[k] , ignore_index=True)
			pt = pt.append(p[k] , ignore_index=True)
			st = st.append(s[k] , ignore_index=True)
			ttmmpp = internal_coord[k]
			ttmmpp.columns = np.arange(54)
			ic = ic.append(ttmmpp, ignore_index=True)
	else:
		dim_label = ['S1','S2','S3'] + atom_labels(base_atoms[opt.which_seq[0]],'1')+ atom_labels(base_atoms[opt.which_seq[1]],'2') +atom_labels(base_atoms[opt.which_seq[2]],'3')+ atom_labels(phos_atoms,'1')+atom_labels(phos_atoms,'2')
		ct = np2df(None,dim_label)

	return [dt, pt, ct, st, ic]

def merge_total_data(opt,load_type):

	print("Reading data ... ... ...")
	if load_type == 'Train':
		data = Parallel(n_jobs=35)(delayed(merge_data)(opt,run_nbr,seq_id) for run_nbr in opt.run_nbr_list for seq_id in opt.seq_id_list)
	elif load_type == 'Test':
		data = Parallel(n_jobs=35)(delayed(merge_data)(opt,run_nbr,seq_id) for run_nbr in opt.run_nbr_list for seq_id in opt.seq_id_list_Test)
	elif load_type == 'val':
		data = Parallel(n_jobs=35)(delayed(merge_data)(opt,run_nbr,seq_id) for run_nbr in opt.run_nbr_list for seq_id in opt.seq_id_list_val)
	else:
		print("which data to load not sepcified -------------------- ")
		sys.exit()
	
	for enum, D in enumerate(data):
		if enum == 0:
			f_dt, f_pt, f_ct, f_st, internal_coord = D[0], D[1], D[2], D[3], D[4] 
		else:
			f_dt, f_pt, f_ct, f_st, internal_coord = f_dt.append(D[0], ignore_index=True), f_pt.append(D[1] , ignore_index=True), f_ct.append(D[2] , ignore_index=True), f_st.append(D[3] , ignore_index=True), internal_coord.append(D[4] , ignore_index=True)
		print(enum, np.shape(f_dt), np.shape(f_pt),np.shape(f_ct), np.shape(f_st))

	print("Reading data completed ... ... ... .")

	return f_dt, f_pt, f_ct, f_st, internal_coord

def split_test_train(inp,out):
	nbr_samples = inp.shape[0]
	index = np.arange(nbr_samples)
	rng = np.random.default_rng()
	rng.shuffle(index)
	test_index  = index[0:nbr_samples//5]  ### 20:80 split
	train_index = index[nbr_samples//5::]
	train_inp, train_out = inp.iloc[train_index], out.iloc[train_index]
	test_inp ,  test_out = inp.iloc[test_index] , out.iloc[test_index]
	return train_inp, train_out, test_inp, test_out


def data_seq_specific(dt, pt, ct, st, seq):
	b = seq_to_binary(seq) 
	b = b['seq']
	ct = ct[(ct['S1']==b[0]) & (ct['S2']==b[1]) & (ct['S3']==b[2])]
	ind = ct.index.to_list()
	return dt.loc[ind], pt.loc[ind], ct, st.loc[ind]

def load_train_test_data(opt,data):
	# X_val[(X_val['S1']==1) & (X_val['S2']==1) & (X_val['S3']==0)]
	opt = check_seq_list(opt)   # this will make the reading faster removing unnecessary seq.
	data.ba_Train, data.pa_Train, data.X_Train, data.Y_Train, _ = merge_total_data(opt,'Train')
	data.ba_val, data.pa_val, data.X_val, data.Y_val, _ = merge_total_data(opt,'val')
	col = data.X_Train.columns.to_list()
	col = col[3::]
	data.X_Train = data.X_Train[col]   ### removing the sequence info
	data.X_val = data.X_val[col]
	return  opt, data


