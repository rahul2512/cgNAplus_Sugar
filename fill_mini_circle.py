import numpy as np, pandas as pd, sys
sys.path.append('./cgDNAplus_python/modules')
sys.path.append('./cgDNAplus_python/classes')
sys.path.append('.')
from cgDNAclass import cgDNA
import cgDNAUtils as tools
import MD_analysis as analysis
from E_transform import Etrans
from Init_MD import init_MD_data
import matplotlib.colors as mcolors
from utilities import pseudo_rot_angle_3DNA, pseudo_rot_angle_CURVESP ,read_hbond_filter_file, seq_to_binary, seq2YR, shift, rotate, dihed_label, pucker_label, sugar_atom_label, atom_labels, np2df, comp, ideal_atoms
from classes import forward_data, Opt, mdDNA_object, Out_stats
from Ideal_coords_2 import base_atoms, phos_atoms
from tensorflow import keras
from dihedral import dihedral
from scipy import stats
from joblib import Parallel, delayed
from plot_results import plot_stats
from scipy.io import loadmat, savemat
import mat73


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

def frame_to_atomistic_coords(d):
        for i in d.data.triw.keys():
                Rc, rc, Rw, rw, Rpw, rpw, Rpc, rpc, seq = d.CGFr.Rc[i], d.CGFr.rc[i], d.CGFr.Rw[i], d.CGFr.rw[i], d.CGFr.Rpw[i], d.CGFr.rpw[i], d.CGFr.Rpc[i], d.CGFr.rpc[i], d.seqs.seqwt[i]
                seqRY = seq2YR(seq)
                Bw = fit_atomistic_coords(Rw , rw , seq)
                Bc = fit_atomistic_coords(Rc , rc , comp(seq))
                Pw = fit_atomistic_coords(Rpw, rpw, 'PPP')
                Pc = fit_atomistic_coords(Rpc, rpc, 'PPP')
        #       vec = atomistic_coords_to_vec([Bw,Bc,Pw,Pc])
                binary = seq_to_binary(seqRY)
                vec = atomistic_coords_to_vec([binary,Bw,Pw])
                dim_label = ['S1','S2','S3'] + atom_labels(base_atoms[seqRY[0]],'1') + atom_labels(base_atoms[seqRY[1]],'2') + atom_labels(base_atoms[seqRY[2]],'3') + atom_labels(phos_atoms,'1') + atom_labels(phos_atoms,'2')
                vec = vec[:,np.newaxis]
                vec=vec.T
                if d.source == 'model':
                        d.MDAt.atmcrd_w[i] = np2df(vec,dim_label)
                elif d.source == 'MD':
                        d.CGAt.atmcrd_w[i] = np2df(vec,dim_label)
        return d

def compute_and_orient_tframes(d):
        for i in d.data.triw.keys():
                _, _, Rc, rc, Rw, rw, Rpw , rpw , Rpc, rpc = tools.frames(d.data.triw[i])
                R, r = Rw[1].T, rw[1]
                d.CGFr.Rc[i]  = rotate(Rc ,R)
                d.CGFr.Rw[i]  = rotate(Rw ,R)
                d.CGFr.rc[i]  = shift(rc ,R,r)
                d.CGFr.rw[i]  = shift(rw ,R,r)
                d.CGFr.Rpc[i] = rotate(Rpc ,R)
                d.CGFr.Rpw[i] = rotate(Rpw ,R)
                d.CGFr.rpw[i] = shift(rpw ,R,r)
                d.CGFr.rpc[i] = shift(rpc ,R,r)
        return d

def decompose_to_trimer(d):
    for i in range(d.nbp-2):  # for Nmer, N-2 trimer
        d.data.triw[i+1]  = d.data.w[24*i:24*(i+3)-18]
        d.data.tric[i+1]  = d.data.c[24*i:24*(i+3)-18]
        d.seqs.seqwt[i+1] = d.seqs.seq[i:i+3]
        d.seqs.seqct[i+1] = d.seqs.seq_comp[i:i+3]
        if d.source == 'model':
            d.CGFr.R[i+1], d.CGFr.r[i+1], d.CGFr.Rc[i+1], d.CGFr.rc[i+1], d.CGFr.Rw[i+1], d.CGFr.rw[i+1], d.CGFr.Rpw[i+1], d.CGFr.rpw[i+1], d.CGFr.Rpc[i+1], d.CGFr.rpc[i+1] = tools.frames(d.data.triw[i+1])
        elif d.source == 'MD':
            d.MDFr.R[i+1], d.MDFr.r[i+1], d.MDFr.Rc[i+1], d.MDFr.rc[i+1], d.MDFr.Rw[i+1], d.MDFr.rw[i+1], d.MDFr.Rpw[i+1], d.MDFr.rpw[i+1], d.MDFr.Rpc[i+1], d.MDFr.rpc[i+1] = tools.frames(d.data.triw[i+1])
    return d

def create_sample(u,index):
        sample = forward_data()
        sample.seqs.seq = u.seq
        sample.seqs.seq_comp = comp(sample.seqs.seq)
        sample.nbp = len(sample.seqs.seq)
        if u.source == 'model':
                sample.source = u.source 
                sample.data.w = u.MC_samples[index].to_numpy()
                sample.CGFr.R, sample.CGFr.r, sample.CGFr.Rc, sample.CGFr.rc, sample.CGFr.Rw, sample.CGFr.rw, sample.CGFr.Rpw , sample.CGFr.rpw , sample.CGFr.Rpc, sample.CGFr.rpc = tools.frames(sample.data.w)
        elif u.source == 'MD':
                sample.source = u.source
                sample.data.w = u.MC_samples[index].to_numpy()	
                tmp_puck = u.pucker[index]
                tmp_dihed = u.dihedrals[index]
                tmp_puck_list  = {}
                tmp_dihed_list = {}
                for n in range(sample.nbp):
                        tmp_label = [z+str(n+1) for z in pucker_label]
                        tmp_puck_list[n] = np.array(tmp_puck.loc[tmp_label])  
                        if n > 0 and n < sample.nbp -1:
                                tmp_label_d = [z+str(n+1) for z in dihed_label]
                                tmp_dihed_list[n] = np.array(tmp_dihed.loc[tmp_label_d])
             
                sample.MDAt.p_angles_w = pd.DataFrame.from_dict(tmp_puck_list,orient='index',columns=pucker_label) 		
                sample.MDAt.b_angles_w = pd.DataFrame.from_dict(tmp_dihed_list,orient='index',columns=dihed_label)
                sample.MDFr.R, sample.MDFr.r, sample.MDFr.Rc, sample.MDFr.rc, sample.MDFr.Rw, sample.MDFr.rw, sample.MDFr.Rpw , sample.MDFr.rpw , sample.MDFr.Rpc, sample.MDFr.rpc = tools.frames(sample.data.w)
        sample.data.c = np.matmul(Etrans(sample.nbp),sample.data.w)
        return sample

def load_models():
	model_path = {}

	model_path['RRR'] = 'models/RRR_model_hyper.15_val_lot.4.h5' 
	model_path['RRY'] = 'models/RRY_model_hyper.15_val_lot.4.h5'
	model_path['RYR'] = 'models/RYR_model_hyper.16_val_lot.4.h5'
	model_path['YRR'] = 'models/YRR_model_hyper.15_val_lot.4.h5'
	model_path['YYR'] = 'models/YYR_model_hyper.15_val_lot.4.h5'
	model_path['YRY'] = 'models/YRY_model_hyper.441_val_lot.4.h5'
	model_path['RYY'] = 'models/RYY_model_hyper.165_val_lot.4.h5'
	model_path['YYY'] = 'models/YYY_model_hyper.46_val_lot.4.h5'

	model = {}
	for t in model_path.keys():
		model[t] = keras.models.load_model(model_path[t])
	return model

def predict_missing_atoms(sample,model):
	tmpw = {}
	if sample.source == 'MD':
		for i in sample.CGAt.atmcrd_w.keys():
			model_tmp = model[seq2YR(sample.seqs.seqwt[i])]
			tmpw[i] = model_tmp.predict(sample.CGAt.atmcrd_w[i].drop(['S1','S2','S3'],axis=1))[0]
		sample.CGAt.s_atoms_w = pd.DataFrame.from_dict(tmpw,orient='index',columns=sugar_atom_label) 
	elif sample.source == 'model':
		for i in sample.MDAt.atmcrd_w.keys():
			model_tmp = model[seq2YR(sample.seqs.seqwt[i])]
			tmpw[i] = model_tmp.predict(sample.MDAt.atmcrd_w[i].drop(['S1','S2','S3'],axis=1))[0]
		sample.MDAt.s_atoms_w = pd.DataFrame.from_dict(tmpw,orient='index',columns=sugar_atom_label)
	return sample

def sugar_pucker_angles(data):
    lc1 = atom_labels(['C1p'])
    lc2 = atom_labels(['C2p'])  
    lc3 = atom_labels(['C3p'])  
    lc4 = atom_labels(['C4p'])  
    lo4 = atom_labels(['O4p'])  

    c1,c2,c3,c4,o4 = data[lc1],data[lc2],data[lc3],data[lc4],data[lo4]
    angles    = np.zeros(6)
    angles[0] = dihedral(c4,o4,c1,c2)
    angles[1] = dihedral(o4,c1,c2,c3)
    angles[2] = dihedral(c1,c2,c3,c4)
    angles[3] = dihedral(c2,c3,c4,o4)
    angles[4] = dihedral(c3,c4,o4,c1)
    tmp = pd.DataFrame(angles[:,np.newaxis].T,columns=pucker_label)
    angles[5] = pseudo_rot_angle_CURVESP(tmp.loc[0]) 
    return angles

def backbone_angles(data1,data2):
	data2 = data2.loc[0]

	lO3p1 = atom_labels(['O3p'],'1')
	lP1   = atom_labels(['P']  ,'1') 
	lO5p1 = atom_labels(['O5p'],'1')
	lC5p  = atom_labels(['C5p'])
	lC4p  = atom_labels(['C4p'])
	lC3p  = atom_labels(['C3p'])
	lO4p  = atom_labels(['O4p'])
	lC1p  = atom_labels(['C1p'])
	lO3p2 = atom_labels(['O3p'],'2')
	lO5p2 = atom_labels(['O5p'],'2')
	lP2   = atom_labels(['P']  ,'2')

	if data2['S2'] == 1:   ### purine
		lN = atom_labels(['N9'],'2')
		lC = atom_labels(['C4'],'2')
	else:
		lN = atom_labels(['N9'],'2')   ### need to correct it
		lC = atom_labels(['C4'],'2')   ### need to correct it ... issue is same labels are used for purine and py // not required this convention any more. 

	O3p1 = data2[lO3p1].T
	P1   = data2[lP1].T
	O5p1 = data2[lO5p1].T
	C5p  = data1[lC5p]
	C4p  = data1[lC4p]
	C3p  = data1[lC3p]
	O4p  = data1[lO4p]
	C1p  = data1[lC1p]
	O3p2 = data2[lO3p2].T
	O5p2 = data2[lO5p2].T
	P2   = data2[lP2].T
	N    = data2[lN].T
	C    = data2[lC].T

	angles    = np.zeros(7)
	angles[0] = dihedral(O3p1,   P1, O5p1,  C5p)
	angles[1] = dihedral(  P1, O5p1,  C5p,  C4p)
	angles[2] = dihedral(O5p1,  C5p,  C4p,  C3p) 	
	angles[3] = dihedral( C5p,  C4p,  C3p, O3p2)
	angles[4] = dihedral( C4p,  C3p, O3p2,   P2)
	angles[5] = dihedral( C3p, O3p2,   P2, O5p2)
	angles[6] = dihedral( O4p,  C1p,    N,    C)

	return angles

def compute_pucker_angles(sample):
	tmp = {}
	if sample.source == 'MD':
		for i in sample.CGAt.s_atoms_w.index.to_list():
			tmp[i] = sugar_pucker_angles(sample.CGAt.s_atoms_w.loc[i])
		sample.CGAt.p_angles_w = pd.DataFrame.from_dict(tmp,orient='index',columns=pucker_label)
	elif sample.source == 'model':
		for i in sample.MDAt.s_atoms_w.index.to_list():
			tmp[i] = sugar_pucker_angles(sample.MDAt.s_atoms_w.loc[i])
		sample.MDAt.p_angles_w = pd.DataFrame.from_dict(tmp,orient='index',columns=pucker_label)
	return sample


def B1_or_B2(sample):
	# epsilon - zeta < 0 for B1 
	try:
		sample.MDAt.b_angles_w['B2form'] = (sample.MDAt.b_angles_w['e'] - sample.MDAt.b_angles_w['z']) > 0
	except:
		None
	try:
		sample.CGAt.b_angles_w['B2form'] = (sample.CGAt.b_angles_w['e'] - sample.CGAt.b_angles_w['z']) > 0
	except:
		None
	return sample

def compute_backbone_angles(sample):
        tmp = {}
        if sample.source == 'MD':
                for i in sample.CGAt.s_atoms_w.index.to_list():
                        tmp[i] = backbone_angles(sample.CGAt.s_atoms_w.loc[i],sample.CGAt.atmcrd_w[i])
                sample.CGAt.b_angles_w = pd.DataFrame.from_dict(tmp,orient='index',columns=dihed_label)
                sample = B1_or_B2(sample)	
        elif sample.source == 'model':
                for i in sample.MDAt.s_atoms_w.index.to_list():
                        tmp[i] = backbone_angles(sample.MDAt.s_atoms_w.loc[i],sample.MDAt.atmcrd_w[i])
                sample.MDAt.b_angles_w = pd.DataFrame.from_dict(tmp,orient='index',columns=dihed_label)
                sample = B1_or_B2(sample)
        return sample 

def read_sample(model,sample,index):
	print("reading sample index --> ", index)
	sample = decompose_to_trimer(sample)
	sample = compute_and_orient_tframes(sample)
	sample = frame_to_atomistic_coords(sample)
	sample = predict_missing_atoms(sample,model)
#	sample = compute_pucker_angles(sample)
#	sample = compute_backbone_angles(sample)
	return sample

def read_all_data(MD_data,model):
	index = np.arange(MD_data.MC_samples.shape[1])
	if MD_data.source == 'model':
		MD_data.MC_samples = pd.DataFrame(MD_data.MC_samples,columns = index)
	samples = []
	for s in index:
		samples.append(create_sample(MD_data,s))
	data =  Parallel(n_jobs=1)([delayed(read_sample)(model,samples[s],s) for s in index])
#	data =  Parallel(n_jobs=1,prefer='threads',verbose=0)([delayed(read_sample)(model,samples[s],s) for s in index])
	return data

def compare_pucker_stats(data):
	out_stats = Out_stats("pucker angle statistics, MD - CG")
	columns = pucker_label
	for enum, D in enumerate(data):
		if enum == 0:
			out_stats.seq = D.seqs.seq
		try:
			index = D.CGAt.p_angles_w.index.to_list()
		except:
			index = D.MDAt.p_angles_w.index.to_list()
		for k in index:
			if enum == 0:
				out_stats.MD[k] = pd.DataFrame(columns=columns)
				out_stats.CG[k] = pd.DataFrame(columns=columns)
			try:
				tmp1 = D.MDAt.p_angles_w.loc[k]
				out_stats.MD[k] = out_stats.MD[k].append(dict(zip(columns, tmp1)),ignore_index=True).round(2)
			except:
				None
			try:
				tmp2 = D.CGAt.p_angles_w.loc[k]
				out_stats.CG[k] = out_stats.CG[k].append(dict(zip(columns, tmp2)),ignore_index=True).round(2)
			except:
				None
	try:
		pc_tot = {}
		for k in index:
			out_stats.difference[k] = out_stats.MD[k] - out_stats.CG[k]
			out_stats.difference[k][out_stats.difference[k] >  180] = out_stats.difference[k] - 360
			out_stats.difference[k][out_stats.difference[k] < -180] = out_stats.difference[k] + 360
			pc = np.zeros(len(columns))
			for enum,c in enumerate(columns):
				try:
					pc[enum] = stats.pearsonr(out_stats.MD[k][c].to_numpy(),out_stats.CG[k][c].to_numpy())[0]
				except:
					pc[enum] = np.nan  # 6,3
			pc_tot[k] = pc
		out_stats.pc = pd.DataFrame.from_dict(pc_tot,orient='index',columns=columns).round(2) 
	except:
		None

	return out_stats


def compare_backbone_stats(data):
        out_stats = Out_stats("backbone angle statistics, MD - CG")
        columns = dihed_label
        for enum, D in enumerate(data):
                if enum == 0:
                        out_stats.seq = D.seqs.seq
                try:
                        index = D.CGAt.b_angles_w.index.to_list()
                except:
                        index = D.MDAt.b_angles_w.index.to_list()
                for k in index:
                        if enum == 0:
                                out_stats.MD[k] = pd.DataFrame(columns=columns)
                                out_stats.CG[k] = pd.DataFrame(columns=columns)
                        try:
                                tmp1 = D.MDAt.b_angles_w.loc[k]
                                out_stats.MD[k] = out_stats.MD[k].append(dict(zip(columns, tmp1)),ignore_index=True).round(2)
                        except:   
                                None
                        try:   
                                tmp2 = D.CGAt.b_angles_w.loc[k]
                                out_stats.CG[k] = out_stats.CG[k].append(dict(zip(columns, tmp2)),ignore_index=True).round(2)
                        except:
                                None

        pc_tot = {}
        try:
                for k in index:
                        out_stats.difference[k] = out_stats.MD[k] - out_stats.CG[k]
                ## The issue is the discontinuity at 180/-180, e.g., that -179 and +179 are 2 degrees but -179-179 = 358
                ## In principle the maximum angle = +/-180, Thus, add or subtract 2*pi if angles is > |pi|
                        out_stats.difference[k][out_stats.difference[k] >  180] = out_stats.difference[k] - 360
                        out_stats.difference[k][out_stats.difference[k] < -180] = out_stats.difference[k] + 360
                        pc = np.zeros(len(columns))
                        for enum,c in enumerate(columns):
                                try:
                                        pc[enum] = stats.pearsonr(out_stats.MD[k][c].to_numpy(),out_stats.CG[k][c].to_numpy())[0]
                                except:
                                        pc[enum] = np.nan  # 6,3
                        pc_tot[k] = pc
                out_stats.pc = pd.DataFrame.from_dict(pc_tot,orient='index',columns=columns).round(2)
        except:
                None
        return out_stats

def mini_circle_to_linear(w, seq):
    #converts n bp minicircle to n+2 bp linear fragment
    # 24(n+2)-18 (linear) otherwise 24n for circle
    s = len(w)
    wl = np.zeros(s+30)
    wl[0:18] = w[-18:]
    wl[18:s+18] = w
    wl[s+18:s+30] = w[0:12]
    seql = seq[-1]+seq + seq[0]
    return wl, seql


def write_pdb(w,sugar):
    R, r, Rc, rc, Rw, rw, Rpw , rpw , Rpc, rpc = tools.frames(a.ground_state)  ## these are the final frames
     
    return None


######## working model for MC 
#a = cgDNA('GGGGGGGGGGGGGGGGGG')  # 18mer
#a.source = 'model'
#a.MC_samples, _ , _ = a.MonteCarlo(10)
#sample = read_sample(a,0,model)

######## working model for MD
#list_length = 1
#model = load_models()
#opt = Opt()
#opt.path='/work/lcvmm/'
#opt.NA='palin.bscl.tip3p.jc'
#opt.seq_id = sys.argv[1]
#opt.run_nbr_list = np.arange(1,1+list_length)
#opt.remove_ends = 3
#opt.sparse=500


model = load_models()
a = cgDNA('GTACGTACGTACGATCGACGTCATGG') ## creating dummy object
a.source = 'model'

#a.MC_samples, _ , _ = a.MonteCarlo(3) 

data = mat73.loadmat('z_minicircle_data_for_MD_Rahul/seq_data1.mat')
a.seq =  data['dataf']['seq']
a.ground_state, a.seq  = mini_circle_to_linear(data['dataf']['minicircleshape'], a.seq)
a.nbp = len(a.seq)
gsc = np.matmul(Etrans(a.nbp),a.ground_state)
a.MC_samples = np.vstack((a.ground_state,gsc)).T ## temporary
D_cg = read_all_data(a,model)
sw = D_cg[0]  #watson strand
sc = D_cg[1]
sw.MDAt.s_atoms_w.index =  sw.MDAt.s_atoms_w.index+1
sw.MDAt.s_atoms_w.loc[1] = np.zeros(18)
sw.MDAt.s_atoms_w.loc[a.nbp] = np.zeros(18)
sc.MDAt.s_atoms_w.index =  sc.MDAt.s_atoms_w.index+1
sc.MDAt.s_atoms_w.loc[1] = np.zeros(18)
sc.MDAt.s_atoms_w.loc[a.nbp] = np.zeros(18)
sw.MDAt.s_atoms_w = sw.MDAt.s_atoms_w.sort_index()
sc.MDAt.s_atoms_w = sc.MDAt.s_atoms_w.sort_index()

mdic = {
        "nbp":a.nbp, 
        "sequence":a.seq,
        "groundstate":a.ground_state[:,np.newaxis],
        "groundstatec":gsc[:,np.newaxis],
        "seqlabel":"seq1",
        "C5pw":sw.MDAt.s_atoms_w[atom_labels(['C5p'],'')].to_numpy(),
        "C4pw":sw.MDAt.s_atoms_w[atom_labels(['C4p'],'')].to_numpy(),
        "O4pw":sw.MDAt.s_atoms_w[atom_labels(['O4p'],'')].to_numpy(),
        "C1pw":sw.MDAt.s_atoms_w[atom_labels(['C1p'],'')].to_numpy(),
        "C3pw":sw.MDAt.s_atoms_w[atom_labels(['C3p'],'')].to_numpy(),
        "C2pw":sw.MDAt.s_atoms_w[atom_labels(['C2p'],'')].to_numpy(),
        "C5pc":sc.MDAt.s_atoms_w[atom_labels(['C5p'],'')].to_numpy(),
        "C4pc":sc.MDAt.s_atoms_w[atom_labels(['C4p'],'')].to_numpy(),
        "O4pc":sc.MDAt.s_atoms_w[atom_labels(['O4p'],'')].to_numpy(),
        "C1pc":sc.MDAt.s_atoms_w[atom_labels(['C1p'],'')].to_numpy(),
        "C3pc":sc.MDAt.s_atoms_w[atom_labels(['C3p'],'')].to_numpy(),
        "C2pc":sc.MDAt.s_atoms_w[atom_labels(['C2p'],'')].to_numpy()
        }

savemat("rand_mini_circile_1.mat", mdic)
