import numpy as np, pandas as pd

from utilities import sugar_atom_label, np2df, atom_labels

def compute_bond_length(atom1,atom2):
	try:
		atom1 = atom1.to_numpy()
		atom2 = atom2.to_numpy()
	except:
		None
	bl = np.linalg.norm(atom1-atom2,axis=1)
	return bl



def compute_bl_pair_sugar(data):
	# bonding in sugar ring
	# O4 -- C1' -- C2' -- C3' -- C4' -- O4'
	#			     |
	#			     C5'		

	df = np2df(data,sugar_atom_label)
	C1 = df[atom_labels(['C1'])]		
	C2 = df[atom_labels(['C2'])]
	C3 = df[atom_labels(['C3'])]
	C4 = df[atom_labels(['C4'])]
	C5 = df[atom_labels(['C5'])]
	O4 = df[atom_labels(['O4'])]
	s = np.shape(C1)
	result = np.zeros((s[0],6))
	result[:,0] = compute_bond_length(C1,C2)
	result[:,1] = compute_bond_length(C2,C3)
	result[:,2] = compute_bond_length(C3,C4)
	result[:,3] = compute_bond_length(C4,O4)
	result[:,4] = compute_bond_length(O4,C1)
	result[:,5] = compute_bond_length(C4,C5)

	index = ['C1-C2','C2-C3','C3-C4','C4-O4','O4-C1','C4-C5']
	result = np2df(result,index)

	return result
	

def bond_constrains_sugar(df):
	df = compute_bl_pair_sugar(df)
	df = df - np.array([1.526, 1.526, 1.526, 1.410, 1.410, 1.526])
	return df






