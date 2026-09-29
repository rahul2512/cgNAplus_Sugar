import sys, numpy as np, pandas as pd
from utilities import bond_length, sugar_atom_label, atom_labels, np2df
from dihedral import dihedral
from classes import stat

def create_setup_XXX(data):
        model     = data.model
        yp_val   = np2df(model.predict(data.X_val),sugar_atom_label)
        yp_train = np2df(model.predict(data.X_Train),sugar_atom_label)
        try:
                yp_test = np2df(model.predict(data.X_Test),sugar_atom_label)
        except:
                yp_test = None
        return yp_val, yp_train, yp_test


###############################
## SUGAR ATOMS
###############################


def sugar_pucker_angles(data):
    c1n = atom_labels(['C1p'])
    c2n = atom_labels(['C2p'])
    c3n = atom_labels(['C3p'])
    c4n = atom_labels(['C4p'])
    o4n = atom_labels(['O4p'])

    c1,c2,c3,c4,o4 = data[c1n],data[c2n],data[c3n],data[c4n],data[o4n]
    angles     = np.zeros(5)
    angles[0]  = dihedral(c4,o4,c1,c2)
    angles[1]  = dihedral(o4,c1,c2,c3)
    angles[2]  = dihedral(c1,c2,c3,c4)
    angles[3]  = dihedral(c2,c3,c4,o4)
    angles[4]  = dihedral(c3,c4,o4,c1)

    return angles


def compute_pucker(df):
	ind = df.index.to_list()
	s = len(ind)
	data = np.zeros((s,5))
	for enum,i in enumerate(ind):
		data[enum] = sugar_pucker_angles(df.loc[i])
	return data


def compare_pucker_angles(data,debug):
        st = stat()
        yp_val, yp_train, yp_test = create_setup_XXX(data)
        try:
                col = data.pa_val.columns.to_list()
        except:
                col = data.pa_Test.columns.to_list()
        col = col[0:5]
        if debug == True:
                print("Debugging on ---- ON")
                pa_val   = np2df(compute_pucker(data.Y_val),col)
                pa_train = np2df(compute_pucker(data.Y_Train),col)
                try:
                        pa_test  = np2df(compute_pucker(data.Y_Test),col)
                except:
                        pa_test = None
        else:
                pa_val   = np2df(compute_pucker(yp_val),col)
                pa_train = np2df(compute_pucker(yp_train),col)
                try:
                        pa_test  = np2df(compute_pucker(yp_test),col)
                except:
                        pa_test = None

        st.stat_val   = data.pa_val   - pa_val
        st.stat_Train = data.pa_Train - pa_train
        try:
                st.stat_Test = data.pa_Test - pa_test
        except:
                None
        return st



#####################################
# Backbone angles
#####################################
#1@O3' :2@P :2@O5' :2@C5' type alpha -- F
#2@P :2@O5' :2@C5' :2@C4' type beta
#2@O5' :2@C5' :2@C4' :2@C3' type gamma

#2@C5' :2@C4' :2@C3' :2@O3' type delta
#2@C4' :2@C3' :2@O3' :3@P type epsilon
#2@C3' :2@O3' :3@P :3@O5' type zeta
#2@O4' :2@C1' :2@N1 :2@C2 type chin

def check_if_neighbours(a1,a2,a3,a4,label):
	b12 = bond_length(a1.to_numpy(),a2.to_numpy())
	b23 = bond_length(a2.to_numpy(),a3.to_numpy())
	b34 = bond_length(a3.to_numpy(),a4.to_numpy())
	print("bond-lengths are for ---",label,"---" ,b12, b23, b34)
	if b12 > 1.8 or b23 > 1.8 or b34 > 1.8:
		print("Specified dihedrals not nieghbours------------",label)
		sys.exit()
	return None

def backbone_angles(dfX,dfY,index):
    a1O3  = dfX[atom_labels(['1O3p'])]
    a2P   = dfX[atom_labels(['1P' ])]
    a2O5  = dfX[atom_labels(['1O5p'])]
    C5p  = dfY[atom_labels(['C5p'])]
    C4p  = dfY[atom_labels(['C4p'])]
    C3p  = dfY[atom_labels(['C3p'])]
    a2O3 = dfX[atom_labels(['2O3p'])]
    a3P   = dfX[atom_labels(['2P' ])]
    a3O5  = dfX[atom_labels(['2O5p'])]
    O4p  = dfY[atom_labels(['O4p'])]
    C1p  = dfY[atom_labels(['C1p'])]
    a2N1  = dfX[atom_labels(['2N9'])]
    a2C2  = dfX[atom_labels(['2C4'])]  ### C2/N1 for py but already taken are of in Ideal_coords

    if index in [0]:#,1,2,3,4,5,6]:
        #['a', 'b', 'g', 'd', 'e', 'z', 'x']
        check_if_neighbours(a1O3,a2P ,a2O5,C5p ,'a')
        check_if_neighbours( a2P,a2O5,C5p ,C4p ,'b')
        check_if_neighbours(a2O5,C5p ,C4p ,C3p ,'g')
        check_if_neighbours( C5p,C4p ,C3p ,a2O3,'d')
        check_if_neighbours( C4p,C3p ,a2O3,a3P ,'e')
        check_if_neighbours( C3p,a2O3,a3P,a3O5 ,'z')
        check_if_neighbours( O4p,C1p ,a2N1,a2C2,'x')

    angles     = np.zeros(7)
    angles[0]  = dihedral(a1O3,a2P ,a2O5,C5p)
    angles[1]  = dihedral( a2P,a2O5,C5p ,C4p)
    angles[2]  = dihedral(a2O5,C5p ,C4p ,C3p)
    angles[3]  = dihedral( C5p,C4p ,C3p ,a2O3)
    angles[4]  = dihedral( C4p,C3p ,a2O3,a3P)
    angles[5]  = dihedral( C3p,a2O3,a3P,a3O5)
    angles[6]  = dihedral( O4p,C1p ,a2N1,a2C2)

    return angles

def compute_backbone_angles(dfX,dfY):
        ind = dfX.index.to_list()
        s = len(ind)
        data = np.zeros((s,7))
        for enum,i in enumerate(ind):
                data[enum] = backbone_angles(dfX.loc[i],dfY.loc[i],enum)
        return data

def compare_backbone_angles(data,debug):
	st = stat()
	yp_val, yp_train, yp_test = create_setup_XXX(data)
	try:
		col = data.ba_val.columns.to_list()
	except:
		col = data.ba_Test.columns.to_list()

	if debug == True:
		print("Debugging on ---- ON")
		ba_val   = np2df(compute_backbone_angles(data.X_val  ,data.Y_val),col)
		ba_train = np2df(compute_backbone_angles(data.X_Train,data.Y_Train),col)
		try:
			ba_test  = np2df(compute_backbone_angles(data.X_Test ,data.Y_Test),col)
		except:
			ba_test = None
	else:   
		ba_val   = np2df(compute_backbone_angles(data.X_val  ,yp_val),col)
		ba_train = np2df(compute_backbone_angles(data.X_Train,yp_train),col)
		try:
			ba_test  = np2df(compute_backbone_angles(data.X_Test ,yp_test),col)
		except:
			ba_test = None

	st.stat_val = data.ba_val - ba_val
	st.stat_Train = data.ba_Train - ba_train
	try:
		st.stat_Test = data.ba_Test - ba_test
	except:
		None
	return st



####################################
############# ATOMS
####################################

def compare_atomic_pos(data):
	st = stat()
	yp_val, yp_train, yp_test = create_setup_XXX(data)
	col = sugar_atom_label
	st.stat_val = data.Y_val - yp_val
	st.stat_val = data.Y_Train - yp_train
	try:
		st.stat_val = data.Y_Test - yp_test
	except:
		None
	return st
	
def compute_3D_atomic_pos(y1,y2):
	try:
		y1,y2 = y1.to_numpy(), y2.to_numpy()
	except:
		None
	b = np.zeros(6)
	for i in range(6):
		b[i] =  bond_length(y1[3*i:3*(i+1)],y2[3*i:3*(i+1)])
	return b


def compare_3D_atomic_pos(data):
        st = stat()
        yp_val, yp_train, yp_test = create_setup_XXX(data)
        col = sugar_atom_label[::3]
        st.stat_val = np.zeros((yp_val.shape[0],6))
        st.stat_Train = np.zeros((yp_train.shape[0],6))
        for i in data.Y_val.index.to_list():
                st.stat_val[i] = compute_3D_atomic_pos(data.Y_val.loc[i], yp_val.loc[i])
        for i in  data.Y_Train.index.to_list():
                st.stat_Train[i] = compute_3D_atomic_pos(data.Y_Train.loc[i], yp_train.loc[i])
        st.stat_Train = np2df(st.stat_Train,col)
        st.stat_val   = np2df(st.stat_val,col)
        try:
                        st.stat_Test = np.zeros((yp_test.shape[0],6))
                        for i in  data.Y_Test.index.to_list():
                                st.stat_Test[i] = compute_3D_atomic_pos(data.Y_Test.loc[i], yp_test.loc[i])
                        st.stat_Test = np2df(st.stat_Test,col)
        except:
                None

        return st


