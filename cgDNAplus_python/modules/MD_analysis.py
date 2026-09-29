import sys, random, re
import itertools, string
#print(sys.path)
#sys.path.append('/Users/rsharma/Dropbox/cgDNAplus_py_rahul/classes')
#print(sys.path)
from cgDNAclass import cgDNA
from Init_MD import init_MD_data
import numpy as np
import pandas as pd
import RotationUtils as Rot
import scipy, scipy.io
from scipy.linalg import sqrtm
from scipy.linalg import norm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import gridspec
import os, sys, time, copy
import scipy.io as sio
path = os.getcwd()
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from E_transform import Etrans
from cgDNAUtils import *
import matplotlib.colors as mcolors
import seaborn as sns#; sns.set()
import tqdm
import time


#############################################################################    
#############################################################################    
DNA_sym = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,False,False,True,True,True,False]
PDNA_sym = DNA_sym[0:16]
RNA_sym = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,False,False,True,True,True]
HYB_sym = [False]*24
MDNA_sym=[True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,True,False,False,False]
MDNA_map = [0,1,2,3,4,5,8,9,12,13,14,16,6,7,10,11,15,17,18,19,20,21]
dimer_16 = ['TA','CG','CA','TG','AA','AG','GA','GG', 'TT','CT','TC','CC', 'GC','AT','AC','GT']
cgDNA_name = ["Buckle","Propeller","Opening","Shear","Stretch","Stagger",
              "WRot1","WRot2","WRot3","WTra1","WTra2","WTra3",
              "Tilt","Roll","Twist","Shift","Slide","Rise",
              "CRot1","CRot2","CRot3","CTra1","CTra2","CTra3",
              "Buckle","Propeller","Opening","Shear","Stretch","Stagger"]
                             
dimer_16_RY = ['YR','RR','YY','RY']
dimer_17 = ['TA','CG','CA','TG','AA','AG','GA','GG', 'TT','CT','TC','CC', 'GC','AT','AC','GT','Avg']
dimer_10 = ['TA','CG','CA','AA','AG','GA','GG','GC','AT','AC']
mon = ['A','T','C','G']
color_16 = ['darkgreen', 'orange', 'slateblue', 'sienna', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','darkgoldenrod','skyblue','navy','lime','tan','gold']
color_16_RY = ['darkgreen', 'darkgreen', 'darkgreen', 'darkgreen', 'magenta', 'magenta', 'magenta', 'magenta', 'gray', 'gray','gray','gray','blue','blue','blue','blue','k']

#############################################################################    
#############################################################################    
# Utilitiy codes
#############################################################################    
#############################################################################    
# this code set ideal limits for x and y
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 0.5
plt.rcParams['axes.facecolor'] = 'white'

def all_Nmers(N):
    return ["".join(item) for item in itertools.product("ATCG", repeat=N)]

def all_YR(N):
    return ["".join(item) for item in itertools.product("RY", repeat=N)]

def Met_all_Nmers(N):
    list_all = ["".join(item) for item in itertools.product(['A','T','C','G','MN','MG','CN'], repeat=N)]
    final_list = [] 
    for seq in list_all:
        if 'M' in seq:
            final_list.append(seq)
    return final_list

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def to_YR(seq):
    seq_YR = ''
    for s in seq:
        if s in "AG":
            seq_YR += "R"
        elif s in "CT":
            seq_YR += "Y"
        else:
            print("Unrecongnized base type --->", s)  
            sys.exit()
    return seq_YR 
                 
def set_lim(x,y,ax,frac=0.09):
    xlim_min = min(x)
    xlim_max = max(x)
    ylim_min = min(y)
    ylim_max = max(y)
    xr = xlim_max - xlim_min
    yr = ylim_max - ylim_min
    xf = (xlim_min - 0.05*xr, xlim_max + 0.05*xr)
    yf = (ylim_min - 0.05*yr, ylim_max + 0.05*yr)
    ax.set_xlim(xf)
    ax.set_ylim(yf)
    lim_r = (xlim_min - (frac+0.01)*xr,ylim_max + frac*yr)
    return lim_r

# return randon sequence of length N
def random_seq(length):
    Base = ['A','T','C','G']
    seq  = ''
    for i in range(length):
        seq = seq + random.choice(Base)
    return seq

def stencil_42(nbp,typ='cg+'):
    x1 = np.zeros(nbp-1,dtype=int)
    x2 = np.zeros(nbp-1,dtype=int)
    if typ=='cg+':
        for i in range(nbp-2):
            x1[i+1] = 18 + 24*i
            x2[i] = 36 + 24*(i)
        x2[nbp-2] = 24*nbp -18
    ############################
    if typ=='cg':
        for i in range(nbp-2):
            x1[i+1] = 12 + 12*i 
            x2[i]   = 18 + 12*(i)
        x2[nbp-2] = 12*nbp -6
    ############################
    if typ=='inter':
        for i in range(nbp-2):
            x1[i+1] = 6 + 6*i 
            x2[i]   = 6 + 6*(i)
        x2[nbp-2] = 6*nbp -6


    return x1,x2


def replacenth(string, sub, wanted, where):
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    newString = before + after
    return newString

def dimer_percent(seq,dim):
    loc = [m.start() for m in re.finditer(dim, seq)] 
    return len(loc)

def mdna_random_seq(how_many):
    for i in range(how_many):
        seq = random_seq(216) 
        loc = [m.start() for m in re.finditer('CG', seq)] 
        print(seq,dimer_percent(seq,'CG'),dimer_percent(seq,'MN'),dimer_percent(seq,'MG') )
        for u in range(1,len(loc),1):
            for q in range(4):
                tmp_seq = seq
                for l in random.choices(loc,k=u):
                    hell = random.choice(['MN','MG'])
                    tmp_seq = replacenth(tmp_seq, 'CG', hell, l)
                print(tmp_seq,dimer_percent(tmp_seq,'CG'),dimer_percent(tmp_seq,'MN'),dimer_percent(tmp_seq,'MG') )
        for u in range(1,len(loc)):
            for s in ['MN','MG']:
                tmp_seq = seq
                for l in random.choices(loc,k=u):
                    tmp_seq = replacenth(tmp_seq, 'CG', s, l)
                print(tmp_seq,dimer_percent(tmp_seq,'CG'),dimer_percent(tmp_seq,'MN'),dimer_percent(tmp_seq,'MG'), "---")


###----------------------------------------------------------------------------------------
## Mahalanobis distance --------------------------------------
###----------------------------------------------------------------------------------------
def Mahal(mu1,mu2,A):
    try:
        A = A.to_numpy(dtype='float')
        mu1, mu2 = mu1.to_numpy(dtype='float')[:,np.newaxis],mu2.to_numpy(dtype='float')[:,np.newaxis]
    except:
        None
    ### Note A must be stiffness matrix
    dis = scipy.spatial.distance.mahalanobis(mu1,mu2,A)/len(mu1)
    return dis
###----------------------------------------------------------------------------------------
## Symmertic Mahalanobis distance --------------------------------------
###----------------------------------------------------------------------------------------
def Mahal_sym(mu1,mu2,A1,A2):
    try:
        A1 = A1.to_numpy(dtype='float')
        A2 = A2.to_numpy(dtype='float')
        mu1, mu2 = mu1.to_numpy(dtype='float')[:,np.newaxis],mu2.to_numpy(dtype='float')[:,np.newaxis]
    except:
        None
    # Note A1,A2 are stiffness matrix
    dis1 = Mahal(mu1,mu2,A1)
    dis2 = Mahal(mu1,mu2,A2)
    dis = (dis1+dis2)/2
    return dis

def kl_mvn(m0, K0, m1, K1):
    N = m0.shape[0]
    S0 = np.linalg.inv(K0)
    diff = m1 - m0
    # kl is made of three terms
    tr_term   = np.trace(np.matmul(S0, K1))
    det_term  = np.linalg.slogdet(K0)[1] - np.linalg.slogdet(K1)[1]
    quad_term = np.matmul(np.matmul(diff.T, K1), diff) 
    # per dof
    kl = .5 * (tr_term + det_term + quad_term - N)
    return kl/N 

def kl_mvn_sym(m0, K0, m1, K1):
    k1 = kl_mvn(m0, K0, m1, K1)
    k2 = kl_mvn(m1, K1, m0, K0)
    kl = 0.5*(k1+k2)
    return kl

def palin_err_mu_norm(data):
    wc = copy.deepcopy(data.shape[0])
    l = np.size(wc)
    nbp = copy.deepcopy(data.nbp[0])
    err = wc - np.matmul(Etrans(nbp),wc)
    err = np.linalg.norm(err)/l
    return err

def palin_err_K_norm(data):

    if hasattr(data, 's1b'):
        wc = copy.deepcopy(data.s1b[0])
        nbp = data.nbp[0]
    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp

    wc_inside = np.zeros((24*nbp-18,24*nbp-18))
    x1,x2 = stencil_42(nbp)
    for i,j in zip(x1,x2):
        wc_inside[i:j,i:j] = wc[i:j,i:j]

    err = wc_inside - np.matmul(np.matmul(Etrans(nbp),wc_inside),Etrans(nbp))
    l = 36*36*2 + 42*42*(nbp-3) - 18*18*(nbp-2)
    err = np.linalg.norm(err)/l
    return err

def palin_err_Mahal_sym(data):
    wc = copy.deepcopy(data.shape[0])
    Kc = copy.deepcopy(data.s1b[0])
    nbp = copy.deepcopy(data.nbp[0])
    E = Etrans(nbp)
    wk = np.matmul(E,wc)
    Kk = np.matmul(np.matmul(E,Kc),E)
    err = Mahal_sym(wk,wc,Kk,Kc)
    return err

def palin_err_KL_sym(data):
    wc = copy.deepcopy(data.shape[0])
    Kc = copy.deepcopy(data.s1b[0])
    nbp = copy.deepcopy(data.nbp[0])
    E = Etrans(nbp)
    wk = np.matmul(E,wc)
    Kk = np.matmul(np.matmul(E,Kc),E)
    err = kl_mvn_sym(wc, Kc, wk, Kk)
    return err

def recons_err_mu_norm(data,ps,sym):
    seq = copy.deepcopy(data.seq[0])
    seq = seq.replace('U','T') 
    if sym==False:
        wm = copy.deepcopy(data.shape[0])
    elif seq==comp(seq):
        wm = copy.deepcopy(data.shape_sym[0])
    else:
        wm = copy.deepcopy(data.shape[0])

    res = cgDNA(seq,ps)
    wr = res.ground_state
    l = np.size(wm)
    err = wm-wr
    err = np.linalg.norm(err)/l
    return err    
    
def recons_err_K_norm(data,ps,sym):

    if hasattr(data, 's1b'):
        seq = copy.deepcopy(data.seq[0])
        nbp = data.nbp[0]
        if sym==False:
            wc = copy.deepcopy(data.s1b[0])
        elif seq==comp(seq):
            wc = copy.deepcopy(data.s1b_sym[0])
        else:
            wc = copy.deepcopy(data.s1b[0])
        seq = seq.replace('U','T') 

    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp
        seq = copy.deepcopy(data.seq)
        seq = seq.replace('U','T') 

    wc_inside = np.zeros((24*nbp-18,24*nbp-18))
    x1,x2 = stencil_42(nbp)
    for i,j in zip(x1,x2):
        wc_inside[i:j,i:j] = wc[i:j,i:j]

    res = cgDNA(seq,ps)
    Kr = res.stiff.todense() 

    err = wc_inside - Kr
    l = 36*36*2 + 42*42*(nbp-3) - 18*18*(nbp-2)
    err = np.linalg.norm(err)/l
    return err

def recons_err_Mahal_sym(data,ps,sym):
    seq = copy.deepcopy(data.seq[0])
    seq = seq.replace('U','T') 
    if sym==False:
        wm = copy.deepcopy(data.shape[0])
        Km = copy.deepcopy(data.s1b[0])
    elif seq==comp(seq):
        wm = copy.deepcopy(data.shape_sym[0])
        Km = copy.deepcopy(data.s1b_sym[0])
    else:
        wm = copy.deepcopy(data.shape[0])
        Km = copy.deepcopy(data.s1b[0])

    res = cgDNA(seq,ps)
    wr, Kr = res.ground_state,res.stiff.todense() 
    err = Mahal_sym(wm,wr,Km,Kr)
    return err


def recons_err_KL_sym(data,ps,sym):
    seq = copy.deepcopy(data.seq[0])
    seq = seq.replace('U','T') 
    if sym==False:
        wm = copy.deepcopy(data.shape[0])
        Km = copy.deepcopy(data.s1b[0])
    elif seq==comp(seq):
        wm = copy.deepcopy(data.shape_sym[0])
        Km = copy.deepcopy(data.s1b_sym[0])
    else:
        wm = copy.deepcopy(data.shape[0])
        Km = copy.deepcopy(data.s1b[0])

    res = cgDNA(seq,ps)
    wr, Kr = res.ground_state,res.stiff.todense() 
    err = kl_mvn_sym(wm, Km, wr, Kr)
    return err



def difference_mu(data1,data2,sym):
    
    if hasattr(data1, 's1b'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym[0])
        else:
            wm = copy.deepcopy(data1.shape[0])
    elif hasattr(data1, 'stiff'):
        wm = copy.deepcopy(data1.ground_state)

    if hasattr(data2, 's1b'):
        if sym==True:
            wr = copy.deepcopy(data2.shape_sym[0])
        else:
            wr = copy.deepcopy(data2.shape[0])
    elif hasattr(data2, 'stiff'):
        wr = copy.deepcopy(data2.ground_state)
        
    l = np.size(wm)
    err = wm-wr
    err = np.linalg.norm(err)/l
    return err

def difference_K(data1,data2,sym):

    if hasattr(data1, 's1b'):
        if sym==True:
            Km = copy.deepcopy(data1.s1b_sym[0])
        else:
            Km = copy.deepcopy(data1.s1b[0])            
        nbp = data1.nbp[0]
    elif hasattr(data1, 'stiff'):
        Km = copy.deepcopy(data1.stiff.todense())
        nbp = data1.nbp

    if hasattr(data2, 's1b'):
        if sym==True:
            Kr = copy.deepcopy(data2.s1b_sym[0])
        else:
            Kr = copy.deepcopy(data2.s1b[0])            
    elif hasattr(data2, 'stiff'):
        Kr = copy.deepcopy(data2.stiff.todense())

    Km_inside = np.zeros((24*nbp-18,24*nbp-18))
    Kr_inside = np.zeros((24*nbp-18,24*nbp-18))
    x1,x2 = stencil_42(nbp)
    for i,j in zip(x1,x2):
        Km_inside[i:j,i:j] = Km[i:j,i:j]
        Kr_inside[i:j,i:j] = Kr[i:j,i:j]

    err = Km_inside - Kr_inside
    l = 36*36*2 + 42*42*(nbp-3) - 18*18*(nbp-2)
    err = np.linalg.norm(err)/l
    return err


def difference_Mahal_sym(data1,data2,sym):
    
    if hasattr(data1, 's1b'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym[0])
            Km = copy.deepcopy(data1.s1b_sym[0])
        else:
            wm = copy.deepcopy(data1.shape[0])
            Km = copy.deepcopy(data1.s1b[0])            
    elif hasattr(data1, 'stiff'):
        wm = copy.deepcopy(data1.ground_state)
        Km = copy.deepcopy(data1.stiff.todense())

    if hasattr(data2, 's1b'):
        if sym==True:
            wr = copy.deepcopy(data2.shape_sym[0])
            Kr = copy.deepcopy(data2.s1b_sym[0])
        else:
            wr = copy.deepcopy(data2.shape[0])
            Kr = copy.deepcopy(data2.s1b[0])            
    elif hasattr(data2, 'stiff'):
        wr = copy.deepcopy(data2.ground_state)
        Kr = copy.deepcopy(data2.stiff.todense())
        
    err = Mahal_sym(wm,wr,Km,Kr)
    return err


def Truncation_KL_sym(data1,sym):
    if hasattr(data1, 's1b'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym[0])
            Km = copy.deepcopy(data1.s1b_sym[0])
            Kr = copy.deepcopy(data1.stiff_me_sym[0])
        else:
            wm = copy.deepcopy(data1.shape[0])
            Km = copy.deepcopy(data1.s1b[0])            
            Kr = copy.deepcopy(data1.stiff_me[0])            
    err = kl_mvn_sym(wm, Km, wm, Kr)
    return err


def Truncation_KL_sym_cg(data1,sym):
    if hasattr(data1, 's1b_cg'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym_cg[0])
            Km = copy.deepcopy(data1.s1b_sym_cg[0])
            Kr = copy.deepcopy(data1.stiff_me_sym_cg[0])
        else:
            wm = copy.deepcopy(data1.shape_cg[0])
            Km = copy.deepcopy(data1.s1b_cg[0])            
            Kr = copy.deepcopy(data1.stiff_me_cg[0])            
    err = kl_mvn_sym(wm, Km, wm, Kr)
    return err

def Truncation_KL_sym_inter(data1,sym):
    if hasattr(data1, 's1b_inter'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym_inter[0])
            Km = copy.deepcopy(data1.s1b_sym_inter[0])
            Kr = copy.deepcopy(data1.stiff_me_sym_inter[0])
        else:
            wm = copy.deepcopy(data1.shape_inter[0])
            Km = copy.deepcopy(data1.s1b_inter[0])            
            Kr = copy.deepcopy(data1.stiff_me_inter[0])            
    err = kl_mvn_sym(wm, Km, wm, Kr)
    return err



def difference_KL_sym(data1,data2,sym):
    if hasattr(data1, 's1b'):
        if sym==True:
            wm = copy.deepcopy(data1.shape_sym[0])
            Km = copy.deepcopy(data1.s1b_sym[0])
        else:
            wm = copy.deepcopy(data1.shape[0])
            Km = copy.deepcopy(data1.s1b[0])            
    elif hasattr(data1, 'stiff'):
        wm = copy.deepcopy(data1.ground_state)
        Km = copy.deepcopy(data1.stiff.todense())

    if hasattr(data2, 's1b'):
        if sym==True:
            wr = copy.deepcopy(data2.shape_sym[0])
            Kr = copy.deepcopy(data2.s1b_sym[0])
        else:
            wr = copy.deepcopy(data2.shape[0])
            Kr = copy.deepcopy(data2.s1b[0])            
    elif hasattr(data2, 'stiff'):
        wr = copy.deepcopy(data2.ground_state)
        Kr = copy.deepcopy(data2.stiff.todense())

    err = kl_mvn_sym(wm, Km, wr, Kr)
    return err





#############################################################################    
#############################################################################    
# CODE BELOW Are main code
#############################################################################    
#############################################################################
def compare_shape(w_list, seq_list, save_name,lss,color,type_of_var='cg+'):
    plt.close()
    RotDeg=0
    if RotDeg == 1:
        y2=' ($^\circ$)'
    elif RotDeg==0:
        y2=' rad/5'
    else:
        print('Wrong value for the Keyword RotDeg.') 
    lww = 0.5
    legend_size = 4
    fs=6
    fs2 = 6
    if type_of_var=='cg+':
        print("plotting only cgDNA variables -  ---- ")
        fig = plt.figure(constrained_layout=False)
        gs1 = gridspec.GridSpec(500, 100)
        gs1.update(left=0.09, right=0.98,top=0.98,bottom=0.06)
        hs,he = 46,54
        vdiff,vshift =15,4

        ax00 = plt.subplot(gs1[0:50,       0:hs])
        ax01 = plt.subplot(gs1[0:50,       he:100])

        ax10 = plt.subplot(gs1[50 + vdiff - vshift:140,    0:hs])
        ax11 = plt.subplot(gs1[50 + vdiff - vshift:140,    he:100])

        ax20 = plt.subplot(gs1[140 + vdiff:260,    0:hs])
        ax21 = plt.subplot(gs1[140 + vdiff:260,    he:100])

        ax30 = plt.subplot(gs1[260 + vdiff:380,    0:hs])
        ax31 = plt.subplot(gs1[260 + vdiff:380,    he:100])

        ax40 = plt.subplot(gs1[380 + vdiff:500,    0:hs])
        ax41 = plt.subplot(gs1[380 + vdiff:500,    he:100])

        ax_list = [ax00,ax10, ax01,ax11, ax20,ax21, ax30,ax31, ax40,ax41]
        ax_list_str = ['ax00','ax10','ax01','ax11','ax20','ax21','ax30','ax31','ax40','ax41']
        ax_sub_list = [ax00,ax10,ax01,ax11,ax20,ax21,ax30,ax31]
        ax_sub_list2 = [ax00,ax10,ax01,ax11]
        ax_min = dict.fromkeys(ax_list_str, 0)
        ax_max = dict.fromkeys(ax_list_str, 0)
        ax_range = dict.fromkeys(ax_list_str, 0)

        count = 0
        for w,seq in zip(w_list,seq_list):
            c=color[count]
            if count > 0:
                coord = ['_no_legend']*12
                coordp = ['_no_legend']*12
            else:
                coord = ['Buckle','Propeller','Opening','Shear','Stretch','Stagger','Tilt','Roll','Twist','Shift','Slide','Rise']
                coordp = ['WRot1','WRot2','WRot3','WTra1','WTra2','WTra3','CRot1','CRot2','CRot3','CTra1','CTra2','CTra3']
            nbp = len(seq)
            ind = np.arange(nbp)
            inds = np.arange(0.5,nbp-1,1)
            intra_r,intra_t,pho_C_r,pho_C_t,inter_r,inter_t,pho_W_r,pho_W_t = DecomposeCoord(w)
            ## think about broken axis

            for i in range(3):
    
                ax20.plot(inds,pho_C_r[i::3].T,color=c[i],lw=lww,label=coordp[0+i],ls = lss[count])
                ax21.plot(inds,pho_W_r[i::3].T,color=c[i],lw=lww,label=coordp[6+i],ls = lss[count])

                if i == 1:
                    ax00.plot(inds,pho_C_t[i::3].T,color=c[i],lw=lww,label=coordp[3+i],ls = lss[count])
                    ax01.plot(inds,pho_W_t[i::3].T,color=c[i],lw=lww,label=coordp[9+i],ls = lss[count])
                else:
                    ax10.plot(inds,pho_C_t[i::3].T,color=c[i],lw=lww,label=coordp[3+i],ls = lss[count])
                    ax11.plot(inds,pho_W_t[i::3].T,color=c[i],lw=lww,label=coordp[9+i],ls = lss[count])
               
                ax40.plot(ind,intra_r[i::3].T,color=c[i],lw=lww,label=coord[0+i],ls = lss[count])
                ax30.plot(ind,intra_t[i::3].T,color=c[i],lw=lww,label=coord[3+i],ls = lss[count])
    
                ax41.plot(inds,inter_r[i::3].T,color=c[i],lw=lww,label=coord[6+i],ls = lss[count])
                ax31.plot(inds,inter_t[i::3].T,color=c[i],lw=lww,label=coord[9+i],ls = lss[count])
    
                for axl in ax_list:
                    axl.tick_params(axis='both', which='major', labelsize=fs2)
                    axl.legend(fontsize=legend_size,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(0.5, 1.1))
                for axl in ax_sub_list2:
                    axl.legend(fontsize=legend_size,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(0.5, 1.25))

            ax41.set_xticks(ind)
            ax41.set_xticklabels(list(seq),fontsize=fs2)
            ax40.set_xticks(ind)
            ax40.set_xticklabels(list(seq),fontsize=fs2)
    
            ax10.set_ylabel(r'$\AA$',fontsize=fs2)
            ax20.set_ylabel(y2,fontsize=fs2)
            ax30.set_ylabel(r'$\AA$',fontsize=fs2)
            ax40.set_ylabel(y2,fontsize=fs2)

            d = .025  # how big to make the diagonal lines in axes coordinates
            # arguments to pass to plot, just so we don't keep repeating them

            kwargs = dict(transform=ax00.transAxes, color='k', clip_on=False,lw=0.5)
            ax00.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax00.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
            
            kwargs.update(transform=ax10.transAxes)  # switch to the bottom axes
            ax10.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax10.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

            kwargs.update(transform=ax01.transAxes)  # switch to the bottom axes
            ax01.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax01.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
            
            kwargs.update(transform=ax11.transAxes)  # switch to the bottom axes
            ax11.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax11.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

            ax00.spines['bottom'].set_visible(False)
            ax10.spines['top'].set_visible(False)    
            ax01.spines['bottom'].set_visible(False)
            ax11.spines['top'].set_visible(False)
            for axl in ax_sub_list:
                axl.set_xticks([])
            for axl in ax_list:
                axl.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')

################function below set the and ylim
            variable = [pho_C_t[1],pho_C_t[[0,2]],pho_W_t[1],pho_W_t[[0,2]],pho_C_r,pho_W_r,intra_t,inter_t,intra_r,inter_r]
            for axl,var_tmp in zip(ax_list_str,variable):
                tmp1,tmp2 = np.amin(var_tmp),np.amax(var_tmp)
                if count == 0:
                    ax_min[axl] = tmp1
                    ax_max[axl] = tmp2
                else:
                    if ax_min[axl] > tmp1:
                        ax_min[axl] = tmp1
                    if ax_max[axl] < tmp2:
                        ax_max[axl] = tmp2
################function above set the and ylim------------------
            count=count+1

        for axl,axl_str in zip(ax_list,ax_list_str):
            ax_range[axl_str] = 0.15*(ax_max[axl_str] - ax_min[axl_str])
            axl.set_ylim(ax_min[axl_str]-ax_range[axl_str],ax_max[axl_str]+ax_range[axl_str])

######################------------type_of_var---------------###################
    if type_of_var=='cg':
        print("plotting only cgDNA variables -  ---- ")
        fig = plt.figure(constrained_layout=False)
        gs1 = gridspec.GridSpec(500, 100)
        gs1.update(left=0.09, right=0.98,top=0.98,bottom=0.06)
        hs,he = 47,53
        vdiff,vshift =15,4


        ax30 = plt.subplot(gs1[0 :250,    0:hs])
        ax31 = plt.subplot(gs1[0 :250,    he:100])

        ax40 = plt.subplot(gs1[250 + vdiff:500,    0:hs])
        ax41 = plt.subplot(gs1[250 + vdiff:500,    he:100])

        ax_list = [ax30,ax31, ax40,ax41]
        ax_list_str = ['ax30','ax31','ax40','ax41']
        ax_min = dict.fromkeys(ax_list_str, 0)
        ax_max = dict.fromkeys(ax_list_str, 0)
        ax_range = dict.fromkeys(ax_list_str, 0)

        count = 0
        for w,seq in zip(w_list,seq_list):
            if count > 0:
                coord = ['_no_legend']*12
                coordp = ['_no_legend']*12
            else:
                coord = ['buckle','propeller','opening','shear','stretch','stagger','tilt','roll','twist','shift','slide','rise']
                coordp = ['W_rot1','W_rot2','W_rot3','W_trans1','W_trans2','W_trans3','C_rot1','C_rot2','C_rot3','C_trans1','C_trans2','C_trans3']
            nbp = len(seq)
            ind = np.arange(nbp)
            inds = np.arange(0.5,nbp-1,1)
            intra_r,intra_t,pho_C_r,pho_C_t,inter_r,inter_t,pho_W_r,pho_W_t = DecomposeCoord(w)
            ## think about broken axis

            for i in range(3):
                   
                ax40.plot(ind,intra_r[i::3].T,color=c[i],lw=lww,label=coord[0+i],ls = lss[count])
                ax30.plot(ind,intra_t[i::3].T,color=c[i],lw=lww,label=coord[3+i],ls = lss[count])
    
                ax41.plot(inds,inter_r[i::3].T,color=c[i],lw=lww,label=coord[6+i],ls = lss[count])
                ax31.plot(inds,inter_t[i::3].T,color=c[i],lw=lww,label=coord[9+i],ls = lss[count])
    
                for axl in ax_list:
                    axl.tick_params(axis='both', which='major', labelsize=fs2)
                    axl.legend(fontsize=legend_size,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(0.5, 1.05))

            ax41.set_xticks(ind)
            ax41.set_xticklabels(list(seq),fontsize=fs2)
            ax40.set_xticks(ind)
            ax40.set_xticklabels(list(seq),fontsize=fs2)
    
            ax30.set_ylabel(r'$\AA$',fontsize=fs2)
            ax40.set_ylabel(y2,fontsize=fs2)

            for axl in [ax30,ax31]:
                axl.set_xticks([])
            for axl in ax_list:
                axl.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')

################function below set the and ylim
            variable = [intra_t,inter_t,intra_r,inter_r]
            for axl,var_tmp in zip(ax_list_str,variable):
                tmp1,tmp2 = np.amin(var_tmp),np.amax(var_tmp)
                if count == 0:
                    ax_min[axl] = tmp1
                    ax_max[axl] = tmp2
                else:
                    if ax_min[axl] > tmp1:
                        ax_min[axl] = tmp1
                    if ax_max[axl] < tmp2:
                        ax_max[axl] = tmp2
################function above set the and ylim------------------
            count=count+1

        for axl,axl_str in zip(ax_list,ax_list_str):
            ax_range[axl_str] = 0.15*(ax_max[axl_str] - ax_min[axl_str])
            axl.set_ylim(ax_min[axl_str]-ax_range[axl_str],ax_max[axl_str]+ax_range[axl_str])


    plt.show()
    fig.savefig("./Plots/" + save_name +".pdf",dpi=600)
    return fig,ax_list

##################################---------------------------------------------
# persistence length
##################################---------------------------------------------
colo = ['red','blue','magenta','green','black','cyan']
def read_persist_data(path):        
    all_files = np.concatenate((np.arange(1,101),np.arange(501,554)))
    all_files = np.arange(1,101)
    print("only reading random files")
    li = []
    for i in all_files:
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame.dropna()

def read_persist_data_rand(path):        
    all_files = np.arange(1,101)
    li = []
    for i in all_files:
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame.dropna()

def read_persist_data_sub(path):        
    all_files = np.arange(1,101)
    li = []
    for i in all_files:
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame.dropna()

def read_persist_data_tandem(path):        
    all_files = np.arange(501,554)
    li = []
    for i in all_files:
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame.dropna()


def read_persist_data_NA(path):        
    all_files = np.concatenate((np.arange(1,101),np.arange(501,554)))
    li = []
    for i in all_files:
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame


def identify_file_index(ind):
    ind = ind+1
    file = ind//20000 + 1
    line = ind%20000
    if file > 100:
        file = file + 400
    return file, line

def read_persis_file_index(path,ind):
    print(path)
    file, line = identify_file_index(ind)
    print(file,line)
    df = pd.read_csv(path+'/results_'+str(file)+'.txt', engine='python',sep=" ",header=None)
    print(df)
    print(df.loc[line-1])

    
def plot_persistence_length(names):
    fig,ax = plt.subplots()
    count=0
    data = {}
    fs = 10
    for n in names:
        path = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/' + n
        data[n] = read_persist_data_rand(path)
        print("Note which data is Reading the ----------")
        print(names,max(data[n][0]),max(data[n][1]),min(data[n][0]),min(data[n][1]), )
        plt.hist(data[n][0] ,histtype = 'step',bins = 1000,color=colo[count] ,lw=1, label = n[:3] +'_' + 'app'  )
        plt.hist(data[n][1] ,histtype = 'step',bins = 500,color=colo[count+1],lw=1, label = n[:3] +'_' + 'dyn'  )
        count = count+2
    ax.legend(fontsize=fs-4)
    ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
    ax.set_xlabel("Persistence length in number of bp",fontsize=fs)
    plt.savefig('./Plots/persistence_length'+".pdf",dpi=600)
    plt.show()
    return data

def plot_persistence_length_tandem_vs_random(names):
    fig,ax = plt.subplots()
    count=0
    data1, data2 = {},{}
    fs = 10
    for n in names:
        path = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/' + n
        data1[n] = read_persist_data_rand(path)
        data2[n] = read_persist_data_tandem(path)
        plt.hist(data1[n][0] ,histtype = 'step',bins = 1000,color=colo[count] ,lw=1, label = n[:3] +'_' + 'app_rand', ls=(0, (5, 10))  )
        plt.hist(data1[n][1] ,histtype = 'step',bins = 500,color=colo[count+1],lw=1, label = n[:3] +'_' + 'dyn_rand', ls=(0, (5, 10))  )
        plt.hist(data2[n][0] ,histtype = 'step',bins = 1000,color=colo[count] ,lw=1, label = n[:3] +'_' + 'app_tand'  )
        plt.hist(data2[n][1] ,histtype = 'step',bins = 500,color=colo[count+1],lw=1, label = n[:3] +'_' + 'dyn_tand'  )
        print(min(data1[n][0]), min(data1[n][1]), min(data2[n][0]), min(data2[n][1]))
        print(max(data1[n][0]), max(data1[n][1]), max(data2[n][0]), max(data2[n][1]))
        count = count+2
    ax.legend(fontsize=fs-4)
    ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
    ax.set_xlabel("Persistence length in number of bp",fontsize=fs)
    plt.savefig('./Plots/'+'_persistence_length_rand_vs_tand'+".pdf",dpi=600)
    plt.show()
    return None


def plot_persistence_length_difference(names):
    fig,ax = plt.subplots()
    count=0
    data = {}
    fs = 10

    for n in names:
        path = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/' + n
        data[n] = read_persist_data_NA(path)
        u0 = data[n][0]- data['DNA_BSTJ_CGF'][0]
        u1 = data[n][1]- data['DNA_BSTJ_CGF'][1]
        if count>0:
            plt.hist(u0.dropna() ,histtype = 'step',bins = 1000,color=colo[count] ,lw=1, label = n[:3] +'_' + 'app' + ' - DNA_app')
            plt.hist(u1.dropna() ,histtype = 'step',bins = 500,color=colo[count+1],lw=1, label = n[:3] +'_' + 'dyn' + ' - DNA_dyn' )
            None
        count = count+2
    ax.legend(fontsize=fs-4)
    ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
    ax.set_xlabel("Persistence length in number of bp",fontsize=fs)
    plt.savefig('./Plots/persistence_length_difference'+".pdf",dpi=600)
    plt.show()
    return data


def read_special_persist_data(path):        

    dimer_files= ['AA','TT','GC','CG', 'TC','CT','TG','GT', 'CC','GG', 'AC','CA','AG','GA', 'AT','TA' ]
    all_files = [501,518,549,539, 521,532,525,546, 535,553, 507,528, 511, 542, 514, 504]
    lines_index = [1,9526,18671,8956, 19431,9146,9336,8766,19051,8576,19811, 19241, 9716,18861, 19621,  9906 ]
    li  = np.zeros((len(all_files),2))
    for enum,i in enumerate(all_files):
        df = pd.read_csv(path+'/results_'+str(i)+'.txt', engine='python',sep=" ",header=None)
        li[enum]=df.loc[lines_index[enum]-1]
    return li, dimer_files

def plot_special_persistence_length(names):
    fig,ax = plt.subplots()
    count=0
    data = {}
    fs = 10
    for n in names:
        path = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/' + n
        data[n], dimer_files = read_special_persist_data(path)
        print(data[n][:,0])
        print(data[n][:,1])
        ax.scatter(np.arange(16),data[n][:,0] ,color=colo[count],s=10, label = n[:3] +'_' + 'app')
        ax.scatter(np.arange(16),data[n][:,1] ,color=colo[count+1],marker='_', label = n[:3] +'_' + 'dyn')
        count = count+2

    ax.legend(fontsize=fs-5)
    ax.set_xticks(np.arange(16))
    ax.set_xticklabels(dimer_files, minor=False,rotation=90)

    ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
    ax.set_ylabel("Persistence length in number of bp",fontsize=fs)
    plt.savefig('./Plots/persistence_length_sp'+".pdf",dpi=600)
    plt.show()
    return data


##################################---------------------------------------------
# plot stencil in stiffness matrix
##################################---------------------------------------------

def fit_stencil_in_matrix(data,sym,save_name):
    if hasattr(data, 's1b'):
        if sym==True:
            wc = copy.deepcopy(data.s1b_sym[0])
        else:
            wc = copy.deepcopy(data.s1b[0])
        nbp = data.nbp[0]
    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp
    wc = np.nan_to_num(wc)

    x1,x2 = stencil_42(nbp)

    x6 = np.arange(0,24*nbp-12,6)
    fig,axr = plt.subplots(1)
    sns.heatmap(wc,ax=axr,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=1,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    lwl,lws = 0.3,0.05
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for j in x6:
        axr.axvline(x=j,color='black', lw=lws)
        axr.axhline(y=j,color='black', lw=lws)

    ind = np.arange(0,24*nbp-18,12)
    axr.set_xticks(ind)         
    axr.set_yticks(ind)    
    axr.set_xticklabels(ind)
    axr.set_yticklabels(ind)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=6, pad=3,length=3,width=0.5,direction= 'inout')
    axr.grid(False)
    plt.savefig('./Plots/'+save_name+'.pdf',dpi=600)


##################################---------------------------------------------
# plot palindromic error
##################################---------------------------------------------
def plot_below1(mat,fs,center,clabel,save_name):
    fig,ax = plt.subplots()
    sns.heatmap(mat,ax=ax,cmap="RdBu_r",cbar=1,annot=True,fmt='.4f',center=center,annot_kws={"size":fs-2},cbar_kws={"shrink": .7,"pad":0.018}).invert_yaxis()
    s = np.shape(mat)
    ax.tick_params(axis='both', length = 3, width=0.5, direction='inout',pad=3)
    ax.figure.axes[-1].set_ylabel(clabel,size=fs)    
    ax.figure.axes[-1].tick_params(labelsize=fs-2)    
    ax.set_xticks(np.arange(s[1])+.5)
    ax.set_yticks(np.arange(s[0])+.5)        
    ax.set_xticklabels(np.arange(s[1])+1,fontsize=fs)         
    ax.set_yticklabels(np.arange(s[0])+1,fontsize=fs)         
    ax.set_ylabel('Index of sequences in the training library',size=fs)
    ax.set_xlabel('Simulation length in $\mu$s',size=fs)
    plt.tight_layout()
    fig.savefig('./Plots/'+save_name+'.pdf',dpi=600)
    plt.show()
    plt.close()
    

def plot_heatmap_palin_err(data_name,name):
    mat1 = np.zeros((16,10))
    mat2 = np.zeros((16,10))
    mat3 = np.zeros((16,10))
    mat4 = np.zeros((16,10))
    for file_index in np.arange(1,11,1):
        data_path_tmp = data_name.replace('XX',str(file_index))
        data_tmp = init_MD_data().load_data(data_path_tmp)
        for seq in np.arange(0,16,1):
            mat1[seq,file_index-1] = palin_err_mu_norm(data_tmp.choose_seq([seq]))
            mat2[seq,file_index-1] = palin_err_K_norm(data_tmp.choose_seq([seq]))
            mat3[seq,file_index-1] = palin_err_Mahal_sym(data_tmp.choose_seq([seq]))
            mat4[seq,file_index-1] = palin_err_KL_sym(data_tmp.choose_seq([seq]))
    plot_below1(mat1,fs=8,center=0.0035,clabel="Palindromic error in groundstate, |$\mu$-E$\mu$|", save_name=name+'_palin_err_mu_norm')
    plot_below1(mat2,fs=8,center=0.005, clabel="Palindromic error in stiffness, |K-EKE|",          save_name=name+'_palin_err_K_norm')
    plot_below1(mat3,fs=8,center=0.0023,clabel="Palindromic error, Symmetric Mahalanobis distance",save_name=name+'_palin_err_Mahal_sym')
    plot_below1(mat4,fs=8,center=0.04,  clabel="Palindromic error, Symmetric KL divergence",       save_name=name+'_palin_err_KL_sym')
    return None

def plot_heatmap_palin_err_epi(data_name,name):
    mat1 = np.zeros((12,10))
    mat2 = np.zeros((12,10))
    mat3 = np.zeros((12,10))
    mat4 = np.zeros((12,10))
    for file_index in np.arange(1,11,1):
        data_path_tmp = data_name.replace('XX',str(file_index))
        data_tmp = init_MD_data().load_data(data_path_tmp)
        for enum,seq in enumerate(MDNA_map[0:12]):
            mat1[enum,file_index-1] = palin_err_mu_norm(data_tmp.choose_seq([seq]))
            mat2[enum,file_index-1] = palin_err_K_norm(data_tmp.choose_seq([seq]))
            mat3[enum,file_index-1] = palin_err_Mahal_sym(data_tmp.choose_seq([seq]))
            mat4[enum,file_index-1] = palin_err_KL_sym(data_tmp.choose_seq([seq]))
    plot_below1(mat1,fs=8,center=0.0035,clabel="Palindromic error in groundstate, |$\mu$-E$\mu$|", save_name=name+'_palin_err_mu_norm')
    plot_below1(mat2,fs=8,center=0.005, clabel="Palindromic error in stiffness, |K-EKE|",          save_name=name+'_palin_err_K_norm')
    plot_below1(mat3,fs=8,center=0.0023,clabel="Palindromic error, Symmetric Mahalanobis distance",save_name=name+'_palin_err_Mahal_sym')
    plot_below1(mat4,fs=8,center=0.04,  clabel="Palindromic error, Symmetric KL divergence",       save_name=name+'_palin_err_KL_sym')
    return None

##################################---------------------------------------------
# Set scale for training/test error, and palindromic error
##################################---------------------------------------------
def set_scale_for_error(path1,sym,label='DNA'):
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(600, 95)
    gs1.update(left=0.03, right=0.95,top=0.98,bottom=0.08)
    ax1 = plt.subplot(gs1[0:280,         0:95])
    ax2 = plt.subplot(gs1[320:600,         0:95])

    DNA = init_MD_data().load_data(path1)
    nseq = 16
    nent =  nseq*(nseq-1)/2
    mat1 = np.zeros((nseq,nseq))
    mat2 = np.zeros((nseq,nseq))
    mat3 = np.zeros((nseq,nseq))
    mat4 = np.zeros((nseq,nseq))
    for i in range(nseq):
        for j in range(nseq):
            if i >j:
#                mat1[i,j]=difference_mu(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
#                mat2[i,j]=difference_K(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat3[i,j]=difference_Mahal_sym(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat4[i,j]=difference_KL_sym(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                None
#    mat3 = np.eye(16)
#    mat4 = np.eye(16)
    clabel = ["SM","SKL"]
    fs=6
    sns.heatmap(mat3+mat3.T,ax=ax1,cmap="RdBu_r",cbar=1,square=1,cbar_kws={"shrink": .7,"pad":0.018},linewidths=0.1, linecolor='k').invert_yaxis()
    ax1.figure.axes[-1].set_ylabel("SM",size=fs)
    ax1.figure.axes[-1].tick_params(labelsize=fs,width=0.5, direction='inout',pad=3)

    sns.heatmap(mat4+mat4.T,ax=ax2,cmap="RdBu_r",cbar=1,square=1,cbar_kws={"shrink": .7,"pad":0.018},linewidths=0.1, linecolor='k').invert_yaxis()
    ax2.figure.axes[-1].set_ylabel("SKL",size=fs)
    ax2.figure.axes[-1].tick_params(labelsize=fs,width=0.5, direction='inout',pad=3)
    for enum,ax in enumerate([ax1,ax2]):
        ax.tick_params(axis='both', length = 3, width=0.5, direction='inout',pad=3)
        ax.set_xticks(np.arange(16)+.5)
        ax.set_yticks(np.arange(16)+.5)
        ax.set_xticklabels(np.arange(16)+1,fontsize=fs)
        ax.set_yticklabels(np.arange(16)+1,fontsize=fs,rotation=0)
        ax.set_ylabel('Index of sequences in the training library',size=fs)
    ax2.set_xlabel('Index of sequences in the training library',size=fs)

    plt.savefig('./Plots/scale_fig_'+label+'.pdf',dpi=600)

    sys.exit()
    scale1, min1 = np.sum(mat1)/nent, np.min(mat1[np.nonzero(mat1)])
    scale2, min2 = np.sum(mat2)/nent, np.min(mat2[np.nonzero(mat2)])
    scale3, min3 = np.sum(mat3)/nent, np.min(mat3[np.nonzero(mat3)])
    scale4, min4 = np.sum(mat4)/nent, np.min(mat4[np.nonzero(mat4)])
    print("\\textbf{Scale1}"," & ",str(scale1)[0:6]," & ",str(scale2)[0:6]," & ",str(scale3)[0:6]," & ",str(scale4)[0:6]," \\\\ ")
    print("\\textbf{Scale2}"," & ",str(min1)[0:6]," & ",str(min2)[0:6]," & ",str(min3)[0:6]," & ",str(min4)[0:6]," \\\\ ")

def set_scale_for_error_epi(DNA,sym):
    nseq = 12
    nent =  nseq*(nseq-1)/2
    mat1 = np.zeros((nseq,nseq))
    mat2 = np.zeros((nseq,nseq))
    mat3 = np.zeros((nseq,nseq))
    mat4 = np.zeros((nseq,nseq))
    which = [0,1,2,3,4,5,8,9,12,13,14,16]
    for enum1,i in enumerate(which):
        for enum2,j in enumerate(which):
            if enum1 > enum2:
                mat1[enum1,enum2]=difference_mu(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat2[enum1,enum2]=difference_K(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat3[enum1,enum2]=difference_Mahal_sym(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
                mat4[enum1,enum2]=difference_KL_sym(DNA.choose_seq([i]),DNA.choose_seq([j]),sym)
    scale1, min1 = np.sum(mat1)/nent, np.min(mat1[np.nonzero(mat1)])
    scale2, min2 = np.sum(mat2)/nent, np.min(mat2[np.nonzero(mat2)])
    scale3, min3 = np.sum(mat3)/nent, np.min(mat3[np.nonzero(mat3)])
    scale4, min4 = np.sum(mat4)/nent, np.min(mat4[np.nonzero(mat4)])
    print("\\textbf{Scale1}"," & ",str(scale1)[0:6]," & ",str(scale2)[0:6]," & ",str(scale3)[0:6]," & ",str(scale4)[0:6]," \\\\ ")
    print("\\textbf{Scale2}"," & ",str(min1)[0:6]," & ",str(min2)[0:6]," & ",str(min3)[0:6]," & ",str(min4)[0:6]," \\\\ ")

##################################---------------------------------------------
# plot training/test error, plot_heatmap_training_err
##################################---------------------------------------------
def plot_below2(mat,fs,center,clabel,save_name):
    fig,ax = plt.subplots(edgecolor='k',facecolor='w')
    sns.heatmap(mat,ax=ax,cmap="RdBu_r",cbar=1,annot=True,fmt='.4f',center=center,annot_kws={"size":fs-2},cbar_kws={"shrink": .7,"pad":0.018},linewidths=0.1, linecolor='k').invert_yaxis()
    s = np.shape(mat)
    ax.tick_params(axis='both', length = 3, width=0.5, direction='inout',pad=3)
    ax.figure.axes[-1].set_ylabel(clabel,size=fs)    
    ax.figure.axes[-1].tick_params(labelsize=fs-2)    
    ax.set_xticks(np.arange(s[1])+.5)
    ax.set_yticks(np.arange(s[0])+.5)        
    ax.set_xticklabels(np.arange(s[1])+1,fontsize=fs)         
    ax.set_yticklabels(np.arange(s[0])+1,fontsize=fs)         
    ax.set_ylabel('Index of sequences in the training library',size=fs)
    ax.set_xlabel('Error definition',size=fs)
    plt.tight_layout()
    fig.savefig('./Plots/'+save_name+'.png',dpi=600)
    plt.show()
    plt.close()

def plot_heatmap_training_err(data_tmp,ps,sym):
    nseq=len(data_tmp.nbp)
    for i in range(nseq):
        print(i+1,"& \\ttfamily",data_tmp.seq[i],"\\\\")
    mat1 = np.zeros((nseq,4))
    for seq in np.arange(0,16,1):
        mat1[seq,0] = recons_err_mu_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,1] = recons_err_K_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,2] = recons_err_Mahal_sym(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,3] = recons_err_KL_sym(data_tmp.choose_seq([seq]),ps,sym)
        print(seq+1, "&",str(mat1[seq,0])[0:6], "&",str(mat1[seq,1])[0:6], "&",str(mat1[seq,2])[0:6], "&",str(mat1[seq,3])[0:6], "\\\\")
    print("\hline")
    print("\\textbf{Average training error}", "&",str(np.mean(mat1[0:16,0]))[0:6], "&",str(np.mean(mat1[0:16,1]))[0:6], "&",str(np.mean(mat1[0:16,2]))[0:6], "&",str(np.mean(mat1[0:16,3]))[0:6], "\\\\")
    print("\hline")
    for seq in np.arange(16,nseq,1):
        mat1[seq,0] = recons_err_mu_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,1] = recons_err_K_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,2] = recons_err_Mahal_sym(data_tmp.choose_seq([seq]),ps,sym)
        mat1[seq,3] = recons_err_KL_sym(data_tmp.choose_seq([seq]),ps,sym)
        print(seq+1, "&",str(mat1[seq,0])[0:6], "&",str(mat1[seq,1])[0:6], "&",str(mat1[seq,2])[0:6], "&",str(mat1[seq,3])[0:6], "\\\\")
    print("\hline")
    print("\\textbf{Average test error}", "&",str(np.mean(mat1[16:nseq,0]))[0:6], "&",str(np.mean(mat1[16:nseq,1]))[0:6], "&",str(np.mean(mat1[16:nseq,2]))[0:6], "&",str(np.mean(mat1[16:nseq,3]))[0:6])


    return mat1[:,2],mat1[:,3]

def unmodify(seq):
    seq = seq.replace('M','C')
    seq = seq.replace('N','G')
    seq = seq.replace('H','C')
    seq = seq.replace('K','G')
    seq = seq.replace('U','T')
    return seq

def plot_heatmap_training_err_epi(data_tmp,ps,sym):
    nseq=len(data_tmp.nbp)
    for enum,seq in enumerate([0,1,2,3,4,5,8,9,12,13,14,16]):
        print(enum+1,"& \\ttfamily",data_tmp.seq[seq],"\\\\")
    for enum,seq in enumerate([6,7,10,11,15,17,18,19,20,21]):
        print(enum+13,"& \\ttfamily",data_tmp.seq[seq],"\\\\")
    mat1 = np.zeros((nseq,4))
    count = 0
    for enum,seq in enumerate([0,1,2,3,4,5,8,9,12,13,14,16]):
        mat1[count,0] = recons_err_mu_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,1] = recons_err_K_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,2] = recons_err_Mahal_sym(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,3] = recons_err_KL_sym(data_tmp.choose_seq([seq]),ps,sym)
        print(count+1, "&",str(mat1[count,0])[0:6], "&",str(mat1[count,1])[0:6], "&",str(mat1[count,2])[0:6], "&",str(mat1[count,3])[0:6], "\\\\")
        count = count+1
    print("\hline")
    print("\\textbf{Average training error}", "&",str(np.mean(mat1[0:12,0]))[0:6], "&",str(np.mean(mat1[0:12,1]))[0:6], "&",str(np.mean(mat1[0:12,2]))[0:6], "&",str(np.mean(mat1[0:12,3]))[0:6], "\\\\")
    print("\hline")
    for enum,seq in enumerate([6,7,10,11,15,17,18,19,20]):
        mat1[count,0] = recons_err_mu_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,1] = recons_err_K_norm(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,2] = recons_err_Mahal_sym(data_tmp.choose_seq([seq]),ps,sym)
        mat1[count,3] = recons_err_KL_sym(data_tmp.choose_seq([seq]),ps,sym)
        print(count+1, "&",str(mat1[count,0])[0:6], "&",str(mat1[count,1])[0:6], "&",str(mat1[count,2])[0:6], "&",str(mat1[count,3])[0:6], "\\\\")
        count = count+1
    print("\hline")
    print("\\textbf{Average test error}", "&",str(np.mean(mat1[12:nseq-1,0]))[0:6], "&",str(np.mean(mat1[12:nseq-1,1]))[0:6], "&",str(np.mean(mat1[12:nseq-1,2]))[0:6], "&",str(np.mean(mat1[12:nseq-1,3]))[0:6])

    return None


##################################---------------------------------------------
# plot_gs_vs_MD_shape 
##################################---------------------------------------------
def Truncation_error(data,NA_type_list):
    for data_tmp, NA_type in zip(data,NA_type_list):
        if NA_type == 'DNA':
            NA_sym = DNA_sym
            pam = np.arange(16)
            print("Note only computing for 16 seq")
        elif NA_type == 'PDNA':
            NA_sym = PDNA_sym
            pam = np.arange(16)
        elif NA_type == 'RNA':
            NA_sym = RNA_sym
            pam = np.arange(24)
        elif NA_type == 'HYB':
            NA_sym = HYB_sym
            pam = np.arange(24)
        elif NA_type == 'MDNA' or  NA_type == 'HDNA':
            NA_sym = MDNA_sym
            pam = MDNA_map
        else:
            print("------------Error provide argiment for sym------------------")
        mat1= []
        for enum,seq in enumerate(pam):
            mat1.append(Truncation_KL_sym(data_tmp.choose_seq([seq]), NA_sym[seq]))
    print(mat1)
    return mat1

def Truncation_error_marg(data,NA_type_list):
    for data_tmp, NA_type in zip(data,NA_type_list):
        if NA_type == 'DNA':
            NA_sym = DNA_sym
            pam = np.arange(16)
            print("Note only computing for 16 seq")
        mat1,mat2,mat3 = [],[],[]
        for enum,seq in enumerate(pam):
            mat1.append(Truncation_KL_sym_inter(data_tmp.choose_seq([seq]), NA_sym[seq]))
            mat2.append(Truncation_KL_sym_cg(data_tmp.choose_seq([seq]), NA_sym[seq]))
            mat3.append(Truncation_KL_sym(data_tmp.choose_seq([seq]), NA_sym[seq]))
        
    return mat1,mat2,mat3

##################################---------------------------------------------
# plot_gs_vs_MD_shape 
##################################---------------------------------------------
def plot_gs_vs_MD_shape(path1,path2,path3):
#########-------------------------DNA------------------
    c1 = ['red','blue','green']
    c2 = ['maroon','dodgerblue','limegreen']
    linestyles= ['-', '--', '-.', ':']
    DNA = init_MD_data().load_data(path1)
    nseq = len(DNA.nbp)
    for i in range(nseq):
        res = cgDNA(DNA.seq[i],'ps2_cgf')
        if comp(DNA.seq[i]) ==DNA.seq[i]:
            compare_shape([DNA.shape_sym[i],res.ground_state],[DNA.seq[i],res.seq],'DNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')
        else:
            compare_shape([DNA.shape[i],res.ground_state],[DNA.seq[i],res.seq],'DNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')

#########-------------------------RNA------------------
    RNA = init_MD_data().load_data(path2)
    nseq = len(RNA.nbp)
    for i in range(nseq):
        seqD = (RNA.seq[i]).replace('U','T')
        res = cgDNA(seqD,'ps_rna')
        if comp(seqD) == seqD:
            compare_shape([RNA.shape_sym[i],res.ground_state],[RNA.seq[i],RNA.seq[i]],'RNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')
        else:
            compare_shape([RNA.shape[i],res.ground_state],[RNA.seq[i],RNA.seq[i]],'RNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')

#########-------------------------HYB------------------
    HYB = init_MD_data().load_data(path3)
    nseq = len(HYB.nbp)
    for i in range(nseq):
        seqD = (HYB.seq[i]).replace('U','T')
        res = cgDNA(seqD,'ps_hyb')
        compare_shape([HYB.shape[i],res.ground_state],[HYB.seq[i],HYB.seq[i]],'HYB_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')

    



def plot_gs_vs_MD_shape_epi(MDNA,HDNA):
#########-------------------------MDNA------------------
    c1 = ['red','blue','green']
    c2 = ['maroon','dodgerblue','limegreen']
    linestyles= ['-', '--', '-.', ':']
    for DNA, dna in zip([MDNA,HDNA],['mdna','hdna']):
        nseq = len(DNA.nbp)
        for enum, i in enumerate(MDNA_map):
            res = cgDNA(DNA.seq[i],'ps_'+dna)
            if comp(DNA.seq[i]) ==DNA.seq[i]:
                compare_shape([DNA.shape_sym[i],res.ground_state],[DNA.seq[i],res.seq],dna.upper() +'_gs_res_compare_'+str(enum+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')
            else:
                compare_shape([DNA.shape[i],res.ground_state],[DNA.seq[i],res.seq],dna.upper() +'_gs_res_compare_'+str(enum+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')

#########-------------------------HDNA------------------
    DNA = HDNA
    nseq = len(DNA.nbp)
    for i in range(nseq):
        res = cgDNA(DNA.seq[i],'ps_mdna')
        if comp(DNA.seq[i]) ==DNA.seq[i]:
            compare_shape([DNA.shape_sym[i],res.ground_state],[DNA.seq[i],res.seq],'MDNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')
        else:
            compare_shape([DNA.shape[i],res.ground_state],[DNA.seq[i],res.seq],'MDNA_gs_res_compare_'+str(i+1),lss=linestyles,color=[c1,c1],type_of_var='cg+')






##################################---------------------------------------------
# plot and compare same sequences across data
##################################---------------------------------------------
def compare_same_seq_across_data(d1,d2,d3,color,seq_id,save_name):
#    lss=['-', '-','-','--','--','--']
    lss=['-', '--',':','--','--','--']
    a = d1.shape_sym[seq_id]
    b = d2.shape_sym[seq_id]
    c = d3.shape[seq_id]
    asq = d1.seq[seq_id]
#    ar = cgDNA(asq,'ps2_cgf').ground_state
#    br = cgDNA(asq,'ps_rna').ground_state
#    cr = cgDNA(asq,'ps_hyb').ground_state
#    shp = [a,b,c,ar,br,cr]
    shp = [a,b,c]
#    seq = [asq]*6
    seq = [asq]*3
    compare_shape(shp,seq,save_name+'_'+str(seq_id+1),lss,color=color*2,type_of_var='cg+')


def compare_same_seq_across_data_epi(d1,d2,color,seq_id,save_name):
#    lss=['-', '-','-','--','--','--']
    lss=['-', '--',':','--','--','--']
    a = d1.shape_sym[seq_id]
    b = d2.shape_sym[seq_id]
    asq = d1.seq[seq_id]
    seq = asq.replace('M','C')
    seq = seq.replace('N','G')
    cr = cgDNA(seq,'ps_mdna').ground_state
#    br = cgDNA(asq,'ps_rna').ground_state
#    cr = cgDNA(asq,'ps_hyb').ground_state
#    shp = [a,b,c,ar,br,cr]
    shp = [cr,a,b]
#    seq = [asq]*6
    seq = [asq]*3
    compare_shape(shp,seq,save_name+'_'+str(seq_id+1),lss,color=color*2,type_of_var='cg+')


##################################---------------------------------------------
# plot and compare different sequences within same data
##################################---------------------------------------------
def compare_diff_seq_within_data(d,seq_id,ps,color,sym,save_name):
    lss=['-', '--','-', '--','-', '--','-', '--']
    shp=[]
    seq=[]
    ids = ''
    for enum,i in enumerate(seq_id):
        s = d.seq[i]
        s = s.replace('U','T')
        if sym[enum] == True:
            shp.append(d.shape_sym[i])
        else:
            shp.append(d.shape[i])
        shp.append(cgDNA(s,ps).ground_state)
        seq.append(s)
        seq.append(s)
        ids = ids + '_'+ str(i+1)
    compare_shape(shp,seq,save_name+ids,lss,color=color,type_of_var='cg+')


##################################---------------------------------------------
# plot oligomer level eigenvalues for DNA,RNA, Hybrid
##################################---------------------------------------------
def plot_eig_olig_first(d1,d2,d3,sym,color,save_name):
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(600, 100)
    gs1.update(left=0.09, right=0.98,top=0.94,bottom=0.06)

    ax0 = plt.subplot(gs1[0:247,       0:100])
    ax1 = plt.subplot(gs1[253:500,       0:100])
    ax2 = plt.subplot(gs1[505:600,       0:100])
    fs=10
    if sym[0] == True:
        s1 = d1.s1b_sym[0]
        e1 = np.linalg.eigvals(s1)
    else:
        s1 = d1.s1b[0]
        e1 = np.linalg.eigvals(s1)
        
    if sym[1] == True:
        s2 = d2.s1b_sym[0]
        e2 = np.linalg.eigvals(s2)
    else:
        s2 = d2.s1b[0]
        e2 = np.linalg.eigvals(s2)
    sq = d1.seq[0]
    s3 = d3.s1b[0]
    e3 = np.linalg.eigvals(s3)
    s4 = cgDNA(sq,'ps2_cgf').stiff.todense()
    s5 = cgDNA(sq,'ps_rna' ).stiff.todense()
    s6 = cgDNA(sq,'ps_hyb' ).stiff.todense()
    e4 = np.linalg.eigvals(s4)
    e5 = np.linalg.eigvals(s5)
    e6 = np.linalg.eigvals(s6)


    g1 = scipy.linalg.eigh(s1, s4, eigvals_only=True)
    g2 = scipy.linalg.eigh(s2, s5, eigvals_only=True)
    g3 = scipy.linalg.eigh(s3, s6, eigvals_only=True)
#    print(g1,g2)

    s = np.shape(e1)
    ax0.scatter(np.arange(s[0])+1,np.sort(e2)[::-1],s=4,color=color[1],label='RNA_MD')
    ax0.scatter(np.arange(s[0])+1,np.sort(e3)[::-1],s=4,color=color[2],label='HYB_MD')
    ax0.scatter(np.arange(s[0])+1,np.sort(e1)[::-1],s=4,color=color[0],label='DNA_MD')
    ax1.scatter(np.arange(s[0])+1,np.sort(e5)[::-1],s=4,color=color[1],label='RNA_cg')
    ax1.scatter(np.arange(s[0])+1,np.sort(e6)[::-1],s=4,color=color[2],label='HYB_cg')
    ax1.scatter(np.arange(s[0])+1,np.sort(e4)[::-1],s=4,color=color[0],label='DNA_cg')

    ax2.scatter(np.arange(s[0])+1,np.sort(g2)[::-1],s=3,color=color[1],label='RNA_gen')
    ax2.scatter(np.arange(s[0])+1,np.sort(g3)[::-1],s=3,color=color[2],label='HYB_gen')
    ax2.scatter(np.arange(s[0])+1,np.sort(g1)[::-1],s=3,color=color[0],label='DNA_gen')

    for ax in [ax0,ax1,ax2]:
        ax.legend(fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')

    ax1.set_ylabel("Eigenvalues for stiffness matrix",fontsize=fs)
    ax2.set_xlabel("Eigenvalue Index",fontsize=fs)
    ax0.set_title(sq+", Seq-length = " + str(len(sq)), fontsize=fs)
    plt.savefig('./Plots/'+save_name+".pdf",dpi=600)
    plt.show()
    plt.close()



##################################---------------------------------------------
# plot oligomer level eigenvalues for DNA,MDNA, HDNA
##################################---------------------------------------------
def plot_eig_olig_first_epi(d1,d2,d3,sym,color,save_name):
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(600, 100)
    gs1.update(left=0.09, right=0.98,top=0.94,bottom=0.06)

    ax0 = plt.subplot(gs1[0:247,       0:100])
    ax1 = plt.subplot(gs1[253:500,       0:100])
    ax2 = plt.subplot(gs1[505:600,       0:100])
    fs=10
    if sym[0] == True:
        s1 = d1.stiff.todense()
        e1 = np.linalg.eigvals(s1)
    else:
        s1 = d1.stiff.todense()
        e1 = np.linalg.eigvals(s1)
        
    if sym[1] == True:
        s2 = d2.s1b_sym[0]
        e2 = np.linalg.eigvals(s2)
    else:
        s2 = d2.s1b[0]
        e2 = np.linalg.eigvals(s2)

    if sym[2] == True:
        s3 = d3.s1b_sym[0]
        e3 = np.linalg.eigvals(s3)
    else:
        s3 = d3.s1b[0]
        e3 = np.linalg.eigvals(s3)

    sq = d2.seq[0]
    e3 = np.linalg.eigvals(s3)
    s4 = s1
    s5 = cgDNA(sq,'ps_mdna' ).stiff.todense()
    s6 = cgDNA(sq,'ps_hdna' ).stiff.todense()
    e4 = np.linalg.eigvals(s4)
    e5 = np.linalg.eigvals(s5)
    e6 = np.linalg.eigvals(s6)


    g1 = scipy.linalg.eigh(s1, s4, eigvals_only=True)
    g2 = scipy.linalg.eigh(s2, s5, eigvals_only=True)
    g3 = scipy.linalg.eigh(s3, s6, eigvals_only=True)
#    print(g1,g2)

    s = np.shape(e1)
    ax0.scatter(np.arange(s[0])+1,np.sort(e2)[::-1],s=4,color=color[1],label='MDNA_MD')
    ax0.scatter(np.arange(s[0])+1,np.sort(e3)[::-1],s=4,color=color[2],label='HDNA_MD')
    ax0.scatter(np.arange(s[0])+1,np.sort(e1)[::-1],s=4,color=color[0],label='DNA_cg')
    ax1.scatter(np.arange(s[0])+1,np.sort(e5)[::-1],s=4,color=color[1],label='MDNA_cg')
    ax1.scatter(np.arange(s[0])+1,np.sort(e6)[::-1],s=4,color=color[2],label='HDNA_cg')
    ax1.scatter(np.arange(s[0])+1,np.sort(e4)[::-1],s=4,color=color[0],label='DNA_cg')

    ax2.scatter(np.arange(s[0])+1,np.sort(g2)[::-1],s=3,color=color[1],label='MDNA_gen')
    ax2.scatter(np.arange(s[0])+1,np.sort(g3)[::-1],s=3,color=color[2],label='HDNA_gen')
    ax2.scatter(np.arange(s[0])+1,np.sort(g1)[::-1],s=3,color=color[0],label='DNA_gen')

    for ax in [ax0,ax1,ax2]:
        ax.legend(fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')

    ax1.set_ylabel("Eigenvalues for stiffness matrix",fontsize=fs)
    ax2.set_xlabel("Eigenvalue Index",fontsize=fs)
    ax0.set_title(sq+", Seq-length = " + str(len(sq)), fontsize=fs)
    plt.savefig('./Plots/'+save_name+".pdf",dpi=600)
    plt.show()
    plt.close()



##################################---------------------------------------------
# Compare groundstate DNA,RNA,HDNA
##################################---------------------------------------------
def extract_dimer_data_from_MD(MD,sym,MD_or_model=None):
    mat = np.zeros((30,17))
    tmp_dict = dict([(key, []) for key in dimer_16])

    for seq_id in range(16):
        tmp_seq = MD.choose_seq([seq_id]).seq[0].replace('U','T')
        if MD_or_model == 'DNA':
            tmp_shape = cgDNA(tmp_seq,'ps2_cgf').ground_state
        if MD_or_model == 'RNA':
            tmp_shape = cgDNA(tmp_seq,'ps_rna').ground_state
        if MD_or_model == 'HYB':
            tmp_shape = cgDNA(tmp_seq,'ps_hyb').ground_state
        if MD_or_model == 'HYB_RNA_strand_model':
            tmp_shape = np.matmul(Etrans(len(tmp_seq)),cgDNA(tmp_seq,'ps_hyb').ground_state)
        if MD_or_model == 'MD':
            if sym == True:
                tmp_shape = MD.choose_seq([seq_id]).shape_sym[0]
            else:
                tmp_shape = MD.choose_seq([seq_id]).shape[0]
        if MD_or_model == 'HYB_RNA_strand_MD':
            tmp_shape = np.matmul(Etrans(len(tmp_seq)),MD.choose_seq([seq_id]).shape[0])
            
        for dim in dimer_16:
            for loc in [m.start() for m in re.finditer(dim, tmp_seq[2:22])]:
                tmp_dict[dim].append(tmp_shape[24*(loc+2):24*(loc+2)+30])

    for enum,dim in enumerate(dimer_16):
        mat[:,enum] = np.mean(tmp_dict[dim],axis=0)
    mat[:,16] = np.mean(mat[:,0:16],axis=1)
    return mat



def compare_gs_DNA_RNA(DNA, RNA, HYB):
    DNA_dimer_data, RNA_dimer_data, HYB_dimer_data,HYB_dimer_data_RNA  = [], [], [], []
    DNA_dimer_data.append(extract_dimer_data_from_MD(DNA,True ,'MD'))
    RNA_dimer_data.append(extract_dimer_data_from_MD(RNA,True ,'MD'))
    HYB_dimer_data.append(extract_dimer_data_from_MD(HYB,False,'MD'))
    HYB_dimer_data_RNA.append(extract_dimer_data_from_MD(HYB,False,'HYB_RNA_strand_MD'))

    DNA_dimer_data.append(extract_dimer_data_from_MD(DNA,True,'DNA'))
    RNA_dimer_data.append(extract_dimer_data_from_MD(RNA,True,'RNA'))
    HYB_dimer_data.append(extract_dimer_data_from_MD(HYB,False,'HYB'))
    HYB_dimer_data_RNA.append(extract_dimer_data_from_MD(HYB,False,'HYB_RNA_strand_model'))

    ##### think about how to read HYB from RNA strand and E_trans method is wrong
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(1802, 215)
    gs1.update(left=0.065, right=0.99,top=0.99,bottom=0.04)
    d = 15
    d2 = 3
    ax10 = plt.subplot(gs1[   0:200 -d2,   0:100])
    ax11 = plt.subplot(gs1[ 200:400 -d2,   0:100])
    ax12 = plt.subplot(gs1[ 400:600 -d2,   0:100])
    ax13 = plt.subplot(gs1[   0:200 -d2,   100+d:200+d])
    ax14 = plt.subplot(gs1[ 200:400 -d2,   100+d:200+d])
    ax15 = plt.subplot(gs1[ 400:600 -d2,   100+d:200+d])

    ax20 = plt.subplot(gs1[ 600:800 -d2,    0:100])
    ax21 = plt.subplot(gs1[ 800:1000-d2,   0:100])
    ax22 = plt.subplot(gs1[1000:1200-d2,   0:100])
    ax23 = plt.subplot(gs1[ 600:800 -d2,    100+d:200+d])
    ax24 = plt.subplot(gs1[ 800:1000-d2,   100+d:200+d])
    ax25 = plt.subplot(gs1[1000:1200-d2,   100+d:200+d])

    ax30 = plt.subplot(gs1[1200:1400-d2,   0:100])
    ax31 = plt.subplot(gs1[1400:1600-d2,   0:100])
    ax32 = plt.subplot(gs1[1600:1800-d2,   0:100])
    ax33 = plt.subplot(gs1[1200:1400-d2,   100+d:200+d])
    ax34 = plt.subplot(gs1[1400:1600-d2,   100+d:200+d])
    ax35 = plt.subplot(gs1[1600:1800-d2,   100+d:200+d])

    ax_list = [ax10,ax11,ax12,ax13,ax14,ax15, ax20,ax21,ax22,ax23,ax24,ax25, ax30,ax31,ax32,ax33,ax34,ax35]
    fs = 5
    color = ['blue','red','k','k']
    line_styles = ['-','-','-','--']
    for enum_MD, MD in enumerate([DNA_dimer_data, RNA_dimer_data,HYB_dimer_data]):
        for enum, ax in enumerate(ax_list):
            if enum_MD == 0:                
                leg = cgDNA_name[enum]
                ax.set_ylabel(leg,fontsize=fs)
                ax.get_yaxis().set_label_coords(-0.09,0.5)
            else:
                leg = '_no_legend_'
            leg1 = '_no_legend_'
            ax.scatter(np.arange(17), MD[0][enum,:],color=color[enum_MD],s=1,label=leg1)
            ax.scatter(np.arange(17), MD[1][enum,:],color=color[enum_MD],s=4,marker="x",lw=0.15)
            ax.plot(np.arange(17), MD[0][enum,:],color=color[enum_MD],lw=0.15,label=leg,ls=line_styles[enum_MD])
            ax.set_xlim(-0.35,16.3)
            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            
    #            ax.legend(fontsize=fs)
    ax32.set_xticks(np.arange(len(dimer_17)))
    ax32.set_xticklabels(dimer_17,fontsize=fs)
    ax35.set_xticks(np.arange(len(dimer_17)))
    ax35.set_xticklabels(dimer_17,fontsize=fs)
    for ticklabel, tickcolor in zip(ax32.get_xticklabels(), color_16_RY):ticklabel.set_color(tickcolor)
    for ticklabel, tickcolor in zip(ax35.get_xticklabels(), color_16_RY):ticklabel.set_color(tickcolor)

#    plt.show()
    fig.savefig("./Plots/compare_dim_gs_DNA_RNA_HYB"  +".pdf",dpi=600)

    return None

    

##################################---------------------------------------------
##################################---------------------------------------------
##################################---------------------------------------------


#################################----------------------------------------------
###########---seq logo
#################################----------------------------------------------


def prob(arr):
    pro = np.zeros((4,2))

    for k1,k2 in zip([0,3],[0,1]):
        for tt in arr:
            if tt[k1] == 'A':
                pro[0,k2] = 1 + pro[0,k2]
            if tt[k1] == 'T':
                pro[1,k2] = 1 + pro[1,k2]
            if tt[k1] == 'C':
                pro[2,k2] = 1 + pro[2,k2]
            if tt[k1] == 'G':
                pro[3,k2] = 1 + pro[3,k2]
    pro = pro/len(arr)
    print("monomer logo for flank -----")
    return pro

def prob_16(arr):
    pro = np.zeros((16,1))
    for tt in arr:
        for i in range(16):
            if tt[0] + tt[3]  == dimer_16[i]:
                pro[i] = pro[i]+1
    pro = pro/len(arr)
    print("dimer logo for flanking -----")
    return pro


def prob_mid(arr):
    pro = np.zeros((16,1))
    for tt in arr:
        for i in range(16):
            if tt[1:3]  == dimer_16[i]:
                pro[i] = pro[i]+1
    pro = pro/len(arr)
    print("dimer logo for middle -----")
    return pro
import logomaker


def logo_plot(data,IC,ax,yaxis):

    if yaxis == "bits":
        H_i = -data*np.nan_to_num(np.log2(data))
        H_i = np.sum(H_i,axis=1)
        R_i = 2 - H_i
        for i in range(np.shape(R_i)[0]):
            data[i] = data[i]*R_i[i]  
    else:
        None
    
    s = 6
    ppm = pd.DataFrame(data, columns=['A', 'T', 'C','G'])
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = False


    color_scheme_mon = {
        'A': 'green' ,
        'T': 'red' ,
        'C': 'blue' ,
        'G': 'orange'
    }
    # create Logo object
    crp_logo = logomaker.Logo(ppm,
                              ax = ax,
                              color_scheme = color_scheme_mon,
                              shade_below=.5,
                              fade_below=.5,
                              font_name='Arial Rounded MT Bold')
    
    # style using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.ax.set_ylim(0,1.05)
#    crp_logo.ax.set_title(IC_coord_name(IC),size=s)
#
#    if which == "ends":
#        for x1 in [1.5]:
#            crp_logo.ax.axvline(x=x1,lw=0.5,color='k',ls='--')
#            crp_logo.ax.set_xticks([0.5,2.5])
#            crp_logo.ax.set_xticklabels([r'$\gamma_{XUVZ} \leq -\sigma$', r'$\sigma < \gamma_{XUVZ}$'])
#    else:
#        for x1 in [1.5,3.5,5.5]:
#            crp_logo.ax.axvline(x=x1,lw=0.5,color='k',ls='--')
#            crp_logo.ax.set_xticks([0.5,2.5,4.5,6.5])
#            crp_logo.ax.set_xticklabels([r'$\gamma_{XUVZ} \leq -\sigma$',r'$-\sigma < \gamma_{XUVZ} \leq 0$' , r'$0 < \gamma_{XUVZ} \leq \sigma$', r'$\sigma < \gamma_{XUVZ}$'])

    return ax


##################################---------------------------------------------
# plot Groovewidths
##################################---------------------------------------------
# In general, Major	Groove	wide	and	deep,	Minor	Groove	narrow	and	deep for BDNA 
# In ADNA, Major Groove	narrow	and	deep,	Minor	Groove	wide	and	shallow
# in cgDNA+ model, interplay between A nd B DNA -- ? 


def create_groovewidths_data(NA):
    if NA == 'DNA':
        ps = 'ps2_cgf'
    if NA == 'RNA':
        ps = 'ps_rna'
    if NA == 'HYB':
        ps = 'ps_hyb'
    seq_list = all_Nmers(10)
    orig_stdout = sys.stdout
    f = open('./Data/grooves_data/' + NA + 'grooves.txt','w')
    sys.stdout = f
    for seq in tqdm.tqdm(seq_list):
        seq = 'GC'+random_seq(4) + seq +random_seq(4)+ 'GC'  
        Dmin, Dmax = GrooveWidths_CS(cgDNA(seq,ps).ground_state)
        print(seq,np.around(Dmin, 3),np.around(Dmax, 3))
    sys.stdout = orig_stdout
    f.close()
    return None

def initiate_groove_data():
    DNA_grooves = pd.read_csv('./Data/grooves_data/DNAgrooves.txt', engine='python',sep=" ",header=None)
    RNA_grooves = pd.read_csv('./Data/grooves_data/RNAgrooves.txt', engine='python',sep=" ",header=None)
    HYB_grooves = pd.read_csv('./Data/grooves_data/HYBgrooves.txt', engine='python',sep=" ",header=None)

    gDNA = {'name': 'DNA', 'seq':DNA_grooves[0], 'minor': DNA_grooves[1].to_numpy(), 'major': DNA_grooves[2].to_numpy(), 'diff': DNA_grooves[2].to_numpy() - DNA_grooves[1].to_numpy()}
    gRNA = {'name': 'RNA', 'seq':RNA_grooves[0], 'minor': RNA_grooves[1].to_numpy(), 'major': RNA_grooves[2].to_numpy(), 'diff': RNA_grooves[2].to_numpy() - RNA_grooves[1].to_numpy()}
    gHYB = {'name': 'HYB', 'seq':HYB_grooves[0], 'minor': HYB_grooves[1].to_numpy(), 'major': HYB_grooves[2].to_numpy(), 'diff': HYB_grooves[2].to_numpy() - HYB_grooves[1].to_numpy()}
    
    return gDNA, gRNA, gHYB

def plot_corr_major_minor():

    gDNA, gRNA, gHYB = initiate_groove_data()
    fig = plt.figure(figsize=(8,8))
    s = 10
    gs1 = gridspec.GridSpec(530, 100)
    gs1.update(left=0.09, right=0.98,top=0.95,bottom=0.1)
    ax0 = plt.subplot(gs1[  0:150,   0:100])
    ax1 = plt.subplot(gs1[190:340,   0:100])
    ax2 = plt.subplot(gs1[380:530,   0:100])
    ax_list = [ax0,ax1,ax2]
    NA = [gDNA, gRNA, gHYB]
    NA_name = ["DNA", "RNA", "HYB"]
    for enum, ax in enumerate(ax_list):
        ax.scatter(NA[enum]['minor'],NA[enum]['major'],s=0.1)
        PC = scipy.stats.pearsonr(NA[enum]['minor'],NA[enum]['major'])
        PC = np.around(PC[0],2)
        ax.set_ylabel("Major Groove (in $\AA$)",fontsize=s)
        ax.tick_params(axis='both', labelsize=s,   pad=3, length=3,width=0.5,direction= 'inout')
        ax.set_title(NA_name[enum]+", PC = "+str(PC),fontsize=s)
    ax2.set_xlabel("Minor Groove (in $\AA$)",fontsize=s)
    fig.savefig("./Plots/Grooves_DNA_RNA_HYB_P_correlation.png",dpi=600)

    plt.show()
    plt.close()
    return None


def plot_groovewidths_all():

    gDNA, gRNA, gHYB = initiate_groove_data()

    for difference in [True,False]:
        fig = plt.figure(constrained_layout=False)
        gs1 = gridspec.GridSpec(500, 100)
        gs1.update(left=0.09, right=0.98,top=0.98,bottom=0.1)
        ax0 = plt.subplot(gs1[0:160,   0:100])
        ax1 = plt.subplot(gs1[167:330, 0:100],sharex=ax0)
        ax2 = plt.subplot(gs1[337:500, 0:100],sharex=ax0)
    
        ax_list = [ax0,ax1,ax2]
        Na = [gDNA, gRNA, gHYB]
        l,bins,fs = 1, 100,10    
        for ax, NA in zip(ax_list,Na):
            ax.hist(NA['minor'],histtype = 'step', color='red' ,lw=l,bins=bins,density=True,label = NA['name']+'_'+'Minor')
            ax.hist(NA['major'],histtype = 'step', color='blue',lw=l,bins=bins,density=True,label = NA['name']+'_''Major')
            if difference == True:
                ax.hist(NA['diff'],histtype = 'step', color='green',lw=l,bins=bins,density=True,label = 'Major - Minor')
            ax.set_ylabel("Norm. hist.")
            ax.set_xlabel("distance in $\AA$")
            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            ax.legend(fontsize=fs)
        plt.show()
        if difference == True:
            fig.savefig("./Plots/Grooves_DNA_RNA_HYB_diff"  +".pdf",dpi=600)
        else:
            fig.savefig("./Plots/Grooves_DNA_RNA_HYB"  +".pdf",dpi=600)
        plt.close()

    return None

def grooves_ind(NA,condition,groove_kind):
    cond_index = {}
    cond_len = len(condition[0])  
    seq_len = len(NA['seq'][0])
    if groove_kind == 'major':
        half_seq_len = seq_len//2 - 1  ### includinde the middle base-pair
        for mid in tqdm.tqdm(condition):
            tmp = []
            for enum, seq in enumerate(NA['seq']):
                if mid == to_YR(seq[half_seq_len:half_seq_len+cond_len]):
                    tmp.append(enum)
            cond_index[mid] = tmp
    elif groove_kind == 'minor':
        half_seq_len = seq_len//2 + 1  ### includinde the middle base-pair
        for mid in tqdm.tqdm(condition):
            tmp = []
            for enum, seq in enumerate(NA['seq']):
                if mid == to_YR(seq[half_seq_len-cond_len:half_seq_len]):
                    tmp.append(enum)
            cond_index[mid] = tmp
    return cond_index

def plot_groovewidths_with_seq_condition(hist=False,box=False,violin=False):
    NA_name = ['DNA','RNA','HYB']
    gDNA, gRNA, gHYB = initiate_groove_data()
    tmp_cond = all_YR(5)
    condition = []
    # following loop is to sort R,Y steps in a desired manner
    for kk in ['RYY','RRY','RRR',  'RYR',  'YRY', 'YYY', 'YRR','YYR']:
        for tmp in tmp_cond:
            # if tmp[-2:] == kk:   ## last two elements
            if tmp[1:4] == kk:   ## middle 3 elements
                condition.append(tmp)

    cond_index_minor = grooves_ind(gDNA,condition,'minor')
    cond_index_major = grooves_ind(gDNA,condition,'major')
    box=True
    hist=True
    
    if hist==True:
        fig = plt.figure(figsize=(7,8))
        gs1 = gridspec.GridSpec(500, 100)
        gs1.update(left=0.09, right=0.98,top=0.98,bottom=0.1)
        ax0 = plt.subplot(gs1[0:160,   0:100])
        ax1 = plt.subplot(gs1[167:330, 0:100],sharex=ax0)
        ax2 = plt.subplot(gs1[337:500, 0:100],sharex=ax0)
        ax_list = [ax0,ax1,ax2]
        Na = [gDNA, gRNA, gHYB]
        l,ls, bins,fs = 1,0.3, 100,10
        for ax, NA in zip(ax_list,Na):
            sns.kdeplot(NA['minor'],color='red' ,linewidth=l,label = NA['name']+'_'+'Minor',ax=ax)
            sns.kdeplot(NA['major'],color='blue',linewidth=l,label = NA['name']+'_''Major' ,ax=ax)
    
            ax.set_ylabel("Norm. hist.")
            ax.set_xlabel("distance in $\AA$")
            ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
            ax.legend(fontsize=fs)
    
            # for enum,mid in enumerate(dimer_16):
            #     sns.kdeplot(NA['minor'][cond_index_minor[mid]], color=color_16_RY[enum], label = '_no_legend_',linewidth=0.3,ax=ax)
            #     sns.kdeplot(NA['major'][cond_index_major[mid]], color=color_16_RY[enum], label = '_no_legend_',linewidth=0.3,ax=ax)
                            
        plt.show()
        fig.savefig("./Plots/Grooves_DNA_RNA_HYB"  +".pdf",dpi=600)
        plt.close()
    
    if box==True:
        Na = [gDNA, gRNA, gHYB]
        for enum1,NA in enumerate(Na):
            fig = plt.figure(figsize=(7,8))
            gs1 = gridspec.GridSpec(500, 100)
            gs1.update(left=0.09, right=0.98,top=0.98,bottom=0.1)
            ax0 = plt.subplot(gs1[0:220,    0:100])
            ax1 = plt.subplot(gs1[280:500,   0:100])
            ylabel = ['Minor groove width in $\AA$', 'Major groove width in $\AA$']
        #    ax_list = [ax0,ax1,ax2]
            data_min,data_max = [], []
            for enum,mid in enumerate(condition):
                data_min.append(NA['minor'][cond_index_minor[mid]])
                data_max.append(NA['major'][cond_index_major[mid]])
            ax0.boxplot(data_min)
            ax1.boxplot(data_max)
            
            for pnum,ax in enumerate([ax0,ax1]):
                ax.set_ylabel(ylabel[pnum])
                ax.set_xticks(1+np.arange(len(condition)))
                ax.set_xticklabels(condition,rotation=90)
            fig.savefig('./Plots/Grooves_box_plot_'+NA_name[enum1]+'.pdf',dpi=600)
            plt.show()
            plt.close()
    
    if violin==True:
        Na = [gDNA, gRNA, gHYB]
        for NA in Na:
            fig = plt.figure(figsize=(7,8))
            gs1 = gridspec.GridSpec(500, 100)
            gs1.update(left=0.09, right=0.98,top=0.98,bottom=0.1)
            ax0 = plt.subplot(gs1[0:500,   0:100])
        #    ax1 = plt.subplot(gs1[167:330, 0:100],sharex=ax0)
        #    ax2 = plt.subplot(gs1[337:500, 0:100],sharex=ax0)
        
        #    ax_list = [ax0,ax1,ax2]
            data_min,data_max = {}, {}
            for enum,mid in enumerate(dimer_16):
                tmp1, tmp2 = np.empty(4**10), np.empty(4**10)
                tmp1[:], tmp2[:] = np.nan, np.nan
                tmp1[cond_index[mid]] =  NA['minor'][cond_index_minor[mid]]
                tmp2[cond_index[mid]] =  NA['major'][cond_index_major[mid]]
                data_min[mid] = tmp1
                data_max[mid] = tmp2
            data_min['Avg'] = NA['minor']
            data_max['Avg'] = NA['major']
            df_min = pd.DataFrame.from_dict(data_min)
            df_max = pd.DataFrame.from_dict(data_max)
            sns.violinplot(data=df_max,ax=ax0,palette="muted",linewidte=0.01)
            sns.violinplot(data=df_min,ax=ax0,palette="muted",linewidte=0.01)
#            ax0.set_xticks(np.arange(17))
#            ax0.set_xticklabels(dimer_17,fontsize=fs)
            for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), color_16_RY):ticklabel.set_color(tickcolor)
            plt.show()
            fig.savefig("./Plots/Grooves_dimer_violin_"+ NA['name']  +".pdf",dpi=600)
            plt.close()

    return None



def compute_k_intervals_seq_logo(df,k,which):
    ### this is based on the quantile 
    # 0 quantile in min and 1 is max and 0.5 is median of the data
    intervals = np.zeros(k+1)
    for i in np.arange(k+1):
        intervals[i] = np.quantile(df[which],i/(k+1))
    return intervals

def compute_prob_list_of_seq(seqs,pos1,pos2):
    prob_mat = np.zeros((5,4))  ### A,T,C,G
    for seq in seqs:
        for enum,base  in enumerate(seq[pos1:pos2]):
            if base == 'A':
                prob_mat[enum,0] = prob_mat[enum,0] + 1 
            if base == 'T':
                prob_mat[enum,1] = prob_mat[enum,1] + 1 
            if base == 'C':
                prob_mat[enum,2] = prob_mat[enum,2] + 1 
            if base == 'G':
                prob_mat[enum,3] = prob_mat[enum,3] + 1 
    prob_mat = prob_mat/np.sum(prob_mat[0,:])  ### division by numer of seq
    return prob_mat  

def assign_seq_to_intervals(df,k,which,pos1,pos2):
    interval = compute_k_intervals_seq_logo(df,k,which)
    comp_prob_mat = []
    for i in range(k):
        tmp_ind = np.where(np.logical_and(df[which]>=interval[i], df[which]<=interval[i+1]))[0]
        tmp_seq_list = df['seq'][tmp_ind]
        comp_prob_mat.append(compute_prob_list_of_seq(tmp_seq_list,pos1,pos2)) 
    return comp_prob_mat

def compute_prob_seq_logo(df,label):
    mpos1,mpos2 = 7,12   ####idea is to include central dimer, minor groove  (7:12)
    Mpos1,Mpos2 = 10,15   ####idea is to include central dimer, major groove (10:15)
    comp_prob_minor = np.array(assign_seq_to_intervals(df,12,'minor',mpos1,mpos2))
    comp_prob_major = assign_seq_to_intervals(df,12,'major',Mpos1,Mpos2)

    q = list(itertools.product(np.arange(3), np.arange(4)))
    fig,ax = plt.subplots(3,4)
    for enum,data in enumerate(comp_prob_minor):
        logo_plot(data,'IC',ax[q[enum][0],q[enum][1]],'bits')
        ax[q[enum][0],q[enum][1]].tick_params(axis='both', labelsize=4, pad=3,length=3,width=0.5,direction= 'inout')

#        ax[q[enum][0],q[enum][1]].set_title("Minor interval "+str(1+enum))
    plt.show()
    fig.savefig("./Plots/"+label+"_minor_groove_seq_logo.pdf",dpi=600)
    plt.close()      
    fig,ax = plt.subplots(3,4)
    for enum,data in enumerate(comp_prob_major):
        logo_plot(data,'IC',ax[q[enum][0],q[enum][1]],'bits')
        ax[q[enum][0],q[enum][1]].tick_params(axis='both', labelsize=4, pad=3,length=3,width=0.5,direction= 'inout')
#        ax[q[enum][0],q[enum][1]].set_title("Major interval "+str(1+enum))
    plt.show()
    fig.savefig("./Plots/"+label+"_major_groove_seq_logo.pdf",dpi=600)
    plt.close()      
    return None

def plot_groovewidths_seq_logo():
    gDNA, gRNA, gHYB = initiate_groove_data()

    compute_prob_seq_logo(gDNA,'DNA')
    compute_prob_seq_logo(gRNA,'RNA')
    compute_prob_seq_logo(gHYB,'HYB')
#    print(gDNA['seq'])
#    ['YR', 'RR', 'YY', 'RY']


#############################################################################
################################### KL Envelope ####################################
###################################################################################
def plot_envelope_KL(ax,mean=0,std=1,lw=0.5,color='red'):
    x_values = np.arange(-4, 4, 0.15)
    y_values = scipy.stats.norm(mean, std)
    
    ax.plot(x_values, y_values.pdf(x_values),lw,color=color)
    return ax

def KL_single(m1,s1,m2,s2):
    K = np.log(s2/s1) + -0.5 + (s1**2 + (m1-m2)**2)/(2*s2**2)
    return K

def envelope_KL():
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(1050, 900)
    gs1.update(left=0.09, right=0.98,top=0.92,bottom=0.1)

    ax00 = plt.subplot(gs1[1:250, 0:400])
    ax01 = plt.subplot(gs1[1:250, 500:900],sharex=ax00)

    ax10 = plt.subplot(gs1[400:650, 0:400])
    ax11 = plt.subplot(gs1[400:650, 500:900],sharex=ax00)

    ax20 = plt.subplot(gs1[800:1050, 0:400])
    ax21 = plt.subplot(gs1[800:1050, 500:900],sharex=ax00)

    what = [0.0001, 0.0005,0.001,0.005,0.01,0.05]
    ax_list = [ax00,ax01,ax10,ax11,ax20,ax21]
    around = [4,4,3,3,2,2]
    ss = [600000,600000,60000,60000,6000,6000]
    for ax,ar,KL,s in zip(ax_list,around,what,ss):
        count=0
        e = np.arange(-0.5,0.5,0.00001) + 0.000000001
        for i in range(s):
            e1 = random.choice(e)
            e2 = 1+ random.choice(e)
            u = 0.5*(KL_single(0,1,e1,e2) + KL_single(e1,e2,0,1))
    
            if KL == np.around(u,ar):
                count=count+1
                plot_envelope_KL(ax,e1,e2,0.02)
        print(count)
    
        plot_envelope_KL(ax,0,1,.025,color='k')
        ax.set_title("sym KL divergence = " + str(KL),fontsize=8)
    plt.show()
    fig.savefig("./Plots/KL_envelope.pdf",dpi=600)
    plt.close()
    return None

###################################################################################
#########  Difference between two MD protocols ####################################
###################################################################################

def compare_MD(DNA1,DNA2):
    A,B = [],[]
    for i in range(np.size(DNA2.seq)):
        A.append(difference_KL_sym(DNA1.choose_seq([i]),DNA2.choose_seq([i]),True))
        B.append(difference_Mahal_sym(DNA1.choose_seq([i]),DNA2.choose_seq([i]),True))
    return A,B

def compare_ps(DNA,ps1,ps2):
    A,B = [],[]
    for seq in DNA.seq:
        res1 = cgDNA(seq,ps1)
        res2 = cgDNA(seq,ps2)
        A.append(difference_KL_sym(res1,res2,True)[0,0])
        B.append(difference_Mahal_sym(res1,res2,True)[0,0])
    return A,B

def palin_training_err(data_tmp,ps):
    a,b= [],[]
    for seq in range(0,16):
        a.append(recons_err_KL_sym(data_tmp.choose_seq([seq]),ps,DNA_sym[seq])[0,0])
        b.append(recons_err_Mahal_sym(data_tmp.choose_seq([seq]),ps,DNA_sym[seq])[0,0])
    return a,b

def make_table(data,label):
    data = np.around(data,4)
    for i in range(len(data[0])):
        if i==0:
            print("Index" ,'&', label[0],'&', label[1], '&', label[2],'&',  label[3], '&',  label[4], '&',  label[5], '&', label[6],'&',  label[7], '&',  label[8], '&',  label[9], '\\\\')  
            print("\\hline")
        print(i+1,'&', data[0][i],'&', data[1][i], '&', data[2][i],'&',  data[3][i], '&',  data[4][i], '&',  data[5][i], '&', data[6][i],'&',  data[7][i], '&',  data[8][i], '&',  data[9][i], '\\\\')     
    print("\\hline")
    avg = np.average(data,axis=1)
    avg = np.around(avg,4)
    print("Average",'&', avg[0],'&', avg[1], '&', avg[2],'&',  avg[3], '&',  avg[4], '&',  avg[5], '&', avg[6],'&',  avg[7], '&',  avg[8], '&',  avg[9], '\\\\')     



def make_table2(data,label):
    avg = np.average(data,axis=1)
    data = np.around(data,4)
    avg = np.around(avg,4)
    print("Index", '&', 1, '&', 2, '&', 3, '&', 4, '&', 5, '&', 6, '&', 7, '&', 8, '&', 9, '&', 10, '&', 11, '&', 12, '&', 13, '&', 14, '&', 15, '&', 16, '&', 'Avg' , '\\\\', "\\hline")
    for enum,i in enumerate(data):
        print(label[enum], '&', data[enum][0], '&', data[enum][1], '&', data[enum][2], '&', data[enum][3], '&', data[enum][4], '&', data[enum][5], '&', data[enum][6], '&', data[enum][7], '&', data[enum][8], '&', data[enum][9], '&', data[enum][10], '&', data[enum][11], '&', data[enum][12], '&', data[enum][13] , '&', data[enum][14] , '&', data[enum][15] , '&', avg[enum] , "\\\\" , "\\hline"             )


def print_palin_err(data_name,DNA_217):
    print("Index", '&', 1, '&', 2, '&', 3, '&', 4, '&', 5, '&', 6, '&', 7, '&', 8, '&', 9, '&', 10, '&', 11, '&', 12, '&', 13, '&', 14, '&', 15, '&', 16, '&', 17, '\\\\', "\\hline")
    print( '&',  '&',  '&',  '&', '&',  '&',  '&', 'SM', '&', '&',  '&',  '&',  '&',  '&',  '&', '&', '&', '&', '\\\\', "\\hline")

    for f in np.arange(1,11,1):
        mat3 = []
        data_path_tmp = data_name.replace('XX',str(f))
        data_tmp = init_MD_data().load_data(data_path_tmp)
        for seq in np.arange(0,17,1):
            mat3.append(palin_err_Mahal_sym(data_tmp.choose_seq([seq])))
        avg3 = np.around(np.mean(mat3),4)
        mat3 = np.around(mat3,4)
        print(f, '$\\mu$s', '&', mat3[0], '&', mat3[1], '&', mat3[2], '&', mat3[3], '&', mat3[4], '&', mat3[5], '&', mat3[6], '&', mat3[7], '&', mat3[8], '&', mat3[9], '&', mat3[10], '&', mat3[11], '&', mat3[12], '&', mat3[13], '&', mat3[14], '&', mat3[15], '&', mat3[16] , '\\\\', "\\hline")
    tmp = palin_err_Mahal_sym(DNA_217.choose_seq([0]))
    print(20, '$\\mu$s', '&',  '&',  '&',  '&', '&',  '&',  '&',  '&', '&',  '&',  '&',  '&',  '&',  '&', '&', '&', '&', np.around(tmp,4) , '\\\\', "\\hline")
    print( '&',  '&',  '&',  '&', '&',  '&',  '&', 'SKL', '&', '&',  '&',  '&',  '&',  '&',  '&', '&', '&', '&', '\\\\', "\\hline")

    for f in np.arange(1,11,1):
        mat4 = []
        data_path_tmp = data_name.replace('XX',str(f))
        data_tmp = init_MD_data().load_data(data_path_tmp)
        for seq in np.arange(0,17,1):
            mat4.append(palin_err_KL_sym(data_tmp.choose_seq([seq])))
        avg4 = np.around(np.mean(mat4),4)
        mat4 = np.around(mat4,4) 

        print(f, '$\\mu$s', '&', mat4[0], '&', mat4[1], '&', mat4[2], '&', mat4[3], '&', mat4[4], '&', mat4[5], '&', mat4[6], '&', mat4[7], '&', mat4[8], '&', mat4[9], '&', mat4[10], '&', mat4[11], '&', mat4[12], '&', mat4[13], '&', mat4[14], '&', mat4[15], '&', mat4[16] , '\\\\', "\\hline")
    tmp = palin_err_KL_sym(DNA_217.choose_seq([0]))
    print(20, '$\\mu$s', '&',  '&',  '&',  '&', '&',  '&',  '&',  '&', '&',  '&',  '&',  '&',  '&',  '&', '&', '&', '&', np.around(tmp,4) , '\\\\', "\\hline")

    return None


def print_end_seq(DNA_ends):
    for i in range(15):
        print(4*i+1, '&', '${\Sfont', DNA_ends.seq[4*i], '}$', '&',  4*i+2, '&','${\Sfont', DNA_ends.seq[4*i+1], '}$', '&',  4*i+1+2, '&', '${\Sfont', DNA_ends.seq[4*i+2], '}$', '&',  4*i+1+3, '&', '${\Sfont', DNA_ends.seq[4*i+3], '}$',        '\\\\')


def hist_seq_training_lib(dna):
    seq_list = dna.choose_seq(np.arange(16)).seq
    mono = dict.fromkeys(['A','G'], 0)
    dimer_dict = dict.fromkeys(dimer_10, 0)
    trimer_list = ['AAA', 'AAT', 'AAC', 'AAG', 'TAA', 'TAT', 'TAC', 'TAG', 'CAA', 'CAT', 'CAC', 'CAG', 'GAA', 'GAT', 'GAC', 'GAG', 'AGA', 'AGT','AGC', 'AGG', 'TGA', 'TGT', 'TGC', 'TGG', 'CGA', 'CGT', 'CGC', 'CGG', 'GGA', 'GGT', 'GGC', 'GGG']
    trimer_dict = dict.fromkeys(trimer_list, 0)

    for seq in seq_list:
        for k in list(mono.keys()):
            mono[k] = mono[k] + seq[2:22].count(k)

        for u in list(dimer_dict.keys()):
            dimer_dict[u] = dimer_dict[u] + seq[2:22].count(u) + seq[2:22].count(comp(u)) 

        for u in list(trimer_dict.keys()):
            trimer_dict[u] = trimer_dict[u] + seq[1:23].count(u)  + seq[1:23].count(comp(u)) 

    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(500, 500)
    gs1.update(left=0.09, right=0.98,top=0.97,bottom=0.12)
    
    ax0 = plt.subplot(gs1[0 :215,    0:160])
    ax1 = plt.subplot(gs1[0 :215,    210:500])
    ax2 = plt.subplot(gs1[285 : 500, 0:500   ])

    ax0.bar(np.arange(2), list(mono.values()) , color = 'grey', width = 0.25)
    ax1.bar(np.arange(10), list(dimer_dict.values()) , color = 'grey', width = 0.25)
    ax2.bar(np.arange(32), list(trimer_dict.values())[0:32] , color = 'grey', width = 0.25)

    fs2 = 10
    ax0.set_xticks(np.arange(2))
    ax0.set_xticklabels(mono.keys(),fontsize=fs2)

    ax1.set_xticks(np.arange(10))
    ax1.set_xticklabels(dimer_dict.keys(),fontsize=fs2,rotation=90)


    ax2.set_xticks(np.arange(32))
    ax2.set_xticklabels(list(trimer_dict.keys())[0:32],fontsize=fs2,rotation=90)
    ax2.set_xlim(-0.5,31.5)

    ax_list = [ax0,ax1,ax2]
    for axl in ax_list:
        axl.tick_params(axis='both', labelsize=fs2, pad=3,length=3,width=0.5,direction= 'inout')

    ax0.text(-0.23, 1, '(a)', transform=ax0.transAxes, size=12)#, weight='bold')
    ax1.text(-0.12, 1, '(b)', transform=ax1.transAxes, size=12)#, weight='bold')
    ax2.text(-0.07, 1.02, '(c)', transform=ax2.transAxes, size=12)#, weight='bold')

    plt.show()
    fig.savefig("./Plots/hist_freq_DNA.pdf",dpi=600)
    plt.close()


    return None



##########---- plot stiffness matrix in 3 coordinates

##################################---------------------------------------------
# plot stencil in stiffness matrix
##################################---------------------------------------------

def fit_stencil_in_matrix_3types_combine(data,sym,save_name):
    if hasattr(data, 's1b'):
        if sym==True:
            wc = copy.deepcopy(data.s1b_sym[0])
        else:
            wc = copy.deepcopy(data.s1b[0])
        nbp = data.nbp[0]
    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp
    seq = data.seq[0]
    wc = np.nan_to_num(wc)
    ind_inter = np.array([24*i+j+12 for i in range(23) for j in range(6)])
    ind_cg = np.array([12*i+j for i in range(24*2-1) for j in range(6)])
    ix_inter =  np.ix_(ind_inter,ind_inter)
    ix_cg    =  np.ix_(ind_cg,ind_cg)
    wc_cov = np.array(np.linalg.inv(wc))
    
    wc_inter = np.linalg.inv(wc_cov[ix_inter])
    wc_cg = np.linalg.inv(wc_cov[ix_cg])



##########----------------------------##################
    labelsize = 8
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(600, 106)
    gs1.update(left=0.03, right=0.98,top=0.995,bottom=0.005)
    ax1 = plt.subplot(gs1[0:600,         0:30])
    ax2 = plt.subplot(gs1[0:600,        35:65])
    ax0 = plt.subplot(gs1[0:600,       70:106])

    x1,x2 = stencil_42(nbp)
    x6 = np.arange(0,24*nbp-12,6)
    shp = int(np.shape(wc)[0]/2)
    sns.heatmap(wc[0:shp,0:shp],ax=ax0,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=1,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    lwl,lws = 0.6,0.05
    for i in range(len(x1)):
        ax0.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax0.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax0.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        ax0.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for uu in [0,shp]:
        ax0.axhline(y=uu, color='k',linewidth=0.5)
        ax0.axvline(x=uu, color='k',linewidth=0.5)

#    for j in x6:
#        axr.axvline(x=j,color='black', lw=lws)
#        axr.axhline(y=j,color='black', lw=lws)
    ind = np.array([24*i+3 for i in range(int(nbp/2))])
#    ind[1::] = ind[1::]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    ax0.set_xticks(ind)         
    ax0.set_yticks(ind)    
    ax0.set_xticklabels(seq_ind)
    ax0.set_yticklabels(seq_ind)
    ax0.set_aspect('equal', adjustable='box')
    ax0.tick_params(axis='both', labelsize=labelsize, pad=3,length=3,width=0.5,direction= 'inout',rotation=0)
    ax0.grid(True)


##########----------------------------##################
    lwl,lws = 0.85,0.05
    x1,x2 = stencil_42(nbp,'inter')
    shp = int(np.shape(wc_inter)[0]/2)
    for i in range(len(x1)):
        ax1.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax1.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax1.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        ax1.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    sns.heatmap(wc_inter[0:shp,0:shp],ax=ax1,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=0,square=True,cbar_kws={"shrink": .25,"pad":0.018})

    ind = [6*i for i in range(int(nbp/2))]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    ax1.set_xticks(ind)         
    ax1.set_yticks(ind)    
    ax1.set_xticklabels(seq_ind)
    ax1.set_yticklabels(seq_ind)
    ax1.set_aspect('equal', adjustable='box')
    ax1.tick_params(axis='both', labelsize=labelsize, pad=3,length=3,width=0.5,direction= 'inout',rotation=0)
    for uu in [0,shp]:
        ax1.axhline(y=uu, color='k',linewidth=0.5)
        ax1.axvline(x=uu, color='k',linewidth=0.5)



##########----------------------------##################
    lwl,lws = 0.7,0.05
    shp = int(np.shape(wc_cg)[0]/2)
    sns.heatmap(wc_cg[0:shp,0:shp],ax=ax2,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=0,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    x1,x2 = stencil_42(nbp,'cg')
    for i in range(len(x1)):
        ax2.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax2.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        ax2.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        ax2.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for uu in [0,shp]:
        ax2.axhline(y=uu, color='k',linewidth=0.5)
        ax2.axvline(x=uu, color='k',linewidth=0.5)
    ind = [12*i+3 for i in range(int(nbp/2))]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    ax2.set_xticks(ind)         
    ax2.set_yticks(ind)    
    ax2.set_xticklabels(seq_ind)
    ax2.set_yticklabels(seq_ind)
    ax2.set_aspect('equal', adjustable='box')
    ax2.tick_params(axis='both', labelsize=labelsize, pad=3,length=3,width=0.5,direction= 'inout',rotation=0)
    
    plt.savefig('./Plots/'+save_name+'_combine.pdf',dpi=600)
    plt.close()
    print('done')



def fit_stencil_in_matrix_3types(data,sym,save_name):
    if hasattr(data, 's1b'):
        if sym==True:
            wc = copy.deepcopy(data.s1b_sym[0])
        else:
            wc = copy.deepcopy(data.s1b[0])
        nbp = data.nbp[0]
    elif hasattr(data, 'stiff'):
        wc = copy.deepcopy(data.stiff.todense())
        nbp = data.nbp
    seq = data.seq[0]
    wc = np.nan_to_num(wc)
    ind_inter = np.array([24*i+j+12 for i in range(23) for j in range(6)])
    ind_cg = np.array([12*i+j for i in range(24*2-1) for j in range(6)])
    ix_inter =  np.ix_(ind_inter,ind_inter)
    ix_cg    =  np.ix_(ind_cg,ind_cg)
    wc_cov = np.array(np.linalg.inv(wc))
    
    wc_inter = np.linalg.inv(wc_cov[ix_inter])
    wc_cg = np.linalg.inv(wc_cov[ix_cg])

##########----------------------------##################
    x1,x2 = stencil_42(nbp)
    x6 = np.arange(0,24*nbp-12,6)
    fig,axr = plt.subplots(1)
    shp = int(np.shape(wc)[0]/2)
    sns.heatmap(wc[0:shp,0:shp],ax=axr,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=1,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    lwl,lws = 0.6,0.05
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for uu in [0,shp]:
        axr.axhline(y=uu, color='k',linewidth=0.5)
        axr.axvline(x=uu, color='k',linewidth=0.5)

#    for j in x6:
#        axr.axvline(x=j,color='black', lw=lws)
#        axr.axhline(y=j,color='black', lw=lws)
    ind = np.array([24*i+3 for i in range(int(nbp/2))])
#    ind[1::] = ind[1::]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    axr.set_xticks(ind)         
    axr.set_yticks(ind)    
    axr.set_xticklabels(seq_ind)
    axr.set_yticklabels(seq_ind)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=10, pad=3,length=3,width=0.5,direction= 'inout')
    axr.grid(True)
    plt.savefig('./Plots/'+save_name+'_cg+.pdf',dpi=600)


##########----------------------------##################
    lwl,lws = 0.85,0.05
    fig,axr = plt.subplots(1)
    x1,x2 = stencil_42(nbp,'inter')
    shp = int(np.shape(wc_inter)[0]/2)
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    sns.heatmap(wc_inter[0:shp,0:shp],ax=axr,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=0,square=True,cbar_kws={"shrink": .25,"pad":0.018})

    ind = [6*i for i in range(int(nbp/2))]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    axr.set_xticks(ind)         
    axr.set_yticks(ind)    
    axr.set_xticklabels(seq_ind)
    axr.set_yticklabels(seq_ind)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=10, pad=3,length=3,width=0.5,direction= 'inout')
    for uu in [0,shp]:
        axr.axhline(y=uu, color='k',linewidth=0.5)
        axr.axvline(x=uu, color='k',linewidth=0.5)

    plt.savefig('./Plots/'+save_name+'_inter.pdf',dpi=600)
    plt.close()


##########----------------------------##################
    lwl,lws = 0.7,0.05
    fig,axr = plt.subplots(1)
    shp = int(np.shape(wc_cg)[0]/2)
    sns.heatmap(wc_cg[0:shp,0:shp],ax=axr,center=0,vmax=20,vmin=-20,cmap='seismic',cbar=0,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    x1,x2 = stencil_42(nbp,'cg')
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for uu in [0,shp]:
        axr.axhline(y=uu, color='k',linewidth=0.5)
        axr.axvline(x=uu, color='k',linewidth=0.5)
    ind = [12*i+3 for i in range(int(nbp/2))]
    seq_ind = [ss for ss in seq[0:int(nbp/2)]]
    axr.set_xticks(ind)         
    axr.set_yticks(ind)    
    axr.set_xticklabels(seq_ind)
    axr.set_yticklabels(seq_ind)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=10, pad=3,length=3,width=0.5,direction= 'inout')

    plt.savefig('./Plots/'+save_name+'_cg.pdf',dpi=600)
    plt.close()
    print('done')



def compare_seq_for_palindrome_article(d,save_name):
#    lss=['-', '-','-','--','--','--']
    lss=['-', '--','-','--',':',':']
    a = d.shape[19]
    b = d.shape[20]
    aseq = d.seq[19]
    bseq = d.seq[20]
    ar = cgDNA(aseq,'ps2_cgf').ground_state
    br = cgDNA(bseq,'ps2_cgf').ground_state

    ar2 = cgDNA(aseq,'dna_mle').ground_state
    br2 = cgDNA(bseq,'dna_mle').ground_state

    shp = [a,ar,b,br,ar2,br2]
    seq = [aseq]*6
    color = ['red','blue','green','maroon','dodgerblue','limegreen']
    c1 = ['red','blue','green']
    c2 = ['maroon','dodgerblue','limegreen']

    fig, ax_list = compare_shape(shp,seq,save_name+'_19_20',lss,color=[c1,c1,c2,c2,c1,c2],type_of_var='cg+')
    s1 = 'GCGGATTACGCAGGC'
    s2 = 'GCGGATTCCGCAGGC'
    sf = list(s1)
    sf[7] = 'A/C'
    seq_l = [sf,sf]
    for enum,a in enumerate([8,9]):
        print(seq_l[enum])
#        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=True)
#        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=8,minor=True)
        ax_list[a].set_xticks(np.arange(len(seq_l[enum])),minor=False)
        ax_list[a].set_xticklabels(list(seq_l[enum]),fontsize=10,minor=False)
        ax_list[a].tick_params(axis='x', which='major', pad=4)
#        ax_list[a].tick_params(axis='x', which='minor', pad=2)
        [t.set_color('k')  for t in ax_list[a].xaxis.get_ticklabels(minor=False)]
#        [t.set_color('maroon') for t in ax_list[a].xaxis.get_ticklabels(minor=True)]

    fig.savefig("./Plots/X" + save_name +".pdf",dpi=600)


    
def plot_persistence_length_palindrome_article(names):
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(100, 100)
    gs1.update(left=0.1, right=0.97,top=0.99,bottom=0.09)

    ax0 = plt.subplot(gs1[0:47,       0:100])
    ax1 = plt.subplot(gs1[53:100,      0:100])

    indices_dim = [0,3,6,9,12,14]
    color_dim = ['cyan','orange','red','blue','gray','magenta']
    count=0
    data,data_sp  = {},{}
    fs = 8
    for n in names:
        path = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/' + n
        data[n] = read_persist_data_sub(path)
        data_sp[n], dimer_files = read_special_persist_data(path)
        print(names,max(data[n][0]),max(data[n][1]),min(data[n][0]),min(data[n][1]), )
    ax0.hist(data['DNA_BSTJ_MLE'][0],histtype = 'step',bins = 1000,color=colo[count] ,lw=1, label = "$\ell_{p}^{\mathcal{P}2}$"  ,density=1 )
    ax0.hist(data['DNA_BSTJ_MLE'][1],histtype = 'step',bins = 500,color=colo[count+1],lw=1, label = "$\ell_{d}^{\mathcal{P}2}$" ,density=1  )
#            ['AA', 'TT', 'GC', 'CG', 'TC', 'CT', 'TG', 'GT', 'CC', 'GG', 'AC', 'CA', 'AG', 'GA', 'AT', 'TA']
    for j in range(6):
        ax0.scatter(data_sp['DNA_BSTJ_MLE'][indices_dim[j],0],[0.165],color=color_dim[j],marker='^', s=20 )
        ax0.scatter(data_sp['DNA_BSTJ_MLE'][indices_dim[j],1],[0.173],color=color_dim[j],marker='o',s=10 )
        print(dimer_files[indices_dim[j]], data_sp['DNA_BSTJ_MLE'][indices_dim[j],0],data_sp['DNA_BSTJ_MLE'][indices_dim[j],1])


    ax1.hist(data['DNA_BSTJ_CGF'][0] - data['DNA_BSTJ_MLE'][0] ,histtype = 'step',bins = 1000,color=colo[0] ,lw=1, label = "$\ell_{p}^{\mathcal{P}1} - \ell_{p}^{\mathcal{P}2}$"  ,density=1 )
    ax1.hist(data['DNA_BSTJ_CGF'][1] - data['DNA_BSTJ_MLE'][1] ,histtype = 'step',bins = 500,color=colo[0+1],lw=1, label = "$\ell_{d}^{\mathcal{P}1} - \ell_{d}^{\mathcal{P}2}$" ,density=1  )


    for j in range(6):
        ax1.scatter(data_sp['DNA_BSTJ_CGF'][indices_dim[j],0] - data_sp['DNA_BSTJ_MLE'][indices_dim[j],0],[0.39],color=color_dim[j],marker='^', s=20 )
        ax1.scatter(data_sp['DNA_BSTJ_CGF'][indices_dim[j],1] - data_sp['DNA_BSTJ_MLE'][indices_dim[j],1],[0.41],color=color_dim[j],marker='o',s=10 )
        print(dimer_files[j])
    ax0.scatter(np.mean(data['DNA_BSTJ_MLE'][0]),[0.004],color='k',marker='^', s=20 )
    ax0.scatter(np.mean(data['DNA_BSTJ_MLE'][1]),[0.004],color='k',marker='o',s=10 )
    ax1.scatter(np.mean(data['DNA_BSTJ_CGF'][0] - data['DNA_BSTJ_MLE'][0]),[0.009],color='k',marker='^', s=20 )
    ax1.scatter(np.mean(data['DNA_BSTJ_CGF'][1] - data['DNA_BSTJ_MLE'][1]),[0.009],color='k',marker='o',s=10 )

    ax1.set_xlim(0,35)
#    ax0.set_ylim(0,0.21)
    for ax in [ax0,ax1]:
        ax.legend(fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
    ax1.set_xlabel("Base pair",fontsize=fs)


    ax0.set_yticks(np.arange(0,0.2,0.1))    
    ax0.set_yticklabels(np.arange(0,0.2,0.1))
    ax1.set_yticks(np.arange(0,0.5,0.1))    
    ax1.set_yticklabels(np.around(np.arange(0,0.5,0.1),1))

    fig.set_size_inches(3.25, 4)

    plt.savefig('./Plots/persistence_length_palin_article'+".pdf",dpi=600)
    plt.show()
    return data



####################################################################
###### Create and plot files for TTC correlation plot
####################################################################
def run_and_save_ttc(A,Drop_base,seq_id,label):
#    _,ttc,tt0 = A.MonteCarlo(1000000,Drop_base)
    _,ttc,tt0 = A.MonteCarlo(100000,Drop_base)
    tt = np.array(np.vstack([ttc,tt0])).T
    np.savetxt('./Data/'+label+'_'+str(seq_id+1)+'.txt',tt,fmt='%1.6f', delimiter=',')
    return tt

def print_TTC_files(NA,Drop_base,label):
#    for seq_id in range(23):
    for seq_id in range(17):
        MD = NA.choose_seq([seq_id])
        seq = MD.seq[0]        
        A = cgDNA(seq,'ps2_cgf')
#        run_and_save_ttc(A,Drop_base,seq_id,label+'_ME')

        B = cgDNA(seq,'dna_mle')
#        run_and_save_ttc(B,Drop_base,seq_id,label+'_MLE')

        if comp(seq) == seq:
            print(seq_id, seq,'Palindromess--------')
            C = A
            C.ground_state =  MD.shape_sym[0]
#            C.stiff =  MD.s1b_sym[0]
#            run_and_save_ttc(B,Drop_base,seq_id,label+'_full_MD')
#            C.stiff =  MD.stiff_me_sym[0]
#            run_and_save_ttc(C,Drop_base,seq_id,label+'_trunc_MD')
            C.stiff =  MD.stiff_mre[0]
            run_and_save_ttc(C,Drop_base,seq_id,label+'_trunc_mre_MD')
        else:
            C = A
            C.ground_state =  MD.shape[0]
            C.stiff =  MD.s1b[0]
            run_and_save_ttc(C,Drop_base,seq_id,label+'_full_MD')

            C.stiff =  MD.stiff_me[0]
            run_and_save_ttc(C,Drop_base,seq_id,label+'_trunc_MD')
            print(seq_id, seq,'NOT Palindromess--------')
        print(seq_id,' --- Done' )

def insert_zero(df):
    df.loc[-1] = [0,0]
    df.index = df.index + 1
    df.sort_index(inplace=True) 
    return df

def plot_TTC_files(NA,label,seq_id):
    path = '/Users/rsharma/Dropbox/cgDNAplus_py_rahul/Data/ttc_data/'
    P1 = pd.read_csv(path+'ttc_'+label+'_ME_'+str(seq_id)+'.txt',header=None)
    P2 = pd.read_csv(path+'ttc_'+label+'_MLE_'+str(seq_id)+'.txt',header=None)
    Gau = pd.read_csv(path+'ttc_'+label+'_full_MD_'+str(seq_id)+'.txt',header=None)
    T2 = pd.read_csv(path+'ttc_'+label+'_trunc_MD_'+str(seq_id)+'.txt',header=None)
    T1 = pd.read_csv(path+'sym_ttc_'+label+'_trunc_mre_MD_'+str(seq_id)+'.txt',header=None)

#    Unsym_df3 = pd.read_csv(path+'Unsym_ttc_'+label+'_full_MD_'+str(seq_id)+'.txt',header=None)
#    Unsym_df4 = pd.read_csv(path+'Unsym_ttc_'+label+'_trunc_MD_'+str(seq_id)+'.txt',header=None)

    test = pd.read_csv(path+'Alessandro_17_ME.txt',header=None)
    test = insert_zero(np.log(test))

    P1 = insert_zero(np.log(P1))
    P2 = insert_zero(np.log(P2))
    Gau = insert_zero(np.log(Gau))
    T2 = insert_zero(np.log(T2))
    T1 = insert_zero(np.log(T1))
#    Unsym_full  = insert_zero(np.log(Unsym_df3))
#    Unsym_trunc = insert_zero(np.log(Unsym_df4))

    md_f = pd.read_csv(path+'log_filter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)
    md_unf = pd.read_csv(path+'log_unfilter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)    

    md_f[1] = Gau[1]
    md_unf[1] = Gau[1]

    ### note that sym or unsym version of MD ttc plots are identical
    sym_md_f = pd.read_csv(path+'sym_log_filter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)
    sym_md_unf = pd.read_csv(path+'sym_log_unfilter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)    
    sym_md_f[1] = Gau[1]
    sym_md_unf[1] = Gau[1]


    ind = np.arange(len(md_f)) + 1
    fig,ax = plt.subplots()
    label = ['$\mathcal{P}1$','$\mathcal{P}2$','$MD_{Gaussian}$','$MD_{filtered}$','$MD_{unfiltered}$']
    col = ['g','r','k','lime','royalblue']
    for enum,data in enumerate([P1,P2,Gau,sym_md_f,sym_md_unf]):
#    for enum,data in enumerate([ME,md_f]):
        mp, _ = np.polyfit(ind, data[0], 1)
        lp = str(np.around(-1/mp,1))
        md, _ = np.polyfit(ind, data[0]-data[1], 1)
        ld = str(np.around(-1/md,1))
        ax.plot(ind,data[0],label=label[enum]+' '+ ', $\ell_{p}$ = ' + lp + ', $\ell_{d}$ = ' + ld  ,color=col[enum],lw=0.5) 
        ax.plot(ind,data[0]-data[1],ls='--',label='_no_legend_',color=col[enum],lw=0.5)
    ax.legend(fontsize=8)
    ax.set_xticks(ind)
    ax.set_xticklabels(list(ind),fontsize=8)
    ax.set_title(NA.seq[seq_id-1])
    ax.set_ylabel(r"$ln langle rangle $")
    ax.tick_params(axis='both', labelsize=8,pad=3,length=3,width=0.5,direction= 'inout')
    plt.savefig('./Plots/ttc_plot_seq_'+str(seq_id)+'.pdf',dpi=600)
    None


def plot_TTC_files_all_ld(NA,label):
    nseq = 23
    ld_comp = np.zeros((nseq,5))
    for seq_id in range(1,nseq+1):
        path = '/Users/rsharma/Dropbox/cgDNAplus_py_rahul/Data/ttc_data/'
        P1 = pd.read_csv(path+'ttc_'+label+'_ME_'+str(seq_id)+'.txt',header=None)
        P2 = pd.read_csv(path+'ttc_'+label+'_MLE_'+str(seq_id)+'.txt',header=None)
        P1 = insert_zero(np.log(P1))
        P2 = insert_zero(np.log(P2))
        ### note that the sequcnes which are not palindromce -- sym is non-palindromce data
        sym_md_f = pd.read_csv(path+'sym_log_filter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)
        sym_md_unf = pd.read_csv(path+'sym_log_unfilter_ttc_out_seq_'+str(seq_id)+'.txt',header=None)    
        Gau = pd.read_csv(path+'ttc_'+label+'_full_MD_'+str(seq_id)+'.txt',header=None)
        Gau = insert_zero(np.log(Gau))
        sym_md_f[1] = Gau[1]
        sym_md_unf[1] = Gau[1]
        ind = np.arange(len(sym_md_f)) + 1
        for enum,data in enumerate([P1,P2,Gau,sym_md_f,sym_md_unf]):
    #    for enum,data in enumerate([ME,md_f]):
    #        mp, _ = np.polyfit(ind, data[0], 1)
    #        lp = str(np.around(-1/mp,1))
            md, _ = np.polyfit(ind, data[0]-data[1], 1)
            ld_comp[seq_id-1,enum] = np.around(-1/md,1)
    fig = plt.figure(constrained_layout=False)
    gs1 = gridspec.GridSpec(100, 100)
    gs1.update(left=0.1, right=0.99,top=0.99,bottom=0.13)

    ax = plt.subplot(gs1[0:100,       0:100])
    ind = np.arange(nseq)
    lw = 0.75
    fs=10
    ax.plot(ind,ld_comp[:,0],label='$\ell_{d}^{\mathcal{P}1}$',color='blue',lw=lw)
    ax.plot(ind,ld_comp[:,1],label='$\ell_{d}^{\mathcal{P}2}$',color='red',lw=lw)
    ax.plot(ind,ld_comp[:,2],label='$\ell_{d}^{MD_{Gaussian}}$',color='k',lw=lw)
    ax.plot(ind,ld_comp[:,3],label='$\ell_{d}^{MD_{filtered}}$',color='green',lw=lw)
    ax.plot(ind,ld_comp[:,4],label='$\ell_{d}^{MD_{unfiltered}}$',color='green',ls='--',lw=lw)
    ax.legend(fontsize=fs,ncol=2)
    ax.set_xticks(ind)         
    ax.set_xticklabels(['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','T1','T2','T3','T4','T5','T6','T7'],rotation=90)
    ax.tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
    ax.set_xlabel("Sequence index",fontsize=fs)
    ax.set_ylabel("Dynamic persistence length (in base-pairs)",fontsize=fs)
    plt.savefig('./Plots/compare_all_ld'+'.pdf',dpi=600)
    for i in range(5):
        for j in range(5):
            print(i,j,np.sqrt(np.mean(np.square(ld_comp[:,i] - ld_comp[:,j])))) 
####################################################################
###### Above --> Create files for TTC correlation plot
####################################################################



####################################################################
###### Below --> Compare two truncation
####################################################################

def compare_two_trunc(data,seq_id,label):
    MD = data.choose_seq([seq_id])
#    wc = MD.stiff_mre[0] - MD.stiff_me_sym[0]
#    wc = MD.stiff_mre[0] -  MD.s1b_sym[0] 
    res2 = cgDNA(MD.seq[0],'dna_mle')
    wc = res2.stiff.todense() - MD.stiff_mre[0] 


    eig = np.linalg.eigvals(wc)
    print(eig)
    nbp = len(MD.seq[0])
    x1,x2 = stencil_42(nbp)

    x6 = np.arange(0,24*nbp-12,6)
    fig,axr = plt.subplots(1)
    sns.heatmap(wc,ax=axr,center=0,cmap='seismic',cbar=1,square=True,cbar_kws={"shrink": .25,"pad":0.018})
    lwl,lws = 0.3,0.05
    for i in range(len(x1)):
        plt.vlines(x=x1[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.vlines(x=x2[i], ymin=x1[i], ymax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x1[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
        plt.hlines(y=x2[i], xmin=x1[i], xmax=x2[i], color='green', lw=lwl)
    for j in x6:
        axr.axvline(x=j,color='black', lw=lws)
        axr.axhline(y=j,color='black', lw=lws)

    ind = np.arange(0,24*nbp-18,12)
    axr.set_xticks(ind)         
    axr.set_yticks(ind)    
    axr.set_xticklabels(ind)
    axr.set_yticklabels(ind)
    axr.set_aspect('equal', adjustable='box')
    axr.tick_params(axis='both', labelsize=6, pad=3,length=3,width=0.5,direction= 'inout')
    axr.grid(False)
    plt.savefig('./Plots/P2-Trunc2_seq_id_'+str(seq_id+1)+'.pdf',dpi=600)

def compare_two_trunc_KL(data):
    T1_T2, T2_T1, T1_T2_sym  = [], [], []
    T1_MD, MD_T1, T1_MD_sym  = [], [], []
    T2_MD, MD_T2, T2_MD_sym  = [], [], []
    T1_P1, P1_T1, T1_P1_sym  = [], [], []
    T2_P2, P2_T2, T2_P2_sym  = [], [], []
    MT1_P1, MP1_T1, MT1_P1_sym  = [], [], []
    MT2_P2, MP2_T2, MT2_P2_sym  = [], [], []


    for seq_id in range(16):
        MD = data.choose_seq([seq_id])
        P1 = cgDNA(MD.seq[0],'ps2_cgf')
        P2 = cgDNA(MD.seq[0],'dna_mle')
        ## difference between two truncation
        T1_T2.append(kl_mvn(MD.shape_sym[0], MD.stiff_me_sym[0], MD.shape_sym[0], MD.stiff_mre[0]))
        T2_T1.append(kl_mvn(MD.shape_sym[0], MD.stiff_mre[0], MD.shape_sym[0], MD.stiff_me_sym[0]))
        T1_T2_sym.append(kl_mvn_sym(MD.shape_sym[0], MD.stiff_me_sym[0], MD.shape_sym[0], MD.stiff_mre[0]))

        ## T1 with MD 
        T1_MD.append(kl_mvn(MD.shape_sym[0], MD.s1b_sym[0], MD.shape_sym[0], MD.stiff_mre[0]))
        MD_T1.append(kl_mvn(MD.shape_sym[0], MD.stiff_mre[0], MD.shape_sym[0], MD.s1b_sym[0]))
        T1_MD_sym.append(kl_mvn_sym(MD.shape_sym[0], MD.s1b_sym[0], MD.shape_sym[0], MD.stiff_mre[0]))

        ## T2 with MD 
        T2_MD.append(kl_mvn(MD.shape_sym[0], MD.s1b_sym[0], MD.shape_sym[0], MD.stiff_me_sym[0]))
        MD_T2.append(kl_mvn(MD.shape_sym[0], MD.stiff_me_sym[0], MD.shape_sym[0], MD.s1b_sym[0]))
        T2_MD_sym.append(kl_mvn_sym(MD.shape_sym[0], MD.s1b_sym[0], MD.shape_sym[0], MD.stiff_me_sym[0]))

        ## T1 with P1
        T1_P1.append(kl_mvn(MD.shape_sym[0], MD.stiff_mre[0], P1.ground_state, P1.stiff.todense()))
        P1_T1.append(kl_mvn( P1.ground_state, P1.stiff.todense(), MD.shape_sym[0],  MD.stiff_mre[0]))
        T1_P1_sym.append(kl_mvn_sym(MD.shape_sym[0], MD.stiff_mre[0], P1.ground_state, P1.stiff.todense()))

        ## T2 with P2
        T2_P2.append(kl_mvn(MD.shape_sym[0], MD.stiff_me_sym[0], P2.ground_state, P2.stiff.todense()))
        P2_T2.append(kl_mvn( P2.ground_state, P2.stiff.todense(), MD.shape_sym[0],  MD.stiff_me_sym[0]))
        T2_P2_sym.append(kl_mvn_sym(MD.shape_sym[0], MD.stiff_me_sym[0], P2.ground_state, P2.stiff.todense()))

        ## T1 with P1
        MT1_P1.append(Mahal(MD.shape_sym[0], P1.ground_state, MD.stiff_mre[0]))
        MP1_T1.append(Mahal( P1.ground_state, MD.shape_sym[0], P1.stiff.todense()))
        MT1_P1_sym.append(Mahal_sym(MD.shape_sym[0], P1.ground_state, MD.stiff_mre[0], P1.stiff.todense()))

        ## T2 with P2
        MT2_P2.append(Mahal(MD.shape_sym[0], P2.ground_state, MD.stiff_mre[0]))
        MP2_T2.append(Mahal( P2.ground_state, MD.shape_sym[0], P2.stiff.todense()))
        MT2_P2_sym.append(Mahal_sym(MD.shape_sym[0], P2.ground_state, MD.stiff_mre[0], P2.stiff.todense()))


    for T in [T1_T2, T2_T1, T1_T2_sym, T1_MD, MD_T1, T1_MD_sym, T2_MD, MD_T2, T2_MD_sym, T1_P1, P1_T1, T1_P1_sym, T2_P2, P2_T2, T2_P2_sym,  MT1_P1, MP1_T1, MT1_P1_sym,  MT2_P2, MP2_T2, MT2_P2_sym]:
        TT = [str(np.around(i,4))+' & '  for i in T]
        TT.append(np.around(np.mean(T),4))
        TT = str(TT).replace("', '",'')
        TT = TT.replace("',","")
        TT = TT.replace("'","")
        TT = TT.replace("[","")
        TT = TT.replace("]","")
        print(TT)

    return None


def check_pos_def_prmset(ps,GC_padding,seq_kind):
    
    for N in range(8,13):
        if seq_kind == 'MDNA':
            seq_list = Met_all_Nmers(N)
        else:    
            seq_list = all_Nmers(N)

        for seq in  tqdm.tqdm(seq_list):
            if GC_padding == True:
                res = cgDNA('GC'+seq+'GC',ps)
            else:
                try:    ##### this is required as when checking for non-GC ends, how to eliminated the one with XM or MX steps
                    res = cgDNA(seq,ps)
                except:
                    # print(seq)
                    continue
                
            if is_pos_def(res.stiff.todense()) == False:
                print(seq, ' ---- Not definite -----')

        print('all Nmers computed for N = ', N)


##################################---------------------------------------------
# Analysis hydroxy/methylation effect on GC islands
##################################---------------------------------------------

def stiffness_analysis_MDNA(DNA,MDNA,HDNA):

    M_D_eig = []
    H_D_eig = []
    H_M_eig = []

    for i in range(3):
        if i==0:
            M_D_eig.append(np.sort(np.linalg.eigvals(MDNA.s1b_sym[i] -  DNA.s1b_sym[0]  )))
            H_D_eig.append(np.sort(np.linalg.eigvals(HDNA.s1b_sym[i] -  DNA.s1b_sym[0]  )))
            H_M_eig.append(np.sort(np.linalg.eigvals(HDNA.s1b_sym[i] -  MDNA.s1b_sym[i] )))
        else:
            M_D_eig.append(np.sort(np.linalg.eigvals(MDNA.s1b[i]     -  DNA.s1b_sym[0] )))
            H_D_eig.append(np.sort(np.linalg.eigvals(HDNA.s1b[i]     -  DNA.s1b_sym[0] )))
            H_M_eig.append(np.sort(np.linalg.eigvals(HDNA.s1b[i]     -  MDNA.s1b[i]    )))

    ss = 24*20-18
    color = ['r','b','g']

    fs = 8
    data = [M_D_eig,H_D_eig, H_M_eig]
    fig,ax = plt.subplots(3,sharex=True)
    for k in range(3):
        for j in range(3):
            ax[k].scatter(np.arange(ss),data[k][j],s=2,color=color[j],label='# of +ve eigs = '+str(np.sum(np.array(data[k][j]) >= 0, axis=0)))
            ax[k].legend(fontsize=6.7,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(0.5, 1.15))
            ax[k].tick_params(axis='both', labelsize=fs, pad=3,length=3,width=0.5,direction= 'inout')
        ax[k].set_ylabel("Eigenvalues" , fontsize=fs)
        ax[k].set_xlim(-3,ss+2)

    ax[2].set_xlabel("Eigenvalue index", fontsize=fs)
    plt.tight_layout()
    plt.savefig('./Plots/stiffness_analysis_MDNA.pdf',dpi=600)


    return None



# a,b = GrooveWidths_CS(cgDNA(random_seq(22),'ps2_cgf').ground_state)
# print(a,b)
