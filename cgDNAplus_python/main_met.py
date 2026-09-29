import sys
sys.path.append('./modules')
sys.path.append('./classes')
from cgDNAclass import cgDNA
import numpy as np
import cgDNAUtils as tools
import MD_analysis as analysis
from E_transform import Etrans
from Init_MD import init_MD_data
import matplotlib.colors as mcolors

c1 = ['red','blue','green']
c2 = ['darkred','navy','olivedrab']
c3 = ['indianred','royalblue','limegreen']
c4 = ['maroon','dodgerblue','limegreen']

#################### Path for 10 mus final files
DNA_path    = './BDNA/palin.bscl.tip3p.jc.comb.sym.stats.10mus.cgF.25_tot.mat'
MDNA_path   = './BDNA/methyl.bscl.tip3p.jc.comb.sym.stats.10mus.cgF.comb.modified.mat'    
HDNA_path   = './BDNA/hmethyl.bscl.tip3p.jc.comb.modified.sym.stats.10mus.cgF.mat'

DNA_path_3  = './BDNA/palin.bscl.tip3p.jc.comb.sym.stats.10mus.cgF.26_28.mat'
MDNA_path_3 = './BDNA/methyl.bscl.tip3p.jc.comb.sym.stats.10mus.30_32.modified.mat'    
HDNA_path_3 = './BDNA/hmethyl.bscl.tip3p.jc.comb.sym.stats.10mus.30_32.modified.mat'
############### Initiate MD data #################################
DNA = init_MD_data().load_data(DNA_path)
MDNA = init_MD_data().load_data(MDNA_path)
HDNA = init_MD_data().load_data(HDNA_path)

DNA_3 = init_MD_data().load_data(DNA_path_3)
MDNA_3 = init_MD_data().load_data(MDNA_path_3)
HDNA_3 = init_MD_data().load_data(HDNA_path_3)

############### comparison of ground state of MD vs reconstruction #################################
def main1():
    analysis.plot_gs_vs_MD_shape_epi(MDNA,HDNA)

############### persistence lengths plots #################################
def main2():

    data = analysis.plot_special_persistence_length(['DNA_BSTJ_CGF','RNA_OL3_CGF','HYB_CGF'])
    data = analysis.plot_persistence_length(['DNA_BSTJ_CGF','RNA_OL3_CGF','HYB_CGF'])

############### palindromic error #################################
def main3():
    path = './BDNA/methyl.bscl.tip3p.jc.comb.sym.stats.XXmus.cgF.mat'
    analysis.plot_heatmap_palin_err_epi(path,name='MDNA')
    path = './BDNA/hmethyl.bscl.tip3p.jc.comb.sym.stats.XXmus.cgF.mat'
    analysis.plot_heatmap_palin_err_epi(path,name='HDNA')

############### training/test error #################################
def main4():
#    analysis.plot_heatmap_training_err(DNA,'ps2_cgf',sym=True)
    analysis.plot_heatmap_training_err_epi(MDNA,'ps_mdna',sym=True)
    analysis.plot_heatmap_training_err_epi(HDNA,'ps_hdna',sym=True)

############### set scale for training/test/palindromic error #################################
def main5():
    analysis.set_scale_for_error_epi(MDNA,sym=True)
    analysis.set_scale_for_error_epi(HDNA,sym=True)

############### compare_same_seq_across_data #################################
linestyles= ['-','-', '--', '-.', ':']
def main6():
    col = [c1,c1,c1]
    for sequence_id in analysis.MDNA_map:
        analysis.compare_same_seq_across_data_epi(MDNA,HDNA,col,sequence_id,'compare_seq_MDNA_HDNA')
    
############### compare_same_seq_across_data #################################
linestyles= ['-','-', '--', '-.', ':']
def main7():
    col = [c1,c1,c2,c2,c3,c3]
    seq_id1=np.array([20,21])
    sym1=[False,False]

    analysis.compare_diff_seq_within_data(DNA,seq_id1-1,'ps2_cgf',col,sym1,'DNA_compare_seq')
    analysis.compare_diff_seq_within_data(RNA,seq_id1-1,'ps_rna',col,sym1,'RNA_compare_seq')
#    analysis.compare_diff_seq_within_data(HYB,seq_id-1,'ps_hyb',col,sym1,'HYB_compare_seq')
    sym2=[False,True]
    seq_id2=np.array([18,19])
    analysis.compare_diff_seq_within_data(DNA,seq_id2-1,'ps2_cgf',col,sym2,'DNA_compare_seq')
    analysis.compare_diff_seq_within_data(RNA,seq_id2-1,'ps_rna',col,sym2,'RNA_compare_seq')
#    analysis.compare_diff_seq_within_data(HYB,seq_id-1,'ps_hyb',col,sym1,'HYB_compare_seq')


##################################---------------------------------------------
# plot oligomer level eigenvalues for DNA,RNA, Hybrid
##################################---------------------------------------------

def main8():
    #### make sure which are sym
    sym=[True,True,True]
    for enum, seq_id in enumerate(analysis.MDNA_map[0:12]):
        seq = analysis.unmodify(MDNA.seq[seq_id])
        res = cgDNA(seq,'ps2_cgf')
        analysis.plot_eig_olig_first_epi(res, MDNA.choose_seq([seq_id]), HDNA.choose_seq([seq_id]), sym, c1, "Epi_olig_eig_"+str(1+enum))


##################################---------------------------------------------
# plot oligomer level stiffness for DNA,RNA, Hybrid
##################################---------------------------------------------
def main9():
    #### make sure which are sym
    for enum, seq_id in enumerate(analysis.MDNA_map[0:12]):
        analysis.fit_stencil_in_matrix(MDNA.choose_seq([seq_id]),True, "MDNA_stiffness_"+str(seq_id+1))
        analysis.fit_stencil_in_matrix(HDNA.choose_seq([seq_id]),True, "HDNA_stiffness_"+str(seq_id+1))


##################################---------------------------------------------
# Truncation error
##################################---------------------------------------------

def main10():
        data = [DNA, MDNA, HDNA]
        NA_type_list = ['DNA', 'MDNA', 'HDNA']
        analysis.Truncation_error(data, NA_type_list,c1)

##################################---------------------------------------------
# Positive def reconstruction tests --- reconstruct seq for given length 
##################################---------------------------------------------
def main11():   
    analysis.check_pos_def_prmset('ps_mdna',False,'MDNA')    
    # analysis.check_pos_def_prmset('ps_hdna',False,'MDNA')

##################################---------------------------------------------
# Analysis hydroxy/methylation effect on GC islands
##################################---------------------------------------------
def main12():
    analysis.stiffness_analysis_MDNA(DNA_3,MDNA_3,HDNA_3)


#main1()  ## comparison of ground state of MD vs reconstruction
#main2()  ## persistence length error
#main3()  ## palindromic error
#main4()  ## training/test error
#main5()  ## set_scale_for_errors
#main6()  ## compare_same_seq_across_data
#main7()  ## compare_diff_seq_within_data
#main8()  ## plot_eig_olig_first
#main9()   ## fit_stencil_in_matrix
#main10()  ### Truncation error
# analysis.mdna_random_seq(1)    #########not completed this yet to do

    
main12()
