from utilities import read_hbond_filter_file, comp, read_seq, read_dihedrals, read_pucker, read_coord
import pandas as pd, numpy as np

class Opt:
  def __init__(self, nbp=None, path=None, seq_specific=True,NA=None, seq_id=None, run_nbr=None, remove_ends = None, seq_id_list=None,seq_id_list_Test=None,seq_id_list_val=None,run_nbr_list=None,iall=None,sparse=None,what=None,which_seq=None):
    self.path = path
    self.NA           = NA
    self.seq_id_list  = seq_id_list 
    self.seq_id_list_val  = seq_id_list_val
    self.seq_id_list_Test  = seq_id_list_Test
    self.run_nbr_list = run_nbr_list
    self.seq_id       = seq_id
    self.run_nbr      = run_nbr
    self.sparse       = sparse   
    self.what         = what
    self.which_seq    = which_seq  
    self.iall         = iall
    self.nbp          = nbp
    self.remove_ends  = remove_ends
    self.seq_specific = seq_specific

class Data:
  def __init__(self, param=None, load_path=None,model=None,save_path=None,hyper_value=None, X_val=None, Y_val=None, X_Train=None, Y_Train=None, X_Test=None, Y_Test=None, out_dim=None, inp_dim=None, model_type=None, ba_Train=None, ba_Test=None,ba_val=None,pa_Train=None, pa_Test=None,pa_val=None):
    self.hyper_value     = hyper_value  
    self.X_val           = X_val  
    self.Y_val           = Y_val  
    self.X_Train         = X_Train
    self.Y_Train         = Y_Train  
    self.X_Test          = X_Test  
    self.Y_Test          = Y_Test  
    self.out_dim         = out_dim   
    self.inp_dim         = inp_dim
    self.model_type      = model_type
    self.ba_Train        = ba_Train
    self.ba_val          = ba_val
    self.ba_Test         = ba_Test
    self.pa_Train        = pa_Train
    self.pa_val          = pa_val
    self.pa_Test         = pa_Test
    self.load_path       = load_path
    self.save_path       = save_path
    self.model           = model

class stat:
	def __init__(self, stat_val=None, stat_Train=None, stat_Test=None):
		self.stat_val = stat_val
		self.stat_Train = stat_Train
		self.stat_Test = stat_Test

class fra:
        def __init__(self,what):
                self.description = "Atoms information from " + what
                self.R      = {}
                self.r      = {}
                self.Rw     = {}
                self.rw     = {}
                self.Rc     = {}
                self.rc     = {}
                self.Rpw    = {}
                self.rpw    = {}
                self.Rpc    = {}
                self.rpc    = {}
class Data:
        def __init__(self):
                self.w    = None
                self.c    = None
                self.triw= {}
                self.tric= {}
                self.description = "Data and data split for trimers"

class atoms:
        def __init__(self,what):
                self.atmcrd_w    = {}
                self.atmcrd_c    = {}
                self.s_atoms_w   = {}
                self.s_atoms_c   = {}
                self.b_angles_w  = None
                self.b_angles_c  = None
                self.p_angles_w  = None
                self.p_angles_c  = None
                self.description = "Atoms information from " + what
class seqs:
        def __init__(self):
                self.seqwt    = {}
                self.seqct    = {}
                self.seq      = None
                self.seq_comp = None
                self.description = "Seq and seq split into trimers"	

class forward_data:
	def __init__(self):

		self.data = Data()
		self.seqs = seqs()
		self.MDFr = fra("MD")
		self.CGFr = fra("cgDNA+")
		
		self.MDAt = atoms("MD")
		self.CGAt = atoms("cgDNA+")

class mdDNA_object:
	def __init__(self,opt):
		self.seq        = read_seq(opt)
		for enum, run_nbr in enumerate(opt.run_nbr_list):
			opt.run_nbr = run_nbr
			opt = read_hbond_filter_file(opt)
			if enum == 0:
				self.MC_samples = read_coord(opt).T[opt.iall]
				self.pucker     = read_pucker(opt).T[opt.iall]
				self.dihedrals  = read_dihedrals(opt).T[opt.iall]
			else:
				self.MC_samples = pd.concat([self.MC_samples, read_coord(opt).T[opt.iall]]     ,axis=1)
				self.pucker     = pd.concat([self.pucker    , read_pucker(opt).T[opt.iall]]    ,axis=1)
				self.dihedrals  = pd.concat([self.dihedrals , read_dihedrals(opt).T[opt.iall]] ,axis=1)
		col = np.arange(self.pucker.shape[1])
		self.MC_samples.columns = col
		self.pucker.columns = col
		self.dihedrals.columns = col

class Out_stats:
	def __init__(self,description):
		self.description = description
		self.difference = {}
		self.pc = None
		self.MD = {}
		self.CG = {}

