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


a = cgDNA('GCATATATATATATATGC')
