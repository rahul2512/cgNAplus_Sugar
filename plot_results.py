import matplotlib.pyplot as plt
import sys, random, re, numpy as np, pandas as pd
from matplotlib import gridspec
import os, time, copy
from utilities import pucker_label, dihed_label

def plot_stats(data,data2,name=None):
	fig = plt.figure(constrained_layout=False,figsize=(6,8))
	if 'pucker' in data.description:
		gs1 = gridspec.GridSpec(600, 100)
		gs1.update(left=0.1, right=0.98,top=0.96,bottom=0.09)
		d = 10
		ax0 = plt.subplot(gs1[000:100, 0:100])
		ax1 = plt.subplot(gs1[100:200, 0:100])
		ax2 = plt.subplot(gs1[200:300, 0:100])
		ax3 = plt.subplot(gs1[300:400, 0:100])
		ax4 = plt.subplot(gs1[400:500, 0:100])
		ax5 = plt.subplot(gs1[500:600, 0:100])
		ax_list = [ax0, ax1, ax2, ax3, ax4, ax5]
		xaxis = list(data.MD.keys())
		label = 'puck_stats'
		plot_label = pucker_label
		plot_label2 = ['$\u03B8_0$', '$\u03B8_1$', '$\u03B8_2$', '$\u03B8_3$', '$\u03B8_4$', 'P']
		xaxis_label = 'Base attached to the sugar ring'
		leg_shift = 1.15
	elif 'backbone' in data.description:
		gs1 = gridspec.GridSpec(700, 100)
		gs1.update(left=0.1, right=0.98,top=0.96,bottom=0.09)
		d = 10
		ax0 = plt.subplot(gs1[000:100, 0:100])
		ax1 = plt.subplot(gs1[100:200, 0:100])
		ax2 = plt.subplot(gs1[200:300, 0:100])
		ax3 = plt.subplot(gs1[300:400, 0:100])
		ax4 = plt.subplot(gs1[400:500, 0:100])
		ax5 = plt.subplot(gs1[500:600, 0:100])
		ax6 = plt.subplot(gs1[600:700, 0:100])
		ax_list = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]
		xaxis = list(data.MD.keys())
		label = 'dihed_stats'
		plot_label = dihed_label
		plot_label2 = ['\u03B1', '\u03B2', '\u03B3','\u03B4','\u03B5','\u03B6','\u03C7']
		xaxis_label = 'Base attached to the sugar ring'
		leg_shift = 1.2
		#### for plotting change the angle from -180 --> 180 to 0 --> 360 by adding 2pi if negative
		B2_MD = np.zeros(len(xaxis))
		B2_NN = np.zeros(len(xaxis))
		B2_CG = np.zeros(len(xaxis))
		for enum,x in enumerate(xaxis):
			B2_MD[enum] = 100*np.mean((data.MD[x]['e'] - data.MD[x]['z']) > 0 )
			B2_NN[enum] = 100*np.mean((data.CG[x]['e'] - data.CG[x]['z']) > 0 )
			B2_CG[enum] = 100*np.mean((data2.MD[x]['e'] - data2.MD[x]['z']) > 0 )
			data.MD[x][data.MD[x] < 0] = 360  + data.MD[x][data.MD[x] < 0]
			data.CG[x][data.CG[x] < 0] = 360  + data.CG[x][data.CG[x] < 0]
			data2.MD[x][data2.MD[x] < 0] = 360  + data2.MD[x][data2.MD[x] < 0]
	legend_MD = ['MD'] + ['_no_legend_']*10
	legend_NN = ['NN'] + ['_no_legend_']*10
	legend_CG = ['CG'] + ['_no_legend_']*10
	legend_size = 10
	fs = 10
	tl = 5
	_, *x_tick, _ = list(data.seq)   ### removes first and last entry
	shift = 0.18
	for enum, ax in enumerate(ax_list):
		tmp_MD = pd.DataFrame(columns=xaxis)
		tmp_NN = pd.DataFrame(columns=xaxis)
		tmp_CG = pd.DataFrame(columns=xaxis)
		for x in xaxis:
			tmp_MD[x] = data.MD[x][plot_label[enum]]
			tmp_NN[x] = data.CG[x][plot_label[enum]]
			tmp_CG[x] = data2.MD[x][plot_label[enum]]
		ax.errorbar(np.array(xaxis)-shift, tmp_MD.mean(), yerr=tmp_MD.std(),fmt='o',c='r',label = legend_MD[enum])
		ax.errorbar(np.array(xaxis),       tmp_NN.mean(), yerr=tmp_NN.std(),fmt='o',c='b',label = legend_NN[enum])
		ax.errorbar(np.array(xaxis)+shift, tmp_CG.mean(), yerr=tmp_CG.std(),fmt='o',c='g',label = legend_CG[enum])
		ax.tick_params(axis='both', labelsize=fs, pad=3,length=tl,width=0.5,direction= 'inout')
		ax.set_xticks(xaxis)
		ax.set_xticklabels(x_tick,fontsize=fs)
#		deg = '(\u00b0)'
		ax.set_ylabel(plot_label2[enum],fontsize=fs)
	ax_list[-1].set_xlabel(xaxis_label,fontsize=fs)
	ax0.legend(fontsize=legend_size,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(0.5, leg_shift))
	fig.align_ylabels()	
	fig.savefig("./Plots/" + label+'_' +name+".pdf",dpi=600)
	plt.close()
	if 'backbone' in data.description:
		fig = plt.figure(constrained_layout=False)
		gs1 = gridspec.GridSpec(600, 100)
		gs1.update(left=0.1, right=0.98,top=0.96,bottom=0.09)
		ax = plt.subplot(gs1[000:600, 0:100])
		ax.plot(   xaxis, B2_MD,c='r', label = legend_MD[0])
		ax.plot(   xaxis, B2_NN,c='b', label = legend_NN[0])
		ax.plot(   xaxis, B2_CG,c='g', label = legend_CG[0])
		ax.scatter(xaxis, B2_MD,c='r', label = legend_MD[1])
		ax.scatter(xaxis, B2_NN,c='b', label = legend_NN[1])
		ax.scatter(xaxis, B2_CG,c='g', label = legend_CG[1])
		ax.tick_params(axis='both', labelsize=fs, pad=3,length=tl,width=0.5,direction= 'inout')
		ax.set_xticks(xaxis)
		ax.set_xticklabels(x_tick,fontsize=fs)
		ax.set_ylabel('B2 %',fontsize=fs)
		ax.set_xlabel(xaxis_label,fontsize=fs)
		ax.legend(fontsize=legend_size,loc='upper center',fancybox=True,ncol=3, frameon=True,framealpha=1, bbox_to_anchor=(0.5, 1.05))
		fig.savefig("./Plots/B1_B2_" + label +'_'+ name+".pdf",dpi=600)
		plt.close()
	return None
