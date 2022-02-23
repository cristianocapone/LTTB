# This code below is to set the structure of the code for Pyramidal Neuron - Larkum style
# Learning through target spikes

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload
from tqdm import trange
import random
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import json
from datetime import datetime
import os as os
import lttb_contesto_both as lttb_module
#from lttb_contesto_sopra import LTTB

run_number = 1
run_noisy_context = True
random_target = True

N, I, O, T = 500, 5, 3, 1000
shape = (N, I, O, T)

dt = .001
tau_m = 20. * dt
tau_s = 2. * dt
tau_ro = 10. * dt
tau_star = 20. * dt
tau_W = 200 * dt

beta = np.exp(-dt/tau_s)
beta_ro = np.exp(-dt/tau_ro)
beta_targ = np.exp(-dt/tau_star)
beta_W = np.exp(-dt/tau_W)

# std of J's used to project context - it used to be 20
sigma_apical_context = 0
sigma_basal_context = 0

sigma_targ = 100 # 10**1.5
sigma_in = 12.

dv = 1 / 500.
alpha = .005
alpha_rout = .01
Vo = - 4
h = - 1
s_inh = 20

sigma_Jrec = 0.
sigma_wrec = 0.
sigma_Jout = 0.1

Ne = int(0.8*N)
Ni = N-Ne

frac_zero_targ = 0.75
frac_zero_cont = 0.75

n_contexts = 1
t_sw = 500

training_specs = [(100,10.0,0.02), (200,5.0,0.01), (200,2.5,0.005), (100,1.25,0.0025), (100,0.625,0.00125)] # 3-uples are composed by (nIterRec,eta,eta_out)
#training_specs = [(1,10.0,0.02), (1,5.0,0.01)] # 3-uples are composed by (nIterRec,eta,eta_out)

now_date = '%s' % datetime.now()

# Here we do some check and then build batch name

batch_name = str(O) + 'D' + '_'

if (sigma_apical_context>0 and sigma_basal_context==0):
	batch_name += 'sigmaApicalCont' + str('%.2f' % sigma_apical_context).replace('.','p') + '_'
elif (sigma_apical_context==0 and sigma_basal_context>0):
	batch_name += 'sigmaBasalCont' + str('%.2f' % sigma_basal_context).replace('.','p') + '_'
elif (sigma_apical_context==0 and sigma_basal_context==0):
	batch_name += 'noCont' + '_'
else:
	print('\nATTENTION!! Context signal is active in both the basal and the apical compartments!\n')
	quit()

batch_name += 'Bfiltrec_'
batch_name += 'Bias_'
batch_name += 'fracZeroTarg' + str('%.2f' % frac_zero_targ).replace('.','p') + '_'
batch_name += 'gradino_'
batch_name += 'N' + str(N) + '_'
batch_name += 'sigmaTarg' + str('%.2f' % sigma_targ).replace('.','p') + '_'
batch_name += 'input' + str(I)

if run_noisy_context:
	batch_name += '_withNoise'

#batch_name = '2D_sigmaApicalCont20_Bfiltrec_Bias_p075_gradino_doubleSize_sigmaTarg30_input50_provaNoisy'

print('\n##### batch_name = %s\n##### run_number = %d\n' % (batch_name,run_number))

wk_dir = './' + 'Results_' + batch_name

if not os.path.isdir(wk_dir):
	os.mkdir(wk_dir)

if os.path.isfile(wk_dir + '/%03d_params.json' % run_number):
	print('ATTENTION!! File(s) with run_number=%d already exist!\n' % run_number)
	quit()

# Here we build the dictionary of the simulation parameters
par = {'dt' : dt, 'tau_m' : tau_m, 'tau_s' : tau_s, 'tau_ro' : tau_ro, 'tau_star' : tau_star,'tau_W' : tau_W,
	   'dv' : dv, 'Vo' : Vo, 'h' : h, 's_inh' : s_inh,
	   'N' : N, 'Ni' : Ni, 'Ne' : Ne, 'T' : T, 'I' : I, 'O' : O, 'shape' : shape,
	   'sigma_Jrec' : sigma_Jrec,'sigma_wrec' : sigma_wrec, 'sigma_Jout' : sigma_Jout,  'n_contexts' : n_contexts,
	   'alpha' : alpha, 'alpha_rout' : alpha_rout,
	   'sigma_in' : sigma_in, 'sigma_targ' : sigma_targ,
	   'sigma_basal_context' : sigma_basal_context, 'sigma_apical_context' : sigma_apical_context, 'h' : h,
	   'frac_zero_targ' : frac_zero_targ, 'frac_zero_cont' : frac_zero_cont, 't_sw' : t_sw,
	   'run_number' : run_number, 'training_specs' : training_specs, 'target' : ('random' if random else './sample_targ.npy'), 'exec_time' : now_date}

with open(wk_dir + '/%03d_params.json' % run_number, 'w') as fp:
	json.dump(par, fp)

TIME = 1.

T = int(np.floor(TIME/dt))

nStepOutTraining = 1000

t_shut = 10

JMAX = 100.
nu_targ = 0.005

apicalFactorTrain = 1
apicalFactorTest = 0

gamma = 10.
def f(x,gamma):
	return np.exp(x*gamma)/(np.exp(x*gamma)+1)

##########
##########

# define clock and target

def init_clock_targ():
	lttb.y_targ_collection = []
	for k in range(n_contexts):
		lttb.init_targ(par)
		if not random_target:
			rt = np.load('./sample_targ.npy')
			if len(rt)!=n_contexts or len(rt[0])!=O or len(rt[0][0])!=T:
				print('ATTENTION!! Loaded targ(s) do not have the suitable shape!\n')
				quit()
			else:
				lttb.y_targ = rt[k]
		lttb.y_targ_collection.append(lttb.y_targ)
	lttb.init_clock(par)

##########
##########

# sparsify target and context [p is the fraction of masked elements]

def sparsify_cont(p):
	mask = np.array([0. if random.random()<p else 1. for _ in range(N*n_contexts)]).reshape((N,n_contexts))
	lttb.j_apical_cont = lttb.j_apical_cont*mask
	mask = np.array([0. if random.random()<p else 1. for _ in range(N*n_contexts)]).reshape((N,n_contexts))
	lttb.j_basal_cont = lttb.j_basal_cont*mask
	return

def sparsify_targ(p):
	mask = np.array([0. if random.random()<p else 1. for _ in range(N*lttb.O)]).reshape((N,lttb.O))
	lttb.j_targ = lttb.j_targ*mask
	return

##########
##########

def training_rec(nIterRec=100, test_every=5, eta=5., eta_out=0.01, etaW=0., eta_bias = 0.0002, print_err=True):
	
	#print('Training', end='\r')
	
	ERRORS = np.zeros((int(nIterRec/test_every),n_contexts))
	
	for iteration in range(nIterRec):
		
		#print('Training: t=%d/%d' % (iteration+1,nIterRec), end='\r')

		###### Online-Training

		#initialize simulation

		for cont_index in range(n_contexts):
	
			lttb.cont = lttb.cont*0
			lttb.cont[cont_index] = 1
		
			lttb.y_targ = lttb.y_targ_collection[cont_index]
		
			lttb.initialize(par)
		
			#run simulation
			dH = 0
			
			for t in range(lttb.T-2):
				
				lttb.step(apicalFactor = apicalFactorTrain)
				
				dH = dH*(1-dt/tau_m) + dt/tau_m*lttb.S_filt[:,t]
				
				#DJ = np.outer(( lttb.S_apic_dist[:,t+1] - f(lttb.VapicRec[:,t],gamma) )*(1-lttb.S_apic_dist[:,t]) ,dH)
				DJ = np.outer( ( lttb.S_apic_dist[:,t+1] - f(lttb.VapicRec[:,t],gamma) )*(1-lttb.S_apic_prox[:,t])*lttb.S_wind_soma[:,t+1] ,dH)
				lttb.J =  lttb.J + eta*DJ
					
				#SR = lttb.S_filtRO[:,t+1]
				SR = lttb.B_filt_rec[:,t+1]
				Y = lttb.Jout@SR + lttb.Bias
						
				DJRO = np.outer(lttb.y_targ[:,t+1] - Y,SR.T)
				dBias = lttb.y_targ[:,t+1] - Y
						
				lttb.Jout = lttb.Jout + eta_out*DJRO
				lttb.Bias = lttb.Bias + eta_bias*dBias

		###### Test

		if (iteration+1)%test_every==0:
	
			if print_err:
				print('t = %d/%d' % (iteration+1,nIterRec))
			
			for cont_index in range(n_contexts):
				
				lttb.cont = lttb.cont*0
				lttb.cont[cont_index] = 1
					
				lttb.y_targ = lttb.y_targ_collection[cont_index]
					
				lttb.initialize(par)
					
				#run simulation
					
				for t in range(lttb.T-2):
						
					lttb.step(apicalFactor = apicalFactorTest)
							
				#SR = lttb.S_filtRO[:,1:-2]
				SR = lttb.B_filt_rec[:,1:-2]
				Y = lttb.Jout@SR + np.tile(lttb.Bias,(lttb.T-3,1)).T
				mse_rec_train = np.std(lttb.y_targ[:,1:-2] - Y)**2
							
				ERRORS[int(iteration/test_every),cont_index] = mse_rec_train
							
				if print_err:
					print('  %.4f' % mse_rec_train, end=' ')
									
			if print_err:
				print()
											
	#print('Training: t=%d/%d ...done.' % (nIterRec,nIterRec))

	"""
	for n in range(n_contexts):
		plt.plot(test_every*np.arange(1,1+len(ERRORS)), ERRORS[:,n], marker='.', color='C'+str(n))
	plt.xlabel('iteration')
	plt.ylabel('mse')
	plt.show()
	"""

##########
##########

def context_experiment(exp_type, t_sw, sigma_noisy_cont=0., show_plots=False):
	
	if exp_type=='switch_context' and n_contexts!=2:
		print('ERROR! \'switch_context\' experiment can not be done with a number of context equal to %d.' % n_contexts)
		return
		
	print('Context experiment=\'%s\', t_sw=%d' % (exp_type,t_sw))

	dt_sw = 100
			
	stats = {}
				
	stats['targs'] = []
	stats['outputs'] = []
	stats['contexts'] = []
	stats['S_somas'] = []
	stats['S_winds'] = []
	stats['B_filt_recs'] = []
			
	if exp_type=='turnoff_context' or exp_type=='switch_context':
		stats['mses_before_onDiag'] = []
		stats['mses_before_offDiag'] = []
		stats['mses_after_onDiag'] = []
		stats['mses_after_offDiag'] = []
	elif exp_type=='full_context' or exp_type=='noisy_full_context':
		stats['mses_onDiag'] = []
		stats['mses_offDiag'] = []
			
	for cont_index in range(n_contexts):
									
		context = np.zeros((lttb.T-2,n_contexts))
								
		if exp_type=='full_context':
			context[:,cont_index] = np.array([1 for _ in range(lttb.T-2)])
		elif exp_type=='noisy_full_context':
			context[:,cont_index] = np.array([1 for _ in range(lttb.T-2)])
			context += np.random.normal(loc=0, scale=sigma_noisy_cont, size=(lttb.T-2)*n_contexts).reshape((np.shape(context)))
		elif exp_type=='turnoff_context':
			#context[:,cont_index] = np.array([f(t_sw-_,0.5) for _ in range(lttb.T-2)])
			context[:,cont_index] = np.array([1 if _<=t_sw else 0 for _ in range(lttb.T-2)])
		elif exp_type=='switch_context':
			#context[:,cont_index] = np.array([f(t_sw-_,0.5) for _ in range(lttb.T-2)])
			context[:,cont_index] = np.array([1 if _<=t_sw else 0 for _ in range(lttb.T-2)])
			#context[:,1-cont_index] = np.array([1-f(t_sw-_,0.5) for _ in range(lttb.T-2)])
			context[:,1-cont_index] = np.array([0 if _<=t_sw else 1 for _ in range(lttb.T-2)])
																
		#lttb.cont = lttb.cont*0
		#lttb.cont[cont_index] = 1
																
		lttb.y_targ = lttb.y_targ_collection[cont_index]
															
		lttb.initialize(par)
																	
		#run simulation
																	
		apicalFactor = 0
																		
		for t in range(lttb.T-2):
																			
			lttb.cont = context[t]
			"""
			if t==t_sw:
				apicalFactor = 0
				#lttb.cont = lttb.cont*0.1
				if exp_type=='switch_context':
					lttb.cont[0] = 1 - lttb.cont[0]
					lttb.cont[1] = 1 - lttb.cont[1]
				elif exp_type=='turnoff_context':
					lttb.cont *= 0
			context[t] = lttb.cont
			#print(lttb.cont)
			"""

			lttb.step(apicalFactor = apicalFactor)
																						
		# SR = lttb.S_filtRO[:,1:-2]
		SR = lttb.B_filt_rec[:,1:-2]
		# Y = lttb.Jout@SR
		Y = lttb.Jout@SR + np.tile(lttb.Bias,(lttb.T-3,1)).T
																							
		stats['outputs'].append(Y)
		stats['contexts'].append(context)
		stats['S_somas'].append(lttb.S_soma)
		stats['S_winds'].append(lttb.S_wind)
		stats['B_filt_recs'].append(lttb.B_filt_rec)

		if exp_type=='turnoff_context':
			right_targ = lttb.y_targ_collection[cont_index][:,1:t_sw+1]
			right_targ = np.append(right_targ, lttb.y_targ_collection[cont_index][:,t_sw+1:-2], 1)
			mse_rec_train_before_onDiag = np.std(lttb.y_targ_collection[cont_index][:,1:t_sw+1-dt_sw] - Y[:,0:t_sw-dt_sw])**2
			mse_rec_train_after_onDiag = np.std(lttb.y_targ_collection[cont_index][:,t_sw+1+dt_sw:-2] - Y[:,t_sw+dt_sw:])**2
			if n_contexts==2:
				wrong_targ = lttb.y_targ_collection[1-cont_index][:,1:t_sw+1]
				wrong_targ = np.append(wrong_targ, lttb.y_targ_collection[1-cont_index][:,t_sw+1:-2], 1)
				mse_rec_train_before_offDiag = np.std(lttb.y_targ_collection[1-cont_index][:,1:t_sw+1-dt_sw] - Y[:,0:t_sw-dt_sw])**2
				mse_rec_train_after_offDiag = np.std(lttb.y_targ_collection[1-cont_index][:,t_sw+1+dt_sw:-2] - Y[:,t_sw+dt_sw:])**2
		elif exp_type=='switch_context':
			right_targ = lttb.y_targ_collection[cont_index][:,1:t_sw+1]
			right_targ = np.append(right_targ, lttb.y_targ_collection[1-cont_index][:,t_sw+1:-2], 1)
			mse_rec_train_before_onDiag = np.std(lttb.y_targ_collection[cont_index][:,1:t_sw+1-dt_sw] - Y[:,0:t_sw-dt_sw])**2
			mse_rec_train_after_onDiag = np.std(lttb.y_targ_collection[1-cont_index][:,t_sw+1+dt_sw:-2] - Y[:,t_sw+dt_sw:])**2
			wrong_targ = lttb.y_targ_collection[1-cont_index][:,1:t_sw+1]
			wrong_targ = np.append(wrong_targ, lttb.y_targ_collection[cont_index][:,t_sw+1:-2], 1)
			mse_rec_train_before_offDiag = np.std(lttb.y_targ_collection[1-cont_index][:,1:t_sw+1-dt_sw] - Y[:,0:t_sw-dt_sw])**2
			mse_rec_train_after_offDiag = np.std(lttb.y_targ_collection[cont_index][:,t_sw+1+dt_sw:-2] - Y[:,t_sw+dt_sw:])**2
		elif exp_type=='full_context' or exp_type=='noisy_full_context':
			right_targ = lttb.y_targ_collection[cont_index][:,1:-2]
			mse_rec_train_onDiag = np.std(lttb.y_targ_collection[cont_index][:,1:-2] - Y)**2
			if n_contexts==2:
				wrong_targ = lttb.y_targ_collection[1-cont_index][:,1:-2]
				mse_rec_train_offDiag = np.std(lttb.y_targ_collection[1-cont_index][:,1:-2] - Y)**2
																															
		stats['targs'].append(right_targ)
																														
		if exp_type=='turnoff_context' or exp_type=='switch_context':
			stats['mses_before_onDiag'].append(mse_rec_train_before_onDiag)
			stats['mses_after_onDiag'].append(mse_rec_train_after_onDiag)
			if n_contexts==2:
				stats['mses_before_offDiag'].append(mse_rec_train_before_offDiag)
				stats['mses_after_offDiag'].append(mse_rec_train_after_offDiag)
			#print('  mse_before[%d]=%.3f\n  mse_after[%d]=%.3f' % (cont_index,mse_rec_train_before,cont_index,mse_rec_train_after))
		elif exp_type=='full_context' or exp_type=='noisy_full_context':
			stats['mses_onDiag'].append(mse_rec_train_onDiag)
			if n_contexts==2:
				stats['mses_offDiag'].append(mse_rec_train_offDiag)
			#print('  mse[%d]=%.3f' % (cont_index,mse_rec_train_onDiag))
																																					
	#mse, Y = lttb.train_ro(par,out_epochs = 1)


		if show_plots:
																																						
			plt.figure(figsize=(12, 4))
			plt.subplot(211)
			for i in range(len(Y)):
				plt.plot(Y[i].T, color='C'+str(i))
				if exp_type=='switch_context' and n_contexts==2:
					plt.plot(lttb.y_targ_collection[cont_index][i].T[0:t_sw], '--', color='C'+str(i))
					plt.plot(range(t_sw,1000),lttb.y_targ_collection[1-cont_index][i].T[t_sw:], '--', color='C'+str(i))
				else:
					plt.plot(lttb.y_targ[i].T, '--', color='C'+str(i))
																																											
			if n_contexts<=2:
				for i in range(n_contexts):
					plt.plot(np.array(context).T[i], '--', lw=2, color=['black','red'][i], zorder=-1)
			"""
			if exp_type=='full_context' or exp_type=='noisy_full_context' or exp_type=='turnoff_context':
				plt.plot(np.array(context).T[cont_index], '--', lw=2, color=['black','red'][cont_index], zorder=-1)
				#plt.plot(np.array(context), '--', lw=2, color=('black' if cont_index==0 else 'red'))
			elif exp_type=='switch_context':
				if cont_index==0:
				
				else:
					plt.plot(np.array(context).T[0], '--', lw=2, color='red', zorder=-1)
					plt.plot(np.array(context).T[1], '--', lw=2, color='black', zorder=-1)
			"""
			
			plt.xlabel('time(s)')
			plt.ylabel('$y_{targ}$ --- $y_{out}$')
			plt.subplot(212)
			#plt.imshow(lttb.S_filtRO[0:20,:],aspect='auto')
			plt.imshow(lttb.B_filt_rec[0:50,:],aspect='auto')
																																																	
			#plt.subplot(313)
			#plt.imshow(lttb.S_wind[0:20,:],aspect='auto')
																																																	
			if par['sigma_apical_context']>0 and par['sigma_basal_context']==0:
				sigma_context = par['sigma_apical_context']
				compartment = 'sopra'
			elif par['sigma_apical_context']==0 and par['sigma_basal_context']>0:
				sigma_context = par['sigma_basal_context']
				compartment = 'sotto'
																																																					
			if exp_type=='turnoff_context':
				fig_title = "turnoff del contesto da %s [sigma=%.0f, contesto n. %d]\nmse_before=%.3f, mse_after=%.3f" % (compartment, sigma_context, cont_index, mse_rec_train_before_onDiag, mse_rec_train_after_onDiag)
				fig_name = "turnoff_%s_sigma%.0f_contesto%d_%d" % (compartment, sigma_context, cont_index+1, n_contexts)
			elif exp_type=='switch_context':
				fig_title = "switch del contesto da %s [sigma=%.0f, contesto n. %d]\nmse_before=%.3f, mse_after=%.3f" % (compartment, sigma_context, cont_index, mse_rec_train_before_onDiag, mse_rec_train_after_onDiag)
				fig_name = "switch_%s_sigma%.0f_contesto%d_%d" % (compartment, sigma_context, cont_index+1, n_contexts)
			elif exp_type=='full_context':
				fig_title = "segnale di contesto stabile da %s [sigma=%.0f, contesto n. %d]\nmse=%.3f" % (compartment, sigma_context, cont_index, mse_rec_train_onDiag)
				fig_name = "full_%s_sigma%.0f_contesto%d_%d" % (compartment, sigma_context, cont_index+1, n_contexts)
			elif exp_type=='noisy_full_context':
				fig_title = "segnale di contesto rumoroso da %s [sigma=%.0f, contesto n. %d]\nmse=%.3f" % (compartment, sigma_context, cont_index, mse_rec_train_onDiag)
				fig_name = "noisy_%s_sigma%.0f_contesto%d_%d" % (compartment, sigma_context, cont_index+1, n_contexts)
																																																											
			plt.suptitle(fig_title)
			plt.savefig('./figures/' + fig_name + '.pdf', transparent=False)
			
			plt.show()
																																																													
	if exp_type=='turnoff_context' or exp_type=='switch_context':
		print('    onDiag: (<mse_before>,<mse_after>) = (%.3f,%.3f)' % (np.mean(stats['mses_before_onDiag']),np.mean(stats['mses_after_onDiag'])))
		print('    offDiag: (<mse_before>,<mse_after>) = (%.3f,%.3f)' % (np.mean(stats['mses_before_offDiag']),np.mean(stats['mses_after_offDiag'])))
	elif exp_type=='full_context' or exp_type=='noisy_full_context':
		print('    onDiag: <mse>=%.3f' % (np.mean(stats['mses_onDiag'])))
		print('    offDiag: <mse>=%.3f' % (np.mean(stats['mses_offDiag'])))

	print()

	return stats

##########
##########

# running

results = {}

print(reload(lttb_module))
lttb = lttb_module.LTTB (par)
	
init_clock_targ()
sparsify_targ(frac_zero_targ)
sparsify_cont(frac_zero_cont)

#for specs in training_specs:
	#training_rec(nIterRec=specs[0], eta=specs[1], eta_out=specs[2], print_err=False)

self_dist_B = []
self_dist_B_rec = []
dist_B = []

context = np.zeros((lttb.T-2,n_contexts))
#context[:,0] = np.array([1 for _ in range(lttb.T-2)])
lttb.y_targ = lttb.y_targ_collection[0]
lttb.initialize(par)
for t in range(lttb.T-2):
	lttb.cont = context[t]
	lttb.step(apicalFactor = 1)

nIterRec = 1000
for n in range(nIterRec):
	print('n = %d/%d' % (n+1,nIterRec), end='\r')
	B_filt_before = lttb.B_filt
	B_filt_rec_before = lttb.B_filt_rec
	training_rec(nIterRec=1, eta=2.5, eta_out=0.0025, print_err=False)
	B_filt_after = lttb.B_filt
	B_filt_rec_after = lttb.B_filt_rec
	self_dist_B.append(np.std(B_filt_after-B_filt_before)**2)
	self_dist_B_rec.append(np.std(B_filt_rec_after-B_filt_rec_before)**2)
	dist_B.append(np.std(B_filt_rec_after-B_filt_after)**2)

print()

"""
for key in ['full','turnoff','switch']:
	res = context_experiment(exp_type=key+'_context', t_sw=t_sw, show_plots=False)
	results[key] = {}
	if key=='full' or key=='noisy':
		results[key]['MSEs_onDiag'] = np.mean(res['mses_onDiag'])
		results[key]['MSEs_offDiag'] = np.mean(res['mses_offDiag'])
	elif key=='turnoff' or key=='switch':
		results[key]['MSEs_onDiag_before'] = np.mean(res['mses_before_onDiag'])
		results[key]['MSEs_onDiag_after'] = np.mean(res['mses_after_onDiag'])
		results[key]['MSEs_offDiag_before'] = np.mean(res['mses_before_offDiag'])
		results[key]['MSEs_offDiag_after'] = np.mean(res['mses_after_offDiag'])
	results[key]['contexts'] = [_.tolist() for _ in res['contexts']]
	results[key]['Ys'] = [_.tolist() for _ in res['outputs']]
	results[key]['targs'] = [_.tolist() for _ in res['targs']]
	#results[key]['S_somas'] = [_.tolist() for _ in res['S_somas']]
	#results[key]['S_winds'] = [_.tolist() for _ in res['S_winds']]
	#results[key]['B_filt_recs'] = [_.tolist() for _ in res['B_filt_recs']]

if run_noisy_context:
	key = 'noisy_full'
	results[key] = {}
	for sigma in np.arange(0.00,1.01,0.025):
		sigma = round(sigma, 4)
		res = context_experiment(exp_type=key+'_context', t_sw=t_sw, sigma_noisy_cont=sigma, show_plots=False)
		results[key][sigma] = {}
		results[key][sigma]['MSEs_onDiag'] = np.mean(res['mses_onDiag'])
		results[key][sigma]['MSEs_offDiag'] = np.mean(res['mses_offDiag'])
		results[key][sigma]['contexts'] = [_.tolist() for _ in res['contexts']]
		results[key][sigma]['Ys'] = [_.tolist() for _ in res['outputs']]
		results[key][sigma]['targs'] = [_.tolist() for _ in res['targs']]
		#results[key][sigma]['S_somas'] = [_.tolist() for _ in res['S_somas']]
		#results[key][sigma]['S_winds'] = [_.tolist() for _ in res['S_winds']]
		#results[key][sigma]['B_filt_recs'] = [_.tolist() for _ in res['B_filt_recs']]
"""

results['self_dist_B'] = self_dist_B
results['self_dist_B_rec'] = self_dist_B_rec
results['dist_B'] = dist_B

results['number_B'] = len([_ for _ in lttb.B.flatten() if _>0])
results['number_B_rec'] = len([_ for _ in lttb.B_rec.flatten() if _>0])

with open(wk_dir + '/%03d_results.json' % run_number, 'w') as fp:
	json.dump(results, fp)



