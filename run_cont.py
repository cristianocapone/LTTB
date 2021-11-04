"""
	© 2021 This work is licensed under a CC-BY-NC-SA license.
	Title: ---
	Authors: Cristiano Capone et al.
"""

import numpy as np
from tqdm import trange
import random

cont = "basal" # Can be "distal" or "basal"
n_contexts = 3
n_trial = 5

if(cont=="distal"):
	from lttb_contesto_sopra import LTTB
elif(cont=="basal"):
	from lttb_contesto_sotto import LTTB

N, I, O, T = 500, 5, 3, 400
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

sigma_context = 20.
sigma_targ = 5. #10**1.5
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

Ne = 400
Ni = 100

N = Ne+Ni

# Here we build the dictionary of the simulation parameters
par = {'dt': dt, 'tau_m': tau_m, 'tau_s': tau_s, 'tau_ro': tau_ro, 'tau_star': tau_star, 'tau_W': tau_W,
	   'dv': dv, 'Vo': Vo, 'h': h, 's_inh': s_inh, 'N': N, 'Ni': Ni, 'Ne': Ne, 'T': T, 'I': I, 'O': O, 'shape': shape,
	   'sigma_Jrec': sigma_Jrec, 'sigma_wrec': sigma_wrec, 'sigma_Jout': sigma_Jout, 'n_contexts': n_contexts,
	   'alpha': alpha, 'alpha_rout': alpha_rout,
	   'sigma_in': sigma_in, 'sigma_targ': sigma_targ, 'sigma_context': sigma_context, 'h': h}

TIME = 1.

T = int(np.floor(TIME/dt))

nStepOutTraining = 1000

t_shut = 10

JMAX = 100.
nu_targ = 0.005

# Here we init our (recurrent) agent

lttb = LTTB (par)

# define clock and target

def init_clock_targ():
	
	lttb.y_targ_collection = []
	
	for k in range(n_contexts):
		lttb.init_targ(par)
		lttb.y_targ_collection.append(lttb.y_targ)
	
	lttb.init_clock(par)

## Training Rec

gamma = 10.
def f(x,gamma):
	return np.exp(x*gamma)/(np.exp(x*gamma)+1)
#f1 = @(x)(gamma*exp(gamma*x))./(exp(gamma*x)+1)-(gamma*exp(2*gamma*x))./(exp(gamma*x)+1).^2;

apicalFactorTrain = 1
apicalFactorTest = 0

nIterRec = 50

eta = 5.
eta_out = 0.1
etaW = .0

def training_rec():
	
	for iter in range(nIterRec):
		
		###### Online-Training
		
		#initialize simulation
		
		for cont_index in range(n_contexts):
			
			lttb.cont = lttb.cont*0
			lttb.cont[cont_index] = 1
				
			lttb.y_targ = lttb.y_targ_collection[cont_index]
				
			lttb.initialize(par)
				
			# run simulation
			dH = 0
					
			for t in range(lttb.T-2):
						
				lttb.step(apicalFactor = apicalFactorTrain)
						
				dH = dH*(1-dt/tau_m) + dt/tau_m*lttb.S_filt[:,t]
						
				DJ = np.outer(( lttb.S_apic_dist[:,t+1] - f(lttb.VapicRec[:,t],gamma) )*(1-lttb.S_apic_dist[:,t]) ,dH)
				lttb.J =  lttb.J + eta*DJ
							
				SR = lttb.S_filtRO[:,t+1]#lttb.B_filt_rec[:,t+1]#
				Y = lttb.Jout@SR
				DJRO = np.outer(lttb.y_targ[:,t+1] - Y,SR.T)
				lttb.Jout =  lttb.Jout + eta_out*DJRO

		###### Test

		if(iter%5 == 0):
			
			for cont_index in range(n_contexts):
		
				lttb.cont = lttb.cont*0
				lttb.cont[cont_index] = 1
		
				lttb.y_targ = lttb.y_targ_collection[cont_index]
		
				lttb.initialize(par)
		
				# run simulation
		
				for t in range(lttb.T-2):
			
					lttb.step(apicalFactor = apicalFactorTest)
	
				SR = lttb.S_filtRO[:,1:-2]# lttb.B_filt_rec[:,1:-2]#
				Y = lttb.Jout@SR
				mse_rec_train = np.std(lttb.y_targ[:,1:-2] - Y)**2

				#print(mse_rec_train)


def run_many_trials():
	
	MSE = np.zeros((n_contexts,n_contexts))

	for cont_index_i in range(n_contexts):
		for cont_index in range(n_contexts):
			
			context = []
		
			lttb.cont = lttb.cont*0
			lttb.cont[cont_index_i] = 1
			
			lttb.y_targ = lttb.y_targ_collection[cont_index]
					
			lttb.initialize(par)
				
			# run simulation
					
			apicalFactor = 0
						
			for t in range(lttb.T-2):
							
				if t==200:
					apicalFactor = 0
					#lttb.cont = lttb.cont*0.1
					#lttb.cont[0] = 1
				context.append(lttb.cont)
									
				lttb.step(apicalFactor = apicalFactor)
											
			SR = lttb.S_filtRO[:,1:-2]#lttb.B_filt_rec[:,1:-2]#
			Y = lttb.Jout@SR
			mse_rec_train = np.std(lttb.y_targ[:,1:-3] - Y[:,0:-1])**2

			#print(mse_rec_train)
			MSE[cont_index_i,cont_index] = mse_rec_train
													
		# mse, Y = lttb.train_ro(par,out_epochs = 1)
		
	return MSE




if(cont=="distal"):
	sigma_context_training = [0, 2.5, 5, 10, 30, 50, 100]
	sigma_context_testing = [0, 2.5, 5, 10, 30, 50, 100]
elif(cont=="basal"):
	sigma_context_training = [0, 2.5, 5, 10, 30, 50, 100]
	sigma_context_testing = [0, 2.5, 5, 10, 30, 50, 100]

filename = cont + "_trial_%03d.csv" % n_trial

print('\nContext entering the ' + cont + ' compartment - trial n. ' + str(n_trial) + '\n')

fp = open("./data/" + filename, "a")
fp.write('sigma_train;sigma_test;onDiag_aver_MSE;offDiag_aver_MSE\n')
fp.close()

for sigma_train in sigma_context_training:
	for sigma_test in sigma_context_testing:
		
		par['sigma_context'] = sigma_train
		lttb = LTTB (par)
		init_clock_targ()
		training_rec()
		
		par['sigma_context'] = sigma_test
		MSE = run_many_trials()
		
		onDiag_aver = np.mean([MSE[i][i] for i in range(n_contexts)])
		offDiag_aver = np.mean([MSE[i][j] for i in range(n_contexts) for j in range(n_contexts) if i!=j])
		
		fp = open("./data/" + filename, "a")
		fp.write(str(sigma_train) + ';' + str(sigma_test) + ';' + str(onDiag_aver) + ';' + str(offDiag_aver) + '\n')
		fp.close()

print('...done!')
