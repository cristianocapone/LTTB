"""
	© 2021 This work is licensed under a CC-BY-NC-SA license.
	Title: ---
	Authors: Cristiano Capone et al.
"""

import numpy as np
from tqdm import trange
import random

class LTTB:
	"""
		This is the base Model ...
	"""

	def init_clock (self, par):
		n_steps = self.I
		T = self.par["T"]

		I_clock = np.zeros((n_steps,T))
		for t in range(T):
			k = int(np.floor(t/T*n_steps))
			I_clock[k,t] = 1;
			self.I_clock = I_clock

	def init_targ (self, par):
		T = self.par["T"]
		dt = self.par["dt"]
		y_targ = []

		for k in range(self.O):
			a1 = .5 +1.5*random.uniform(0,1)
			a2 = .5 +1.5*random.uniform(0,1)
			a3 = .5 +1.5*random.uniform(0,1)
			a4 = .5 +1.5*random.uniform(0,1)
			w1 = 1*2*np.pi
			w2 = 2 *2*np.pi
			w3 = 3*2*np.pi
			w4 = 5*2*np.pi
			f1 = random.uniform(0,1)*2*np.pi;
			f2 = random.uniform(0,1)*2*np.pi;
			f3 = random.uniform(0,1)*2*np.pi;
			f4 = random.uniform(0,1)*2*np.pi;
			y_targ.append([a1*np.cos(_*dt*w1+f1) + a2*np.cos(_*dt*w2+f2) + a3*np.cos(_*dt*w3+f3)+ a4*np.cos(_*dt*w4+f4) for _ in range(T)])
			y_targ[k] = (-min(y_targ[k]) + y_targ[k])/(max(y_targ[k])-min(y_targ[k]))*2-1
			self.y_targ = np.array(y_targ)

	def __init__ (self, par):
		# This are the network size N, input I, output O and max temporal span T
		self.N, self.I, self.O, self.T = par['shape']

		self.ndxE = range(par['Ne'])
		self.ndxI = range(par['Ne'],par['N'])

		self.h = par['h']

		self.dt = par['dt']

		self.tau_m = par['tau_m']

		self.itau_m = np.exp (-self.dt / par['tau_m'])
		self.itau_s = np.exp (-self.dt / par['tau_s'])
		self.itau_ro = np.exp (-self.dt / par['tau_ro'])
		self.itau_star = np.exp (-self.dt / par['tau_star'])

		self.dv = par['dv']

		# This is the network connectivity matrix
		self.J = np.random.normal (0., par['sigma_Jrec'], size = (self.N, self.N))
		self.w = np.random.normal (0., par['sigma_wrec'], size = (self.N, self.N))

		# self.J = np.zeros ((self.N, self.N))

		# This is the network input, teach and output matrices
		self.j_in = np.random.normal (0., par['sigma_in'], size = (self.N, self.I))
		self.j_targ = np.random.normal (0., par['sigma_targ'], size = (self.N, self.O))
		self.j_targ[self.ndxI] = 0

		self.Jdiag = np.diag(-20*np.ones(self.N))

		# This is the hint signal
		try:
			self.H = par['hint_shape']
			self.Jhint = np.random.normal (0., par['sigma_hint'], size = (self.N, self.H))
		except KeyError:
			self.H = 0
			self.Jhint = None

		self.Jout = np.random.normal (0., par['sigma_Jout'], size = (self.O, self.N))
		# self.Jout = np.zeros ((self.O, self.N))

		# Remove self-connections
		np.fill_diagonal (self.J, 0.)

		# Impose reset after spike
		self.s_inh = -par['s_inh']
		self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

		# This is the external field
		#h = par['h']

		#assert type (h) in (np.ndarray, float, int)
		#self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h

		# Membrane potential
		self.H = np.ones (self.N) * par['Vo']
		self.Vo = par['Vo']

		# These are the spikes train and the filtered spikes train
		self.S = np.zeros (self.N)
		self.S_hat = np.zeros (self.N)

		# This is the single-time output buffer
		self.out = np.zeros (self.N)

		# Here we save the params dictionary
		self.par = par

		self.b = 100


		self.apicalFactor = 1

		self.me = 0.01
		self.mi = 4 * self.me

		self.href = -5

	def initialize (self, par):

		self.t = 0

		self.S_soma = np.zeros((self.N,self.T))
		self.S_apic_dist = np.zeros((self.N,self.T))
		self.S_apic_prox = np.zeros((self.N,self.T))
		self.W = np.zeros((self.N,self.T))

		self.Isoma = np.zeros((self.N,self.T))

		self.S_wind_apic = np.zeros((self.N,self.T))
		self.S_wind_cont = np.zeros((self.N,self.T))
		self.S_wind_soma = np.zeros((self.N,self.T))
		self.S_wind = np.zeros((self.N,self.T))

		self.B_filt = np.zeros((self.N,self.T))
		self.B_filt_rec = np.zeros((self.N,self.T))

		self.S_wind_targ = np.zeros((self.N,self.T))

		self.S_filt_targ = np.zeros((self.N,self.T))
		self.S_filt = np.zeros((self.N,self.T))
		self.S_filt_soma = np.zeros((self.N,self.T))
		self.S_filt_apic = np.zeros((self.N,self.T))
		self.S_filt_apicCont = np.zeros((self.N,self.T))
		self.S_filtRO = np.zeros((self.N,self.T))

		self.S_wind_targ_filt = np.zeros((self.N,self.T))

		self.Vapic = np.zeros((self.N,self.T)) + self.h
		self.VapicRec = np.zeros((self.N,self.T)) + self.h
		self.Vsoma = np.zeros((self.N,self.T)) + self.h

		self.Vreset = -20
		self.VresetApic = -80*2
		self.apicalThreshold = .05/4/4
		self.somaticThreshold = .05/2
		self.burstThreshold = .05/4
		self.targThreshold = .08

	def step (self):
		self.t += 1
		t = self.t

		itau_m = self.itau_m
		itau_s = self.itau_s

		# Qui mi da' errore
		# Credo sia dovuto al fatto che J e' una matrice NxN, e moltiplica il vettore N-dimensionale S_filt.
		# Bisogna quindi inserire il prodotto matriciale?

		self.VapicRec[:,t] = self.VapicRec[:,t-1]*(1-self.dt/self.tau_m) + self.dt/self.tau_m*(self.J@self.S_filt[:,t-1] +  self.h + self.href) + self.VresetApic*self.S_apic_prox[:,t-1]

		self.Vapic[:,t] = self.Vapic[:,t-1]*(1-self.dt/self.tau_m) + self.dt/self.tau_m*(self.j_targ@self.y_targ[:,t-1]*self.apicalFactor + self.h + self.href) + self.VresetApic*self.S_apic_dist[:,t-1]

		self.Isoma[:,t] = self.w@self.S_filt[:,t-1] + self.h + self.j_in@self.I_clock[:,t-1] + self.S_wind[:,t-1]*20 - self.b*self.W[:,t-1]
		self.Vsoma[:,t] = (self.Vsoma[:,t-1]*(1-self.dt/self.tau_m)+ self.dt/self.tau_m*( self.Isoma[:,t-1] ) ) * (1-self.S_soma[:,t-1]) + self.Vreset*self.S_soma[:,t]/(1 + 2*self.S_wind[:,t-1])
