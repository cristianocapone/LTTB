"""
	© 2021 This work is licensed under a CC-BY-NC-SA license.
	Title: ---
	Authors: Cristiano Capone et al.
"""

import numpy as np
from tqdm import trange

class LTTB:
	"""
		This is the base Model ...
	"""

	def __init__ (self, par):
		# This are the network size N, input I, output O and max temporal span T
		self.N, self.I, self.O, self.T = par['shape']

		self.dt = par['dt']

		self.itau_m = np.exp (-self.dt / par['tau_m'])
		self.itau_s = np.exp (-self.dt / par['tau_s'])
		self.itau_ro = np.exp (-self.dt / par['tau_ro'])
		self.itau_star = np.exp (-self.dt / par['tau_star'])

		self.dv = par['dv']

		# This is the network connectivity matrix
		self.J = np.random.normal (0., par['sigma_Jrec'], size = (self.N, self.N))
		# self.J = np.zeros ((self.N, self.N))

		# This is the network input, teach and output matrices
		self.J_in = np.random.normal (0., par['sigma_in'], size = (self.N, self.I))
		self.J_targ = np.random.normal (0., par['sigma_targ'], size = (self.N, self.O))

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
		h = par['h']

		assert type (h) in (np.ndarray, float, int)
		self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h

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

		self.t = 0

		self.apicalFactor = 1

		self.me = 0.01
		self.mi = 4 * self.me

		self.href = -5

		self.S_soma = np.zeros((self.N,self.T))
		self.S_apic_dist = np.zeros((self.N,self.T))
		self.S_apic_prox = np.zeros((self.N,self.T))
		self.W = np.zeros((self.N,self.T))

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

		self.Vapic = np.zeros((self.N,self.T)) + h
		self.VapicRec = np.zeros((self.N,self.T)) + h
		self.Vsoma = np.zeros((self.N,self.T)) + h

		self.Vreset = -20
		self.VresetApic = -80*2
		self.apicalThreshold = .05/4/4
		self.somaticThreshold = .05/2
		self.burstThreshold = .05/4
		self.targThreshold = .08

	def step (self ):
		self.t += 1

		itau_m = self.itau_m
		itau_s = self.itau_s


		VapicRec[:,self.t] = VapicRec[:,self.t-1]*(1-self.dt/self.tau_m)+ self.dt/self.tauH*(self.J*self.S_filt[:,self.t-1] +  self.h + self.href  ) + self.VresetApic*self.S_apic_prox[:,self.t-1]

	"""


    %% apical rec comp

    VapicRec = VapicRec*(1-dt/tauH)+ dt/tauH*(J*S_filt(:,t)+  h + href  ) + VresetApic*SapicRec(:,t);

    %% apical cont comp

    Vapic = Vapic*(1-dt/tauH)+ dt/tauH*( jIn*X(:,t)*apicalFactor + h + href  ) + VresetApic*Sapic(:,t);

    %% somatic comp

    Isoma = w*S_filt(:,t) + h + jInClock*xClock(:,t) + S_wind(:,t)*20 - b*W(:,t);%15;
    Vsoma = (Vsoma*(1-dt/tauH)+ dt/tauH*( Isoma ) ).*(1-Ssoma(:,t)) + Vreset*Ssoma(:,t)./(1 + 2*S_wind(:,t)) ;%

    %%

    SapicRec(:,t+1) = heaviside( f(VapicRec) - .5 );

    Sapic(:,t+1) =  heaviside( f(Vapic) - .5 )  ;
    Sapic_targ(:,t+1) =  Sapic(:,t+1);
    Ssoma(:,t+1) = heaviside( f(Vsoma) - .5 );
    Ssoma_targ(:,t+1) =  Ssoma(:,t+1);

    S_filt(:,t+1) = S_filt(:,t)*beta + Ssoma(:,t+1)*(1-beta);
    S_filtRO(:,t+1) = S_filtRO(:,t)*betaRO + Ssoma(:,t+1)*(1-betaRO);

    S_filt_soma(:,t+1) = S_filt_soma(:,t)*beta_targ + Ssoma(:,t+1)*(1-beta_targ);
    %   S_filt_apic(:,t+1) = S_filt_apic(:,t)*beta_targ + Sapic(:,t+1)*(1-beta_targ);
    %   S_filt_apicCont(:,t+1) = S_filt_apicCont(:,t)*beta_targ + Sapic(:,t+1)*(1-beta_targ);
    W(:,t+1) = W(:,t)*beta_W + Ssoma(:,t+1)*(1-beta_W);

    %   S_wind_apic(:,t+1) = heaviside(S_filt_apic(:,t+1) - apicalThreshold);
    %   S_wind_apicCont(:,t+1) = heaviside(S_filt_apicCont(:,t+1) - apicalThreshold);
    S_wind_soma(:,t+1) = heaviside(S_filt_soma(:,t+1) - somaticThreshold);

    B(:,t+1) = S_wind_soma(:,t+1).*Sapic(:,t+1) ;
    %B(:,t+1) = Sapic(:,t+1) ;

    B_rec(:,t+1) = S_wind_soma(:,t+1).*SapicRec(:,t+1) ;

    B_filt(:,t+1)  = B_filt(:,t)*beta_targ + B(:,t+1)*(1-beta_targ);
    B_filt_rec(:,t+1)  = B_filt_rec(:,t)*beta_targ + B_rec(:,t+1)*(1-beta_targ);

    %%

    S_wind_pred(:,t+1) = heaviside( B_filt_rec(:,t+1) - burstThreshold) ;
    S_wind_targ(:,t+1) = heaviside( B_filt(:,t+1) - burstThreshold) ;%S_wind_apic(:,t+1).*S_wind_soma(:,t+1);%heaviside(S_filtRO(:,t+1)-targThreshold);%.* S_wind_apic(:,t+1);

    S_wind_targ_filt(:,t+1) = S_wind_targ_filt(:,t)*betaRO + S_wind_targ(:,t+1)*(1-betaRO);

    %S_wind(:,t+1) = S_wind_targ(:,t+1);
    S_wind(:,t+1) = min( S_wind_pred(:,t+1)+S_wind_targ(:,t+1) ,1);

    V(:,t) = Vsoma;
    Vap(:,t) = Vapic;


	"""

	"""
    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv

        # If dv is too small, no need for elaborate computation, just return
        # the theta function
        if dv < 1 / 30:
            return x > 0

        # Here we apply numerically stable version of signoid activation
        # based on the sign of the potential
        y = x / dv

        out = np.zeros (x.shape)
        mask = x > 0
        out [mask] = 1. / (1. + np.exp (-y [mask]))
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]))

        return out

    def step (self ):
        itau_m = self.itau_m
        itau_s = self.itau_s

        self.S_hat [:] = self.S_hat [:] * itau_s + self.S [:] * (1. - itau_s)
        self.H [:] = self.H [:] * itau_m + (1. - itau_m) * (self.J @ self.S_hat [:]  + self.h) + self.Jreset @ self.S [:]

        self.S [:] = self._sigm (self.H [:], dv = self.dv) - 0.5 > 0.
		"""
