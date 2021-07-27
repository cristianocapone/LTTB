"""
    © 2021 This work is licensed under a CC-BY-NC-SA license.
    Title: ---
    Authors: Anonymus
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
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I))
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O))

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
