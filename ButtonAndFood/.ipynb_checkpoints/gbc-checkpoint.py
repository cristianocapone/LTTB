"""
    © 2021 This work is licensed under a CC-BY-NC-SA license.
    Title: *"Behavioral cloning in recurrent spiking networks: A comprehensive framework"*
    Authors: Anonymus
"""

import numpy as np
import utils as ut
from optimizer import Adam, SimpleGradient
from tqdm import trange

class GBC:
    """
        This is the base Model class which represent a recurrent network
        of binary {0, 1} stochastic spiking units with intrinsic potential. A
        nove target-based training algorithm is used to perform temporal sequence
        learning via likelihood maximization.
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

        self.Jout = np.zeros ((self.O, self.N)) 

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

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv)) 

    def step (self, inp, t):
        itau_m = self.itau_m 
        itau_s = self.itau_s 

        self.S_hat [:] = self.S_hat [:] * itau_s + self.S [:] * (1. - itau_s)

        self.H [:] = self.H [:] * itau_m + (1. - itau_m) * (self.J @ self.S_hat [:] + self.Jin @ inp + self.h)\
                                                          + self.Jreset @ self.S [:] 

        self.S [:] = self._sigm (self.H [:], dv = self.dv) - 0.5 > 0. 

        # Here we use our policy to suggest a novel action given the system
        # current state
        action = self.policy (self.S) 

        # Here we return the chosen next action
        return action, self.S.copy ()

    def policy (self, state):
        self.out = self.out * self.itau_ro  + state * (1 - self.itau_ro) 

        return self.Jout @ self.out 

    def compute (self, inp, init = None, rec = True):
        '''
            This function is used to compute the output of our model given an
            input.
            Args:
                inp : numpy.array of shape (N, T), where N is the number of neurons
                      in the network and T is the time length of the input.
                init: numpy.array of shape (N, ), where N is the number of
                      neurons in the network. It defines the initial condition on
                      the spikes. Should be in range [0, 1]. Continous values are
                      casted to boolean.
            Keywords:
                Tmax: (default: None) Optional time legth of the produced output.
                      If not provided default is self.T
                Vo  : (default: None) Optional initial condition for the neurons
                      membrane potential. If not provided defaults to external
                      field h for all neurons.
                dv  : (default: None) Optional different value for the dv param
                      to compute the sigmoid activation.
        '''
        # Check correct input shape
        assert inp.shape[0] == self.N 

        N, T = inp.shape 

        itau_m = self.itau_m 
        itau_s = self.itau_s 

        self.reset (init) 

        Sout = np.zeros ((N, T)) 

        for t in range (T - 1):
            self.S_hat [:] = self.S_hat [:] * itau_s + self.S [:] * (1. - itau_s)

            self.H [:] = self.H * itau_m + (1. - itau_m) * ((self.J @ self.S_hat [:] if rec else 0)
                                                            + inp [:, t] + self.h)\
                                                         + self.Jreset @ self.S 

            self.S [:] = self._sigm (self.H, dv = self.dv) - 0.5 > 0. 
            Sout [:, t + 1] = self.S.copy () 

        return Sout, self.Jout @ ut.sfilter (Sout, itau = self.itau_ro) 

    def implement (self, experts, hint = None, adapt = False, rec = False):
        if (self.J != 0).any () and rec:
            print ('WARNING: Implement expert with non-zero recurrent weights\n') 

        # First we parse experts input-output pair
        inps = np.array ([exp[0] for exp in experts]) 
        outs = np.array ([exp[1] for exp in experts]) 

        # Here we extract the maxima of input and output which are used to balance
        # the signal injection using the sigma_input and sigma_teach variables
        if adapt:
            self.sigma_input = 5. / np.max (np.abs (inps)) 
            self.sigma_teach = 5. / np.max (np.abs (outs)) 

            self.par['sigma_input'] = self.sigma_input 
            self.par['sigma_teach'] = self.sigma_teach 

            self.Jin = np.random.normal (0., self.sigma_input, size = (self.N, self.I)) 
            self.Jteach = np.random.normal (0., self.sigma_teach, size = (self.N, self.O)) 

        # We then build a target network dynamics
        Inp = np.einsum ('ij, njt->nit', self.Jin, inps) 
        Tar = np.einsum ('ij, njt->nit', self.Jteach, outs)

        tInp = Inp + Tar + (0 if hint is None else self.Jhint @ hint)

        Targ = [self.compute (t_inp, rec = rec)[0] for t_inp in tInp] 

        return Targ, Inp 

    def clone (self, experts, targets, epochs = 500, rank = None, clump = False,
               feedback = 'random', track = False, validate = None, verbose = False):
        assert len (experts) == len (targets) 

        # Here we clone this behaviour
        itau_m = self.itau_m 
        itau_s = self.itau_s
        itau_star = self.itau_star 

        alpha = self.par['alpha'] 
        alpha_rout = self.par['alpha_rout'] 
        
        adam_out = Adam (alpha = alpha_rout, drop = .9, drop_time = max (epochs // 10, 1)) 
        adam_rec = Adam (alpha = alpha, drop = .9, drop_time = max (epochs // 10, 1) * self.T)

        targets = np.array (targets)
        inps = np.array ([self.Jin @ exp[0] for exp in experts]) 
        outs = np.array ([exp[1] for exp in experts]) 

        dH = np.zeros (self.N)
        S_pred = np.zeros (self.N)
        T_pred = np.zeros (self.N) 

        train_err = np.zeros (epochs) 
        valid_err = np.zeros (epochs)
        delta_err = np.zeros (epochs)

        # Here we define the rank of the feedback matrix B for training
        rank = self.N if rank is None else rank

        if feedback == 'random':
            B = np.random.normal (0.,  1. / np.sqrt(rank), size = (rank, self.N)) 
            B = B.T @ B
        elif feedback == 'diagonal':
            B = np.eye (self.N) / rank
            B[rank:] = 0 
        else:
            raise ValueError (f'Unsupported feedback structure {feedback}.')

        # Here we train the network
        iterator = trange (epochs, leave = False, desc = 'Cloning') if verbose else range(epochs)
        
        for epoch in iterator:
            ut.shuffle ((inps, outs, targets)) 

            for inp, out, targ in zip (inps, outs, targets):
                self.reset () 
                dH *= 0
                S_pred *= 0
                T_pred *= 0

                for t in range (self.T - 1):
                    S = np.hstack ((targ [:rank, t], self.S[rank:])) if clump else self.S

                    self.S_hat [:] = self.S_hat * itau_s + S * (1. - itau_s) 
                    self.H [:] = self.H * itau_m + (1. - itau_m) * (self.J @ self.S_hat + inp [:, t] + self.h)\
                                                                 + self.Jreset @ S 

                    # dH [:] = dH  * (1. - itau_m) + itau_m * self.S_hat 
                    dH [:] = dH  * itau_m + self.S_hat * (1. - itau_m)

                    self.S [:] = self._sigm (self.H, dv = self.dv) - 0.5 > 0.
                    S_pred [:] = S_pred * itau_star + self.S * (1. - itau_star)
                    T_pred [:] = T_pred * itau_star + targ [:, t + 1] * (1. - itau_star)

                    # dJ = np.outer (B @ (targ [:, t + 1] - self._sigm(self.H, dv = self.dv)), dH)
                    dJ = np.outer (B @ (T_pred - S_pred), dH)

                    self.J = adam_rec.step (self.J, dJ) 
                    np.fill_diagonal (self.J, 0.) 

            # Here we validate the recurrent training via DS inspection
            def DS (inp, tar): return np.sum(np.abs(self.compute(inp)[0] - tar))
            delta_err[epoch] = np.mean ([DS (inp, tar) for inp, tar in zip (inps, targets)])

        # ======= READOUT TRAINING: AFTER RECURRENT WEIGHTS ARE FIXED =======
        iterator = trange(epochs, desc = 'Readout Training', leave = False) if verbose else range(epochs)
        
        for epoch in iterator:
            ut.shuffle ((inps, outs)) 

            for inp, out in zip(inps, outs):
                s, s_out = self.compute (inp)

                # Readout Training
                dJ = (out - s_out) @ s.T 
                self.Jout = adam_out.step (self.Jout, dJ)
                 
            # Here we track the training measures: training and validation error
            def MSE(inp, out): return np.mean ((out - self.compute (inp)[1])**2)

            if track: 
                train_err[epoch] = np.mean ([MSE(inp, out) for inp, out in zip ( inps,  outs)])
            
            if validate:
                vinps = np.array ([self.Jin @ exp[0] for exp in validate]) 
                vouts = np.array ([           exp[1] for exp in validate])

                valid_err[epoch] = np.mean ([MSE(inp, out) for inp, out in zip (vinps, vouts)])

        return train_err, valid_err, delta_err

    def train_rec (self, experts, targets, epochs = 500, rank = None, feedback = 'random', 
                    clump = False, validate = False, verbose = False):
        assert len (experts) == len (targets) 

        itau_m = self.itau_m 
        itau_s = self.itau_s
        itau_star = self.itau_star 

        alpha = self.par['alpha'] 
        
        adam_rec = Adam (alpha = alpha, drop = .9, drop_time = max (epochs // 10, 1) * self.T)

        targets = np.array (targets)
        inps = np.array ([self.Jin @ exp[0] for exp in experts]) 
        outs = np.array ([exp[1] for exp in experts]) 

        dH = np.zeros (self.N)
        S_pred = np.zeros (self.N)
        T_pred = np.zeros (self.N) 

        if validate: delta_err = np.zeros ((epochs, self.N))

        # Here we define the rank of the feedback matrix B for training
        rank = self.N if rank is None else rank

        if feedback == 'random':
            B = np.random.normal (0.,  1. / np.sqrt(self.N), size = (rank, self.N)) 
            B = B.T @ B
        elif feedback == 'diagonal':
            B = np.eye (self.N) / self.N
            B[rank:] = 0 
        else:
            raise ValueError (f'Unsupported feedback structure {feedback}.')

        # Here we train the network
        iterator = trange (epochs, leave = False, desc = 'Rec. Cloning') if verbose else range(epochs)
        
        for epoch in iterator:
            ut.shuffle ((inps, outs, targets)) 

            for inp, _, targ in zip (inps, outs, targets):
                self.reset () 
                dH *= 0
                S_pred *= 0
                T_pred *= 0

                for t in range (self.T - 1):
                    S = np.hstack ((targ [:rank, t], self.S[rank:])) if clump else self.S

                    self.S_hat [:] = self.S_hat * itau_s + S * (1. - itau_s) 
                    self.H [:] = self.H * itau_m + (1. - itau_m) * (self.J @ self.S_hat + inp [:, t] + self.h)\
                                                        + self.Jreset @ S 

                    dH [:] = dH  * itau_m + self.S_hat * (1. - itau_m)

                    self.S [:] = self._sigm (self.H, dv = self.dv) - 0.5 > 0. 
                    S_pred [:] = S_pred * itau_star + self.S * (1. - itau_star)
                    T_pred [:] = T_pred * itau_star + targ [:, t + 1] * (1. - itau_star)

                    # dJ = np.outer (B @ (targ [:, t + 1] - self._sigm(self.H, dv = self.dv)), dH)
                    dJ = np.outer (B @ (T_pred - S_pred), dH)

                    self.J = adam_rec.step (self.J, dJ) 
                    np.fill_diagonal (self.J, 0.) 

            if validate:
                # Here we validate the recurrent training via DS inspection
                # def DS (inp, tar): return np.sum(np.abs(self.compute(inp)[0] - tar))
                def DS (inp, tar): return np.mean(np.abs(self.compute(inp)[0] - tar), axis = 1)
                delta_err[epoch, :] = np.mean ([DS (inp, tar) for inp, tar in zip (inps, targets)], axis = 0)

        return delta_err if validate else None

    def errclone(self, experts, epochs = 500, feedback = 'random', burnin = False, track = False, verbose = False):
        # Here we clone this behaviour
        itau_m = self.itau_m 
        itau_s = self.itau_s 
        itau_ro = self.par['beta_ro']

        alpha = self.par['alpha'] 
        alpha_rout = self.par['alpha_rout'] 
        
        inps = np.array ([self.Jin @ exp[0] for exp in experts]) 
        outs = np.array ([exp[1] for exp in experts]) 

        dH = np.zeros (self.N)
        S_pred = np.zeros (self.N) 

        train_err = np.zeros (epochs)

        adam_out = Adam (alpha = alpha_rout, drop = .99, drop_time = max (epochs // 10, 1) * len(inps)) 
        adam_rec = Adam (alpha = alpha, drop = .99, drop_time = max (epochs // 10, 1) * self.T * len(inps))

        # Here we define the rank of the feedback matrix B for training
        if feedback == 'random':
            B = np.random.normal (0.,  1. / np.sqrt(self.N), size = (self.N, self.O)) 
        else:
            raise ValueError (f'Unsupported feedback structure {feedback}.')

        # Pretrain the readout
        if burnin:
            adam_burnin = Adam (alpha = alpha_rout, drop = .9, drop_time = max (epochs // 10, 1) * len(inps)) 

            for _ in range (burnin):
                for inp, out in zip (inps, outs):
                    # Readout Training
                    s, s_out = self.compute (inp)

                    dJ = (out - s_out) @ s.T 
                    self.Jout = adam_burnin.step (self.Jout, dJ)

        # Here we train the network
        iterator = trange (epochs, leave = False, desc = 'Error-Based Cloning') if verbose else range(epochs)

        for epoch in iterator:
            ut.shuffle ((inps, outs)) 

            for inp, out in zip (inps, outs):
                self.reset () 
                dH *= 0
                S_pred [:] *= 0

                # Readout Training
                s, s_out = self.compute (inp)

                dJ = (out - s_out) @ s.T 
                self.Jout = adam_out.step (self.Jout, dJ)

                for t in range (self.T - 1):
                    self.S_hat [:] = self.S_hat * itau_s + self.S * (1. - itau_s) 
                    self.H [:] = self.H * (1. - itau_m) + itau_m * (self.J @ self.S_hat + inp [:, t] + self.h)\
                                                        + self.Jreset @ self.S 

                    dH [:] = dH  * (1. - itau_m) + itau_m * self.S_hat 

                    self.S [:] = self._sigm (self.H, dv = self.dv) - 0.5 > 0. 
                    S_pred [:] = S_pred [:] * itau_ro + self.S * (1. - itau_ro)

                    # Recurrent Training
                    yout = self.Jout @ S_pred
                    yerr = (out[:, t] - yout)
                    dJ = np.outer ((B @ yerr) * self._dsigm(self.H, dv = self.dv), dH)

                    self.J = adam_rec.step (self.J, dJ) 
                    np.fill_diagonal (self.J, 0.)

            if track:
                # Here we track the training measures: training and validation error
                def MSE(inp, out): return np.mean ((out - self.compute (inp)[1])**2)

                train_err[epoch] = np.mean ([MSE(inp, out) for inp, out in zip ( inps,  outs)])

        return train_err if track else None

    def reset (self, init = None):
        self.S [:] = init if init else np.zeros (self.N) 
        self.S_hat [:] = self.S [:] * self.itau_s if init else np.zeros (self.N) 

        self.out [:] *= 0 

        self.H [:] = self.Vo 

    def forget (self, J = None, Jout = None):
        self.J = np.random.normal (0., self.par['sigma_Jrec'], size = (self.N, self.N)) if J is None else J.copy()
        self.Jout = np.zeros ((self.O, self.N)) if Jout is None else Jout.copy()

    def save (self, filename):
        # Here we collect the relevant quantities to store
        data_bundle = (self.Jin, self.Jteach, self.Jout, self.J, self.par)

        np.save (filename, np.array (data_bundle, dtype = np.object)) 

    @classmethod
    def load (cls, filename):
        data_bundle = np.load (filename, allow_pickle = True) 

        Jin, Jteach, Jout, J, par = data_bundle 

        obj = GBC (par) 

        obj.Jin = Jin.copy () 
        obj.Jteach = Jteach.copy () 
        obj.Jout = Jout.copy () 
        obj.J = J.copy () 

        return obj 
