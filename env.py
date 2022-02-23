"""
	This is the Learning Through Target Bursts (LTTB) repository for code associated to the arXiv 2201.11717 preprint paper:
	Cristiano Capone*, Cosimo Lupo*, Paolo Muratore, Pier Stanislao Paolucci (2022)
	"Burst-dependent plasticity and dendritic amplification support target-based learning and hierarchical imitation learning"
	
	Please give credit to this paper if you use or modify the code in a derivative work.
	This work is licensed under the Creative Commons Attribution 4.0 International License.
	To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
	or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch

from matplotlib.gridspec import GridSpec

from time import sleep

class Unlock:
    '''
        This class represent and Unlock environment where the goal is to first
        go an push a button (by reaching a particular spot) which `unlocks` the
        target, and then go and actually reach the target itself. If target is
        reached prior to unlocking, no reward is obtained.
        Possible action is the agent move in the next time step.
    '''
    def __init__(self, init = None, targ = None, vtarg = None, btn = None, unit = (1., 1.),
                 max_T = 1500, extent = ((-1, -1), (1, 1)), res = 25, render = False, rs = (0.1, 0.03), N = 100):
        # Here we collect environment duration and span
        self.max_T = max_T 
        self.extent = np.array (extent) 
        self.scale = np.mean (self.extent[1] - self.extent[0])
        self.dt, self.dx = unit 
        self.res = res 
        self.max_T = max_T
        self.r_lock, self.r_targ = rs

        # Keep internal time
        self.t = 0 

        # Here we init observation array and done flag
        self.obv = np.empty (4 * res) 
        self.done = False 

        # Here we init the position of target and agent and button
        self.targ = np.array (targ) if targ is not None else np.random.uniform (*extent) 
        self.agen = np.array (init) if init is not None else np.zeros (2) 
        self.btn  = np.array (btn) if btn is not None else np.random.uniform (*extent) 

        self.vtarg = np.array (vtarg) if vtarg is not None else np.zeros (2) 

        # Here we initialize the lock flag to True to signal that target is locked
        self.locked = True 

        # Here we init the reward to zero
        self.r = 0. 

        # Here we init the rendering variables
        self.atraj = np.empty ((2, max_T)) 
        self.ttraj = np.empty ((2, max_T)) 

        self.atraj [:, 0] = self.agen 
        self.ttraj [:, 0] = self.targ 

        # Here we prepare for rendering
        self.do_render = render 

        if self.do_render:
            self.fig = plt.figure (figsize = (6, 8)) 
            self.gs = GridSpec (nrows = 2, ncols = 2, height_ratios = [0.7, 0.3],
                                hspace = 0.05, wspace = 0.05, figure = self.fig) 

            self.ax = self.fig.add_subplot (self.gs[0, :]) 

            self.actv_ax = self.fig.add_subplot(self.gs[1, 0]) 
            self.actn_ax = self.fig.add_subplot(self.gs[1, 1]) 

            self.polar_ax = self.fig.add_axes ((0.7, 0.37, 0.2, 0.2), projection = 'polar') 
            self.polar_ax.set_yticklabels ([]) 

            self.actv_ax.set_xticks ([]) 
            self.actv_ax.set_yticks ([]) 
            self.actv_ax.set_xticklabels ([]) 
            self.actv_ax.set_yticklabels ([]) 

            self.ax.set_xlim (self.extent[0][0] + self.agen[0] - 0.1,
                              self.extent[1][0] + self.agen[1] + 0.1) 
            self.ax.set_ylim (self.extent[0][1] + self.agen[0] - 0.1,
                              self.extent[1][1] + self.agen[1] + 0.1) 

            self.ax.set_xticks (np.arange (-10, 10, 0.5)) 
            self.ax.set_yticks (np.arange (-10, 10, 0.5)) 
            self.ax.set_xticklabels ([]) 
            self.ax.set_yticklabels ([]) 

            self.ptarg = self.ax.scatter (*self.targ, s = 500, alpha = 0.5, marker = 'X', color = 'darkred') 
            self.pagen = self.ax.scatter (*self.agen, s = 60, marker = 'p', color = 'darkblue') 
            self.pbtn  = self.ax.scatter (*self.btn,  s = 300, marker = 'D', color = 'darkgreen') 

            self.pttraj, = self.ax.plot (*self.targ, c = 'C3', ls = '--') 
            self.patraj, = self.ax.plot (*self.agen, c = 'C0') 

            self.buff_len = 100 
            self.abuff = np.zeros ((self.buff_len, 2)) 
            self.sbuff = np.zeros ((self.buff_len, N)) 

            self.psbuff = self.actv_ax.imshow (self.sbuff.T, aspect = 'auto',
                                  extent = (0, self.buff_len, 0, N), cmap = 'binary', vmin = 0, vmax = 1) 
            self.pabuff_x, = self.actn_ax.plot (*self.abuff[0], c = 'C1') 
            self.pabuff_y, = self.actn_ax.plot (*self.abuff[0], c = 'C3') 

            self.actn_ax.legend ((self.pabuff_x, self.pabuff_y), ('$v_x$', '$v_y$'),
                                 loc = 1, frameon = False) 

            self.actn_ax.set_yticklabels ([]) 

            self.pol_act_N = plt.Polygon (((0, 1),  (-np.pi, 0.05), (np.pi, 0.05)), color = 'r') 
            self.pol_act_S = plt.Polygon (((0, -1), (-np.pi, 0.05), (np.pi, 0.05)), color = 'b') 

            self.polar_ax.add_patch (self.pol_act_N) 
            self.polar_ax.add_patch (self.pol_act_S) 

            self.time = self.ax.text (-0.2, 1.1, 'Time 0', fontsize = 16) 
            self.fig.subplots_adjust (left = 0.05, right = 0.95, bottom = 0.05, top = 0.98) 
            plt.ion() 

    def step(self, action):
        # Here we apply the provided action to the agent state and update the target.
        self.agen += np.array (action) * self.dt / self.dx
        self.targ += self.vtarg * self.dt / self.dx

        self.atraj [:, self.t] = self.agen 
        self.ttraj [:, self.t] = self.targ 

        # Here we compute the distance to target and button
        t_dist = np.sqrt (np.square (self.targ - self.agen).sum ()) 
        b_dist = np.sqrt (np.square (self.btn - self.agen).sum ()) 

        # Here we build the observation of this environment
        self.obv [:] = np.hstack ([self.encode (self.targ - self.agen),
                                   self.encode (self.btn - self.agen)]) 

        self.locked = b_dist > self.r_lock if self.locked else False 
        self.done = (t_dist < self.r_targ or self.t > self.max_T) and not self.locked

        self.r = 0. if self.locked else 1 / t_dist  

        # Here we increase the env time
        self.t += 1

        return self.obv, self.r, self.done, self.agen 

    def render(self, cam = 'middle', save = None):
        self.patraj.set_data (*self.atraj[:, :self.t]) 
        self.pttraj.set_data (*self.ttraj[:, :self.t]) 

        self.pagen.set_offsets (self.atraj[:, self.t - 1]) 
        self.ptarg.set_offsets (self.ttraj[:, self.t - 1]) 

        self.ptarg.set_alpha (0.5 if self.locked else 1.) 
        self.pbtn.set_sizes ([300] if self.locked else [100]) 

        self.psbuff.set_data (self.sbuff.T) 
        t = np.linspace (max (self.t - self.buff_len, 0), self.t, num = min (self.buff_len, self.t)) 
        self.pabuff_x.set_data ([t, self.abuff[:self.t, 0]]) 
        self.pabuff_y.set_data ([t, self.abuff[:self.t, 1]]) 

        pol = self.abuff[min (self.t, self.buff_len - 1)] 
        r = np.sqrt (np.sum (pol**2)) 
        t = np.arctan2 (*pol[::-1]) 

        self.pol_act_N.set_xy (((t, r), (t + np.pi * .5, r * .1), (t - np.pi / 2, r * .1))) 
        self.pol_act_S.set_xy (((t + np.pi, r), (t + np.pi * .5, r * .1), (t - np.pi * .5, r * .1))) 

        self.polar_ax.set_ylim (0, r + 0.5) 

        if cam == 'agen':
            min_x = self.extent[0][0] + self.atraj[0, self.t - 1] - 0.1 
            max_x = self.extent[1][0] + self.atraj[0, self.t - 1] + 0.1 
            min_y = self.extent[0][1] + self.atraj[1, self.t - 1] - 0.1 
            max_y = self.extent[1][1] + self.atraj[1, self.t - 1] + 0.1 

        elif cam == 'middle':
            min_x = min (self.ttraj[0, self.t - 1], self.atraj[0, self.t - 1], self.btn[0]) - 0.5 
            max_x = max (self.ttraj[0, self.t - 1], self.atraj[0, self.t - 1], self.btn[0]) + 0.5 
            min_y = min (self.ttraj[1, self.t - 1], self.atraj[1, self.t - 1], self.btn[1]) - 0.5 
            max_y = max (self.ttraj[1, self.t - 1], self.atraj[1, self.t - 1], self.btn[1]) + 0.5 

        else:
            raise ValueError('Unknwon camera option: {}'.format (cam)) 

        self.time.set_text ('Time {}'.format (self.t)) 
        self.time.set_position ((0.5 * (min_x + max_x) - 0.05 * (max_x - min_x),
                                 max_y - 0.1 * (max_y - min_y))) 

        self.ax.set_xlim (min_x, max_x) 
        self.ax.set_ylim (min_y, max_y) 

        self.actn_ax.set_xlim (max (self.t - self.buff_len, 0), max (self.t, self.buff_len)) 
        self.actn_ax.set_ylim (np.min (self.abuff[:self.t] - 0.1),
                               np.max (self.abuff[:self.t]) + 0.1) 

        if save: self.fig.savefig (save + str(self.t).zfill(3) + '.png') 
        else:
            # Here we signal redraw
            self.fig.canvas.draw () 
            self.fig.canvas.flush_events() 
            plt.show () 
            sleep (0.02) 

        return self.fig 

    def buffer(self, data):
        S, act = data 

        # Here we roll the buffers
        if self.t >= self.buff_len:
            self.sbuff = np.roll (self.sbuff, -1, axis = 0) 
            self.abuff = np.roll (self.abuff, -1, axis = 0) 

            self.sbuff[-1] = S 
            self.abuff[-1] = act 
        else:
            self.sbuff[self.t] = S 
            self.abuff[self.t] = act 

    def dense_r(self, dist):
        # Dense reward is defined as decaying with the distance between agent
        # position and target.
        return np.exp (-dist * self.inv_scale) 

    def encode(self, pos, res = None):
        if res is None: res = self.res 

        pos = np.array (pos) 
        if len (pos.shape) == 1: pos = pos.reshape (-1, 1) 
        shape = pos.shape 

        x, y = np.clip (pos.T, *self.extent).T 

        mu_x, mu_y = np.linspace (*self.extent, num = res).T 
        s_x, s_y = np.diff (self.extent, axis = 0).T / (res) 

        enc_x = np.exp (-0.5 * ((x.reshape (-1, 1) - mu_x) / s_x)**2).T 
        enc_y = np.exp (-0.5 * ((y.reshape (-1, 1) - mu_y) / s_y)**2).T 

        return np.array ((enc_x, enc_y)).reshape(-1, shape[-1]).squeeze () 

    def reset_target (self, new_targ = None, new_vtarg = None):
        self.targ = np.array (new_targ) if new_targ is not None else np.random.uniform (-1, 1, size = 2) 
        self.vtarg = np.array (new_vtarg) if new_vtarg is not None else np.random.uniform (-1, 1, size = 2) 

        self.ptarg.set_offsets (self.targ) 

    def reset (self, init = None, targ = None, btn = None, vtarg = None):
        self.agen = np.array (init) if init is not None else np.random.uniform (*self.extent) 
        self.targ = np.array (targ) if targ is not None else np.random.uniform (*self.extent) 
        self.btn = np.array (btn) if btn is not None else np.random.uniform (*self.extent) 

        self.vtarg = np.array (vtarg) if vtarg is not None else np.zeros (2) 

        self.atraj = np.empty ((2, self.max_T)) 
        self.ttraj = np.empty ((2, self.max_T)) 

        self.atraj [:, 0] = self.agen 
        self.ttraj [:, 0] = self.targ 

        if self.do_render:
            self.pagen.set_offsets (self.agen) 
            self.ptarg.set_offsets (self.targ) 
            self.pbtn.set_offsets (self.btn) 

            self.patraj.set_data (*self.agen) 
            self.pttraj.set_data (*self.targ) 

        self.locked = True 

        self.t = 0 

    def build_expert (self, targ, init, btn, steps = (200, 200), T = (300, 300), offT = (0, 0), vtarg = None):
        assert np.sum (T) >= (np.sum (steps) + np.sum (offT)) 
        vtarg = np.array (vtarg) * self.dx / self.dt if vtarg else np.zeros (2) 
        vbtn = np.zeros (2) 

        def line_to_targ (init, targ, steps, offT, T, vtarg = None):
            # Calculate final position of target
            v = np.tile ((targ - init) / (steps), (steps, 1)).T 
            t = np.linspace (0, steps, num = steps) 

            _inp = init.reshape (2, -1) + v * t
            _out = v * self.dx / self.dt

            _inp = np.pad (_inp, ((0, 0), (offT, T - offT - _inp.shape [-1])), mode = 'edge') 
            _out = np.pad (_out, ((0, 0), (offT, T - offT - _out.shape [-1])))

            _out[0, :offT] = np.linspace (0, _out[0, offT], num = offT)
            _out[1, :offT] = np.linspace (0, _out[1, offT], num = offT)

            return _inp, _out 

        # First aim at the button
        inp, out1 = line_to_targ (init, btn, steps[0], offT[0], T[0], vtarg = vbtn) 

        # Compute relative position to both target and button
        mid_targ = (targ + vtarg * T[0])
        mid_btn  = (btn + vbtn * T[0])

        inp2targ1 = np.linspace (targ, mid_targ, num = T[0]).T - inp
        inp2btn1  = np.linspace (btn, mid_btn, num = T[0]).T - inp

        # Then aim at the target
        end_targ = mid_targ + vtarg * T[1] 
        end_btn  = mid_btn + vbtn * T[1]

        inp, out2 = line_to_targ (inp[:, -1], end_targ, steps[1], offT[1], T[1], vtarg = vtarg) 

        # Compute relative position to both target and button
        inp2targ2 = np.linspace (mid_targ, end_targ, num = T[1]).T - inp 
        inp2btn2  = np.linspace (mid_btn, end_btn, num = T[1]).T - inp


        # Here we compose the two trajectories
        inp2targ = self.encode (np.hstack ((inp2targ1, inp2targ2))) 
        inp2btn  = self.encode (np.hstack ((inp2btn1, inp2btn2))) 

        out = np.hstack ((out1, out2)) 
        inp = np.vstack ((inp2targ, inp2btn))

        return inp, out 
