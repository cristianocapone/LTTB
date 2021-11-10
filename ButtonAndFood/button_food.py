"""
    © 2021 This work is licensed under a CC-BY-NC-SA license.
    Title: *"Behavioral cloning in recurrent spiking networks: A comprehensive framework"*
    Authors: Anonymus
"""

import json
import numpy as np

from gbc import GBC #--> replace with LTTB
from env import Unlock

from tqdm import tqdm, trange
from itertools import product as Prod

# ============= TEST FUNCTION =================
def test (gbc, env, testset, par):
    rt, rb = r = (par['rt'], par['rb'])
    size = np.shape(testset)[-1]

    hist = {'agent'  : np.zeros ((size, par['T'], 2)),
            'action' : np.zeros ((size, par['T'], par['O'])),
            'theta'  : np.zeros (size),
            'R'      : np.zeros (size)}

    tars = np.array ([(rt * np.cos (t), rt * np.sin (t)) for t in testset])
    btns =  [(0., rb)] * len (testset)

    for i, (targ, btn) in enumerate (zip (tars, btns)):
        env.reset (init = init, targ = targ, btn = btn)
        gbc.reset ()
        R = 0

        state = np.hstack ((env.encode (targ - init), env.encode (btn - init)))

        for t in range (par['T']):
            action, _ = gbc.step (state, t)
            state, r, done, agen = env.step (action)

            R = max (R, r)

            hist['action'][i, t] = action
            hist['agent'][i, t]  = agen

            if done: break

        hist['action'][i, t:] = np.nan
        hist['agent'][i, t:]  = agen
        hist['theta'][i]      = testset[i]
        hist['R'][i]          = R

    return hist

# Loading configuration file
path = 'config.json'
config = 'BUTTON_FOOD'

with open (path, 'r') as f:
    par = json.load (f)[config]

par['hint'] = par['hint'] == 'True'
par['clump'] = par['clump'] == 'True'
par['validate'] = par['validate'] == 'True'
par['verbose'] = par['verbose'] == 'True'

# ==== Environment Initialization ======
init = np.array ((0., 0.))
targ = np.array ((0., 1.))
btn = np.array ((0., 0.))

env = Unlock (init = init, targ = targ, btn = btn, unit = (par['dt'], par['dx']), res = 20)

rt, rb = par['rt'], par['rb']

trainset = np.array (par['trainset'])
validset = np.array (par['validset'])
testset  = np.linspace (*par['testset'])

train_theta = trainset * np.pi / 180.
valid_theta = validset * np.pi / 180.
test_theta  = testset * np.pi / 180.

train_targs = np.array ([(rt * np.cos (t), rt * np.sin (t)) for t in train_theta])
valid_targs = np.array ([(rt * np.cos (t), rt * np.sin (t)) for t in valid_theta])

train_bttns = np.array ([(0, rb) for t in train_theta])
valid_bttns = np.array ([(rb * np.cos (t), rb * np.sin (t)) for t in valid_theta])

# Here we ask the env for the expert behaviour
epar = {'offT' : (1, 1), 'steps' : (29, 49), 'T' : (30, 50)}
train_exp = [env.build_expert (targ, init, btn, **epar) for targ, btn in zip (train_targs, train_bttns)]
valid_exp = [env.build_expert (targ, init, btn, **epar) for targ, btn in zip (valid_targs, valid_bttns)]


# ====== Cloning Behavior for different B ranks ========
rep    = par['rep']
scan   = par['scan']
ranks  = np.linspace (1, par['N'], num = par['scan'], endpoint = True, dtype = np.int)

Test      = np.zeros ((rep, scan), dtype = np.object)

train_par = {'epochs' : par['epochs'], 'validate' : par['validate'],
             'clump' : par['clump'], 'feedback' : par['feedback'],
             'verbose' : par['verbose']}

if par['hint']:
    hint = np.zeros ((par['H'], par['T']))
    hint[0, :epar['T'][0]] = 1.
    hint[1, epar['T'][0]:] = 1.
else:
    hint = None

for i in trange (rep, desc = 'Repeating', leave = False):
    # Here we init our model
    gbc = GBC (par)

    # Construct the target input-output pairs for train and validation
    itrain, train_inps = gbc.implement (train_exp, hint = hint)
    ivalid, valid_inps = gbc.implement (valid_exp, hint = hint)

    # Store a copy of synaptic matrix for fair rank comparison
    J_init = gbc.J.copy()

    tqbar = tqdm (ranks, leave = False)
    for r, rank in enumerate (tqbar):
        # Start cloning from scratch
        gbc.forget (J = J_init)

        # HERE WE EVALUATE THE PERFORMANCES OF GBC WITH GIVEN RANK
        msg = 'Scanning Ranks (Training)'
        tqbar.set_description (msg)

        _, _, _ = gbc.clone (train_exp, itrain, rank = rank, **train_par)

        # Here we test the model on the suite of different angles
        msg = 'Scanning Ranks (Testing)'
        tqbar.set_description (msg)

        Test[i, r] = test (gbc, env, test_theta, par)

# ====== Save the results to file ========
import pickle

savepath = par['savepath']

rewards = {ranks[r] : np.array ([t['R'] for t in Test[:, r]])
            for r in range(scan)}

# Find best trajectory and save it
R = np.array ([t['R'] for tr in Test for t in tr]).reshape (rep, scan, -1)

best_idx = np.unravel_index(np.argmax (R.mean (axis = -1)), (rep, scan))
best_traj = Test[best_idx]['agent']

with open (savepath + '_rewards.pkl', 'wb') as f:
    data = (par, rewards)
    pickle.dump (data, f)


with open (savepath + '_best_traj.pkl', 'wb') as f:
    idx, rank = best_idx
    data = ((ranks[rank], idx), best_traj)
    pickle.dump (data, f)

print ('Experiment Completed.')
