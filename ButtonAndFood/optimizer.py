"""
    © 2021 This work is licensed under a CC-BY-NC-SA license.
    Title: *"Behavioral cloning in recurrent spiking networks: A comprehensive framework"*
    Authors: Anonymus
"""

import numpy as np

# This is a very simple class to implement the Adam Optimizer
class Adam:
    def __init__ (self, alpha = .03, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8,
                  drop = 0.8, drop_time = 5000):
        self.alpha = alpha 
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 
        self.epsilon = epsilon 
        self.drop = drop 
        self.drop_time = drop_time 

        self.m = 0. 
        self.v = 0. 

        self.t = 1 

    def step (self, theta_t, g_t, step_time = True):
        if self.t % self.drop_time == 0:
            self.alpha *= self.drop 

        self.m = self.beta_1 * self.m + (1. - self.beta_1) * g_t 
        self.v = self.beta_2 * self.v + (1. - self.beta_2) * g_t**2 

        m_hat = self.m / (1. - self.beta_1**self.t) 
        v_hat = self.v / (1. - self.beta_2**self.t) 

        if step_time:
            self.t += 1 

        return theta_t + self.alpha * m_hat / (np.sqrt (v_hat) + self.epsilon) 

    # This is the Adam optimization
    def optimize (self, grad, *args, init = 0., t_max = 1000):
        # Initialize time step and moment estimations
        self.m = 0. 
        self.v = 0. 

        theta = init 

        for t in range (1, t_max):
            if t % self.drop_time == 0:
                self.alpha *= self.drop 

            g_t = grad (theta, *args) 

            # First we update the first and second moment gradient estimate
            self.m = self.beta_1 * self.m + (1. - self.beta_1) * g_t 
            self.v = self.beta_2 * self.v + (1. - self.beta_2) * g_t**2 

            # Compute bias-corrected estimate for first and second moment
            m_hat = self.m / (1. - self.beta_1**t) 
            v_hat = self.v / (1. - self.beta_2**t) 

            theta += self.alpha * m_hat / (np.sqrt (v_hat) + self.epsilon) 

        return theta 

class SimpleGradient:
    def __init__ (self, alpha, momentum = 0., drop = 0., drop_thr = 1e8):
        self.alpha = alpha 

        self.momentum = momentum 
        self.drop = drop 
        self.drop_thr = drop_thr 

        self.last_g_t = 0 

    def step (self, theta_t, g_t, curr_time = 0):
        if curr_time > self.drop_thr:
            self.alpha *= self.drop 

        theta_t += self.momentum * self.last_g_t + self.alpha * g_t 

        self.last_g_t = g_t 

        return theta_t 
