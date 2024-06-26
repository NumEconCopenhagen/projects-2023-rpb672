
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class Household:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. Extention
        par.extention = False
        par.ext_F = 1

        # e. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:      
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = min(HM,HF)
        else:
            H = ( (1-par.alpha)*HM**( (par.sigma-1)/par.sigma )+par.alpha*HF**( (par.sigma-1) /par.sigma) )**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        # Added extention. The original disutillity function is a special case of this, where par.extention=False, i.e. 0, thus negating the last term.
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)+par.extention*(par.ext_F*LF**epsilon_/epsilon_)
        

        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_con(self,do_print=False):
        """ solve model continously """

        par = self.par
        opt = SimpleNamespace()

        def objectivefunction(x):
            LM,HM,LF,HF = x
            penalty = 0
            u = self.calc_utility(LM,HM,LF,HF) + penalty  
            # c. set to minus infinity if constraint is broken
            if (LM+HM > 24) | (LF+HF > 24):
                penalty += 10000
            return -u
           
        
        x_guess = [3,3,3,3] # Initial guess
        solve = optimize.minimize(objectivefunction,x_guess,method='Nelder-Mead')

        opt.LM = solve.x[0]
        opt.HM = solve.x[1]
        opt.LF = solve.x[2]
        opt.HF = solve.x[3]

        return opt


    def solve_wF_vec(self):
        """ solve model for vector of female wages """
        
        par = self.par
        sol = self.sol

        # Looping over female wages
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF

            # Reporting optimal values to solution vectors
            opt = self.solve_con()
            sol.HF_vec[i]=opt.HF
            sol.HM_vec[i]=opt.HM

        return sol.HF_vec, sol.HM_vec


    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """
        par = self.par
        sol = self.sol


        # Objective function to be minimized
        def deviation(x):
            par.alpha = x[0]
            par.sigma = x[1]
            # Calculating the deviation
            self.solve_wF_vec()
            self.run_regression()
            sol.err = ((par.beta0_target - sol.beta0)**2+(par.beta1_target-sol.beta1)**2)
            return sol.err       
        bounds = ((0.5,1),(0.05,0.5)) # bounds chosen by trial and error
        guess = [0.75,0.3] # initial guess
        result = optimize.minimize(deviation, x0 = guess, method = 'Nelder-Mead', bounds=bounds, tol = 10e-8)

        #unpack results
        sol.alpha = result.x[0]
        sol.sigma = result.x[1]
        #print results
        print(f'Estimated alpha from data = {sol.alpha:6.4f}, and sigma = {sol.sigma:6.4f}. Deviation was = {sol.err:6.4f}')

    
    def estimate_ext(self,alpha=0.5,sigma=None):
        """ estimate ext_F and sigma """
        par = self.par
        sol = self.sol

        # Objective function to be minimized
        def deviation_ext(x):
            par.ext_F = x[0]
            par.sigma = x[1]
            # Calculating the deviation of the regression
            self.solve_wF_vec()
            self.run_regression()
            sol.err = ((par.beta0_target - sol.beta0)**2+(par.beta1_target-sol.beta1)**2)
            return sol.err       
        bounds = ((0,5),(0,1)) # bounds chosen through trial and error
        guess = [1,0.5]
        result = optimize.minimize(deviation_ext, x0 = guess, method = 'Nelder-Mead', bounds=bounds, tol = 10e-8)

        #unpack results
        sol.ext_F = result.x[0]
        sol.sigma = result.x[1]
        #print results
        print(f'Estimated ext_F from data = {sol.ext_F:6.4f}, and sigma = {sol.sigma:6.4f} Deviation was = {sol.err:6.4f}')
