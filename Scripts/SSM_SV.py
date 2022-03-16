import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.stats as ss

########## NOTATION
# alpha_t = States
# v_t = Prediction Error
# F_t = var(v_t)

# K_t = Kalman Gain

# a_t / a_tt = mean(alpha_t) Filter State
# p_t / p_tt = var(alpha_t)

# u_t = Smoothing Error
# D_t = var(u_t)

# r_t = Smoothing Cumulant
# N_t = var(r_t)

# alpha_hat_t = Smoothed State
# V_t = Smoothed State Variance

# e_hat_t = Smoothed Observation Disturbance / Error
# h_hat_t = Smoothed Mean of the Disturbance / State Error

# u_star_t = Observation Residual
# r_star_t = State Residual
# e_t = Standardised Prediction Errors


class SSM_SV:
    def __init__(self, data:pd.DataFrame, H_t:float=None, c_t:float=None, Q_t:float=None, d_t:float=None,T_t:float=None,R_t:float=None, a_1:float=None, P_1:float=None, theta0:list=None, Beta:float=None, use_Beta:bool=False):
        """Initialize Local Level Model Object
        Args:
            data (pd.DataFrame): Time Series Data
            H_t (float, optional): Variance of Epsilon. Defaults to None.
            Q_t (float, optional): Variance of Eta. Defaults to None.
            a_1 (float, optional): Starting value for a_t. Defaults to None.
            P_1 (float, optional): Starting value for P_t. Defaults to None.
            q0 (list, optional): Starting value for q for maximum likelihood estimation of form [q0]
        """
        # Assign final attributes of the data and the number of observations    
        self.df = data.copy(deep=True)
        self.N = self.df.shape[0]
        self.c_t = c_t
        self.H_t = H_t
        
        # If no variances are specified, then we estimate them through maximum likelihood
        if Q_t is None or T_t is None or d_t is None:
            
            mle_results = self.qml_sv(theta0) # Peform MLE
            self.Q_t = mle_results[0] # Variance of Eta from MLE
            self.d_t = mle_results[1] # Omega from MLE
            self.T_t = mle_results[2] # Phi from MLE
        else:
            self.Q_t = Q_t # Var E hat updated in MLE
            self.d_t = d_t # Omega from MLE
            self.T_t = T_t # Phi from MLE
        
        self.R_t = np.sqrt(self.Q_t) # Calculate sigma_eta
        self.xi = self.d_t/(1-self.T_t) # Calculate xi
        self.d_t = self.d_t * np.ones(self.N)
        
        # If starting values for P and a are not specified, we use diffuse initialization
        if a_1 is None or P_1 is None:
            # Diffuse Initialization
            self.a_1 = self.df['y_t'][0]
            self.P_1 = self.H_t + self.Q_t
        else:
            self.a_1 = a_1
            self.P_1 = P_1

        if use_Beta:
            self.beta_df = data.copy(deep=True)
            self.estimate_Beta()
            
    def k_filter(self, y_t: float, a_t: float, P_t: float, H_t: float, Q_t: float, T_t, c_t: float, d_t: float, R_t: float):
        """Kalman Filter at a single point in time given values at time t-1

        Args:
            y_t (float): True value
            a_t (float): 
            P_t (float): 
            var_e (float): Variance of Epsilon
            var_h (float): Variance of Eta

        Returns:
            _type_: New values at time t
        """
                
        # Kalman gain calculation
        F_t = P_t + H_t
        M_t = P_t / F_t
        K_t = T_t * M_t
    
        v_t = y_t - c_t - a_t
        a_tt = a_t + (M_t * v_t) # L3.S22
        P_tt = P_t - (M_t * F_t * M_t) # L3.S22

        # Prediction Step
        a_t = d_t + T_t * a_tt # L3.S22
        P_t = T_t * P_tt * T_t + R_t * Q_t* R_t # L3.S22

        return a_tt, P_tt, a_t, P_t, v_t, F_t, K_t
    
    def kalman_filter(self, vars=['a_t','P_t', 'v_t','F_t','K_t'], beta=False,y='y_t'):
        """Kalman Filter for the entire time series
        """
        df = self.beta_df if beta else self.df
        
        df[vars] = np.nan
        
        # Loop through each observation of y_t
        for t, y_t in enumerate(df[y]):
            # Initialize Values
            if t == 0 :
                a_t = self.a_1 
                P_t = self.P_1
            
            # Apply the filter and save updated values for next iteration
            a_tt, P_tt, a_t, P_t, v_t, F_t, K_t = self.k_filter(y_t=y_t,a_t=a_t,P_t=P_t,H_t=self.H_t,
                                                       Q_t=self.Q_t,T_t=self.T_t,c_t=self.c_t,d_t=self.d_t[t], R_t = self.R_t)
            
            # Store output
            
            df.loc[t, vars] = [a_tt, P_tt, v_t, F_t, K_t]

    def state_smooth(self):
        
        # Create Helper variable L_t
        self.df['L_t'] = self.T_t - self.df['K_t']
        
        # Initialize columns for recursion
        self.df[['r_t','N_t', 'alpha_hat_t']] = np.nan

        # Smoothing goes in reverse (from t=n to t=1)
        for t in reversed(range(self.N)):
            if t == self.N-1: # Since the last observation has no 't+1', set it to 0
                self.df.loc[t, ['r_t', 'N_t']] = [0,0]
            else:
            # Apply the formulas using t+1 when necessary
                self.df.loc[t, 'N_t'] = (1 / self.df.loc[t+1, 'F_t']) + ((self.df.loc[t+1, 'L_t'] ** 2) * self.df.loc[t+1, 'N_t'])
                self.df.loc[t, 'r_t'] = (self.df.loc[t+1, 'v_t'] / self.df.loc[t+1, 'F_t']) + \
                                        (self.df.loc[t+1, 'L_t'] * self.df.loc[t+1, 'r_t'])
        
        # Set first value and then recurse
        # We thus start at the second value for a_t and P_t
        # These equations require r_{t-1} and N_{t-1} so we do not use the last values and start at first value
        self.df.loc[0, 'alpha_hat_t'] = self.a_1
        self.df.loc[1:,'alpha_hat_t'] = self.df.loc[1:, 'a_t'] + self.df.loc[1:, 'P_t'] * self.df.loc[:self.N, 'r_t']
    
    def qml_sv(self, theta0:list):
        def calc_neg_log_lk(theta0:list):
            Q_t = theta0[0]
            d_t = theta0[1]
            T_t = theta0[2]
            R_t = np.sqrt(Q_t)
            
            for t, y_t in enumerate(self.df['y_t'][1:]):
                # Initialize Values
                if t == 0 :
                    a_t = d_t/(1-T_t)
                    P_t = Q_t/(1-T_t ** 2)
                
                # Apply the filter
                a_tt, P_tt, a_t, P_t, v_t, F_t, K_t = self.k_filter(y_t=y_t, a_t=a_t, P_t=P_t, H_t=self.H_t,
                                                                    Q_t=Q_t, T_t=T_t, c_t=self.c_t, d_t=d_t, R_t=R_t)
                    
                # Store output
                self.df.loc[t, ['a_t','P_t','v_t','F_t','K_t']] = [a_tt, P_tt, v_t, F_t, K_t]
            
            # Equation 7.2
            ll = -self.N/2 * np.log(2 * np.pi) - \
                    0.5 * np.sum(np.log(self.df['F_t']) + self.df['v_t'] ** 2 / self.df['F_t'])
            
            self.mle_round +=1
                    
            return -ll # Return the negative since we are minimizing
        self.mle_round = 1           
        bnds = [(0, None), (None, None), (-0.999999999999, 0.999999999)] # Make sure variance of eta is positive and phi between -1 and 1
        results = minimize(calc_neg_log_lk, theta0, method = 'L-BFGS-B', bounds = bnds) # Perform optimization
        
        return results['x'] # Return optimal q value
    
    def estimate_Beta(self):
        self.kalman_filter(vars=['a_y_t','P_y_t','v_star_t','F_y_t','K_y_t'], beta=True, y='y_t')
        
        self.beta_df['X_t_plus_d_t'] = self.beta_df['X_t'] + self.d_t
        self.kalman_filter(vars=['a_x_t','P_x_t', 'x_star_t','F_x_t','K_x_t'], beta=True, y='X_t_plus_d_t')
        
        self.Beta = np.sum(self.beta_df['x_star_t'] * self.beta_df['F_y_t'] * self.beta_df['v_star_t']) / \
            np.sum(self.beta_df['x_star_t'] * self.beta_df['F_y_t'] * self.beta_df['x_star_t']) # Equation 6.2
            
        self.d_t = self.Beta * self.df['X_t'] + self.d_t
        print(f'Beta: {self.Beta}')

    
    
    
