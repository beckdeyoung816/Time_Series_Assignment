import pandas as pd
import numpy as np
from scipy.optimize import minimize
class SSM_SV:
    def __init__(self, data:pd.DataFrame, H_t:float=None, c_t:float=None, Q_t:float=None, d_t:float=None,T_t:float=None,R_t:float=None, a_1:float=None, P_1:float=None, theta0:list=None):
        """Initialize State Space Model for Stochastic Volatility

        Args:            
            data (pd.DataFrame): Time Series Data
            H_t (float, optional): Variance of Epsilon. Defaults to None.
            c_t (float, optional): Trend of observation equation. Defaults to None.
            Q_t (float, optional): Variance of Eta. Defaults to None.
            d_t (float, optional): Trend of state equation. Defaults to None.
            T_t (float, optional): System Matrix. Defaults to None.
            R_t (float, optional): System Matrix. Defaults to None.
            a_1 (float, optional): Starting value for a_t. Defaults to None.
            P_1 (float, optional): Starting value for P_t. Defaults to None.
            theta0 (list, optional): Starting values for [Q_t, d_t, phi] for QML. Defaults to None.
        """
        
        # Assign attributes of the data and the number of observations    
        self.df = data.copy(deep=True)
        self.N = self.df.shape[0]
        self.c_t = c_t * np.ones(self.N)
        self.H_t = H_t
        
        # If the desired parameters are not specified, then we estimate them through maximum likelihood
        if Q_t is None or T_t is None or d_t is None:
            
            mle_results = self.qml_sv(theta0) # Peform MLE
            self.Q_t = mle_results[0] # Variance of Eta from MLE
            self.d_t = mle_results[1] # Omega from MLE
            self.T_t = mle_results[2] # Phi from MLE
        else:
            # We take the inputted ones
            self.Q_t = Q_t 
            self.d_t = d_t 
            self.T_t = T_t 
        
        self.R_t = np.sqrt(self.Q_t) # Calculate sigma_eta
        self.xi = self.d_t/(1-self.T_t) # Calculate xi
        self.d_t = self.d_t # Turn d_t into a vector (as it will be time dependent when RV is used)
        
        # If starting values for P and a are not specified, we use diffuse initialization
        if a_1 is None or P_1 is None:
            # Diffuse Initialization
            self.a_1 = self.df['y_t'][0]
            self.P_1 = self.H_t + self.Q_t
        else:
            self.a_1 = a_1
            self.P_1 = P_1
            
    def k_filter(self, y_t: float, a_t: float, P_t: float, H_t: float, Q_t: float, T_t, c_t: float, d_t: float, R_t: float):
        """Kalman Filter at a single point in time given values at time t-1
        Args:
            y_t (float): True value
            a_t (float): 
            P_t (float): 
            H_t (float): Variance of Epsilon
            Q_t (float): Variance of Eta
            T_t (_type_): phi
            c_t (float): observation equation trend
            d_t (float): omega
            R_t (float): sigma_eta

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
    
    def kalman_filter(self, var_names=['a_t','P_t', 'v_t','F_t','K_t'], beta=False,y_name='y_t'):
        """Kalman Filter for the entire time series

        Args:
            var_names (list, optional): Variable names for storage. Defaults to ['a_t','P_t', 'v_t','F_t','K_t'].
            beta (bool, optional): Whether or not this is the model with beta and RV. Defaults to False.
            y_name (str, optional): Name of y variable. Defaults to 'y_t'.
        """
        # Add beta to the name of the variables if this is for the model with beta * RV in it
        var_names = [var + '_beta' for var in var_names] if beta else var_names
        
        # Initialize columns for the variables in the recursion
        self.df[var_names] = np.nan
        
        # Loop through each observation of y_t
        for t, y_t in enumerate(self.df[y_name]):
            # Initialize Values
            if t == 0 :
                a_t = self.a_1 
                P_t = self.P_1
            
            # Apply the filter and save updated values for next iteration
            a_tt, P_tt, a_t, P_t, v_t, F_t, K_t = self.k_filter(y_t=y_t,a_t=a_t,P_t=P_t,H_t=self.H_t,
                                                       Q_t=self.Q_t,T_t=self.T_t,c_t=self.c_t[t],d_t=self.d_t, R_t = self.R_t)
            
            # Store output
            self.df.loc[t, var_names] = [a_tt, P_tt, v_t, F_t, K_t]

    def state_smooth(self, beta=False):
        """Peform State Smoothing

        Args:
            beta (bool, optional): Whether or not this is the model with beta and RV. Defaults to False.
        """
        
        # Add beta to the name of the variables if this is for the model with beta * RV in it
        b = '_beta' if beta else ''
        
        # Create Helper variable L_t
        self.df['L_t'+b] = self.T_t - self.df['K_t'+b]
        
        # Initialize columns for recursion
        
        self.df[['r_t'+b,'N_t'+b, 'alpha_hat_t'+b]] = np.nan

        # Smoothing goes in reverse (from t=n to t=1)
        for t in reversed(range(self.N)):
            if t == self.N-1: # Since the last observation has no 't+1', set it to 0
                self.df.loc[t, ['r_t'+b, 'N_t'+b]] = [0,0]
            else:
            # Apply the formulas using t+1 when necessary
            # We know N_t isn't required for this assignment, but we are including it for flexibility in the future
                self.df.loc[t, 'N_t'+b] = (1 / self.df.loc[t+1, 'F_t'+b]) + ((self.df.loc[t+1, 'L_t'+b] ** 2) * self.df.loc[t+1, 'N_t'+b])
                self.df.loc[t, 'r_t'+b] = (self.df.loc[t+1, 'v_t'+b] / self.df.loc[t+1, 'F_t'+b]) + \
                                        (self.df.loc[t+1, 'L_t'+b] * self.df.loc[t+1, 'r_t'+b])
        
        # Set first value and then recurse
        # We thus start at the second value for a_t and P_t
        # These equations require r_{t-1} and N_{t-1} so we do not use the last values and start at first value
        self.df.loc[0, 'alpha_hat_t'+b] = self.a_1
        self.df.loc[1:,'alpha_hat_t'+b] = self.df.loc[1:, 'a_t'+b] + self.df.loc[1:, 'P_t'+b] * self.df.loc[:self.N, 'r_t'+b]
    
    def qml_sv(self, theta0:list):
        """Peform Quasi-ML for Qt, dt, and Tt

        Args:
            theta0 (list): Initial values for [Qt, dt, Tt]
        """
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
            
            # Equation 7.2 for the likelihood
            ll = -self.N/2 * np.log(2 * np.pi) - \
                    0.5 * np.sum(np.log(self.df['F_t']) + self.df['v_t'] ** 2 / self.df['F_t'])
                    
            return -ll # Return the negative since we are minimizing
         
        bnds = [(0, None), (None, None), (-0.999999999999, 0.999999999)] # Make sure variance of eta is positive and phi between -1 and 1
        results = minimize(calc_neg_log_lk, theta0, method = 'L-BFGS-B', bounds = bnds) # Perform optimization
        
        return results['x'] # Return optimal q value
    
    def estimate_Beta(self):
        """Estimate Beta for model with RV
        """
        # Run the kalman filter with y_t as the observation
        self.kalman_filter(var_names=['a_y_t','P_y_t','v_star_t','F_y_t','K_y_t'], y_name='y_t')
        
        # run the kalman filter with Xt + dt = log(RV_t) + dt as the observation
        self.df['X_t_plus_d_t'] = self.df['X_t'] + self.d_t # create new observation variable
        self.kalman_filter(var_names=['a_x_t','P_x_t', 'x_star_t','F_x_t','K_x_t'], y_name='X_t_plus_d_t')
        
        # Equation 6.2 to estimate beta 
        self.Beta = np.sum(self.df['x_star_t'] / self.df['F_y_t'] * self.df['v_star_t']) / \
            np.sum(self.df['x_star_t'] / self.df['F_y_t'] * self.df['x_star_t']) 
        
        # Update dt to now how B*Xt + dt in it
        self.c_t = self.Beta * self.df['X_t'] + self.c_t
        print(f'Beta: {self.Beta}')

    
    
    
