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


class LLM:
    def __init__(self, data:pd.DataFrame, var_e:float=None, var_h:float=None, a_1:float=None, P_1:float=None, q0:list=None):
        """Initialize Local Level Model Object
        Args:
            data (pd.DataFrame): Time Series Data
            var_e (float, optional): Variance of Epsilon. Defaults to None.
            var_h (float, optional): Variance of Eta. Defaults to None.
            a_1 (float, optional): Starting value for a_t. Defaults to None.
            P_1 (float, optional): Starting value for P_t. Defaults to None.
            q0 (list, optional): Starting value for q for maximum likelihood estimation of form [q0]
        """
        # Assign final attributes of the data and the number of observations    
        self.df = data
        self.N = self.df.shape[0]
        
        # If no variances are specified, then we estimate them through maximum likelihood
        if var_e is None or var_h is None:
            if q0 is None:
                raise ValueError('Please specify q0 or var_e and var_h')
            self.var_e_hat = 1 # Just an intialization for a variable calculated in the MLE
                     
            self.q = self.mle_q(q0)[0] # Get q from MLE
            self.var_e = self.var_e_hat # Var E hat updated in MLE
            self.var_h = self.var_e * self.q # Calculate variance of h
        else:
            self.var_e = var_e
            self.var_h = var_h
            self.q = self.var_h / self.var_e
            
        # If starting values for P and a are not specified, we use diffuse initialization
        if a_1 is None or P_1 is None:
            # Diffuse Initialization
            self.a_1 = self.df['y_t'][0]
            self.P_1 = self.var_e + self.var_h
        else:
            self.a_1 = a_1
            self.P_1 = P_1
    
    def k_filter(self, y_t: float, a_t: float, P_t: float, var_e: float, var_h: float):
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
        F_t = P_t + var_e
        K_t = P_t / F_t
        
        # If y_t is missing, use Equation 2.54 for the filtering step
        if np.isnan(y_t):
            v_t = 0 # set as missing/0 because no y_t        
            a_tt = a_t
            P_tt = P_t
  
        else:
            v_t = y_t - a_t
            a_tt = a_t + (K_t * v_t)
            P_tt = P_t * (1-K_t)

        # Prediction Step
        a_t = a_tt
        P_t = P_tt + var_h

        return a_tt, P_tt, a_t, P_t, v_t, F_t, K_t
    
    def kalman_filter(self, missing:bool=False):
        """Kalman Filter for the entire time series

        Args:
            missing (bool, optional): Whether or not this is for missing data. Defaults to False.
        """
        m = 'm' if missing else '' # If working with missing data, add 'm' to end of column names
        
        self.df[['a_t' + m,'P_t' + m, 'v_t' + m,'F_t' + m,'K_t' + m]] = np.nan
        
        # Loop through each observation of y_t
        for t, y_t in enumerate(self.df['y_t' + m]):
            # Initialize Values
            if t == 0 :
                a_t = self.a_1 
                P_t = self.P_1
            
            # Apply the filter and save updated values for next iteration
            a_tt, P_tt, a_t, P_t, v_t, F_t, K_t = self.k_filter(y_t=y_t, a_t=a_t, P_t=P_t,
                                                                 var_e=self.var_e, var_h=self.var_h)
            
            # Store output
            self.df.loc[t, ['a_t' + m,'P_t' + m,'v_t' + m,'F_t' + m,'K_t' + m]] = [a_tt, P_tt, v_t, F_t, K_t]

    def state_smooth(self, missing:bool=False):
        """Perform State Smoothing
        
        Args:
            missing (bool, optional): Whether or not this is for missing data. Defaults to False.
        """
        m = 'm' if missing else '' # If working with missing data, add 'm' to end of column names
        
        # Create Helper variable L_t
        self.df['L_t' + m] = 1 - self.df['K_t' + m]
        
        # Initialize columns for recursion
        self.df[['r_t' + m,'N_t' + m]] = np.nan

        # Smoothing goes in reverse (from t=n to t=1)
        for t in reversed(range(self.N)):
            if t == self.N-1: # Since the last observation has no 't+1', set it to 0
                self.df.loc[t, ['r_t' + m, 'N_t' + m]] = [0,0]
            else:
            # Apply the formulas using t+1 when necessary
                self.df.loc[t, 'N_t' + m] = (1 / self.df.loc[t+1, 'F_t' + m]) + ((self.df.loc[t+1, 'L_t' + m] ** 2) * self.df.loc[t+1, 'N_t' + m])
                self.df.loc[t, 'r_t' + m] = (self.df.loc[t+1, 'v_t' + m] / self.df.loc[t+1, 'F_t' + m]) + \
                                        (self.df.loc[t+1, 'L_t' + m] * self.df.loc[t+1, 'r_t' + m])
                                        
    def disturbance_smooth(self, missing:bool=False):
        """Perform Disturbance Smoothing
        Args:
            missing (bool, optional): Whether or not this is for missing data. Defaults to False.
        """
        
        m = 'm' if missing else '' # If working with missing data, add 'm' to end of column names
        
        # Initliaze columns 
        self.df[['alpha_hat_t' + m, 'V_t' + m]] = np.nan
        
        # Set first value and then recurse
        # We thus start at the second value for a_t and P_t
        # These equations require r_{t-1} and N_{t-1} so we do not use the last values and start at first value
        self.df.loc[0, 'alpha_hat_t' + m] = self.a_1
        self.df.loc[1:,'alpha_hat_t' + m] = self.df.loc[1:, 'a_t' + m] + self.df.loc[1:, 'P_t' + m] * self.df.loc[:self.N, 'r_t' + m]
        
        # Set first value then recurse
        self.df.loc[0, 'V_t' + m] = self.P_1
        self.df.loc[1:,'V_t' + m] = self.df.loc[1:, 'P_t' + m] - ((self.df.loc[1:, 'P_t' + m] ** 2) * self.df.loc[:self.N, 'N_t' + m])

        # Smoothing Error and Variance
        self.df['u_t' + m] = (self.df['v_t' + m] / self.df['F_t' + m]) - (self.df['K_t' + m] * self.df['r_t' + m])
        self.df['D_t' + m] = (1 / self.df['F_t' + m]) + ((self.df['K_t' + m] ** 2) * self.df['N_t' + m])

        # Smoothed Observation Error and State Error
        self.df['e_hat_t' + m] = self.var_e * self.df['u_t' + m]
        self.df['h_hat_t' + m] = self.var_h * self.df['r_t' + m]

        # Variances and SDs of Observation and State Errors
        self.df['var_e_hat_t' + m] = self.var_e - ((self.var_e ** 2) * self.df['D_t' + m])
        self.df['sd_e_hat_t' + m] = np.sqrt(self.df['var_e_hat_t' + m])
        
        self.df['var_h_hat_t' + m] = self.var_h - ((self.var_h ** 2) * self.df['N_t' + m])
        self.df['sd_h_hat_t' + m] = np.sqrt(self.df['var_h_hat_t' + m])
    
    def auxilary_residuals(self):
        """Calculate extra residual variables u_t*, r_t*, and e_t
        """
        self.df['u_star_t'] = self.df['u_t'] / np.sqrt(self.df['D_t'])
        self.df['r_star_t'] = self.df['r_t'] / np.sqrt(self.df['N_t'])
        self.df['e_t'] = self.df['v_t'] / np.sqrt(self.df['F_t'])
    
    def filter_with_missing_vals(self, missing_ranges:list):
        """Perform Kalman Filter with missing data

        Args:
            missing_ranges (list): A list of dictionaries of the form {'start': val, 'stop': val} which says which indices to make NaN
        """
        # Initialize new y column that will have missing values
        self.df['y_tm'] = self.df['y_t']
        
        # Set desired values to missing values
        for m_range in missing_ranges:
            self.df.loc[m_range['start']:m_range['stop'], 'y_tm'] = np.nan
        
        # Call kalman filter
        self.kalman_filter(missing=True)
    
    def smooth_with_missing_vals(self):
        """Perform smoothing with missing values
        """
        # Make sure the filtering was done first
        if 'y_tm' not in self.df.columns:
            raise ValueError('Please Filter with missing values first')
        
        self.state_smooth(missing=True)
        self.disturbance_smooth(missing=True)
        
    def forecast(self, j:int):
        """Perform State Forecasting

        Args:
            j (int): Number of samples to forecast
        """
        # Create a copy of the df because we need to add rows and we dont want to affect the original df
        # Also add j blank rows to the end of the dataset
        self.forecast_df = self.df.copy(deep=True).reindex(list(range(0, self.N + j))).reset_index(drop=True)
        
        # Generate new x-values for these additional rows
        time_interval = self.forecast_df['x'][1] - self.forecast_df['x'][0] # Calculate time interval
        forecast_time = np.array([time_interval * i for i in range(j)]) # Generate n new x values incrementing by 1 time interval
        self.forecast_df.loc[self.N:self.N + j, 'x'] = self.forecast_df.loc[self.N-1, 'x'] + forecast_time # add these to largest time interval

        # Initialize Columns
        self.forecast_df[['a_tf','P_tf','v_tf','F_tf']] = np.nan       

        # Perform filter
        for t, y_t in enumerate(self.forecast_df['y_t']):
            if t == 0 :
                a_t = self.a_1 
                P_t = self.P_1
                
            a_tt, P_tt, a_t, P_t, v_t, F_t, K_t = self.k_filter(y_t=y_t, a_t=a_t, P_t=P_t, 
                                                    var_e=self.var_e, var_h=self.var_h)
            # Store Values
            self.forecast_df.loc[t, ['a_tf','P_tf','v_tf','F_tf','K_tf']] = [a_tt, P_tt, v_t, F_t, K_t]
            
        # Compute Confidence Intervals
        z = ss.norm.ppf((1 + 0.5) / 2)
        self.forecast_df['a_tf_upper_c'] = self.forecast_df['a_tf'] + z * np.sqrt(self.forecast_df['P_tf'])
        self.forecast_df['a_tf_lower_c'] = self.forecast_df['a_tf'] - z * np.sqrt(self.forecast_df['P_tf'])
        
    def get_conf_intervals(self, col:str, var:str, pct:float):
        """Calculate Confidence Intervals for a given variable

        Args:
            col (str): Column name of the variable
            var (str): Column name of the variance of the variable
            pct (float): Confidence Interval Percentage
        """
        z = ss.norm.ppf((1 + pct) / 2) # Get quantile of desired percentage
        self.df[col+'_upper_c'] = self.df[col] + z * np.sqrt(self.df[var])
        self.df[col+'_lower_c'] = self.df[col] - z * np.sqrt(self.df[var])
    
    def mle_q(self, q0:list):
        """Compute optimal signal-noise ratio and associated error variances through MLE

        Args:
            q0 (list): Initialization for q. Must be in a list as [q0]
        """
        def calc_neg_log_lk(q:list):
            """Calculate negative log likelihood for a given q value

            Args:
                q (list): Initialization for q. Must be in a list as [q0]

            Returns:
                float: negative log likelihood
            """
            var_e = 1
            var_h = q[0]
            q = var_h/var_e
            for t, y_t in enumerate(self.df['y_t'][1:]):
                # Initialize Values
                if t == 0 :
                    a_t = self.df['y_t'][0] # Initialize at y1
                    P_t = var_e + var_h
                
                # Apply the filter
                a_tt, P_tt, a_t, P_t, v_t, F_t, K_t = self.k_filter(y_t=y_t, a_t=a_t, P_t=P_t, 
                                                            var_e=var_e, var_h=var_h)
                
                # Store output
                self.df.loc[t, ['a_t','P_t','v_t','F_t','K_t']] = [a_tt, P_tt, v_t, F_t, K_t]
            
            # Calculate helper variable for log likelihood calculation
            self.df['F_star_t'] = self.df['F_t'] / var_e
            # We store the var_e_hat in self because this is the optimal var_e and we will then multiple it with the outputted q to get var_h
            self.var_e_hat = 1 / (self.N - 1) * np.sum(self.df['v_t'][1:] ** 2 / self.df['F_star_t'][1:])
            
            # Equation 2.63
            ll = -self.N/2 * np.log(2 * np.pi) - (self.N - 1) / 2 - \
                (self.N - 1) / 2 * np.log(self.var_e_hat) - \
                    0.5 * np.sum(np.log(self.df['F_star_t'][1:]))
                    
            return -ll # Return the negative since we are minimizing
                    
        bnds = [(1e-10, None)] # Make sure q is positive
        results = minimize(calc_neg_log_lk, [q0], method = 'L-BFGS-B', bounds = bnds) # Perform optimization
        
        return results['x'] # Return optimal q value