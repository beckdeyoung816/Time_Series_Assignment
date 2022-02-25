import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.stats as ss

########## NOTATION

# v_t = Prediction Error
# F_t = var(v_t)

# K_t = Kalman Gain

# a_t / a_tt = mean(alpha_t) Filter State
# p_t / p_tt = var(alpha_t)

# u_t = Smoothing Error
# D_t = var(u_t)

# r_t = Smoothing Cumulant
# N_t = var(r_t)

# alpha_t = 
# V_t =

# e_hat_t = Smoothed Observation Disturbance / Error
# h_hat_t = Smoothed Mean of the Disturbance / State Error


# u_star_t = Observation Residual
# r_star_t = State Residual
# e_t = Standardised Prediction Errors


class LLM:
    def __init__(self, data: pd.DataFrame, var_e:float=None, var_h:float=None, a_1:float=None, P_1:float=None, ML_start:list=None):
        """Initialize Local Level Model Object
        Args:
            data (pd.DataFrame): Time Series Data
            var_e (float, optional): Variance of Epsilon. Defaults to None.
            var_h (float, optional): Variance of Eta. Defaults to None.
            a_1 (float, optional): Starting value for a_t. Defaults to None.
            P_1 (float, optional): Starting value for P_t. Defaults to None.
            ML_start (list, optional): Starting values for maximum likelihood estimation
        """
        # Assign final attributes of the data and the number of observations    
        self.df = data
        self.N = self.df.shape[0]
        # If no variances are specified, then we estimate them through maximum likelihood
        # a_1 and P_1 are implied in this case
        if var_e is None or var_h is None:
            if ML_start is None:
                raise ValueError('Please specify ML_Start or var_e and var_h')         
            self.res = self.Kalman_ML(ML_start)
            self.var_e, self.var_h = self.res.x
            
        # If variances are specified, starting values for a_t and P_t must be specified too 
        elif a_1 is None or P_1 is None:
            raise ValueError('Please specify a_1 and/or P_1')
        
        # Otherwise, assign the given variances and starting values
        else:
            self.var_e = var_e
            self.var_h = var_h
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
        # If y_t is missing, set values for kalman gain
        if np.isnan(y_t):
            v_t = 0
            F_t = 10 ** 7
            K_t = 0
        else:
            # Kalman gain calculation
            v_t = y_t - a_t
            F_t = P_t + var_e
            K_t = P_t / F_t

        # State Update / Filtering Step
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
        self.df['u_star_t'] = self.df['u_t'] / np.sqrt(self.df['D_t'])
        self.df['r_star_t'] = self.df['r_t'] / np.sqrt(self.df['N_t'])
        self.df['e_t'] = self.df['v_t'] / np.sqrt(self.df['F_t'])
    
    def missing_filter(self, missing_ranges:list):
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
    
    def missing_smooth(self):
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
        self.forecast_df = self.df.copy(deep=True).reindex(list(range(0, self.N + j))).reset_index(drop=True)
        
        # Generate new t-values
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
        self.forecast_df['a_tf_upper_c'] = self.forecast_df['a_tf'] + .67 * np.sqrt(self.forecast_df['P_tf'])
        self.forecast_df['a_tf_lower_c'] = self.forecast_df['a_tf'] - .67 * np.sqrt(self.forecast_df['P_tf'])
        
        
    def get_conf_intervals(self, col:str, var:str, pct:float):
        """Calculate Confidence Intervals for a given variable

        Args:
            col (str): Column name of the variable
            var (str): Column name of the variance of the variable
            pct (float): Confidence Interval Percentage
        """
        z = ss.norm.ppf(pct) # Get quantile of desired percentage
        self.df[col+'_upper_c'] = self.df[col] + z * np.sqrt(self.df[var])
        self.df[col+'_lower_c'] = self.df[col] - z * np.sqrt(self.df[var])
        
    def Kalman_ML(self, starting_vars):
        def ML_fun(vars):
            var_e = vars[0]
            var_h = vars[1]
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
            
            self.df['F_star_t'] = self.df['F_t'] / var_e
            var_e_hat = 1 / (self.N - 1) * np.sum(self.df['v_t'][1:] ** 2 / self.df['F_star_t'][1:])

            return -self.N/2 * np.log(2 * np.pi) - (self.N - 1) / 2 - \
                (self.N - 1) / 2 * np.log(var_e_hat) - \
                    0.5 * np.sum(np.log(self.df['F_star_t'][1:]))
                    
        bnds = [(1e-10, None), (1e-10, None)]
        results = minimize(ML_fun, starting_vars, method = 'L-BFGS-B', bounds = bnds)
        return results
    
    def Kalman_ML_q(self, q0):
        def ML_fun_q(q):
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
            
            self.df['F_star_t'] = self.df['F_t'] / var_e
            var_e_hat = 1 / (self.N - 1) * np.sum(self.df['v_t'][1:] ** 2 / self.df['F_star_t'][1:])

            return -self.N/2 * np.log(2 * np.pi) - (self.N - 1) / 2 - \
                (self.N - 1) / 2 * np.log(var_e_hat) - \
                    0.5 * np.sum(np.log(self.df['F_star_t'][1:]))
                    
        bnds = [(1e-10, None)]
        results = minimize(ML_fun_q, [q0], method = 'L-BFGS-B', bounds = bnds)
        return results
    
    
    def Kalman_avg_ML(self, starting_vars):
        def ML_avg_fun(vars):
            var_e = vars[0]
            var_h = vars[1]
            q = var_h/var_e
            
            total_ll = 0
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
                
                # Total Log likelihood
                total_ll += -self.N/2 * np.log(2 * np.pi) - \
                    (1/2) * np.sum(np.log(self.df['F_t'][1:])) - \
                    (1/2) * np.sum(self.df['v_t'][1:] ** 2 * self.df['F_t'][1:])
            
            return total_ll / self.N # Average LL
                    
                    
        bnds = [(1e-10, None), (1e-10, None)]
        results = minimize(ML_avg_fun, starting_vars, method = 'L-BFGS-B', bounds = bnds)
        return results