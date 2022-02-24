
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import minimize
import scipy.stats as ss

# %%
########## NOTATION

# v_t = Prediction Error
# F_t = Prediction Error Variance
# K_t = Kalman Gain
# a_t / a_tt = Filter Density Mean
# p_t / p_tt = Filter Density Variance
#

# %%

data = pd.read_excel('../Data/Nile.xlsx', names = ['year', 'y_t'])
    
VAR_E = 15099
VAR_H = 1469.1
A1 = 0
P1 = 10 ** 7
N_OBS = data.shape[0]

# %%
def Kalman_Filter_ML(q):
    
    
    for t, y_t in enumerate(data['flow'][1:]):
        # Initialize Values
        if t == 0 :
            a_t = y_t 
            p_t = 1 + q
        
        # Apply the filter
        a_tt, p_tt, a_t, p_t, v_t, F_t, K_t = kf_filter_q(y_t=y_t, a_t=a_t, p_t=p_t, q=q) 
                                                            
        # Store output
        data.loc[t, ['a_t','p_t','v_t','F_t','K_t']] = [a_tt, p_tt, v_t, F_t, K_t]
    
    #data['F_star_t'] = data['F_t'] / var_e
    var_e_hat = 1 / (N_OBS - 1) * np.sum(data['v_t'][1:] ** 2 / data['F_t'][1:])

    return -N_OBS/2 * np.log(2 * np.pi) - (N_OBS - 1) / 2 - \
        (N_OBS - 1) / 2 * np.log(var_e_hat) - \
            0.5 * np.sum(np.log(data['F_t'][1:]))

# %%
#bnds = ((0, np.inf), (0, np.inf))

#q = 1

#options={'disp': True}
#test = minimize(Kalman_Filter_ML, q , method = 'BFGS')

#print(Kalman_Filter_ML(q))

# %%
def kf_filter(y_t, a_t, P_t, var_e, var_h):
    
    if np.isnan(y_t):
        v_t = 0
        F_t = 10 ** 7
        K_t = 0
    else:
        # Kalman gain calculation
        v_t = y_t - a_t
        F_t = P_t + var_e
        K_t = P_t / F_t
    
    # State update
    a_tt = a_t + (K_t * v_t)
    P_tt = P_t * (1-K_t)
    
    # Prediction Step
    a_t = a_tt
    P_t = P_t * (1-K_t) + var_h
    
    return a_tt, P_tt, a_t, P_t, v_t, F_t, K_t

# %%
def kf_filter_q(y_t, a_t, p_t, q):
    
    # Kalman gain calculation
    v_t = y_t - a_t
    F_t = p_t + 1
    K_t = p_t / F_t
    
    # State update
    a_tt = a_t + (K_t * v_t)
    p_tt = p_t * (1-K_t)
    
    # Prediction Step
    a_t = a_tt
    p_t = p_t * (1-K_t) + q
    
    return a_tt, p_tt, a_t, p_t, v_t, F_t, K_t

# %%
data[['a_t','p_t', 'v_t','F_t','K_t']] = np.nan

# %%
for t, y_t in enumerate(data['flow']):
    # Initialize Values
    if t == 0 :
        a_t = A1 
        p_t = P1
    
    # Apply the filter
    a_tt, p_tt, a_t, p_t, v_t, F_t, K_t = kf_filter(y_t=y_t, a_t=a_t, P_t=p_t, 
                                                 var_e=VAR_E, var_h=VAR_H)
    
    # Store output
    data.loc[t, ['a_t','p_t','v_t','F_t','K_t']] = [a_tt, p_tt, v_t, F_t, K_t]

data['L_t'] = 1 - data['K_t']
data[['r_t','N_t']] = np.nan

# %%
for t in reversed(range(N_OBS)):
    if t == N_OBS-1: # Since the last observation has no 't+1', set it to 0
        data.loc[t, ['r_t', 'N_t']] = [0,0]
    else:
        data.loc[t, 'N_t'] = (1 / data.loc[t+1, 'F_t']) + ((data.loc[t+1, 'L_t'] ** 2) * data.loc[t+1, 'N_t'])
        data.loc[t, 'r_t'] = (data.loc[t+1, 'v_t'] / data.loc[t+1, 'F_t']) + (data.loc[t+1, 'L_t'] * data.loc[t+1, 'r_t'])
# %%

# %%
data[['alpha_hat_t', 'V_t']] = np.nan
data.loc[0, 'alpha_hat_t'] = A1
data.loc[1:,'alpha_hat_t'] = data.loc[1:, 'a_t'] + data.loc[1:, 'p_t'] * data.loc[:N_OBS, 'r_t']
data.loc[0, 'V_t'] = P1
data.loc[1:,'V_t'] = data.loc[1:, 'p_t'] - ((data.loc[1:, 'p_t'] ** 2) * data.loc[:N_OBS, 'N_t'])



#%%

data['D_t'] = (1 / data['F_t']) + ((data['K_t'] ** 2) * data['N_t'])

data['u_t'] = (data['v_t'] / data['F_t']) - (data['K_t'] * data['r_t'])

data['e_hat_t'] = VAR_E * data['u_t']
data['h_hat_t'] = VAR_H * data['r_t']

data['var_e_hat_t'] = VAR_E - ((VAR_E ** 2) * data['D_t'])
data['sd_e_hat_t'] = np.sqrt(data['var_e_hat_t'])
data['var_h_hat_t'] = VAR_H - ((VAR_H ** 2) * data['N_t'])
data['sd_h_hat_t'] = np.sqrt(data['var_h_hat_t'])

# %%
data['u_star_t'] = data['u_t'] / np.sqrt(data['D_t'])
data['r_star_t'] = data['r_t'] / np.sqrt(data['N_t'])
data['e_t'] = data['v_t'] / np.sqrt(data['F_t'])


# %%
# Missing Data Filterd
data[['a_tm','P_tm','v_tm','alpha_hat_tm']] = np.nan
# %%
data['y_tm'] = data['y_t']
data.loc[21:40, 'y_tm'] = np.nan
data.loc[61:80, 'y_tm'] = np.nan
# %%

for t, y_t in enumerate(data['y_tm']):
    if t == 0 :
        a_t = A1 
        P_t = P1
    a_tt, p_tt, a_t, P_t, v_t, F_t, K_t = kf_filter(y_t=y_t, a_t=a_t, P_t=P_t, 
                                            var_e=VAR_E, var_h=VAR_H)
    data.loc[t, ['a_tm','P_tm','v_tm','F_tm','K_tm']] = [a_tt, p_tt, v_t, F_t, K_t]

#%%
data['L_tm'] = 1 - data['K_tm']
data[['r_tm','N_tm']] = np.nan

for t in reversed(range(N_OBS)):
    if t == N_OBS-1: # Since the last observation has no 't+1', set it to 0
        data.loc[t, ['r_tm', 'N_tm']] = [0,0]
    else:
        data.loc[t, 'N_tm'] = (1 / data.loc[t+1, 'F_tm']) + ((data.loc[t+1, 'L_tm'] ** 2) * data.loc[t+1, 'N_tm'])
        data.loc[t, 'r_tm'] = (data.loc[t+1, 'v_tm'] / data.loc[t+1, 'F_tm']) + (data.loc[t+1, 'L_tm'] * data.loc[t+1, 'r_tm'])

# %%
data[['alpha_hat_tm', 'V_tm']] = np.nan
data.loc[0, 'alpha_hat_tm'] = A1
data.loc[1:,'alpha_hat_tm'] = data.loc[1:, 'a_tm'] + data.loc[1:, 'P_tm'] * data.loc[:N_OBS, 'r_tm']
data.loc[0, 'V_tm'] = P1
data.loc[1:,'V_tm'] = data.loc[1:, 'P_tm'] - ((data.loc[1:, 'P_tm'] ** 2) * data.loc[:N_OBS, 'N_tm'])

# %%
#Forcasting 
future_data = data.copy(deep=True).reindex(list(range(0, N_OBS + 30))).reset_index(drop=True)
time_length = int(future_data['year'][1] - future_data['year'][0])
future_data.loc[N_OBS:N_OBS+30, 'year'] = future_data.loc[N_OBS-1, 'year'] + time_length * np.linspace(1,30,int(30/time_length))
# %%

future_data[['a_tf','P_tf','v_tf','F_tf']] = np.nan       


for t, y_t in enumerate(future_data['y_t']):
    if t == 0 :
        a_t = A1 
        p_t = P1
    a_tt, p_tt, a_t, P_t, v_t, F_t, K_t = kf_filter(y_t=y_t, a_t=a_t, P_t=P_t, 
                                            var_e=VAR_E, var_h=VAR_H)
    future_data.loc[t, ['a_tf','P_tf','v_tf','F_tf','K_tf']] = [a_tt, p_tt, v_t, F_t, K_t]
        

# %%
future_data['a_tf_upper_c'] = future_data['a_tf'] + .67 * np.sqrt(future_data['P_tf'])
future_data['a_tf_lower_c'] = future_data['a_tf'] - .67 * np.sqrt(future_data['P_tf'])
        
# %%
data['u_star_t'] = data['u_t'] / np.sqrt(data['D_t'])
data['r_star_t'] = data['r_t'] / np.sqrt(data['N_t'])
data['e_t'] = data['v_t'] / np.sqrt(data['F_t'])







# %%

# Confidence intervals

data['a_t_upper_c'] = data['a_t'] + 1.96 * np.sqrt(data['p_t'])
data['a_t_lower_c'] = data['a_t'] - 1.96 * np.sqrt(data['p_t'])

data['alpha_hat_t_upper_c'] = data['alpha_hat_t'] + 1.96 * np.sqrt(data['V_t'])
data['alpha_hat_t_lower_c'] = data['alpha_hat_t'] - 1.96 * np.sqrt(data['V_t'])




# %%
######## Figure 2.1
fig, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()

# Data and Filtered State
data.plot(ax = axes[0], x = 'year', y = 'flow', linestyle = 'none', marker = '.', color = 'red',
         ylim = (450,1400), xlim = (1865,1975), legend = None)
data.plot(ax = axes[0], x = 'year', y = 'a_t_upper_c', color = 'black', linewidth = 0.4, legend = None)
data.plot(ax = axes[0], x = 'year', y = 'a_t', color = 'blue', legend = None)
data.plot(ax = axes[0], x = 'year', y = 'a_t_lower_c', color = 'black', linewidth = 0.4, legend = None)

# Filtered State Variance Pt
data.plot(ax = axes[1], x = 'year', y = 'p_t')

# Prediction Errors vt
data.plot(ax = axes[2], x = 'year', y = 'v_t', ylim = (-450,450), legend = None)
axes[2].axhline(y = 0, color = 'b', linestyle = ':')

# Prediction Variance Ft
data.iloc[1:].plot(ax = axes[3], x='year', y='F_t', ylim =(20000,32500), legend = None) # Ignore initialization

# Save Figure
plt.savefig('Figures/Fig_2_1.png', facecolor='w')

# %%
# FIGURE 2.2

fig, axes = plt.subplots(2,2, figsize = (10,10))
axes = axes.ravel()

# Data and Smoothed State alpha_hat
data.plot(ax = axes[0], x = 'year', y = 'flow', linestyle = 'none', marker = '.', color = 'red',
         ylim = (450,1400), xlim = (1865,1975), legend = None)
data.iloc[1:].plot(ax = axes[0], x = 'year', y = 'alpha_hat_t_upper_c', color = 'black', linewidth = 0.4, legend = None)
data.iloc[1:].plot(ax = axes[0], x = 'year', y = 'alpha_hat_t', color = 'blue', legend = None)
data.iloc[1:].plot(ax = axes[0], x = 'year', y = 'alpha_hat_t_lower_c', color = 'black', linewidth = 0.4, legend = None)
# Smoothed State Variance Vt
data.plot(ax = axes[1], x = 'year', y = 'V_t', ylim = (2200, 4100), legend =None)

# Smoothing Cumulant rt

data.plot(ax = axes[2], x = 'year', y = 'r_t', legend = None)
axes[2].axhline(y = 0, color = 'black', linestyle = ':')
# Smoothing Cumulant Variant Nt
data.plot(ax = axes[3], x = 'year', y = 'N_t', ylim = (6e-5, .00011), legend =None)


# Save Figure
plt.savefig('Figures/Fig_2_2.png', facecolor = 'w')

# %%
# FIGURE 2.3
fig, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()

# Observation Error e hat
data.plot(ax = axes[0], x = 'year', y = 'e_hat_t', ylim = (-375,300), xlim = (1865,1975), legend = None)
axes[0].axhline(y = 0, color = 'black')

# Observation Error Standard Deviation
data.plot(ax = axes[1], x = 'year', y = 'sd_e_hat_t', legend = None)

# State error h hat
data.plot(ax = axes[2], x = 'year', y = 'h_hat_t', ylim = (-43,40), xlim = (1865,1975), legend = None)
axes[2].axhline(y = 0, color = 'black')

# State Error Standard Deviation
data.plot(ax = axes[3], x = 'year', y = 'sd_h_hat_t', legend = None)

# Save Figure
plt.savefig('Figures/Fig_2_3.png', facecolor = 'w')

# %%
# FIGURE 2.5
fig, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()

data.plot(ax = axes[0], x = 'year', y = 'missing_vals', linestyle = '-', marker = '.', color = 'red',
         ylim = (450,1400), xlim = (1865,1975), legend = None)
data.plot(ax = axes[0], x = 'year', y = 'a_tm', color = 'blue', legend = None)

data.plot(ax = axes[1], x = 'year', y = 'P_tm')
# Prediction Errors vt

data.plot(ax = axes[2], x = 'year', y = 'missing_vals', linestyle = '-', marker = '.', color = 'red',
         ylim = (450,1400), xlim = (1865,1975), legend = None)
data.iloc[1:].plot(ax = axes[2], x = 'year', y = 'alpha_hat_tm', color = 'blue', legend = None)

# Smoothed State Variance Vt
data.plot(ax = axes[3], x = 'year', y = 'V_tm', ylim = (2200, 10000), legend =None)

# Save Figure
plt.savefig('Figures/Fig_2_5.png', facecolor = 'w')

# %%
# FIGURE 2.6
fig, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()


future_data.iloc[1:].plot(ax = axes[0], x = 'year', y = 'a_tf_upper_c', color = 'black', linewidth = .4, legend = None)
future_data.iloc[1:].plot(ax = axes[0], x = 'year', y = 'a_tf', color = 'blue', legend = None)
future_data.iloc[1:].plot(ax = axes[0], x = 'year', y = 'a_tf_lower_c', color = 'black', linewidth = .4, legend = None)
future_data.plot(ax = axes[0], x = 'year', y = 'y_t', linestyle = 'none', marker = '.', color = 'red',
         ylim = (450,1400), xlim = (1865,1975+30), legend = None)


# Filtered State Variance Pt
future_data.plot(ax = axes[1], x = 'year', y = 'P_tf')

# Prediction Errors vt
future_data.iloc[1:].plot(ax = axes[2], x = 'year', y = 'a_tf', ylim = (700,1200), legend = None)
axes[2].axhline(y = 0, color = 'b', linestyle = ':')

# Prediction Variance Ft
future_data.iloc[1:].plot(ax = axes[3], x='year', y='F_tf', ylim =(20000,32500), legend = None) # Ignore initialization
# Save Figure

plt.savefig('Figures/Fig_2_6.png', facecolor = 'w')

# %%
# FIGURE 2.7
fig, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()

# Standardized Residual
data.plot(ax = axes[0], x = 'year', y = 'e_t', ylim=(-2.8,2.8), legend=None)
axes[0].axhline(y = 0, color = 'black', linestyle = ':')

# Histogram plus estimated density
sns.histplot(ax = axes[1], data=data.iloc[1:], x = 'e_t', stat='density', kde=True, bins=17)

# Ordered residuals
sm.qqplot(data.loc[9:, 'e_t'], ax = axes[2], line ='45')

# Correlogram

#### DOESN"T LOOK LIKE RIGHT TYPE OF PLOT
pd.plotting.autocorrelation_plot(data['e_t'], ax = axes[3]).plot()

# Save Figure
plt.savefig('Figures/Fig_2_7.png', facecolor = 'w')

# %%
# FIGURE 2.8
fig, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()

# Observation Residual ut^*
data.plot(ax = axes[0], x = 'year', y = 'u_star_t', ylim=(-3,2.2), legend=None)
axes[0].axhline(y = 0, color = 'black', linestyle = ':')

# Histogram and estimated density for ut^*
sns.histplot(ax = axes[1], data=data.iloc[2:], x = 'u_star_t', stat='density', kde=True, bins=17, binwidth = 0.4)


# State Residual rt^*
data.plot(ax = axes[2], x = 'year', y = 'r_star_t', ylim=(-3,2.2), legend=None)
axes[2].axhline(y = 0, color = 'black', linestyle = ':')

# Histogram and estimated density for rt^*
sns.histplot(ax = axes[3], data=data.iloc[2:], x = 'r_star_t', stat='density', kde=True, bins=17, binwidth = 0.4)

# Save Figure
plt.savefig('Figures/Fig_2_8.png', facecolor = 'w')

