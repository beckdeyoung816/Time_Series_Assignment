

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# %%
########## NOTATION

# v_t = Prediction Error
# F_t = Prediction Error Variance
# K_t = Kalman Gain
# a_t / a_tt = Filter Density Mean
# p_t / p_tt = Filter Density Variance
#

# %%

data = pd.read_excel('../Data/Nile.xlsx', names = ['year', 'flow'])
    
VAR_E = 15099
VAR_H = 1469.1
A1 = 0
P1 = 10 ** 7
N_OBS = data.shape[0]

# %%

def initialize_cols(df, col_names):
    # Initialize empty columns to help with recursion
    return df.reindex(df.columns.tolist() + col_names, axis=1)

# %%
def filter(y_t, a_t, p_t, var_e, var_h):
    
    # Kalman gain calculation
    v_t = y_t - a_t
    F_t = p_t + var_e
    K_t = p_t / F_t
    
    # State update
    a_tt = a_t + (K_t * v_t)
    p_tt = p_t * (1-K_t)
    
    # Prediction Step
    a_t = a_tt
    p_t = p_tt + var_h
    
    return a_tt, p_tt, a_t, p_t, v_t, F_t, K_t

# %%
data = initialize_cols(data, col_names=['a_t','p_t', 'v_t','F_t','K_t'])

# %%
for t, y_t in enumerate(data['flow']):
    # Initialize Values
    if t == 0 :
        a_t = A1 
        p_t = P1
    
    # Apply the filter
    a_tt, p_tt, a_t, p_t, v_t, F_t, K_t = filter(y_t=y_t, a_t=a_t, p_t=p_t, 
                                                 var_e=VAR_E, var_h=VAR_H)
    
    # Store output
    data.loc[t, ['a_t','p_t','v_t','F_t','K_t']] = [a_tt, p_tt, v_t, F_t, K_t]

data['L_t'] = 1 - data['K_t']
data = initialize_cols(data,col_names=['r_t','N_t'])

# %%
for t in reversed(range(N_OBS)):
    if t == N_OBS-1: # Since the last observation has no 't+1', set it to 0
        data.loc[t, ['r_t', 'N_t']] = [0,0]
    else:
        data.loc[t, 'N_t'] = (1 / data.loc[t+1, 'F_t']) + ((data.loc[t+1, 'L_t'] ** 2) * data.loc[t+1, 'N_t'])
        data.loc[t, 'r_t'] = (data.loc[t+1, 'v_t'] / data.loc[t+1, 'F_t']) + (data.loc[t+1, 'L_t'] * data.loc[t+1, 'r_t'])
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
######## Figure 2.1
fig, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()

# Data and Filtered State
########### STILL NEED CONFIDENCE INTERVALS
data.plot(ax = axes[0], x = 'year', y = 'flow', linestyle = 'none', marker = '.', color = 'red',
         ylim = (450,1400), xlim = (1865,1975), legend = None)
data.plot(ax = axes[0], x = 'year', y = 'a_t', color = 'blue', legend = None)

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
fig, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()




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



# Save Figure
plt.savefig('Figures/Fig_2_5.png', facecolor = 'w')

# %%
# FIGURE 2.6
fig, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()



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
sns.histplot(ax = axes[1], data=data.iloc[2:], x = 'e_t', stat='density', kde=True, bins=17)

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

# %%
