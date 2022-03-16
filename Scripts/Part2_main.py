# %%
import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from SSM_SV import SSM_SV

# %%
# Load in the data
sv = pd.read_excel('../Data/SvData.xlsx')
sv.columns = ['r_t']
sv['y_t'] = sv['r_t'] / 100 # R_t is misscaled
mu = np.mean(sv['y_t']) # Calculate the mean to detrend the data
sv['x_t'] = np.log((sv['y_t']-mu) ** 2) # Transform returns

# %%
plt.figure(figsize=(15,6))
plt.plot(sv.y_t)
plt.ylim((-.03, 0.05))
plt.xlim((0,950))
# %%
plt.figure(figsize=(15,6))
plt.scatter(sv.index, sv.x_t)
plt.ylim((-30, -8))
plt.xlim((0,950))
# %%
df_sv = sv[['x_t']].rename(columns={'x_t': 'y_t'}, copy=True)
sv_ssm = SSM_SV(data = df_sv, H_t = (np.pi ** 2) / 2, c_t = -1.27, 
                  Q_t = 0.04006997037080177,
                  d_t = -0.1186102356182056,
                  T_t = 0.9883814337893937,
                 )#,
                  #theta0=[0.8,0.0077,0.995])

# %%
sv_ssm.kalman_filter()
sv_ssm.state_smooth()

# %%
fig, axes = plt.subplots(2,1, figsize=(15,12))
axes[0].scatter(sv_ssm.df.index, sv_ssm.df.y_t, color='black', alpha=.5)
axes[0].plot(sv_ssm.df.alpha_hat_t[2:], 'red', linewidth=2, label = 'Smoothed ht')
axes[0].legend(loc='lower right')


# xi = w/(1-phi) = -10.2089
# Ht = ht - xi
axes[1].plot(sv_ssm.df.a_t[4:] - sv_ssm.xi, 'blue', label = 'Filtered Ht')
axes[1].plot(sv_ssm.df.alpha_hat_t[2:] - sv_ssm.xi, 'red', label = 'Smoothed Ht')
axes[1].legend()

# Part e
# %%
# Data prep
rv = pd.read_csv('../Data/Realized_Volatility_Indices.csv')
rv = rv[rv['Symbol'] == '.DJI'] # Select desired index
rv = rv[['Date', 'close_price', 'rv5']]
# Fix dates and filter for desired years
rv['Date'] = pd.to_datetime(rv.Date) 
rv['Date'] = rv['Date'].dt.date
rv = rv[(rv['Date'] >= dt.date(2010,1,1)) & (rv['Date'] < dt.date(2016,1,1))].reset_index(drop=True)

# %%
rv['y_t'] = np.log(rv.close_price / rv.close_price.shift(1))
rv = rv.iloc[1:].reset_index(drop=True)
mu = np.mean(rv['y_t']) # Calculate the mean to detrend the data
rv['y_t'] = np.log((rv['y_t']-mu) ** 2) # Transform returns
rv['X_t'] = np.log(rv['rv5'])

# %%
df_rv = rv[['X_t', 'y_t']].copy(deep=True)
rv_ssm = SSM_SV(data = df_rv, H_t = (np.pi ** 2) / 2, c_t = -1.27, 
                # theta0=[0.8,0.0077,0.995]
                  Q_t = 0.4247156788750475,
                  d_t = -0.8739300473341086,
                  T_t = 0.9122642976429152
                 )
# %%
rv_ssm.kalman_filter()
rv_ssm.state_smooth()

# %%
fig, axes = plt.subplots(2,1, figsize=(15,12))
axes[0].scatter(rv_ssm.df.index, rv_ssm.df.y_t, color='black', alpha=.5)
axes[0].plot(rv_ssm.df.alpha_hat_t[2:], 'red', linewidth=2, label = 'Smoothed ht')
axes[0].legend(loc='lower right')


# xi = w/(1-phi) = -10.2089
# Ht = ht - xi
axes[1].plot(rv_ssm.df.a_t[4:] - rv_ssm.xi, 'blue', label = 'Filtered Ht')
axes[1].plot(rv_ssm.df.alpha_hat_t[2:] - rv_ssm.xi, 'red', label = 'Smoothed Ht')
axes[1].legend()

# %%
dji_ssm_rv = SSM_SV(data = df_rv, H_t = (np.pi ** 2) / 2, c_t = -1.27, 
                # theta0=[0.8,0.0077,0.995]
                  Q_t = 0.4247156788750475,
                  d_t = -0.8739300473341086,
                  T_t = 0.9122642976429152,
                  use_Beta=True
                 )
# %%
dji_ssm_rv.kalman_filter()
dji_ssm_rv.state_smooth()
# %%
fig, axes = plt.subplots(2,1, figsize=(15,12))
axes[0].scatter(dji_ssm_rv.df.index, rv_ssm.df.y_t, color='black', alpha=.5)
axes[0].plot(dji_ssm_rv.df.alpha_hat_t[2:], 'red', linewidth=2, label = 'Smoothed ht')
axes[0].legend(loc='lower right')


# xi = w/(1-phi) = -10.2089
# Ht = ht - xi
axes[1].plot(dji_ssm_rv.df.a_t[4:] - dji_ssm_rv.xi, 'blue', label = 'Filtered Ht')
axes[1].plot(dji_ssm_rv.df.alpha_hat_t[2:] - dji_ssm_rv.xi, 'red', label = 'Smoothed Ht')
axes[1].legend()

# %%
test = rv_ssm.df[['y_t', 'a_t', 'alpha_hat_t']]
test[['X_t', 'a_beta_t', 'alpha_hat_beta_t']] = dji_ssm_rv.df[['X_t', 'a_t', 'alpha_hat_t']]
# %%
