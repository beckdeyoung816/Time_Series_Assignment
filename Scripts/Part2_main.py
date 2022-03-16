# %%
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from SSM_SV import SSM_SV

# %%

############################################ PART A THROUGH D
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

# Ht = ht - xi WHERE xi = w/(1-phi) = -10.2089 from QML
axes[1].plot(sv_ssm.df.a_t[4:] - sv_ssm.xi, 'blue', label = 'Filtered Ht')
axes[1].plot(sv_ssm.df.alpha_hat_t[2:] - sv_ssm.xi, 'red', label = 'Smoothed Ht')
axes[1].legend()

############################################## Part e
# %%
# Data prep
# Choices for stock and RV measure
rv_index = '.DJI'
rv_measure = 'rv5'
START_YEAR = 2010
END_YEAR = 2016

rv = pd.read_csv('../Data/Realized_Volatility_Indices.csv')
rv = rv[rv['Symbol'] == rv_index] # Select desired index
rv = rv[['Date', 'close_price', rv_measure]]

# Fix dates and filter for desired years
rv['Date'] = pd.to_datetime(rv.Date) 
rv['Date'] = rv['Date'].dt.date
rv = rv[(rv['Date'] >= dt.date(START_YEAR,1,1)) & (rv['Date'] < dt.date(END_YEAR,1,1))].reset_index(drop=True)

# %%
# Transform the returns to use in the state space model
rv['y_t'] = np.log(rv.close_price / rv.close_price.shift(1)) # log(Pt/Pt-1)
rv = rv.iloc[1:].reset_index(drop=True) # Remove the first value it has no lag

mu = np.mean(rv['y_t']) # Calculate the mean to detrend the data
rv['y_t'] = np.log((rv['y_t']-mu) ** 2) # Transform returns
rv['X_t'] = np.log(rv[rv_measure]) # Calculate log of RV measure to be used as Xt

# %%
# Create state space model with this new data
df_rv = rv[['X_t', 'y_t']].copy(deep=True)
rv_ssm = SSM_SV(data = df_rv, H_t = (np.pi ** 2) / 2, c_t = -1.27, 
                # theta0=[0.8,0.0077,0.995]
                  Q_t = 0.4247156788750475,
                  d_t = -0.8739300473341086,
                  T_t = 0.9122642976429152
                 )
# %%

# Regular part a-d
rv_ssm.kalman_filter()
rv_ssm.state_smooth()

# %%
fig, axes = plt.subplots(2,1, figsize=(15,12))
axes[0].scatter(rv_ssm.df.index, rv_ssm.df.y_t, color='black', alpha=.5)
axes[0].plot(rv_ssm.df.alpha_hat_t[2:], 'red', linewidth=2, label = 'Smoothed ht')
axes[0].legend(loc='lower right')


# Ht = ht - xi WHERE xi = w/(1-phi) = -10.2089 from QML
axes[1].plot(rv_ssm.df.a_t[4:] - rv_ssm.xi, 'blue', label = 'Filtered Ht')
axes[1].plot(rv_ssm.df.alpha_hat_t[2:] - rv_ssm.xi, 'red', label = 'Smoothed Ht')
axes[1].legend()



# %%

# model with beta and RV added

rv_ssm.estimate_Beta()
rv_ssm.kalman_filter(beta=True)
rv_ssm.state_smooth(beta=True)
# %%
fig, axes = plt.subplots(2,1, figsize=(15,12))
axes[0].scatter(rv_ssm.df.index, rv_ssm.df.y_t, color='black', alpha=.5)
axes[0].plot(rv_ssm.df.alpha_hat_t_beta[2:], 'red', linewidth=2, label = 'Smoothed ht')
axes[0].legend(loc='lower right')

# Ht = ht - xi WHERE xi = w/(1-phi) = -10.2089 from QML
axes[1].plot(rv_ssm.df.a_t_beta[4:] - rv_ssm.xi, 'blue', label = 'Filtered Ht')
axes[1].plot(rv_ssm.df.alpha_hat_t_beta[2:] - rv_ssm.xi, 'red', label = 'Smoothed Ht')
axes[1].legend()

# %%
fig, axes = plt.subplots(2,1, figsize=(15,12))
axes[0].plot(rv_ssm.df.a_t[4:] - rv_ssm.xi, 'blue', label = 'Filtered Ht')
axes[0].legend()

axes[1].plot(rv_ssm.df.alpha_hat_t[2:] - rv_ssm.xi, 'red', label = 'Smoothed Ht')
axes[1].legend()
# %%
