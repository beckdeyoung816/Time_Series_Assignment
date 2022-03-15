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
df = sv[['x_t']].rename(columns={'x_t': 'y_t'}, copy=True)
sv_ssm = SSM_SV(data = df, H_t = (np.pi ** 2) / 2, c_t = -1.27, 
                #   Q_t = 0.099,
                #   d_t = -0.1458,
                #   T_t = 0.9867,
                  Q_t = 0.345,
                  d_t = -2.032,
                  T_t = 0.8005,
                  )#,
                  #theta0=[0.8,0.0077,0.995])

# %%
sv_ssm.kalman_filter()
sv_ssm.state_smooth()

# %%
plt.figure(figsize=(15,6))
#plt.scatter(sv_ssm.df.index, sv_ssm.df.y_t)
plt.plot(sv_ssm.df.a_t[1:], 'red', label = 'Filtered')
plt.plot(sv_ssm.df.alpha_hat_t[2:], 'green', label = 'Smoothed')
plt.legend()
# %%
# %%


# %%
