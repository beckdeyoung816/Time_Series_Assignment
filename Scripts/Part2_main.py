# %%
import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats as ss
import matplotlib.pyplot as plt

from SSM import SSM

# %%
# Load in the data
sv = pd.read_excel('../Data/SvData.xlsx')
sv.columns = ['r_t']

# %%
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
