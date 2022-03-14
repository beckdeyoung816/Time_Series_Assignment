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
sv['y_t'] = np.log((sv['y_t']-mu) ** 2) # Transform returns

# %%
plt.figure(figsize=(15,6))
plt.plot(sv.y_t)
plt.ylim((-.03, 0.05))
plt.xlim((0,950))
# %%
plt.figure(figsize=(15,6))
plt.scatter(sv.index, sv.y_t)
plt.ylim((-30, -8))
plt.xlim((0,950))
# %%

test_llm = SSM_SV(data = sv[['y_t']], H_t = (np.pi ** 2) / 2, c_t = -1.27, theta0=[0.8,0.0077,0.995])

# %%
test_llm.kalman_filter()


# %%
plt.figure(figsize=(15,6))
plt.scatter(test_llm.df.index, test_llm.df.y_t)
#plt.plot(test_llm.df.y_t)
plt.plot(test_llm.df.a_t, 'red')
# %%

mle_results = test_llm.mle_sv([0.8,0.0077,0.995])
# %%


