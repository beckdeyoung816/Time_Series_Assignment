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
mu = np.mean(sv.r_t)
sv['y_t'] = np.log((sv.r_t-mu) ** 2)
sv['y_t2'] = np.log(sv.r_t ** 2)

# %%
plt.plot(sv.y_t2)
# %%
