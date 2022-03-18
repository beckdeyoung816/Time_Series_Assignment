# %%
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from SSM_SV import SSM_SV
import Part2_Figures as p2f

# %%

############################################ PART A THROUGH D SV Data
# Load in the data
sv = pd.read_excel('../Data/SvData.xlsx')
sv.columns = ['r_t']
sv['y_t'] = sv['r_t'] / 100 # R_t is misscaled
mu = np.mean(sv['y_t']) # Calculate the mean to detrend the data
sv['x_t'] = np.log((sv['y_t']-mu) ** 2) # Transform returns



# %%
df_sv = sv[['x_t']].rename(columns={'x_t': 'y_t'}).copy(deep=True)
sv_ssm = SSM_SV(data = df_sv, H_t = (np.pi ** 2) / 2, c_t = -1.27, 
                #theta0=[0.8,0.0077,0.995] # Our initial values used in QML
                  Q_t = 0.04006997037080177,
                  d_t = -0.1186102356182056,
                  T_t = 0.9883814337893937
                 )

# %%
sv_ssm.kalman_filter()
sv_ssm.state_smooth()
  
# %%
p2f.part_a_fig(sv,y_lim=(-.03, 0.05), data = 'sv')
p2f.part_a_descr_stats(sv['y_t'])
p2f.part_b_fig(sv,y_lim=(-30, -8), data = 'sv')
p2f.part_d_fig(sv_ssm, data = 'sv')

############################################## Extension 1
# %%
# Data prep
# Choices for stock, RV measure, and years to examine
rv_index = '.DJI'
rv_measure = 'rv5'
START_YEAR = 2010
END_YEAR = 2016

rv = pd.read_csv('../Data/Realized_Volatility_Indices.csv')
rv = rv.loc[rv['Symbol'] == rv_index, ['Date', 'close_price', rv_measure]] # Select desired index

# Fix dates and filter for desired years
rv['Date'] = pd.to_datetime(rv.Date) 
rv['Date'] = rv['Date'].dt.date
rv = rv[(rv['Date'] >= dt.date(START_YEAR,1,1)) & \
        (rv['Date'] < dt.date(END_YEAR,1,1))].reset_index(drop=True)

# %%
# Transform the returns to use in the state space model
rv['y_t'] = np.log(rv.close_price / rv.close_price.shift(1)) # log(Pt/Pt-1)
rv = rv.iloc[1:].reset_index(drop=True) # Remove the first value it has no lag

mu = np.mean(rv['y_t']) # Calculate the mean to detrend the data
rv['x_t'] = np.log((rv['y_t']-mu) ** 2) # Transform returns
rv['X_t'] = np.log(rv[rv_measure]) # Calculate log of RV measure to be used as Xt

# %%
# Create state space model with this new data
df_rv = rv[['X_t', 'x_t']].rename(columns={'x_t': 'y_t'}).copy(deep=True)
rv_ssm = SSM_SV(data = df_rv, H_t = (np.pi ** 2) / 2, c_t = -1.27, 
                # theta0=[0.8,0.0077,0.995] # Our initial values used in QML
                  Q_t = 0.4247156788750475,
                  d_t = -0.8739300473341086,
                  T_t = 0.9122642976429152
                 )
# %%
# Regular SV model with no Beta
rv_ssm.kalman_filter()
rv_ssm.state_smooth()

# %%
p2f.part_a_fig(rv,y_lim = (-.03, 0.05), data = 'rv')
p2f.part_a_descr_stats(rv['y_t'])
p2f.part_b_fig(rv, data = 'rv')
p2f.part_d_fig(rv_ssm, data = 'rv')

# %%
# Model with beta and RV added

rv_ssm.estimate_Beta() # Get our estimate for Beta
rv_ssm.kalman_filter(beta=True)
rv_ssm.state_smooth(beta=True)

# %%
p2f.part_d_fig(rv_ssm, data = 'rv', beta = '_beta')


# %%
