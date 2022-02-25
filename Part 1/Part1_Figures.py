
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from LLM import LLM
    
# %%
def fig_1(data, df_name='', ylim_0=None, xlim_0=None, ylim_2=None, ylim_3=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Data and Filtered State
    ## Observed Data
    data.plot(ax = axes[0], x = 'x', y = 'y_t', linestyle = 'none', marker = '.', color = 'red',
            ylim = ylim_0, xlim = xlim_0, legend = None)
    ## Filtered State
    data.plot(ax = axes[0], x = 'x', y = 'a_t', color = 'blue', legend = None)
    ## Confidence Intervals
    data.plot(ax = axes[0], x = 'x', y = 'a_t_upper_c', color = 'black', linewidth = 0.4, legend = None)
    data.plot(ax = axes[0], x = 'x', y = 'a_t_lower_c', color = 'black', linewidth = 0.4, legend = None)

    # Filtered State Variance Pt
    data.plot(ax = axes[1], x = 'x', y = 'P_t', legend = None)

    # Prediction Errors vt
    data.plot(ax = axes[2], x = 'x', y = 'v_t', ylim = ylim_2, legend = None)
    axes[2].axhline(y = 0, color = 'black', linestyle = ':') # Line at x axis

    # Prediction Variance Ft
    data.iloc[1:].plot(ax = axes[3], x='x', y='F_t', ylim=ylim_3, legend = None) # Ignore initialization

    # Save Figure
    plt.savefig(f'Figures/Fig_2_1_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_2(data, df_name='', ylim_0=None, xlim_0=None, ylim_1=None, ylim_3=None):
    fig, axes = plt.subplots(2,2, figsize = (10,10))
    axes = axes.ravel()

    # Data and Smoothed State alpha_hat
    ## Observed Data
    data.plot(ax = axes[0], x = 'x', y = 'y_t', linestyle = 'none', marker = '.', color = 'red',
            ylim = ylim_0, xlim = xlim_0, legend = None)
    ## Smoothed State
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'alpha_hat_t', color = 'blue', legend = None)
    ## Confidence Intervals
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'alpha_hat_t_upper_c', color = 'black', linewidth = 0.4, legend = None)
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'alpha_hat_t_lower_c', color = 'black', linewidth = 0.4, legend = None)

    # Smoothed State Variance Vt
    data.plot(ax = axes[1], x = 'x', y = 'V_t', ylim = ylim_1, legend = None)

    # Smoothing Cumulant rt
    data.plot(ax = axes[2], x = 'x', y = 'r_t', legend = None)
    axes[2].axhline(y = 0, color = 'black', linestyle = ':') # Line at x axis
    
    # Smoothing Cumulant Variant Nt
    data.plot(ax = axes[3], x = 'x', y = 'N_t', ylim = ylim_3, legend = None)

    # Save Figure
    plt.savefig(f'Figures/Fig_2_2_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_3(data, df_name='', ylim_0=None, xlim_0=None, ylim_2=None, xlim_2=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Observation Error e hat
    data.plot(ax = axes[0], x = 'x', y = 'e_hat_t', ylim = ylim_0, xlim = xlim_0, legend = None)
    axes[0].axhline(y = 0, color = 'black')

    # Observation Error Standard Deviation
    data.plot(ax = axes[1], x = 'x', y = 'sd_e_hat_t', legend = None)

    # State error h hat
    data.plot(ax = axes[2], x = 'x', y = 'h_hat_t', ylim = ylim_2, xlim = xlim_2, legend = None)
    axes[2].axhline(y = 0, color = 'black')

    # State Error Standard Deviation
    data.plot(ax = axes[3], x = 'x', y = 'sd_h_hat_t', legend = None)

    # Save Figure
    plt.savefig(f'Figures/Fig_2_3_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_5(data, df_name='', ylim_0=None, xlim_0=None, ylim_2=None, xlim_2=None, ylim_3=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Data and Filtered State
    data.plot(ax = axes[0], x = 'x', y = 'y_tm', linestyle = ":", marker = '.', color = 'red',
            ylim = ylim_0, xlim = xlim_0, legend = None)
    data.plot(ax = axes[0], x = 'x', y = 'a_tm', color = 'blue', legend = None)

    # Filtered State Variance Ptm
    data.plot(ax = axes[1], x = 'x', y = 'P_tm', legend = None)
    
    # Data and Smoothed State
    data.plot(ax = axes[2], x = 'x', y = 'y_tm', linestyle = ":", marker = '.', color = 'red',
            ylim = ylim_2, xlim = xlim_2, legend = None)
    data.iloc[1:].plot(ax = axes[2], x = 'x', y = 'alpha_hat_tm', color = 'blue', legend = None)

    # Smoothed State Variance Vt
    data.plot(ax = axes[3], x = 'x', y = 'V_tm', ylim = ylim_3, legend =None)


    # Save Figure
    plt.savefig(f'Figures/Fig_2_5_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_6(data, df_name='', ylim_0=None, xlim_0=None, ylim_2=None, ylim_3=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Data and State Forecast
    ## Observed Data
    data.plot(ax = axes[0], x = 'x', y = 'y_t', linestyle = 'none', marker = '.', color = 'red',
        ylim = ylim_0, xlim = xlim_0, legend = None)
    ## State Forecast
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'a_tf_upper_c', color = 'black', linewidth = .4, legend = None)
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'a_tf', color = 'blue', legend = None)
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'a_tf_lower_c', color = 'black', linewidth = .4, legend = None)

    # State Variance Pt
    data.plot(ax = axes[1], x = 'x', y = 'P_tf', legend = None)

    # Observation Forecast
    data.iloc[1:].plot(ax = axes[2], x = 'x', y = 'a_tf', ylim = ylim_2, legend = None)
    axes[2].axhline(y = 0, color = 'b', linestyle = ':')

    # Observation Forecast Variance Ft
    data.iloc[1:].plot(ax = axes[3], x='x', y='F_tf', ylim =ylim_3, legend = None) # Ignore initialization

    # Save Figure
    plt.savefig(f'Figures/Fig_2_6_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_7(data, df_name='', ylim_0=None, ylim_3=None, xlim_3=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Standardized Residual
    data.plot(ax = axes[0], x = 'x', y = 'e_t', ylim=ylim_0, legend = None)
    axes[0].axhline(y = 0, color = 'black', linestyle = ':')

    # Histogram plus estimated density
    sns.histplot(ax = axes[1], data=data.iloc[2:], x = 'e_t', stat='density', kde=True, bins=17)

    # Ordered residuals
    sm.qqplot(data.loc[9:, 'e_t'], ax = axes[2], line ='45')

    # Correlogram
    correls = np.correlate(data['e_t'], data['e_t'], mode="full")
    correls /= np.dot(data['e_t'], data['e_t'])

    maxlags = 10
    N_OBS = data.shape[0]
    lags = np.arange(-maxlags, maxlags + 1)
    correls = correls[N_OBS - 1 - maxlags:N_OBS + maxlags]

    axes[3].vlines(lags, [0], correls, linewidth = 15)
    axes[3].axhline()
    axes[3].set_ylim(ylim_3)
    axes[3].set_xlim(xlim_3)

    # Save Figure
    plt.savefig(f'Figures/Fig_2_7_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_8(data, df_name='', ylim_0=None, ylim_2=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Observation Residual ut^*
    data.plot(ax = axes[0], x = 'x', y = 'u_star_t', ylim=ylim_0, legend = None)
    axes[0].axhline(y = 0, color = 'black', linestyle = ':')

    # Histogram and estimated density for ut^*
    sns.histplot(ax = axes[1], data=data.iloc[2:], x = 'u_star_t', stat='density', kde=True, bins=17, binwidth = 0.4)


    # State Residual rt^*
    data.plot(ax = axes[2], x = 'x', y = 'r_star_t', ylim=ylim_2, legend = None)
    axes[2].axhline(y = 0, color = 'black', linestyle = ':')

    # Histogram and estimated density for rt^*
    sns.histplot(ax = axes[3], data=data.iloc[2:], x = 'r_star_t', stat='density', kde=True, bins=17, binwidth = 0.4)

    # Save Figure
    plt.savefig(f'Figures/Fig_2_8_{df_name}.png', facecolor = 'w')
    plt.show()

# %%
#data = pd.read_csv('../Data/AirPassengers.csv')
data = pd.read_excel('../Data/Nile.xlsx', names = ['x', 'y_t'])
nile_llm = LLM(data=data, var_e=15099, var_h=1469.1, a_1=0, P_1=10**7)

# Set parameters
missing_vals_nile = [{'start': 21, 'stop': 40},
                {'start': 61, 'stop': 80}] # Indices for missing vals
forecast_n_nile = 30 # Number of samples to forecast
# %%
nile_llm.kalman_filter()
nile_llm.state_smooth()
nile_llm.disturbance_smooth()
nile_llm.auxilary_residuals()
nile_llm.missing_filter(missing_ranges=missing_vals_nile)
nile_llm.missing_smooth()
nile_llm.forecast(j=forecast_n_nile)

# Confidence intervals
nile_llm.get_conf_intervals('a_t', 'P_t', pct=.90)
nile_llm.get_conf_intervals('alpha_hat_t', 'V_t', pct=.90)
# nile_llm.get_conf_intervals('a_t', 'P_t', pct=.50)
# nile_llm.get_conf_intervals('alpha_hat_t', 'V_t', pct=.50)
# %%
# Generate and Save Figures
fig_1(nile_llm.df, df_name = 'nile', ylim_0=(450,1400), xlim_0=(1865,1975),ylim_2=(-450,450), ylim_3=(20000,32500))
fig_2(nile_llm.df, df_name = 'nile', ylim_0=(450,1400), xlim_0=(1865,1975),ylim_1=(2200, 4100), ylim_3=(6e-5, .00011))
fig_3(nile_llm.df, df_name = 'nile', ylim_0=(-375,300), xlim_0=(1865,1975),ylim_2=(-43,40), xlim_2=(1865,1975))
fig_5(nile_llm.df, df_name = 'nile', ylim_0=(450,1400), xlim_0=(1865,1975), ylim_2=(450,1400), xlim_2=(1865,1975), ylim_3=(2200, 10000))
fig_6(nile_llm.forecast_df, df_name = 'nile', ylim_0=(450,1400), xlim_0=(1865,1975 + forecast_n), ylim_2=(700,1200), ylim_3=(20000,60000))
fig_7(nile_llm.df, df_name = 'nile', ylim_0=(-2.8,2.8), ylim_3=(-1,1), xlim_3=(.5,11))
fig_8(nile_llm.df, df_name = 'nile', ylim_0=(-3,2.2), ylim_2=(-3,2.2))

# %%
data2= pd.read_csv('../Data/AirPassengers.csv')
data2['Month'] = pd.to_datetime(data2['Month']).dt.to_period('M') # Convert datatime to months
data2.columns = ['x', 'y_t'] # Rename columns
# Set parameters
missing_vals_air = [{'start': 21, 'stop': 40},
                {'start': 61, 'stop': 80}] # Indices for missing vals
forecast_n_air = 30 # Number of samples to forecast
# %%
air_llm = LLM(data=data2, ML_start = [1])
air_llm.kalman_filter()
air_llm.state_smooth()
air_llm.disturbance_smooth()
air_llm.auxilary_residuals()
air_llm.missing_filter(missing_ranges=missing_vals_air)
air_llm.missing_smooth()
air_llm.forecast(j=forecast_n_air)

# Confidence intervals
air_llm.get_conf_intervals('a_t', 'P_t', pct=.90)
air_llm.get_conf_intervals('alpha_hat_t', 'V_t', pct=.90)
# air_llm.get_conf_intervals('a_t', 'P_t', pct=.50)
# air_llm.get_conf_intervals('alpha_hat_t', 'V_t', pct=.50)
# %%
# Generate and Save Figures
fig_1(air_llm.df, df_name = 'air')
fig_2(air_llm.df, df_name = 'air')
fig_3(air_llm.df, df_name = 'air')
fig_5(air_llm.df, df_name = 'air')
fig_6(air_llm.forecast_df, df_name = 'air')
fig_7(air_llm.df, df_name = 'air')
fig_8(air_llm.df, df_name = 'air')

# %%
