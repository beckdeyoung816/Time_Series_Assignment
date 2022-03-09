import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime as dt

FIG_PATH = f'../Figures/Part_1'
    
def fig_1(data, df_name='', ylim_0=None, xlim_0=None, ylim_1=None, xlim_1=None, xlim_2=None, ylim_2=None,xlim_3=None, ylim_3=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Data and Filtered State
    ## Observed Data
    data.plot(ax = axes[0], x = 'x', y = 'y_t', linestyle = 'none', marker = '.', color = 'red', legend = None)
    ## Filtered State
    data.plot(ax = axes[0], x = 'x', y = 'a_t', color = 'blue', legend = None)
    ## Confidence Intervals
    data.plot(ax = axes[0], x = 'x', y = 'a_t_upper_c', color = 'black', linewidth = 0.4, legend = None)
    data.plot(ax = axes[0], x = 'x', y = 'a_t_lower_c', color = 'black', linewidth = 0.4, legend = None)
    axes[0].set_xbound(xlim_0)
    axes[0].set_ybound(ylim_0)
    
    # Filtered State Variance Pt
    data.plot(ax = axes[1], x = 'x', y = 'P_t', legend = None)
    axes[1].set_xbound(xlim_1) 
    axes[1].set_ybound(ylim_1)

    # Prediction Errors vt
    axes[2].axhline(y = 0, color = 'black', linestyle = ':') # Line at x axis
    data.plot(ax = axes[2], x = 'x', y = 'v_t', legend = None)
    axes[2].set_xbound(xlim_2) 
    axes[2].set_ybound(ylim_2)
    
    # Prediction Variance Ft
    data.iloc[1:].plot(ax = axes[3], x='x', y='F_t', legend = None) # Ignore initialization
    axes[3].set_xbound(xlim_3)
    axes[3].set_ybound(ylim_3)
    
    # Save Figure
    plt.savefig(f'{FIG_PATH}/Fig_2_1_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_2(data, df_name='', ylim_0=None, xlim_0=None, ylim_1=None, xlim_1=None, xlim_2=None, ylim_2=None,xlim_3=None, ylim_3=None):
    fig, axes = plt.subplots(2,2, figsize = (10,10))
    axes = axes.ravel()

    # Data and Smoothed State alpha_hat
    ## Observed Data
    data.plot(ax = axes[0], x = 'x', y = 'y_t', linestyle = 'none', marker = '.', color = 'red', legend = None)
    ## Smoothed State
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'alpha_hat_t', color = 'blue', legend = None)
    ## Confidence Intervals
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'alpha_hat_t_upper_c', color = 'black', linewidth = 0.4, legend = None)
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'alpha_hat_t_lower_c', color = 'black', linewidth = 0.4, legend = None)
    axes[0].set_xbound(xlim_0)
    axes[0].set_ybound(ylim_0)
    
    # Smoothed State Variance Vt
    data.plot(ax = axes[1], x = 'x', y = 'V_t', legend = None)
    axes[1].set_xbound(xlim_1) 
    axes[1].set_ybound(ylim_1)
    
    # Smoothing Cumulant rt
    data.plot(ax = axes[2], x = 'x', y = 'r_t', legend = None)
    axes[2].axhline(y = 0, color = 'black', linestyle = ':') # Line at x axis
    axes[2].set_xbound(xlim_2) 
    axes[2].set_ybound(ylim_2)
    
    # Smoothing Cumulant Variant Nt
    data.plot(ax = axes[3], x = 'x', y = 'N_t', legend = None)
    axes[3].set_xbound(xlim_3)
    axes[3].set_ybound(ylim_3)
    
    # Save Figure
    plt.savefig(f'{FIG_PATH}/Fig_2_2_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_3(data, df_name='', ylim_0=None, xlim_0=None, ylim_1=None, xlim_1=None, xlim_2=None, ylim_2=None,xlim_3=None, ylim_3=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Observation Error e hat
    data.plot(ax = axes[0], x = 'x', y = 'e_hat_t', legend = None)
    axes[0].axhline(y = 0, color = 'black')
    axes[0].set_xbound(xlim_0)
    axes[0].set_ybound(ylim_0)

    # Observation Error Standard Deviation
    data.plot(ax = axes[1], x = 'x', y = 'sd_e_hat_t', legend = None)
    axes[1].set_xbound(xlim_1) 
    axes[1].set_ybound(ylim_1)
    
    # State error h hat
    data.plot(ax = axes[2], x = 'x', y = 'h_hat_t', legend = None)
    axes[2].axhline(y = 0, color = 'black')
    axes[2].set_xbound(xlim_2) 
    axes[2].set_ybound(ylim_2)
    
    # State Error Standard Deviation
    data.plot(ax = axes[3], x = 'x', y = 'sd_h_hat_t', legend = None)
    axes[3].set_xbound(xlim_3)
    axes[3].set_ybound(ylim_3)
    
    # Save Figure
    plt.savefig(f'{FIG_PATH}/Fig_2_3_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_5(data, df_name='', ylim_0=None, xlim_0=None, ylim_1=None, xlim_1=None, xlim_2=None, ylim_2=None,xlim_3=None, ylim_3=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Data and Filtered State
    data.plot(ax = axes[0], x = 'x', y = 'y_tm', linestyle = ":", marker = '.', color = 'red', legend = None)
    data.plot(ax = axes[0], x = 'x', y = 'a_tm', color = 'blue', legend = None)
    axes[0].set_xbound(xlim_0)
    axes[0].set_ybound(ylim_0)
    
    # Filtered State Variance Ptm
    data.plot(ax = axes[1], x = 'x', y = 'P_tm', legend = None)
    axes[1].set_xbound(xlim_1) 
    axes[1].set_ybound(ylim_1)
    
    # Data and Smoothed State
    data.plot(ax = axes[2], x = 'x', y = 'y_tm', linestyle = ":", marker = '.', color = 'red', legend = None)
    data.iloc[1:].plot(ax = axes[2], x = 'x', y = 'alpha_hat_tm', color = 'blue', legend = None)
    axes[2].set_xbound(xlim_2) 
    axes[2].set_ybound(ylim_2)
    
    # Smoothed State Variance Vt
    data.plot(ax = axes[3], x = 'x', y = 'V_tm', legend = None)
    axes[3].set_xbound(xlim_3)
    axes[3].set_ybound(ylim_3)

    # Save Figure
    plt.savefig(f'{FIG_PATH}/Fig_2_5_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_6(data, df_name='', ylim_0=None, xlim_0=None, ylim_1=None, xlim_1=None, xlim_2=None, ylim_2=None,xlim_3=None, ylim_3=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Data and State Forecast
    ## Observed Data
    data.plot(ax = axes[0], x = 'x', y = 'y_t', linestyle = 'none', marker = '.', color = 'red', legend = None)
    ## State Forecast
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'a_tf_upper_c', color = 'black', linewidth = .4, legend = None)
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'a_tf', color = 'blue', legend = None)
    data.iloc[1:].plot(ax = axes[0], x = 'x', y = 'a_tf_lower_c', color = 'black', linewidth = .4, legend = None)
    axes[0].set_xbound(xlim_0)
    axes[0].set_ybound(ylim_0)
    
    # State Variance Pt
    data.plot(ax = axes[1], x = 'x', y = 'P_tf', legend = None)
    axes[1].set_xbound(xlim_1) 
    axes[1].set_ybound(ylim_1)
    
    # Observation Forecast
    data.iloc[1:].plot(ax = axes[2], x = 'x', y = 'a_tf', legend = None)
    axes[2].axhline(y = 0, color = 'b', linestyle = ':')
    axes[2].set_xbound(xlim_2) 
    axes[2].set_ybound(ylim_2)
    
    # Observation Forecast Variance Ft
    data.iloc[1:].plot(ax = axes[3], x='x', y='F_tf', ylim =ylim_3, legend = None) # Ignore initialization
    axes[3].set_xbound(xlim_3)
    axes[3].set_ybound(ylim_3)
    
    # Save Figure
    plt.savefig(f'{FIG_PATH}/Fig_2_6_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_7(data, df_name='', ylim_0=None, xlim_0=None, ylim_1=None, xlim_1=None, xlim_2=None, ylim_2=None,xlim_3=None, ylim_3=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Standardized Residual
    data.plot(ax = axes[0], x = 'x', y = 'e_t', legend = None)
    axes[0].axhline(y = 0, color = 'black', linestyle = ':')
    axes[0].set_xbound(xlim_0)
    axes[0].set_ybound(ylim_0)
    
    # Histogram plus estimated density
    sns.histplot(ax = axes[1], data=data.iloc[2:], x = 'e_t', stat='density', kde=True, bins=17)
    axes[1].set_xbound(xlim_1) 
    axes[1].set_ybound(ylim_1)
    
    # Ordered residuals
    sm.qqplot(data.loc[9:, 'e_t'], ax = axes[2], line ='45')
    axes[2].set_xbound(xlim_2) 
    axes[2].set_ybound(ylim_2)
    
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
    plt.savefig(f'{FIG_PATH}/Fig_2_7_{df_name}.png', facecolor = 'w')
    plt.show()

def fig_8(data, df_name='', ylim_0=None, xlim_0=None, ylim_1=None, xlim_1=None, xlim_2=None, ylim_2=None,xlim_3=None, ylim_3=None):
    fig, axes = plt.subplots(2,2,figsize=(10,10))
    axes = axes.ravel()

    # Observation Residual ut^*
    data.plot(ax = axes[0], x = 'x', y = 'u_star_t', legend = None)
    axes[0].axhline(y = 0, color = 'black', linestyle = ':')
    axes[0].set_xbound(xlim_0)
    axes[0].set_ybound(ylim_0)

    # Histogram and estimated density for ut^*
    sns.histplot(ax = axes[1], data=data.iloc[2:], x = 'u_star_t', stat='density', kde=True, bins=17, binwidth = 0.4)
    axes[1].set_xbound(xlim_1) 
    axes[1].set_ybound(ylim_1)
    
    # State Residual rt^*
    data.plot(ax = axes[2], x = 'x', y = 'r_star_t',  legend = None)
    axes[2].axhline(y = 0, color = 'black', linestyle = ':')
    axes[2].set_xbound(xlim_2) 
    axes[2].set_ybound(ylim_2)
    
    # Histogram and estimated density for rt^*
    sns.histplot(ax = axes[3], data=data.iloc[2:], x = 'r_star_t', stat='density', kde=True, bins=17, binwidth = 0.4)
    axes[3].set_xbound(xlim_3)
    axes[3].set_ybound(ylim_3)
    
    # Save Figure
    plt.savefig(f'{FIG_PATH}/Fig_2_8_{df_name}.png', facecolor = 'w')
    plt.show()
    