import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew

FIG_PATH = '../Figures/Part_2'
TICK_SIZE = 20

def part_a_fig(df, data, y_lim=None, x_lim=(0,950)):
    """Part A plot of data in returns (original y_t)
    """
    
    plt.figure(figsize=(15,6))
    plt.plot(df['y_t'])
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)

    
    plt.savefig(f'{FIG_PATH}/part_a_{data}.png')
    
def part_a_descr_stats(y):
    ### Descriptive statistics
    
    print(f'Num Obs: {len(y)}')
    print(f'Mean: {np.mean(y)}')
    print(f'Std Dev: {np.std(y)}')
    print(f'Variance: {np.var(y)}')
    print(f'Minimum: {np.min(y)}')
    print(f'Maximum: {np.max(y)}')
    print(f'Kurtosis: {kurtosis(y, fisher=False)}')
    print(f'Skew: {skew(y)}')
      
def part_b_fig(df, data, y_lim=None, x_lim=(0,950)):
    """Part B scatter plot of transformed data (x_t)
    """
    
    plt.figure(figsize=(15,6))
    plt.scatter(df.index, df.x_t, color='black')
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    
    plt.savefig(f'{FIG_PATH}/part_b_{data}.png')
  
  
def part_d_fig(ssm, data, beta = ''):
    """Part D Figures of 
        i) transformed data (x_t) and smoothed state h_t &
        ii) Filtered state of H_t and smoothed state of H_t
    """

    fig, axes = plt.subplots(2,1, figsize=(15,12))
    axes[0].scatter(ssm.df.index, ssm.df['y_t'], color='black', alpha=.5)
    axes[0].plot(ssm.df.loc[2:, 'alpha_hat_t' + beta], 'red', linewidth=2, label = 'Smoothed ht')
    axes[0].legend(loc='lower right', prop={'size': 18})
    axes[0].tick_params(axis='both', labelsize=TICK_SIZE)

    # Ht = ht - xi WHERE xi = w/(1-phi) = -10.2089 from QML
    axes[1].plot(ssm.df.loc[4:, 'a_t'] - ssm.xi, 'blue', label = 'Filtered Ht')
    axes[1].plot(ssm.df.loc[2:, 'alpha_hat_t' + beta] - ssm.xi, 'red', label = 'Smoothed Ht')
    axes[1].legend(prop={'size': 18})
    axes[1].tick_params(axis='both', labelsize=TICK_SIZE)

    plt.savefig(f'{FIG_PATH}/part_d_{data}{beta}.png')