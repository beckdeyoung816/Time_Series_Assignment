
# %%
import pandas as pd
import datetime as dt
import scipy.stats as ss


from LLM import LLM
import Part1_Figures as p1f

# %%
nile_data = pd.read_excel('../Data/Nile.xlsx', names = ['x', 'y_t'])
nile_llm = LLM(data=nile_data, var_e=15099, var_h=1469.1, a_1=0, P_1=10**7)

# Set parameters
missing_vals_nile = [{'start': 21, 'stop': 40},
                     {'start': 61, 'stop': 80}] # Indices for missing vals
forecast_n_nile = 30 # Number of samples to forecast
# %%
nile_llm.kalman_filter()
nile_llm.state_smooth()
nile_llm.disturbance_smooth()
nile_llm.auxilary_residuals()
nile_llm.filter_with_missing_vals(missing_ranges=missing_vals_nile)
nile_llm.smooth_with_missing_vals()
nile_llm.forecast(j=forecast_n_nile)

# Confidence intervals
nile_llm.get_conf_intervals('a_t', 'P_t', pct=.90)
nile_llm.get_conf_intervals('alpha_hat_t', 'V_t', pct=.90)
# nile_llm.get_conf_intervals('a_t', 'P_t', pct=.50)
# nile_llm.get_conf_intervals('alpha_hat_t', 'V_t', pct=.50)
# %%
# Generate and Save Figures
p1f.fig_1(nile_llm.df, df_name = 'nile', ylim_0=(450,1400), xlim_0=(1865,1975),ylim_2=(-450,450), ylim_3=(20000,32500))
p1f.fig_2(nile_llm.df, df_name = 'nile', ylim_0=(450,1400), xlim_0=(1865,1975),ylim_1=(2200, 4100), ylim_3=(6e-5, .00011))
p1f.fig_3(nile_llm.df, df_name = 'nile', ylim_0=(-375,300), xlim_0=(1865,1975),ylim_2=(-43,40), xlim_2=(1865,1975))
p1f.fig_5(nile_llm.df, df_name = 'nile', ylim_0=(450,1400), xlim_0=(1865,1975), ylim_2=(450,1400), xlim_2=(1865,1975), ylim_3=(2200, 10000))
p1f.fig_6(nile_llm.forecast_df, df_name = 'nile', ylim_0=(450,1400), xlim_0=(1865,1975 + forecast_n_nile), ylim_2=(700,1200), ylim_3=(20000,60000))
p1f.fig_7(nile_llm.df, df_name = 'nile', ylim_0=(-2.8,2.8), ylim_3=(-1,1), xlim_3=(.5,11))
p1f.fig_8(nile_llm.df, df_name = 'nile', ylim_0=(-3,2.2), ylim_2=(-3,2.2))

# %%
cpi_data= pd.read_csv('../Data/CPI.csv')
cpi_data.columns = ['x', 'y_t']
cpi_data['x'] = pd.to_datetime(cpi_data['x'], format= '%Y-%m-%d')
cpi_data['y_t'] = cpi_data['y_t'].pct_change() * 100

cpi_llm = LLM(data=cpi_data.iloc[1:].reset_index(), q0 = [1]) # Initialize q = 1 for MLE

# Set parameters
missing_vals_cpi = [{'start': 21, 'stop': 40},
                    {'start': 161, 'stop': 180}] # Indices for missing vals
forecast_n_cpi = 30 # Number of samples to forecast
# %%
cpi_llm.kalman_filter()
cpi_llm.state_smooth()
cpi_llm.disturbance_smooth()
cpi_llm.auxilary_residuals()
cpi_llm.filter_with_missing_vals(missing_ranges=missing_vals_cpi)
cpi_llm.smooth_with_missing_vals()
cpi_llm.forecast(j=forecast_n_cpi)

# Confidence intervals
cpi_llm.get_conf_intervals('a_t', 'P_t', pct=.90)
cpi_llm.get_conf_intervals('alpha_hat_t', 'V_t', pct=.90)
# cpi_llm.get_conf_intervals('a_t', 'P_t', pct=.50)
# cpi_llm.get_conf_intervals('alpha_hat_t', 'V_t', pct=.50)
# %%
# Generate and Save Figures
p1f.fig_1(cpi_llm.df, df_name = 'cpi', ylim_0=(-3,5),ylim_2=(-5,3), xlim_0=(dt.date(1950,1,1), dt.date(2030,1,1)))
# %%
p1f.fig_2(cpi_llm.df, df_name = 'cpi', ylim_0=(-3,5))
p1f.fig_3(cpi_llm.df, df_name = 'cpi')
p1f.fig_5(cpi_llm.df, df_name = 'cpi')
p1f.fig_6(cpi_llm.forecast_df, df_name = 'cpi')
p1f.fig_7(cpi_llm.df, df_name = 'cpi', ylim_3=(-2,3))
p1f.fig_8(cpi_llm.df, df_name = 'cpi')