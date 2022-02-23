
# %%

from email import header
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm
import pylab as py

# %%

data = pd.read_excel('../Data/Nile.xlsx', names = ['year', 'Nile'])
x,y =  data['year'], data['Nile']
    
VAR_E = 15099
VAR_H = 1469.1
# %%
# FIGURE 2.1

def filter(true_value, a, p, var_e, var_h):
    
    #Kalman gain calculation
    v = true_value - a
    Ft = p + var_e
    K = p / Ft
    
    #state update
    filter_a = a + K * v
    filter_p = p * (1-K)
    
    #predictions
    a = a + K * v
    p = (p - p * K) + var_h 
    
    return filter_a, filter_p, a, p, v, Ft, K

# %%


f_a, f_p, f_v, f_Ft, f_k = [],[],[],[],[]

for i, true_value in enumerate(y):
    # Initialize Values
    if i == 0 :
        a = 0 
        p = 10**7
        
        
    filter_a, filter_p, a, p, v, Ft ,K = filter(true_value, a=a, p=p, 
                                       var_e=VAR_E, var_h=VAR_H)
    
    # Store output
    f_a.append(filter_a)
    f_p.append(filter_p)
    f_v.append(v)
    f_Ft.append(Ft)
    f_k.append(K)


# %%

plt.subplot(2,2,1)
plt.plot(x, y, linestyle = 'none', marker = '.', color = 'red')
plt.plot(x,f_a, color = 'blue')
plt.ylim(450,1400)
plt.xlim(1865,1975)


plt.subplot(2,2, 2)
plt.plot(x, f_p)

plt.subplot(2,2,3)
plt.plot(x, f_v)
plt.axhline(y = 0, color = 'b', linestyle = ':')
plt.ylim(-450,450)

plt.subplot(2,2,4)
plt.plot(x[1:], f_Ft[1:])
plt.ylim(20000,32500)

plt.show()

# %%
# FIGURE 2.2

# %%
# FIGURE 2.3
Lt = [1 - f_k[i] for i in range(len(f_k))]

Nt, rt = [], []
for i in range(len(f_Ft)):
    if i == 0:
        n=0
        r=0
    else:
        n = 1 / f_Ft[100-i] + Lt[100-i] ** 2 * n
        r = f_v[100-i] / f_Ft[100-i] + Lt[100-i] * r
    Nt.append(n)
    rt.append(r)
    
Nt.reverse()
rt.reverse()
#%%

Dt = [1 / f_Ft[i] + f_k[i] ** 2 * Nt[i] for i in range(len(Nt))]

#Dt2 = 1 / f_Ft + f_k ** 2 * Nt

ut = [f_v[i] / f_Ft[i] + f_k[i] * rt[i] for i in range(len(rt))]

e_hat_t = [VAR_E * u for u in ut]
n_hat_t = [VAR_H * r for r in rt]
var_e_hat = [VAR_E - (VAR_E ** 2 * D) for D in Dt]
var_n_hat = [VAR_H - (VAR_H ** 2 * N) for N in Nt]

utstar = [ut[i] / math.sqrt(Dt[i]) for i in range(len(Dt))]

rtstar = [rt[i] / math.sqrt(Nt[i]) for i in range(98)]

# %%
plt.subplot(2,2,1)
plt.plot(x, e_hat_t)
plt.axhline(y = 0, color = 'black')
plt.ylim(-375,300)
plt.xlim(1865,1975)


plt.subplot(2,2, 2)
plt.plot(x, [math.sqrt(var) for var in var_e_hat])

plt.subplot(2,2,3)
plt.plot(x, n_hat_t)
plt.axhline(y = 0, color = 'black')
plt.ylim(-43,40)
plt.xlim(1865,1975)

plt.subplot(2,2,4)
plt.plot(x, [math.sqrt(var) for var in var_n_hat])

plt.show()

# %%
# FIGURE 2.5

# %%
# FIGURE 2.6

# %%
# FIGURE 2.7

# %%
# FIGURE 2.8

# %%

errors = [f_v[i] / math.sqrt(f_Ft[i]) for i in range(len(f_Ft))]  
plt.plot(x, errors)
plt.axhline(y = 0, color = 'b', linestyle = ':')
plt.ylim(-2.8,2.8)
plt.show()


sns.histplot(data=errors, kde=True, bins=17, binwidth = 0.4)
plt.show()


error_plot = pd.plotting.autocorrelation_plot(errors)
error_plot.plot()

# %%

e = np.array(errors)
sm.qqplot(e[9:], line ='45')
py.show()


# %%

sns.histplot(data=errors[2:], stat='density', kde=True, bins=17)
plt.show()

# %%


Lt = [15099 / f_Ft[i] for i in range(len(f_Ft))]

N = []
for i in range(len(f_Ft)):
    if i == 0:
        n=0
    else:
        n = 1 / f_Ft[100-i] + Lt[100-i] ** 2 * n
    N.append(n)
    

#%%

rt = []
for i in range(len(f_Ft)):
    if i == 0:
        r=0
    else:
        r = f_v[100-i] / f_Ft[100-i] + Lt[100-i] * r
    rt.append(r)
    
#%%

N.reverse()
rt.reverse()

#%%

Dt = [1 / f_Ft[i] + f_k[i] ** 2 * N[i] for i in range(len(N))]

ut = [f_v[i] / f_Ft[i] + f_k[i] * rt[i] for i in range(len(rt))]

utstar = [ut[i] / math.sqrt(Dt[i]) for i in range(len(Dt))]

rtstar = [rt[i] / math.sqrt(N[i]) for i in range(98)]

#%%

plt.plot(x, utstar)
plt.axhline(y = 0, color = 'b', linestyle = ':')
plt.ylim(-3,2.2)
plt.show()

#%%

sns.histplot(data=utstar, stat='density', kde=True, bins=17, binwidth = 0.4)
plt.show()

#%%

plt.plot(x [2:], rtstar)
plt.axhline(y = 0, color = 'b', linestyle = ':')
plt.ylim(-3,2.2)
plt.show()

#%%

sns.histplot(data=rtstar, stat='density', kde=True, bins=17, binwidth = 0.4)
plt.show()


# %%
