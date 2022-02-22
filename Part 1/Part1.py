#!/usr/bin/env python
# coding: utf-8

# In[33]:


from email import header
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm
import pylab as py

# In[34]:

data = pd.read_excel('../Data/Nile.xlsx', names = ['year', 'Nile'])
x,y =  data['year'], data['Nile']
    
    
# %%
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

# In[41]:


f_a, f_p, f_v, f_Ft, f_k = [],[],[],[],[]

for i, true_value in enumerate(y):
    # Initialize Values
    if i == 0 :
        a = 0 
        p = 10**7
        
        
    filter_a, filter_p, a, p, v, Ft ,K = filter(true_value, a=a, p=p, 
                                       var_e=15099, var_h=1469.1)
    
    # Store output
    f_a.append(filter_a)
    f_p.append(filter_p)
    f_v.append(v)
    f_Ft.append(Ft)
    f_k.append(K)


# In[48]:

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

errors = [f_v[i] / math.sqrt(f_Ft[i]) for i in range(len(f_Ft))]  
plt.plot(x, errors)
plt.axhline(y = 0, color = 'b', linestyle = ':')
plt.ylim(-2.8,2.8)
plt.show()


sns.histplot(data=errors, kde=True, bins=17, binwidth = 0.4)
plt.show()


x = pd.plotting.autocorrelation_plot(errors)
x.plot()

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
