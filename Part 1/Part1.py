#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[34]:

data = pd.read_excel('../Data/Nile.xlsx', names = ['year', 'Nile'])
x,y =  data['year'], data['Nile']
    
    
# %%

def filter(true_value, filter_a, filter_p, v, var_e, var_h):
    #a prediction
    a_p = filter_a + (filter_p / (filter_p * var_e)) * v
    
    #p prediction
    p_p = (filter_p * var_e) / (filter_p + var_e) + var_h
    
    #F calculation
    Ft = filter_p + var_e
    
    #Kalman gain calculation 
    k = p_p / Ft
    
    #residual calculation 
    v = true_value - a_p
    
    #filtered prediction 
    
    filter_a = a_p + k * v
    
    #filtered variance
    filter_p = p_p - p_p * k
    
    return filter_a, filter_p, v, Ft

# In[41]:


f_a, f_p, f_v, f_Ft = [],[],[],[]

for i, true_value in enumerate(y):
    # Initialize Values
    if i == 0 :
        filter_a = 0 
        filter_p = 10**7
        v = true_value - filter_a
        
    filter_a, filter_p, v, Ft = filter(true_value, filter_a=filter_a,filter_p=filter_p, 
                                       var_e=15099, var_h=1469.1, v=v)
    
    # Store output
    f_a.append(filter_a)
    f_p.append(filter_p)
    f_v.append(v)
    f_Ft.append(Ft)


# In[48]:

plt.subplot(2,2,1)
plt.plot(x, y, linestyle = 'none', marker = '.', color = 'red')
plt.plot(x[4:],f_a[4:], color = 'blue')
plt.xlim(1865,1975)
plt.ylim(550,1400)

plt.subplot(2,2, 2)
plt.plot(x, f_p)

plt.subplot(2,2,3)
plt.plot(x, f_v)
plt.ylim(-400,400)

plt.subplot(2,2,4)
plt.plot(x, f_Ft)

plt.show()

# %%
