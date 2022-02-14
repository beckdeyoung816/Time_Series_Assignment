#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[34]:


data = pd.read_excel('../Data/Nile.xlsx')


# In[35]:


data.columns = ['year', 'Nile']
y = data['Nile']

x = data['year']


# In[ ]:





# In[39]:


def filter(true_value, updatenumber):
    global filter_a, filter_p, v, F, var_e, var_h
    if updatenumber == 0 :
        filter_a = 0 
        filter_p = 10**7
        var_e = 15099
        var_h = 1469.1
        v = true_value - filter_a
     
    
    #a prediction
    a_p = filter_a + (filter_p / (filter_p * var_e)) * v
    
    #p prediction
    p_p = (filter_p * var_e) / (filter_p + var_e) + var_h
    
    #F calculation
    F = filter_p + var_e
    
    #Kalman gain calculation 
    K = p_p / F
    
    #residual calculation 
    v = true_value - a_p
    
    #filtered prediction 
    
    filter_a = a_p + K * v
    
    #filtered variance
    filter_p = p_p - p_p * K
    
    return (filter_a, filter_p, v, F)
    
    


# In[41]:


f_a = []
f_p = []
f_v = []
f_F = []

num_of_measurements = len(y) 

for i in range(num_of_measurements):
    true_value = y[i]
        
        #call filter
    results_of_filter = filter(true_value, i)
        
        #put the results in lists
    f_a.append(results_of_filter[0])
    f_p.append(results_of_filter[1])
    f_v.append(results_of_filter[2])
    f_F.append(results_of_filter[3])
    filter_a = results_of_filter[0]
    filter_p = results_of_filter[1]
    v = results_of_filter[2]
    F = results_of_filter[3]
    var_e = 15099
    var_h = 1469.1


# In[48]:


plt.plot(x, f_p)


# In[50]:


plt.plot(x, f_v)


# In[51]:


plt.plot(x, f_F)


# In[52]:


_ = plt.plot(x, y, linestyle = 'none', marker = '.', color = 'red')
_ = plt.plot(x,f_a, color = 'blue')

