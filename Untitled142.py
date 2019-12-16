#!/usr/bin/env python
# coding: utf-8

# ## logistic regression
# 

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


raw_data = pd.read_csv('2.01. Admittance.csv')


# In[40]:


data = raw_data.copy()


# In[41]:


data.head()


# In[42]:


data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0})


# In[43]:


data['Admitted']


# In[44]:


y = data['Admitted']
x1 = data['SAT']


# In[45]:


x1


# In[46]:


x = sm.add_constant(x1)


# In[47]:


reg_log = sm.Logit(y,x)


# In[48]:


results_log = reg_log.fit()


# In[50]:


results_log.summary()


# In[ ]:


raw

