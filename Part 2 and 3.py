#!/usr/bin/env python
# coding: utf-8

# In[3]:
'''
Teammates: Anthony Bedi and Harsha Mangnani
'''
#importing packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from textblob import TextBlob
from datetime import datetime
import time


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('tweets1.csv', index_col = 0)


# In[8]:


df.info()


# In[9]:


#converting account creation date to datetime

df['usersincedate'] = df['usersincedate'].astype('datetime64[ns]') 


# In[15]:


#converting datetime to ordinal
#https://stackoverflow.com/questions/53893323/how-to-convert-pandas-data-frame-datetime-column-to-int

df['usersincedate'] = df['usersincedate'].apply(lambda x:x.toordinal())


# In[16]:


df


# In[26]:


print("Polarity vs Account Creation Date Correlation:")
np.corrcoef(df.usersincedate, df.Polarity)


# In[18]:


print("Subjectivity vs Account Creation Date Correlation:")
np.corrcoef(df.usersincedate, df.Subjectivity)


# In[19]:


print("Polarity vs Word Count Correlation:")
np.corrcoef(df.count_word, df.Polarity)


# In[20]:


print("Subjectivity vs Word Count Correlation:")
np.corrcoef(df.count_word, df.Subjectivity)


# In[23]:


print("Followers vs Word Count Correlation:")
np.corrcoef(df.followers, df.count_word)


# In[28]:


plt.scatter(df.count_word, df.Subjectivity)


# In[30]:


#Visualizing the strongest relationship, Word Count vs Subjectivity

y = df.Subjectivity #outcome variable
x = df.count_word #predictor
x = sm.add_constant(x)
lr_model = sm.OLS(y,x).fit()
print(lr_model.summary())


# In[31]:


plt.scatter(df.count_word, df.Subjectivity)
plt.plot(df.count_word, 0.0123 * df.Subjectivity + 0.0886)
plt.xlabel('count_word')
plt.ylabel('Subjectivity')
np.corrcoef(df.count_word, df.Subjectivity)[0,1]


# In[37]:


df2 = pd.read_csv('cases.csv', index_col = 0)


# In[38]:


df2


# In[42]:


print("Case Count vs Polarity Correlation:")
np.corrcoef(df2.Cases, df2.Polarity)


# In[44]:


plt.scatter(df2.Cases, df2.Polarity)


# In[46]:


y = df2.Polarity #outcome variable
x = df2.Cases #predictor
x = sm.add_constant(x)
lr_model = sm.OLS(y,x).fit()
print(lr_model.summary())


# In[49]:


plt.scatter(df2.Cases, df2.Polarity)
plt.plot(df2.Cases, -1.994e-06 * df2.Cases + 0.1092)
plt.xlabel('Cases')
plt.ylabel('Polarity')
np.corrcoef(df2.Cases, df2.Polarity)[0,1]


# In[ ]:




