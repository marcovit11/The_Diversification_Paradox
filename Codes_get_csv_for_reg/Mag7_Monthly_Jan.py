#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


mag7 = pd.read_csv("Mag7 Monthly Returns.csv")


# In[12]:


mag7.head()
mag7.columns
mag7.info()


# In[13]:


mag7.columns
mag7.head()


# In[14]:


mag7 = mag7[[
    "(tic) Ticker Symbol",
    "(datadate) Data Date - Security Monthly",
    "(trt1m) Monthly Total Return"
]]


# In[15]:


mag7.columns = ["ticker", "date", "trt1m"]


# In[16]:


mag7["date"] = pd.to_datetime(
    mag7["date"],
    format="%d.%m.%Y"
)


# In[17]:


mag7.info()


# In[18]:


mag7["ret_dec"] = mag7["trt1m"] / 100


# In[19]:


mag7["ret_dec"].describe()


# In[20]:


import numpy as np

mag7["log_ret"] = np.log(1 + mag7["ret_dec"])


# In[21]:


mag7["log_ret"].describe()


# In[22]:


mag7_avg = (
    mag7
    .groupby("date")["log_ret"]
    .mean()
    .rename("mag7_ret")
    .reset_index()
)


# In[23]:


mag7_avg.head()


# In[24]:


import matplotlib.pyplot as plt

# make sure date is index
mag7_avg = mag7_avg.set_index("date")

plt.figure(figsize=(12,5))
plt.plot(mag7_avg.index, mag7_avg["mag7_ret"])
plt.axhline(0, linestyle="--")
plt.title("Mag7 Average Monthly Log Returns")
plt.ylabel("Log Return")
plt.xlabel("Date")
plt.tight_layout()
plt.show()


# In[25]:


mag7_avg.describe()


# In[26]:


mag7_avg.head()


# In[27]:


mag7_avg = mag7_avg.reset_index()
mag7_avg.to_csv("mag7_clean.csv", index=False)


# In[ ]:




