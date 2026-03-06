#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

y10 = pd.read_excel("DGS10 Monthly.xlsx")


# In[11]:


y10.columns = ["date", "y10"]


# In[12]:


y10.head()
y10.info()


# In[13]:


y10["date"] = pd.to_datetime(y10["date"])

# Move to end-of-month timestamp
y10["date"] = y10["date"] + pd.offsets.MonthEnd(0)

y10 = y10.sort_values("date")


# In[14]:


y10.head()


# In[15]:


y10["dy10"] = y10["y10"] - y10["y10"].shift(1)


# In[16]:


y10 = y10.dropna()


# In[17]:


y10 = y10[y10["date"] >= "2015-12-31"]


# In[18]:


y10.shape


# In[19]:


y10["dy10"].describe()


# In[20]:


plt.figure(figsize=(12,5))

plt.plot(y10["date"], y10["dy10"])
plt.axhline(0, linestyle="--")

plt.title("Monthly Change in 10-Year Treasury Yield")
plt.xlabel("Date")
plt.ylabel("Δ 10Y Yield (percentage points)")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[21]:


y10.head()


# In[22]:


y10_final = y10[["date", "dy10"]].copy()
y10_final = y10_final.sort_values("date")

y10_final.to_csv("dgs10_clean.csv", index=False)


# In[ ]:




