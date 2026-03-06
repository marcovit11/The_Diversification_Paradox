#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

vix = pd.read_excel("VIX Monthly.xlsx")


# In[24]:


vix.columns = ["date", "vix"]

vix["date"] = pd.to_datetime(vix["date"], dayfirst=True)

vix = vix.sort_values("date")


# In[25]:


vix.head()


# In[26]:


vix["log_vix"] = np.log(vix["vix"])

vix["dlog_vix"] = vix["log_vix"] - vix["log_vix"].shift(1)


# In[27]:


vix = vix.dropna()


# In[28]:


vix = vix[vix["date"] >= "2015-12-29"]


# In[29]:


vix.shape
vix.head()
vix["dlog_vix"].describe()


# In[30]:


plt.figure(figsize=(12,5))

plt.plot(vix["date"], vix["dlog_vix"])
plt.axhline(0, linestyle="--")

plt.title("Monthly Change in Log VIX")
plt.xlabel("Date")
plt.ylabel("Δ log(VIX)")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# In[31]:


vix = vix.copy()
vix["date"] = pd.to_datetime(vix["date"])
vix["date"] = vix["date"] + pd.offsets.MonthEnd(0)


# In[32]:


vix.head()


# In[33]:


vix_final = vix[["date", "dlog_vix"]].copy()
vix_final = vix_final.sort_values("date")

vix_final.to_csv("vix_clean.csv", index=False)


# In[ ]:




