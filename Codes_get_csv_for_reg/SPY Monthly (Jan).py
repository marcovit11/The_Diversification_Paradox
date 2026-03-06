#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


spy = pd.read_excel("SPY Monthly.xlsx")


# In[20]:


spy.head()
spy.columns
spy.info()


# In[21]:


spy = spy.rename(columns={
    "Date": "date",
    "Total Net Assets as of Month End": "tna",
    "Total Return per Share as of Month End": "ret",
    "Monthly Net Asset Value Per Share": "nav"
})


# In[22]:


spy["ret"].describe()


# In[23]:


spy.columns


# In[24]:


spy.columns = [
    "date",
    "fund_id",
    "tna",
    "ret",
    "nav"
]


# In[25]:


spy.info()


# In[26]:


spy["tna_lag"] = spy["tna"].shift(1)

spy["net_flow"] = spy["tna"] - (spy["tna_lag"] * (1 + spy["ret"]))

spy["flow_spy"] = spy["net_flow"] / spy["tna_lag"]

spy = spy.dropna()


# In[27]:


spy[["tna", "ret", "net_flow", "flow_spy"]].head()
spy["flow_spy"].describe()


# In[28]:


spy_flow = spy[["date", "flow_spy"]]


# In[29]:


plt.figure(figsize=(12,5))

plt.plot(spy["date"], spy["flow_spy"])
plt.axhline(0, linestyle="--")

plt.title("SPY Monthly Scaled Net Fund Flows")
plt.xlabel("Date")
plt.ylabel("Scaled Net Flow")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# In[30]:


spy[["date", "flow_spy"]].head()


# In[31]:


spy = spy.copy()

spy["date"] = pd.to_datetime(spy["date"])
spy["date"] = spy["date"] + pd.offsets.MonthEnd(0)


# In[32]:


spy[["date", "flow_spy"]].head()


# In[33]:


spy_final = spy[["date", "flow_spy"]].copy()
spy_final = spy_final.sort_values("date")

spy_final.to_csv("spy_clean.csv", index=False)


# In[34]:


spy_final.head()
spy_final.tail()


# In[ ]:




