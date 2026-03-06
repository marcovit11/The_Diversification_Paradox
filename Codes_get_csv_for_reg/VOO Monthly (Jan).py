#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


voo = pd.read_excel("VOO Monthly.xlsx")


# In[3]:


voo.head()
voo.columns
voo.info()


# In[4]:


voo = voo.rename(columns={
    "Date": "date",
    "Total Net Assets as of Month End": "tna",
    "Total Return per Share as of Month End": "ret"
})


# In[5]:


voo["ret"].describe()


# In[6]:


voo.info()


# In[7]:


voo["tna_lag"] = voo["tna"].shift(1)

voo["net_flow"] = voo["tna"] - (voo["tna_lag"] * (1 + voo["ret"]))

voo["flow_voo"] = voo["net_flow"] / voo["tna_lag"]

voo = voo.dropna()


# In[8]:


voo["ret"].describe()


# In[9]:


voo[["tna", "ret", "net_flow", "flow_voo"]].head()
voo["flow_voo"].describe()


# In[10]:


voo_flow = voo[["date", "flow_voo"]]


# In[11]:


plt.figure(figsize=(12,5))

plt.plot(voo["date"], voo["flow_voo"])
plt.axhline(0, linestyle="--")

plt.title("VOO Monthly Scaled Net Fund Flows")
plt.xlabel("Date")
plt.ylabel("Scaled Net Flow")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# In[12]:


voo[["date", "flow_voo"]].head()


# In[13]:


voo = voo.copy()

voo["date"] = pd.to_datetime(voo["date"])
voo["date"] = voo["date"] + pd.offsets.MonthEnd(0)


# In[14]:


voo[["date", "flow_voo"]].head()


# In[15]:


voo_final = voo[["date", "flow_voo"]].copy()
voo_final = voo_final.sort_values("date")

voo_final.to_csv("voo_clean.csv", index=False)


# In[ ]:




