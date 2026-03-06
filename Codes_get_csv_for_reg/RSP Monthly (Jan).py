#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[28]:


rsp = pd.read_excel("RSP Monthly.xlsx")


# In[29]:


rsp.head()
rsp.columns
rsp.info()


# In[30]:


rsp = rsp.rename(columns={
    "Date": "date",
    "Total Net Assets as of Month End": "tna",
    "Total Return per Share as of Month End": "ret",
    "Monthly Net Asset Value Per Share": "nav"
})


# In[31]:


rsp["ret"].describe()


# In[32]:


rsp.columns


# In[33]:


rsp.columns = [
    "date",
    "fund_id",
    "tna",
    "ret",
    "nav"
]


# In[34]:


rsp.info()


# In[35]:


rsp["tna_lag"] = rsp["tna"].shift(1)

rsp["net_flow"] = rsp["tna"] - (rsp["tna_lag"] * (1 + rsp["ret"]))

rsp["flow_rsp"] = rsp["net_flow"] / rsp["tna_lag"]

rsp = rsp.dropna()


# In[36]:


rsp[["tna", "ret", "net_flow", "flow_rsp"]].head()
rsp["flow_rsp"].describe()


# In[37]:


rsp_flow = rsp[["date", "flow_rsp"]]


# In[38]:


plt.figure(figsize=(12,5))

plt.plot(rsp["date"], rsp["flow_rsp"])
plt.axhline(0, linestyle="--")

plt.title("RSP Monthly Scaled Net Fund Flows")
plt.xlabel("Date")
plt.ylabel("Scaled Net Flow")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# In[39]:


rsp[["date", "flow_rsp"]].head()


# In[40]:


rsp = rsp.copy()
rsp["date"] = pd.to_datetime(rsp["date"])
rsp["date"] = rsp["date"] + pd.offsets.MonthEnd(0)


# In[41]:


rsp.head()


# In[42]:


rsp_final = rsp[["date", "flow_rsp"]].copy()
rsp_final = rsp_final.sort_values("date")

rsp_final.to_csv("rsp_clean.csv", index=False)


# In[ ]:




