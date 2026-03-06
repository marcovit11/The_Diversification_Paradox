#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from pytrends.request import TrendReq


# In[22]:


pytrends = TrendReq(
    hl='en-US',
    tz=360,              
)


# In[23]:


kw_list = ["AI Stocks"]

timeframe = "2015-04-01 2025-12-31"
geo = "US"


# In[24]:


pytrends.build_payload(
    kw_list=kw_list,
    cat=0,
    timeframe=timeframe,
    geo=geo,
    gprop=""
)


# In[25]:


monthly_svi = pytrends.interest_over_time()

monthly_svi = monthly_svi.drop(columns=["isPartial"])

monthly_svi.head()


# In[26]:


monthly_svi = monthly_svi.reset_index()
monthly_svi.rename(columns={"date": "Month"}, inplace=True)

monthly_svi.head()


# In[27]:


monthly_svi["Month"].diff().value_counts()


# In[28]:


monthly_svi.tail()


# In[29]:


monthly_svi["Month"] = pd.to_datetime(monthly_svi["Month"])

monthly_svi.head()


# In[30]:


import matplotlib.pyplot as plt

# Make sure data is sorted
monthly_svi = monthly_svi.sort_values("Month")

plt.figure()
plt.plot(monthly_svi["Month"], monthly_svi["AI Stocks"])
plt.title("Monthly Google Trends SVI: 'AI Stocks' (US)")
plt.xlabel("Month")
plt.ylabel("Search Volume Index")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[31]:


import numpy as np

# Ensure sorted
monthly_svi = monthly_svi.sort_values("Month").reset_index(drop=True)

# Avoid log(0) — replace zeros with small positive number 
monthly_svi["SVI_adj"] = monthly_svi["AI Stocks"].replace(0, 0.5)

# Compute rolling 8-month median of past values (t-1 to t-4)
monthly_svi["Median_8"] = (
    monthly_svi["SVI_adj"]
    .rolling(window=8)
    .median()
    .shift(1)
)

# Compute ASVI
monthly_svi["ASVI8"] = np.log(monthly_svi["SVI_adj"]) - np.log(monthly_svi["Median_8"])

monthly_svi.head(12)


# In[32]:


monthly_svi = monthly_svi.dropna(subset=["ASVI8"]).reset_index(drop=True)


# In[33]:


monthly_svi.head()


# In[34]:


plt.figure()
plt.plot(monthly_svi["Month"], monthly_svi["ASVI8"])
plt.title("Monthly ASVI8 (AI Stocks)")
plt.xlabel("Month")
plt.ylabel("ASVI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[35]:


monthly_svi.head(12)


# In[36]:


monthly_export = monthly_svi[["Month", "ASVI8"]].copy()

monthly_export = monthly_export.rename(columns={
    "Month": "date"
})


# In[37]:


monthly_export["date"] = (
    monthly_export["date"]
        .dt.to_period("M")
        .dt.to_timestamp("M")
)


# In[38]:


monthly_export.head()


# In[39]:


monthly_export.tail()


# In[40]:


monthly_export.to_csv("AIStocks_ASVI8_Monthly.csv", index=False)

