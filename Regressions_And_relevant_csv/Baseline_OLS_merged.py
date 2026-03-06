#!/usr/bin/env python
# coding: utf-8

# In[159]:


import pandas as pd
import numpy as np
import statsmodels.api as sm


# In[160]:


spy = pd.read_csv("spy_clean.csv", parse_dates=["date"])
voo = pd.read_csv("voo_clean.csv", parse_dates=["date"])
rsp = pd.read_csv("rsp_clean.csv", parse_dates=["date"])
vix = pd.read_csv("vix_clean.csv", parse_dates=["date"])
y10 = pd.read_csv("dgs10_clean.csv", parse_dates=["date"])
asvi4 = pd.read_csv("AIStocks_ASVI4_Monthly.csv", parse_dates=["date"])
asvi8 = pd.read_csv("AIStocks_ASVI8_Monthly.csv", parse_dates=["date"])
mag7 = pd.read_csv("mag7_clean.csv", parse_dates=["date"])


# In[161]:


spy.shape, voo.shape, rsp.shape, vix.shape, y10.shape, asvi4.shape, asvi8.shape, mag7.shape


# In[162]:


df = spy.merge(rsp, on="date", how="inner", suffixes=("_spy", "_rsp"))
df = df.merge(voo, on="date", how="inner")
df = df.merge(vix,  on="date", how="inner")
df = df.merge(y10,  on="date", how="inner")
df = df.merge(asvi4, on="date", how="inner")
df = df.merge(asvi8, on="date", how="inner")
df = df.merge(mag7, on="date", how="inner")


# In[163]:


df.head(10)


# In[164]:


df.shape


# In[165]:


df["ASVI4_lag"] = df["ASVI4"].shift(1)
df["ASVI8_lag"] = df["ASVI8"].shift(1)
df["flow_spy_lag"] = df["flow_spy"].shift(1)
df["flow_voo_lag"] = df["flow_voo"].shift(1)
df["flow_rsp_lag"] = df["flow_rsp"].shift(1)
df["mag7_ret_lag"] = df["mag7_ret"].shift(1)


# In[166]:


df = df.dropna()


# In[167]:


df.shape


# In[168]:


#SPY as dependent variable
#2016-2025

Y = df["flow_spy"]

X = df[["ASVI4_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model1 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model1.summary())


# In[169]:


df_2020 = df[df["date"] >= "2020-01-31"].copy()
df_2021 = df[df["date"] >= "2021-01-31"].copy()
df_2022 = df[df["date"] >= "2022-01-31"].copy()
df_2023 = df[df["date"] >= "2023-01-31"].copy()
df_2024 = df[df["date"] >= "2024-01-31"].copy()


# In[170]:


#2020 Flow OLS

Y = df_2020["flow_spy"]
X = sm.add_constant(df_2020[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[171]:


#2021 Flow OLS

Y = df_2021["flow_spy"]
X = sm.add_constant(df_2021[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[172]:


#2022 Flow OLS

Y = df_2022["flow_spy"]
X = sm.add_constant(df_2022[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[173]:


#2023 Flow OLS

Y = df_2023["flow_spy"]
X = sm.add_constant(df_2023[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[174]:


#RSP as dependent variable
#2016-2025

Y = df["flow_rsp"]

X = df[["ASVI4_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model1 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model1.summary())


# In[175]:


#2020 RSP Flow OLS

Y = df_2020["flow_rsp"]
X = sm.add_constant(df_2020[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[176]:


#2021 RSP Flow OLS

Y = df_2021["flow_rsp"]
X = sm.add_constant(df_2021[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[177]:


#2022 RSP Flow OLS

Y = df_2022["flow_rsp"]
X = sm.add_constant(df_2022[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[178]:


#2023 RSP Flow OLS

Y = df_2023["flow_rsp"]
X = sm.add_constant(df_2023[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[179]:


#Mag7 Returns SPY OLS 2016-2025

Y = df["mag7_ret"]

X = df[["flow_spy_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model2 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model2.summary())


# In[180]:


#2020 Mag7 Ret SPY OLS

Y = df_2020["mag7_ret"]
X = sm.add_constant(df_2020[["flow_spy_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[181]:


#2021 Mag7 Ret SPY OLS

Y = df_2021["mag7_ret"]
X = sm.add_constant(df_2021[["flow_spy_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[182]:


#2022 Mag7 Ret SPY OLS

Y = df_2022["mag7_ret"]
X = sm.add_constant(df_2022[["flow_spy_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[183]:


#2023 Mag7 Ret SPY OLS

Y = df_2023["mag7_ret"]
X = sm.add_constant(df_2023[["flow_spy_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[184]:


#Mag7 Returns RSP OLS 2016-2025

Y = df["mag7_ret"]

X = df[["flow_rsp_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model2 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model2.summary())


# In[185]:


#2020 Mag7 Ret RSP OLS

Y = df_2020["mag7_ret"]
X = sm.add_constant(df_2020[["flow_rsp_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[186]:


#2021 Mag7 Ret RSP OLS

Y = df_2021["mag7_ret"]
X = sm.add_constant(df_2021[["flow_rsp_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[187]:


#2022 Mag7 Ret RSP OLS

Y = df_2022["mag7_ret"]
X = sm.add_constant(df_2022[["flow_rsp_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[188]:


#2023 Mag7 Ret RSP OLS

Y = df_2023["mag7_ret"]
X = sm.add_constant(df_2023[["flow_rsp_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[189]:


#ASVI lagged Mag7_ret OLS 2016-2025

Y = df["ASVI4"]

X = df[["mag7_ret_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model2 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model2.summary())


# In[190]:


#2020 ASVI lagged Mag7_ret OLS

Y = df_2020["ASVI4"]
X = sm.add_constant(df_2020[["mag7_ret_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[191]:


#2021 ASVI lagged Mag7_ret OLS

Y = df_2021["ASVI4"]
X = sm.add_constant(df_2021[["mag7_ret_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[192]:


#2022 ASVI lagged Mag7_ret OLS

Y = df_2022["ASVI4"]
X = sm.add_constant(df_2022[["mag7_ret_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[193]:


#2023 ASVI lagged Mag7_ret OLS

Y = df_2023["ASVI4"]
X = sm.add_constant(df_2023[["mag7_ret_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[194]:


#VOO as dependent variable
#2016-2025

Y = df["flow_voo"]

X = df[["ASVI4_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model1 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model1.summary())


# In[195]:


#2020 VOO Flow OLS

Y = df_2020["flow_voo"]
X = sm.add_constant(df_2020[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[196]:


#2021 VOO Flow OLS

Y = df_2021["flow_voo"]
X = sm.add_constant(df_2021[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[197]:


#2022 VOO Flow OLS

Y = df_2022["flow_voo"]
X = sm.add_constant(df_2022[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[198]:


#2023 VOO Flow OLS

Y = df_2023["flow_voo"]
X = sm.add_constant(df_2023[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[199]:


#Mag7 Returns VOO OLS 2016-2025

Y = df["mag7_ret"]

X = df[["flow_voo_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model2 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model2.summary())


# In[200]:


#2020 Mag7 Ret VOO OLS

Y = df_2020["mag7_ret"]
X = sm.add_constant(df_2020[["flow_voo_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[201]:


#2021 Mag7 Ret VOO OLS

Y = df_2021["mag7_ret"]
X = sm.add_constant(df_2021[["flow_voo_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[202]:


#2022 Mag7 Ret VOO OLS

Y = df_2022["mag7_ret"]
X = sm.add_constant(df_2022[["flow_voo_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[203]:


#2023 Mag7 Ret VOO OLS

Y = df_2023["mag7_ret"]
X = sm.add_constant(df_2023[["flow_voo_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[204]:


#Mag7 Ret affected by ASVI?

Y = df["mag7_ret"]
X = sm.add_constant(df[["ASVI4_lag", "dlog_vix", "dy10"]])

model = sm.OLS(Y, X).fit(cov_type="HC3")
print(model.summary())


# In[205]:


#2020 Mag7 Ret ASVI OLS

Y = df_2020["mag7_ret"]
X = sm.add_constant(df_2020[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[206]:


#2021 Mag7 Ret ASVI OLS

Y = df_2021["mag7_ret"]
X = sm.add_constant(df_2021[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[207]:


#2022 Mag7 Ret ASVI OLS

Y = df_2022["mag7_ret"]
X = sm.add_constant(df_2022[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[208]:


#2023 Mag7 Ret ASVI OLS

Y = df_2023["mag7_ret"]
X = sm.add_constant(df_2023[["ASVI4_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[209]:


#SPY as dependent variable
#2016-2025

Y = df["flow_spy"]

X = df[["ASVI8_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model1 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model1.summary())


# In[210]:


#2020 Flow OLS

Y = df_2020["flow_spy"]
X = sm.add_constant(df_2020[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[211]:


#2021 Flow OLS

Y = df_2021["flow_spy"]
X = sm.add_constant(df_2021[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[212]:


#2022 Flow OLS

Y = df_2022["flow_spy"]
X = sm.add_constant(df_2022[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[213]:


#2023 Flow OLS

Y = df_2023["flow_spy"]
X = sm.add_constant(df_2023[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[214]:


#RSP as dependent variable
#2016-2025

Y = df["flow_rsp"]

X = df[["ASVI8_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model1 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model1.summary())


# In[215]:


#2020 RSP Flow OLS

Y = df_2020["flow_rsp"]
X = sm.add_constant(df_2020[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[216]:


#2021 RSP Flow OLS

Y = df_2021["flow_rsp"]
X = sm.add_constant(df_2021[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[217]:


#2022 RSP Flow OLS

Y = df_2022["flow_rsp"]
X = sm.add_constant(df_2022[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[218]:


#2023 RSP Flow OLS

Y = df_2023["flow_rsp"]
X = sm.add_constant(df_2023[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[219]:


#ASVI lagged Mag7_ret OLS 2016-2025

Y = df["ASVI8"]

X = df[["mag7_ret_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model2 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model2.summary())


# In[220]:


#2020 ASVI lagged Mag7_ret OLS

Y = df_2020["ASVI8"]
X = sm.add_constant(df_2020[["mag7_ret_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[221]:


#2021 ASVI lagged Mag7_ret OLS

Y = df_2021["ASVI8"]
X = sm.add_constant(df_2021[["mag7_ret_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[222]:


#2022 ASVI lagged Mag7_ret OLS

Y = df_2022["ASVI8"]
X = sm.add_constant(df_2022[["mag7_ret_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[223]:


#2023 ASVI lagged Mag7_ret OLS

Y = df_2023["ASVI8"]
X = sm.add_constant(df_2023[["mag7_ret_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[224]:


#VOO as dependent variable
#2016-2025

Y = df["flow_voo"]

X = df[["ASVI8_lag", "dlog_vix", "dy10"]]
X = sm.add_constant(X)

model1 = sm.OLS(Y, X).fit(cov_type="HC3")
print(model1.summary())


# In[225]:


#2020 VOO Flow OLS

Y = df_2020["flow_voo"]
X = sm.add_constant(df_2020[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[226]:


#2021 VOO Flow OLS

Y = df_2021["flow_voo"]
X = sm.add_constant(df_2021[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[227]:


#2022 VOO Flow OLS

Y = df_2022["flow_voo"]
X = sm.add_constant(df_2022[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[228]:


#2023 VOO Flow OLS

Y = df_2023["flow_voo"]
X = sm.add_constant(df_2023[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())


# In[229]:


#Mag7 Ret affected by ASVI?

Y = df["mag7_ret"]
X = sm.add_constant(df[["ASVI8_lag", "dlog_vix", "dy10"]])

model = sm.OLS(Y, X).fit(cov_type="HC3")
print(model.summary())


# In[230]:


#2020 Mag7 Ret ASVI OLS

Y = df_2020["mag7_ret"]
X = sm.add_constant(df_2020[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2020 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2020.summary())


# In[231]:


#2021 Mag7 Ret ASVI OLS

Y = df_2021["mag7_ret"]
X = sm.add_constant(df_2021[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2021 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2021.summary())


# In[232]:


#2022 Mag7 Ret ASVI OLS

Y = df_2022["mag7_ret"]
X = sm.add_constant(df_2022[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2022 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2022.summary())


# In[233]:


#2023 Mag7 Ret ASVI OLS

Y = df_2023["mag7_ret"]
X = sm.add_constant(df_2023[["ASVI8_lag", "dlog_vix", "dy10"]])
model_2023 = sm.OLS(Y, X).fit(cov_type="HC3")

print(model_2023.summary())

