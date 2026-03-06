import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pytrends.request import TrendReq

pytrends = TrendReq(
    hl='en-US',
    tz=360,              
)

kw_list = ["AI Stocks"]

timeframe = "2015-08-01 2025-12-31"
geo = "US"

pytrends.build_payload(
    kw_list=kw_list,
    cat=0,
    timeframe=timeframe,
    geo=geo,
    gprop=""
)

monthly_svi = pytrends.interest_over_time()
monthly_svi = monthly_svi.drop(columns=["isPartial"])
monthly_svi.head()

monthly_svi = monthly_svi.reset_index()
monthly_svi.rename(columns={"date": "Month"}, inplace=True)
monthly_svi.head()

monthly_svi["Month"].diff().value_counts()
monthly_svi.tail()

monthly_svi["Month"] = pd.to_datetime(monthly_svi["Month"])
monthly_svi.head()


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


# Ensure sorted
monthly_svi = monthly_svi.sort_values("Month").reset_index(drop=True)

# Avoid log(0) — replace zeros with small positive number 
monthly_svi["SVI_adj"] = monthly_svi["AI Stocks"].replace(0, 0.5)

# Compute rolling 4-month median of past values (t-1 to t-4)
monthly_svi["Median_4"] = (
    monthly_svi["SVI_adj"]
    .rolling(window=4)
    .median()
    .shift(1)
)

# Compute ASVI
monthly_svi["ASVI4"] = np.log(monthly_svi["SVI_adj"]) - np.log(monthly_svi["Median_4"])
monthly_svi.head(12)

monthly_svi = monthly_svi.dropna(subset=["ASVI4"]).reset_index(drop=True)

monthly_svi.head()

plt.figure()
plt.plot(monthly_svi["Month"], monthly_svi["ASVI4"])
plt.title("Monthly ASVI (AI Stocks)")
plt.xlabel("Month")
plt.ylabel("ASVI4")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

monthly_svi.head(12)

monthly_export = monthly_svi[["Month", "ASVI4"]].copy()

monthly_export = monthly_export.rename(columns={
    "Month": "date"
})

monthly_export["date"] = (
    monthly_export["date"]
        .dt.to_period("M")
        .dt.to_timestamp("M")
)

monthly_export.head()
monthly_export.tail()
monthly_export.to_csv("AIStocks_ASVI4_Monthly.csv", index=False)
