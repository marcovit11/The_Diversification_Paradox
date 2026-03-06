import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mag7 = pd.read_csv("Mag7 Monthly Returns.csv")

mag7.head()
mag7.columns
mag7.info()

mag7.columns
mag7.head()

mag7 = mag7[[
    "(tic) Ticker Symbol",
    "(datadate) Data Date - Security Monthly",
    "(trt1m) Monthly Total Return"
]]

mag7.columns = ["ticker", "date", "trt1m"]

mag7["date"] = pd.to_datetime(
    mag7["date"],
    format="%d.%m.%Y"
)

mag7.info()

mag7["ret_dec"] = mag7["trt1m"] / 100

mag7["ret_dec"].describe()


mag7["log_ret"] = np.log(1 + mag7["ret_dec"])

mag7["log_ret"].describe()

mag7_avg = (
    mag7
    .groupby("date")["log_ret"]
    .mean()
    .rename("mag7_ret")
    .reset_index()
)

mag7_avg.head()

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

mag7_avg.describe()

mag7_avg.head()

mag7_avg = mag7_avg.reset_index()
mag7_avg.to_csv("mag7_clean.csv", index=False)
