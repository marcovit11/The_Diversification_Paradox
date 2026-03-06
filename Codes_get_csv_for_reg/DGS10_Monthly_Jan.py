import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

y10 = pd.read_excel("DGS10 Monthly.xlsx")

y10.columns = ["date", "y10"]
y10.head()
y10.info()

y10["date"] = pd.to_datetime(y10["date"])

# Move to end-of-month timestamp
y10["date"] = y10["date"] + pd.offsets.MonthEnd(0)

y10 = y10.sort_values("date")

y10.head()

y10["dy10"] = y10["y10"] - y10["y10"].shift(1)

y10 = y10.dropna()

y10 = y10[y10["date"] >= "2015-12-31"]

y10.shape

y10["dy10"].describe()

plt.figure(figsize=(12,5))

plt.plot(y10["date"], y10["dy10"])
plt.axhline(0, linestyle="--")

plt.title("Monthly Change in 10-Year Treasury Yield")
plt.xlabel("Date")
plt.ylabel("Δ 10Y Yield (percentage points)")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


y10.head()


y10_final = y10[["date", "dy10"]].copy()
y10_final = y10_final.sort_values("date")

y10_final.to_csv("dgs10_clean.csv", index=False)
