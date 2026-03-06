import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

spy = pd.read_excel("SPY Monthly.xlsx")
spy.head()
spy.columns
spy.info()

spy = spy.rename(columns={
    "Date": "date",
    "Total Net Assets as of Month End": "tna",
    "Total Return per Share as of Month End": "ret",
    "Monthly Net Asset Value Per Share": "nav"
})

spy["ret"].describe()

spy.columns
spy.columns = [
    "date",
    "fund_id",
    "tna",
    "ret",
    "nav"
]

spy.info()

spy["tna_lag"] = spy["tna"].shift(1)
spy["net_flow"] = spy["tna"] - (spy["tna_lag"] * (1 + spy["ret"]))
spy["flow_spy"] = spy["net_flow"] / spy["tna_lag"]
spy = spy.dropna()

spy[["tna", "ret", "net_flow", "flow_spy"]].head()
spy["flow_spy"].describe()

spy_flow = spy[["date", "flow_spy"]]


plt.figure(figsize=(12,5))

plt.plot(spy["date"], spy["flow_spy"])
plt.axhline(0, linestyle="--")

plt.title("SPY Monthly Scaled Net Fund Flows")
plt.xlabel("Date")
plt.ylabel("Scaled Net Flow")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

spy[["date", "flow_spy"]].head()

spy = spy.copy()

spy["date"] = pd.to_datetime(spy["date"])
spy["date"] = spy["date"] + pd.offsets.MonthEnd(0)

spy[["date", "flow_spy"]].head()

spy_final = spy[["date", "flow_spy"]].copy()
spy_final = spy_final.sort_values("date")

spy_final.to_csv("spy_clean.csv", index=False)

spy_final.head()
spy_final.tail()

