import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rsp = pd.read_excel("RSP Monthly.xlsx")

rsp.head()
rsp.columns
rsp.info()

rsp = rsp.rename(columns={
    "Date": "date",
    "Total Net Assets as of Month End": "tna",
    "Total Return per Share as of Month End": "ret",
    "Monthly Net Asset Value Per Share": "nav"
})

rsp["ret"].describe()

rsp.columns

rsp.columns = [
    "date",
    "fund_id",
    "tna",
    "ret",
    "nav"
]

rsp.info()

rsp["tna_lag"] = rsp["tna"].shift(1)
rsp["net_flow"] = rsp["tna"] - (rsp["tna_lag"] * (1 + rsp["ret"]))
rsp["flow_rsp"] = rsp["net_flow"] / rsp["tna_lag"]
rsp = rsp.dropna()

rsp[["tna", "ret", "net_flow", "flow_rsp"]].head()
rsp["flow_rsp"].describe()

rsp_flow = rsp[["date", "flow_rsp"]]

plt.figure(figsize=(12,5))

plt.plot(rsp["date"], rsp["flow_rsp"])
plt.axhline(0, linestyle="--")

plt.title("RSP Monthly Scaled Net Fund Flows")
plt.xlabel("Date")
plt.ylabel("Scaled Net Flow")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

rsp[["date", "flow_rsp"]].head()

rsp = rsp.copy()
rsp["date"] = pd.to_datetime(rsp["date"])
rsp["date"] = rsp["date"] + pd.offsets.MonthEnd(0)

rsp.head()

rsp_final = rsp[["date", "flow_rsp"]].copy()
rsp_final = rsp_final.sort_values("date")

rsp_final.to_csv("rsp_clean.csv", index=False)
