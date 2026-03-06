import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

voo = pd.read_excel("VOO Monthly.xlsx")

voo.head()
voo.columns
voo.info()

voo = voo.rename(columns={
    "Date": "date",
    "Total Net Assets as of Month End": "tna",
    "Total Return per Share as of Month End": "ret"
})

voo["ret"].describe()

voo.info()

voo["tna_lag"] = voo["tna"].shift(1)
voo["net_flow"] = voo["tna"] - (voo["tna_lag"] * (1 + voo["ret"]))
voo["flow_voo"] = voo["net_flow"] / voo["tna_lag"]
voo = voo.dropna()

voo["ret"].describe()

voo[["tna", "ret", "net_flow", "flow_voo"]].head()
voo["flow_voo"].describe()

voo_flow = voo[["date", "flow_voo"]]


plt.figure(figsize=(12,5))

plt.plot(voo["date"], voo["flow_voo"])
plt.axhline(0, linestyle="--")

plt.title("VOO Monthly Scaled Net Fund Flows")
plt.xlabel("Date")
plt.ylabel("Scaled Net Flow")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

voo[["date", "flow_voo"]].head()

voo = voo.copy()

voo["date"] = pd.to_datetime(voo["date"])
voo["date"] = voo["date"] + pd.offsets.MonthEnd(0)

voo[["date", "flow_voo"]].head()

voo_final = voo[["date", "flow_voo"]].copy()
voo_final = voo_final.sort_values("date")

voo_final.to_csv("voo_clean.csv", index=False)
