import pandas as pd #used for tabular data
import numpy #dependency for statsmodels
import statsmodels.api as sm #provides regression engine 
import pickle # to make this able to be queried 

#read in CSV data
spy = pd.read_csv("spy_clean.csv", parse_dates=["date"])
voo = pd.read_csv("voo_clean.csv", parse_dates=["date"])
rsp = pd.read_csv("rsp_clean.csv", parse_dates=["date"])
vix = pd.read_csv("vix_clean.csv", parse_dates=["date"])
y10 = pd.read_csv("dgs10_clean.csv", parse_dates=["date"])
asvi4 = pd.read_csv("AIStocks_ASVI4_Monthly.csv", parse_dates=["date"])
asvi8 = pd.read_csv("AIStocks_ASVI8_Monthly.csv", parse_dates=["date"])
mag7 = pd.read_csv("mag7_clean.csv", parse_dates=["date"])

#sequential inner merges: align rows by matching date values, keep only dates from both data frames.
df = spy.merge(rsp, on="date", how="inner", suffixes=("_spy", "_rsp"))
#chain of merges, keeping only dates present across all sources 
for frame in [voo,vix,y10, asvi4,asvi8, mag7]:
    df = df.merge(frame, on="date", how="inner")

#this moves every value down one row, creates a one period lag 
for col in ["ASVI4", "ASVI8", "flow_spy", "flow_voo", "flow_rsp", "mag7_ret"]:
    df[f"{col}_lag"] = df[col].shift(1)

#drop the NaN at the top if each lag col
df = df.dropna()

#year slices
year_dfs = {
    "2016_2025": df,
    "2020":      df[df["date"] >= "2020-01-31"].copy(),
    "2021":      df[df["date"] >= "2021-01-31"].copy(),
    "2022":      df[df["date"] >= "2022-01-31"].copy(),
    "2023":      df[df["date"] >= "2023-01-31"].copy(),
}
#model specifications: (label, y_col, x_cols)
specs = [
    # ASVI4 models
    ("SPY_flow ~ ASVI4",        "flow_spy",  ["ASVI4_lag",    "dlog_vix", "dy10"]),
    ("RSP_flow ~ ASVI4",        "flow_rsp",  ["ASVI4_lag",    "dlog_vix", "dy10"]),
    ("VOO_flow ~ ASVI4",        "flow_voo",  ["ASVI4_lag",    "dlog_vix", "dy10"]),
    ("mag7_ret ~ flow_spy",     "mag7_ret",  ["flow_spy_lag", "dlog_vix", "dy10"]),
    ("mag7_ret ~ flow_rsp",     "mag7_ret",  ["flow_rsp_lag", "dlog_vix", "dy10"]),
    ("mag7_ret ~ flow_voo",     "mag7_ret",  ["flow_voo_lag", "dlog_vix", "dy10"]),
    ("mag7_ret ~ ASVI4",        "mag7_ret",  ["ASVI4_lag",    "dlog_vix", "dy10"]),
    ("ASVI4 ~ mag7_ret",        "ASVI4",     ["mag7_ret_lag", "dlog_vix", "dy10"]),

    # ASVI8 models
    ("SPY_flow ~ ASVI8",        "flow_spy",  ["ASVI8_lag",    "dlog_vix", "dy10"]),
    ("RSP_flow ~ ASVI8",        "flow_rsp",  ["ASVI8_lag",    "dlog_vix", "dy10"]),
    ("VOO_flow ~ ASVI8",        "flow_voo",  ["ASVI8_lag",    "dlog_vix", "dy10"]),
    ("mag7_ret ~ ASVI8",        "mag7_ret",  ["ASVI8_lag",    "dlog_vix", "dy10"]),
    ("ASVI8 ~ mag7_ret",        "ASVI8",     ["mag7_ret_lag", "dlog_vix", "dy10"]),
]

#run and store models
models = {}

for label, y_col, x_cols in specs:
    models[label] = {}
    for year, data in year_dfs.items():
        Y = data[y_col]
        X = sm.add_constant(data[x_cols])
        result = sm.OLS(Y, X).fit(cov_type="HC3")
        models[label][year] = result

        print(f"\n{'-'*60}")
        print(f"  Model : {label}")
        print(f"  Period: {year}")
        print(f"{'-'*60}")
        print(result.summary())

#to query from another file:
with open("models.pkl", "wb") as f:
    pickle.dump(models, f)

#uncomment to test if saved.
#print("Models saved to models.pkl")



# You can use these lines to access models in this file and view what you want 
# by changing the specifications or duplicating lines.

#reading a model later (ex: "SPY_flow ~ ASVI4", 2022):
#print(models["SPY_flow ~ ASVI4"]["2022"].summary())

# Get the 2022 SPY flow ~ ASVI4 model
#m = models["SPY_flow ~ ASVI4"]["2022"]
#print(m.params)      # coefficients, covered in summary but accessible individually
#print(m.rsquared)    # R², covered in summary but accessible individually
#print(m.pvalues)     # p-values, covered in summary but accessible individually