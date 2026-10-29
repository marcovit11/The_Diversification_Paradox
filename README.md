# The Diversification Paradox: replication package

Replication package for my bachelor's thesis "The Diversification Paradox: Retail AI Narrative Attention, Passive Concentration, and Magnificent 7 Returns" at the University of St. Gallen (2026).

The package contains all data, code and outputs. With it you can reproduce every table and figure in the thesis. The data is monthly, from December 2015 to December 2025 (121 months).


## How to run

Install the packages:

```
pip install -r requirements.txt
```

Then open Jupyter and run the notebooks in their numbered order, first the four in "Codes to get csvs for regression" and then the five in "Regressions and relevant csvs". In each notebook, restart the kernel and run all cells. Open the notebooks from inside their folder, because the file paths are relative. Everything together takes about a minute.

The notebooks in the first folder build the clean csv files from the raw data and save them in the second folder. The clean files are already included, so the analysis notebooks also run without the first step.

I used Python 3.13 with the package versions in requirements.txt. I also tested it with an older and a newer set of package versions and got the same results.


## Folders

**Codes to get csvs for regression**: raw data and the four data notebooks

- 1_ASVI_Google_Trends: attention measure (ASVI8 and ASVI4)
- 2_ETF_Flows: net flows of SPY, RSP and VOO
- 3_Mag7_Returns: Mag7, Mag7B and Mag7C returns
- 4_Macro_Controls: VIX and the 10-year Treasury yield

**Regressions and relevant csvs**: clean csv files and the five analysis notebooks. Tables (csv) and figures (png) are saved in the output folder.

- 1_Descriptive_Statistics: Table 1, Figure 1, Tables A.2 and A.3, Figure A.1
- 2_Baseline_OLS: Table 2, Tables C.5 and C.6
- 3_VAR_Primary: Tables 3 to 5, Figure 2, Tables B.1, B.3, C.1 and C.2, Figures D.1 and D.2
- 4_VAR_Robustness: Table 6, Tables B.2, C.3, C.4 and C.7
- 5_Bootstrap_IRF: Table D.1, Figure D.3

**Code PDFs**: a printed version of every notebook with its output.


## Data

All raw files are in "Codes to get csvs for regression".

| File | Source | Content |
|---|---|---|
| AIStocks_SVI_raw.csv | Google Trends | Search volume index (0 to 100) for "AI Stocks", United States, all categories, web search, April 2015 to December 2025. Columns: month, SVI |
| SPY Monthly.xlsx, RSP Monthly.xlsx, VOO Monthly.xlsx | CRSP Mutual Fund database (WRDS) | Month-end total net assets (USD million), monthly total return per share and NAV per share, November 2015 to December 2025 |
| Mag7 Monthly Returns.csv | Compustat Security Monthly (WRDS) | Monthly total return in percent (trt1m) of AAPL, AMZN, GOOGL, META, MSFT, NVDA and TSLA, December 2015 to December 2025 |
| VIX Monthly.xlsx | CBOE Indexes (WRDS) | Month-end close of the VIX, November 2015 to December 2025 |
| DGS10 Monthly.xlsx | FRED, series DGS10 | 10-year Treasury yield in percent, monthly, end of period, November 2015 to December 2025 |

The data from WRDS (CRSP, Compustat and CBOE) are included only so the results can be replicated. The license does not allow sharing them further, which is also why they are not in the public GitHub version of the code.

Google Trends: I downloaded the series with pytrends on March 4, 2026. Google Trends gives slightly different numbers every time you download, so the notebook uses the saved file and does not download again (the download code is in the notebook as a comment). I did not save the raw download as a separate file at first. The first eight months (April to November 2015, all 0) are in the output of my original notebook, and the other months were recovered from the saved ASVI8 series. The recovered file gives exactly the same ASVI8 and ASVI4 values.


## Variables

All clean files are in "Regressions and relevant csvs". Each file has a date column (month end) and one variable, December 2015 to December 2025.

| Variable | File | Definition |
|---|---|---|
| ASVI8 | AIStocks_ASVI8_Monthly.csv | log(SVI in month t) minus log(median SVI of months t-8 to t-1). SVI values of 0 are set to 0.5 before taking logs. Main attention measure |
| ASVI4 | AIStocks_ASVI4_Monthly.csv | Same as ASVI8 with the median of months t-4 to t-1. Robustness check |
| SVI | AIStocks_SVI_Monthly.csv | Raw Google Trends index in the sample period (Figure A.1) |
| flow_spy, flow_rsp, flow_voo | spy_clean.csv, rsp_clean.csv, voo_clean.csv | Implied net flow as a share of last month's total net assets: (TNA(t) - TNA(t-1) x (1 + R(t))) / TNA(t-1), where R is the total return per share |
| mag7_ret | mag7_clean.csv | Equal-weighted average of the monthly log returns of AAPL, AMZN, GOOGL, META, MSFT, NVDA and TSLA |
| mag7b_ret | mag7b_clean.csv | Same without Tesla |
| mag7c_ret | mag7c_clean.csv | Same for GOOGL, META, MSFT and NVDA only |
| dlog_vix | vix_clean.csv | log(VIX in month t) minus log(VIX in month t-1) |
| d_dgs10 | dgs10_clean.csv | Change in the 10-year Treasury yield from month t-1 to t, in percentage points |

The output folder also contains bootstrap_irf_Full_sample_raw.csv and bootstrap_irf_2021-2025_raw.csv. They hold the full bootstrap bands for every horizon (2,000 replications, seed 42) and are written by 5_Bootstrap_IRF.
