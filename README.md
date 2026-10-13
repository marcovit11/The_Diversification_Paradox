# The Diversification Paradox

Code for my bachelor's thesis "The Diversification Paradox: Retail AI Narrative Attention, Passive Concentration, and Magnificent 7 Returns" at the University of St. Gallen (2026).

With these notebooks you can reproduce all tables and figures in the thesis. The data is monthly, from December 2015 to December 2025.


## Folders

**Codes to get csvs for regression**

These notebooks build the monthly data set from the raw files and save the clean csv files in the second folder.

- 1_ASVI_Google_Trends: attention measure (ASVI8 and ASVI4)
- 2_ETF_Flows: net flows of SPY, RSP and VOO
- 3_Mag7_Returns: Mag7, Mag7B and Mag7C returns
- 4_Macro_Controls: VIX and the 10-year Treasury yield

**Regressions and relevant csvs**

Here are the clean csv files and the analysis. Tables and figures are saved in the output folder.

- 1_Descriptive_Statistics: Table 1, Figure 1, Tables A.2 and A.3, Figure A.1
- 2_Baseline_OLS: Table 2, Tables C.5 and C.6
- 3_VAR_Primary: Tables 3 to 5, Figure 2, Tables B.1, B.3, C.1 and C.2, Figures D.1 and D.2
- 4_VAR_Robustness: Table 6, Tables B.2, C.3, C.4 and C.7
- 5_Bootstrap_IRF: Table D.1, Figure D.3

**Code PDFs** has a printed version of every notebook with its output.


## How to run

Install the packages:

```
pip install -r requirements.txt
```

Then run the notebooks in their numbered order, first the four in "Codes to get csvs for regression" and then the five in "Regressions and relevant csvs". Everything together takes about a minute.


## Data

Google Trends: search term "AI Stocks", United States, monthly, April 2015 to December 2025. I downloaded it with pytrends on March 4, 2026 (AIStocks_SVI_raw.csv). Google Trends gives slightly different numbers every time you download, so the notebook uses the saved file and does not download again. I did not save the raw download as a separate file at first. The first eight months (April to November 2015, all 0) are in the output of my original notebook, and the other months were recovered from the saved ASVI8 series. The recovered file gives exactly the same ASVI8 and ASVI4 values.

FRED: 10-year Treasury yield (DGS10), monthly, end of period (DGS10 Monthly.xlsx).

The following data comes from WRDS and is not included in this repository because the license does not allow sharing it:

- CRSP Mutual Fund database: total net assets and monthly returns of SPY, RSP and VOO
- Compustat: monthly returns of Apple, Amazon, Alphabet, Meta, Microsoft, Nvidia and Tesla
- CBOE: month-end VIX

The clean csv files built from this data are also excluded (see .gitignore). With WRDS access you can download the same series and run the first folder again.


## Software

Python 3.13 with pandas, numpy, statsmodels, scipy and matplotlib (versions in requirements.txt). I tested it with an older and a newer set of package versions and got the same results.
