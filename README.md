# The Diversification Paradox: code and data

Code for the master's thesis "The Diversification Paradox: Retail AI Narrative Attention, Passive Concentration, and Magnificent 7 Returns" by Marco Vitulli, University of St. Gallen, 2026.

## What is in this repository

| File or folder | What it does | Thesis output |
| --- | --- | --- |
| `1_VAR_Primary.ipynb` | Five-variable VAR(1) for SPY, RSP, and VOO, full sample and 2021-2025: stationarity tests, lag selection, Granger tests, impulse responses, variance decomposition, bias-corrected bootstrap bands | Tables 3 to 5, Figure 2, Table D.1, Figure D.3 |
| `2_VAR_Baskets.ipynb` | Same VAR with the Mag7B (without Tesla) and Mag7C (NVIDIA, Microsoft, Alphabet, Meta) baskets | Table 6, Panel B |
| `3_VAR_ASVI4.ipynb` | Same VAR with the 4-month attention measure (ASVI4) | Table 6, Panel D |
| `4_VAR_Restricted3var.ipynb` | Three-variable VAR (VIX, ASVI8, Mag7 returns) | Table 6, Panel C |
| `5_Appendix_Tables.ipynb` | Revised Table 1, Appendices A to D, and a check of the key numbers reported in the thesis | Table 1, Tables A.2 to C.7, Figures A.1, D.1, D.2 |
| `bootstrap_irf_corrected.py` | Recursive residual bootstrap with the bias correction of Kilian (1998), used in notebook 1 | Table D.1, Figure D.3 |
| `Baseline OLS Merged-2.ipynb` | Baseline OLS regressions with HC3 standard errors | Table 2 |
| `data_preparation/` | Notebooks that build the clean monthly CSV files from the raw downloads | |
| `appendix_output/` | Tables (CSV) and figures (PNG) written by notebook 5 | |
| `Code PDFs/` | PDF exports of the executed notebooks | |

## Data

Included (public):

- Google Trends search volume for "AI Stocks", United States, monthly, retrieved with the pytrends library on [DATE]: `AIStocks_ASVI8_Monthly.csv`, `AIStocks_ASVI4_Monthly.csv`, and the raw index `AIStocks_SVI_raw_reconstructed.csv`, recovered from the ASVI8 file by inverting Equation (1) of the thesis.
- 10-year US Treasury constant maturity yield from FRED (series DGS10): `dgs10_clean.csv`.

Not included (licensed through WRDS):

- CRSP Mutual Fund database: total net assets and monthly total returns of SPY, VOO, and RSP, used to compute net fund flows (`spy_clean.csv`, `voo_clean.csv`, `rsp_clean.csv`).
- Compustat: monthly total returns of the Magnificent 7 (`mag7_clean.csv`, `mag7b_clean.csv`, `mag7c_clean.csv`, `Mag7 Monthly Returns.csv`).
- CBOE Volatility Index downloaded through WRDS (`vix_clean.csv`).

Users with WRDS access can rebuild these files with the notebooks in `data_preparation/`.

## How to reproduce the results

1. Place all CSV files and the notebooks in one folder.
2. Open each notebook (1 to 5) in Jupyter and choose Kernel > Restart Kernel and Run All Cells.
3. Notebook 5 ends with a check of the key numbers. Every line should say OK.

Do not rerun the Google Trends notebooks in `data_preparation/`. Google Trends draws a new sample of searches for each download, so a new download gives slightly different numbers.

## Software

Python [VERSION], pandas [VERSION], numpy [VERSION], statsmodels [VERSION], matplotlib [VERSION].
