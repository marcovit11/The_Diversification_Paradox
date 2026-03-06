import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, kpss   # stationarity tests
from statsmodels.tsa.vector_ar.var_model import VAR     # VAR estimator

# ═══════════════════════════════════════════════════════════════════════
# VAR NOTEBOOK — Diversification Paradox
# ───────────────────────────────────────────────────────────────────────
# Reduced-form Vector Autoregression (VAR)
# 3 separate 5-variable ([dlog_vix, dy10, ASVI8, flow_x, mag7_ret]) VARs — one per ETF flow fund (SPY, RSP, VOO)
# ═══════════════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore", category=UserWarning) # Suppress KPSS interpolation warnings

# Load each clean CSV (Sequential inner merge on date — keeps only months present in all series)
spy  = pd.read_csv("spy_clean.csv",              parse_dates=["date"])  # SPY net flows (scaled by TNA)
voo  = pd.read_csv("voo_clean.csv",              parse_dates=["date"])  # VOO net flows
rsp  = pd.read_csv("rsp_clean.csv",              parse_dates=["date"])  # RSP net flows (equal-weight benchmark)
vix  = pd.read_csv("vix_clean.csv",              parse_dates=["date"])  # log-differenced VIX (macro risk proxy)
y10  = pd.read_csv("dgs10_clean.csv",            parse_dates=["date"])  # first-differenced 10Y Treasury yield
asvi8= pd.read_csv("AIStocks_ASVI8_Monthly.csv", parse_dates=["date"])  # ASVI8: abnormal AI search volume, 8-month median baseline
mag7 = pd.read_csv("mag7_clean.csv",             parse_dates=["date"])  # equal-weighted Mag7 log returns

df = (spy
      .merge(rsp,   on="date", how="inner")
      .merge(voo,   on="date", how="inner")
      .merge(vix,   on="date", how="inner")
      .merge(y10,   on="date", how="inner")
      .merge(asvi8, on="date", how="inner")
      .merge(mag7,  on="date", how="inner")
)

df = df.sort_values("date").reset_index(drop=True) # Sort chronologically and reset index to 0-based integer index

#print(f"Merged shape: {df.shape}")   # expect 121 rows × 8+ columns
#print(df.head())                     # visual check: dates start Dec 2015

# ═══════════════════════════════════════════════════════════════════════
# PRE-ESTIMATION: STATIONARITY TESTS
# ═══════════════════════════════════════════════════════════════════════
# We run two complementary tests:
#   ADF: null = unit root exists → want to REJECT (p < 0.05)
#   KPSS: null = stationary     → want to FAIL TO REJECT (p > 0.05)

all_vars = ["dlog_vix", "dy10", "ASVI8", "flow_spy", "flow_rsp", "flow_voo", "mag7_ret"]

print("═" * 68)
print("ADF TEST  (want p < 0.05 → stationary)")
print("═" * 68)
for name in all_vars:
    r = adfuller(df[name].dropna(), autolag="AIC")
    status = "Good" if r[1] < 0.05 else "Reject: NON-STATIONARY"
    print(f"{name:<12} | stat: {r[0]:8.3f} | p: {r[1]:.4f} | lags: {r[2]} | {status}")

print()
print("═" * 68)
print("KPSS TEST  (want p > 0.05 → stationary)")
print("═" * 68)
for name in all_vars:
    r = kpss(df[name].dropna(), regression="c", nlags="auto")
    status = "Good" if r[1] > 0.05 else "CHECK — possible unit root"
    print(f"{name:<12} | stat: {r[0]:8.3f} | p: {r[1]:.4f} | {status}")

# ═══════════════════════════════════════════════════════════════════════
# SAMPLE CHECK
# ═══════════════════════════════════════════════════════════════════════
# Build the master VAR dataframe including all variables + date column

var_df = df[all_vars + ["date"]].dropna().copy()

#print(f"Final sample: {len(var_df)} observations")          # expect 121 (Dec 2015 – Dec 2025)
#print(f"Date range:   {var_df['date'].min().date()} → {var_df['date'].max().date()}")
#print(f"Missing values:\n{var_df[all_vars].isnull().sum()}")  # must all be 0

# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTION — run_var()
# ═══════════════════════════════════════════════════════════════════════
# Encapsulates the full VAR pipeline for one ETF flow variable.
# Called once per fund (SPY, RSP, VOO) for primary and robustness runs.
# Arguments:
#   var_name  : label for print headers and plot titles
#   flow_col  : which ETF flow to include ("flow_spy", "flow_rsp", "flow_voo")
#   data      : dataframe to use — defaults to full var_df, pass var_df_2020
#               for the subsample robustness check

def run_var(var_name, flow_col, data=None):

    if data is None:
        data = var_df   # use full sample by default

    cols = ["dlog_vix", "dy10", "ASVI8", flow_col, "mag7_ret"]
    idx  = {c: i for i, c in enumerate(cols)}  # maps variable name → column index for IRF/FEVD slicing
    sub  = data[cols].copy()                    # extract only the 5 variables for this VAR

    print("\n" + "═"*68)
    print(f"  VAR — {var_name}  |  N={len(sub)}  |  Variables: {cols}")
    print("═"*68)

    # LAG SELECTION
    model   = VAR(sub)
    lag_sel = model.select_order(maxlags=4)  # test lags 0 through 4
    print("\nLAG SELECTION:")
    print(lag_sel.summary())                 # full table: AIC, BIC, HQIC, FPE for each lag

    bic_lag  = lag_sel.bic   # BIC-optimal lag
    aic_lag  = lag_sel.aic   # AIC-optimal lag 
    selected = max(bic_lag, 1)  # enforce minimum of 1 — VAR(0) is meaningless
    print(f"BIC → {bic_lag}  |  AIC → {aic_lag}  |  Using: {selected}")

    # ESTIMATION: Fits the reduced-form VAR: each variable regressed on p lags of all variables
    results = model.fit(maxlags=selected, ic="bic")
    print("\nVAR SUMMARY:")
    print(results.summary())  # full coefficient table for all 5 equations

    # STABILITY CHECK 
    print(f"\nStability check : {results.is_stable()}")              # must be True
    print(f"Max eigenvalue  : {np.abs(results.roots).max():.4f}")   # must be < 1.0

    # GRANGER CAUSALITY
    print("\nGRANGER CAUSALITY:")
    print(f"{'Test':<35} | {'F-stat':>8} | {'p-value':>8} | Result")
    print("─" * 70)

    gc_tests = [
        ("ASVI8",    flow_col,   f"ASVI8 → {flow_col}"),   # H1: narrative drives flows?
        ("ASVI8",    "mag7_ret", "ASVI8 → mag7_ret"),       # direct attention → price channel
        (flow_col,   "mag7_ret", f"{flow_col} → mag7_ret"), # flow → price channel
        ("mag7_ret", "ASVI8",    "mag7_ret → ASVI8"),       # feedback loop: returns drive attention?
        ("dlog_vix", "mag7_ret", "VIX → mag7_ret"),         # macro risk → Mag7 (expected significant)
        ("dy10",     "mag7_ret", "dy10 → mag7_ret"),        # yield → Mag7 (expected significant)
    ]
    for cause, effect, label in gc_tests:
        r   = results.test_causality(effect, [cause], kind="f")  # F-test version (preferred for small N)
        sig = "Good, p<0.05" if r.pvalue < 0.05 else ("Check:  p<0.10" if r.pvalue < 0.10 else "—")
        print(f"{label:<35} | {r.test_statistic:>8.3f} | {r.pvalue:>8.4f} | {sig}")

    # RESIDUAL DIAGNOSTICS
    print("\nWHITENESS TEST (Portmanteau, nlags=10):")
    print(results.test_whiteness(nlags=10).summary())

    # Normality test
    print("\nNORMALITY TEST:")
    print(results.test_normality().summary())

    # IMPULSE RESPONSE FUNCTIONS (IRFs)
    irf = results.irf(12)

    # Full 5×5 IRF grid — shows all impulse-response pairs
    irf.plot(orth=True, figsize=(14, 10))
    plt.suptitle(f"Orthogonalised IRFs — {var_name}", y=1.01, fontsize=13)
    plt.tight_layout()
    plt.show()

    # Focused plot — the 3 economically relevant relationships
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    focused = [
        ("ASVI8",    "mag7_ret", "ASVI8 shock → Mag7 returns"),  # direct narrative → price
        ("ASVI8",    flow_col,   f"ASVI8 shock → {flow_col}"),   # narrative → flow (H1)
        ("mag7_ret", "ASVI8",    "Mag7 shock → ASVI8"),          # feedback loop
    ]
    for ax, (imp, resp, title) in zip(axes, focused):
        irf_vals = irf.orth_irfs[:, idx[resp], idx[imp]]           # point estimate of IRF
        stderr   = irf.stderr(orth=True)[:, idx[resp], idx[imp]]   # standard error at each horizon
        ci_lo    = irf_vals - 1.96 * stderr  # lower bound of approximate 95% confidence band
        ci_hi    = irf_vals + 1.96 * stderr  # upper bound of approximate 95% confidence band
        horizons = np.arange(len(irf_vals))  # 0 to 12 months

        ax.plot(horizons, irf_vals, color="navy", linewidth=2, label="IRF")
        ax.fill_between(horizons, ci_lo, ci_hi, alpha=0.2, color="steelblue", label="95% CI")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")  # zero reference line
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Months after shock")
        ax.legend(fontsize=8)

    plt.suptitle(f"Key IRFs — {var_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)
    fevd = results.fevd(12)

    fevd.plot(figsize=(12, 8))
    plt.suptitle(f"FEVD — {var_name}", y=1.01, fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print clean FEVD table for mag7_ret at horizons 1, 6, 12
    print(f"\nFEVD — share of mag7_ret forecast error variance explained by each variable:")
    print(f"{'Variable':<12} | {'t=1':>8} | {'t=6':>8} | {'t=12':>8}")
    print("─" * 45)
    fevd_df = pd.DataFrame(
        fevd.decomp[idx["mag7_ret"]],  # slice FEVD for the mag7_ret equation
        columns=cols
    )
    for var in cols:
        print(f"{var:<12} | {fevd_df[var].iloc[0]:>7.1%} | {fevd_df[var].iloc[5]:>7.1%} | {fevd_df[var].iloc[11]:>7.1%}")

    return results  # return results object for further querying if needed

# ═══════════════════════════════════════════════════════════════════════
# PRIMARY RUNS — ASVI8, FULL SAMPLE (Dec 2015 – Dec 2025)
# ═══════════════════════════════════════════════════════════════════════

results_spy = run_var("VAR-SPY", "flow_spy")
results_rsp = run_var("VAR-RSP", "flow_rsp")
results_voo = run_var("VAR-VOO", "flow_voo")


# ═══════════════════════════════════════════════════════════════════════
# ROBUSTNESS — 2021+ SUBSAMPLE
# ═══════════════════════════════════════════════════════════════════════
# Re-runs all three VARs on the post-COVID subsample (Jan 2021 onwards)
var_df_2021 = var_df[var_df["date"] >= "2021-01-31"].copy()
print(f"2021+ subsample: {len(var_df_2021)} observations")  # expect ~60

results_spy_2021 = run_var("VAR-SPY 2021+", "flow_spy", data=var_df_2021)
results_rsp_2021 = run_var("VAR-RSP 2021+", "flow_rsp", data=var_df_2021)
results_voo_2021 = run_var("VAR-VOO 2021+", "flow_voo", data=var_df_2021)
