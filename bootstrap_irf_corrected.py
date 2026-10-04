# ═══════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE BANDS FOR THE ASVI8 -> MAG7 IMPULSE RESPONSE
# Replaces the bootstrap cell (In [12]) in 1_VAR_Primary.
# Run from the folder that holds the clean CSVs.
# ───────────────────────────────────────────────────────────────────────
# What was wrong with the old cell
#   It built each bootstrap sample as  Y_boot = fittedY + resampled residuals.
#   The fitted values come from the ORIGINAL data, so the bootstrap series do
#   not follow the VAR dynamics y_t = c + A1 y_(t-1) + u_t. The re-estimated
#   persistence is too low, the bands collapse toward zero at longer horizons,
#   and from h = 5 the point estimate fell outside its own 95% band.
#
# What this cell does instead
#   1. Estimate the VAR(p) and keep coefficients and residuals.
#   2. Center the residuals.
#   3. Rebuild every bootstrap series recursively from the actual first p
#      observations: y*_t = c + A1 y*_(t-1) + ... + Ap y*_(t-p) + u*_t.
#   4. Re-estimate the VAR(p) on y* and store the orthogonalized IRF.
#   5. Report two versions of the 95% band:
#        "percentile": plain percentile bootstrap
#        "kilian":     bias-corrected bootstrap-after-bootstrap (Kilian, 1998)
#      Least squares VAR coefficients are biased toward zero in small samples,
#      which shifts the plain bootstrap distribution toward zero. Kilian's
#      method removes that bias first and is widely used for small samples
#      like these (60 and 120 monthly observations).
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR

COLS = ["dlog_vix", "dy10", "ASVI8", "flow_spy", "mag7_ret"]   # Cholesky order
REPS = 2000          # bootstrap replications for the bands
BIAS_REPS = 1000     # first-stage replications for Kilian's bias estimate
HORIZON = 12
SEED = 42


# ── VAR helpers (least squares, identical to statsmodels VAR.fit) ──────────
def fit_var(Y, p):
    """Return intercept c, list of lag matrices [A1..Ap], residuals U, Sigma_u."""
    T, K = Y.shape
    X = np.column_stack([np.ones(T - p)] + [Y[p - j - 1:T - j - 1] for j in range(p)])
    B = np.linalg.lstsq(X, Y[p:], rcond=None)[0]
    U = Y[p:] - X @ B
    sigma = U.T @ U / (T - p - X.shape[1])       # degrees-of-freedom adjusted
    A = [B[1 + j * K:1 + (j + 1) * K].T for j in range(p)]
    return B[0], A, U, sigma


def orth_irf(A, sigma, horizon):
    """Orthogonalized (Cholesky) impulse responses, shape (horizon+1, K, K)."""
    K = sigma.shape[0]
    p = len(A)
    P = np.linalg.cholesky(sigma)
    Phi = [np.eye(K)]
    for h in range(1, horizon + 1):
        Phi.append(sum(A[j] @ Phi[h - 1 - j] for j in range(min(p, h))))
    return np.array([Ph @ P for Ph in Phi])


def simulate(c, A, U, Y_init, T, rng):
    """Recursive bootstrap sample of length T started at the actual first p rows."""
    p, K = len(A), len(c)
    Y = np.empty((T, K))
    Y[:p] = Y_init
    u_star = U[rng.integers(0, len(U), size=T - p)]
    for t in range(p, T):
        Y[t] = c + sum(A[j] @ Y[t - 1 - j] for j in range(p)) + u_star[t - p]
    return Y


def max_modulus(A):
    """Largest eigenvalue modulus of the companion matrix (stable if < 1)."""
    K, p = A[0].shape[0], len(A)
    C = np.zeros((K * p, K * p))
    C[:K] = np.hstack(A)
    if p > 1:
        C[K:, :-K] = np.eye(K * (p - 1))
    return np.abs(np.linalg.eigvals(C)).max()


def bias_correct(A, bias):
    """Kilian (1998) correction, shrunk if it would make the VAR non-stationary."""
    delta = 1.0
    A_bc = [a - b for a, b in zip(A, bias)]
    while max_modulus(A_bc) >= 1 and delta > 0:
        delta -= 0.01
        A_bc = [a - delta * b for a, b in zip(A, bias)]
    return A_bc


# ── Main routine ────────────────────────────────────────────────────────────
def bootstrap_irf_bands(data, cols, impulse, response, p=1, horizon=HORIZON,
                        reps=REPS, bias_reps=BIAS_REPS, seed=SEED):
    Y = data[cols].to_numpy(dtype=float)
    T = len(Y)
    i_imp, i_resp = cols.index(impulse), cols.index(response)
    rng = np.random.default_rng(seed)

    c, A, U, sigma = fit_var(Y, p)
    U = U - U.mean(axis=0)
    point = orth_irf(A, sigma, horizon)[:, i_resp, i_imp]

    # Asymptotic (delta-method) bands from statsmodels, as in the main text
    irf_sm = VAR(Y).fit(p).irf(horizon)
    assert np.allclose(irf_sm.orth_irfs[:, i_resp, i_imp], point)
    se = irf_sm.stderr(orth=True)[:, i_resp, i_imp]

    # (a) Plain percentile bootstrap
    draws_pct, A_draws = [], []
    for _ in range(reps):
        _, A_b, _, s_b = fit_var(simulate(c, A, U, Y[:p], T, rng), p)
        A_draws.append(A_b)
        draws_pct.append(orth_irf(A_b, s_b, horizon)[:, i_resp, i_imp])
    draws_pct = np.array(draws_pct)

    # (b) Kilian (1998) bias-corrected bootstrap-after-bootstrap
    bias = [np.mean([Ad[j] for Ad in A_draws[:bias_reps]], axis=0) - A[j] for j in range(p)]
    A_bc = bias_correct(A, bias)
    c_bc = (np.eye(len(cols)) - sum(A_bc)) @ Y.mean(axis=0)   # keeps the sample mean
    draws_kil = []
    for _ in range(reps):
        _, A_b, _, s_b = fit_var(simulate(c_bc, A_bc, U, Y[:p], T, rng), p)
        draws_kil.append(orth_irf(bias_correct(A_b, bias), s_b, horizon)[:, i_resp, i_imp])
    draws_kil = np.array(draws_kil)

    z = 1.959964
    out = pd.DataFrame({"h": np.arange(horizon + 1), "point": point,
                        "asym_lo": point - z * se, "asym_hi": point + z * se})
    out["pct_lo"], out["pct_hi"] = np.percentile(draws_pct, [2.5, 97.5], axis=0)
    out["kilian_lo"], out["kilian_hi"] = np.percentile(draws_kil, [2.5, 97.5], axis=0)
    for m in ["asym", "pct", "kilian"]:
        out[f"{m}_excl_0"] = (out[f"{m}_lo"] > 0) | (out[f"{m}_hi"] < 0)
    return out


if __name__ == "__main__":
    df = None
    for f in ["spy_clean.csv", "vix_clean.csv", "dgs10_clean.csv",
              "AIStocks_ASVI8_Monthly.csv", "mag7_clean.csv"]:
        part = pd.read_csv(f, parse_dates=["date"])
        df = part if df is None else df.merge(part, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    samples = {
        "full": ("Full sample, Jan 2016 to Dec 2025", df),
        # Dec 2020 is kept as the initial lag: estimation runs Jan 2021 to Dec 2025 (60 obs)
        "2021": ("2021-2025 subsample", df[df["date"] >= "2020-12-31"]),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (key, (title, d)) in zip(axes, samples.items()):
        tab = bootstrap_irf_bands(d, COLS, impulse="ASVI8", response="mag7_ret")
        tab.to_csv(f"bootstrap_irf_{key}.csv", index=False)
        print(f"\n{title}: ASVI8 shock -> Mag7 returns ({REPS} replications)")
        print(tab.round(4).to_string(index=False))

        h = tab["h"]
        ax.fill_between(h, tab["kilian_lo"], tab["kilian_hi"], color="#2a78d6", alpha=0.18,
                        linewidth=0, label="Bias-corrected bootstrap 95% (Kilian, 1998)")
        ax.plot(h, tab["asym_lo"], color="#eb6834", lw=1.5, ls="--", label="Asymptotic 95%")
        ax.plot(h, tab["asym_hi"], color="#eb6834", lw=1.5, ls="--")
        ax.plot(h, tab["pct_lo"], color="#1baf7a", lw=1.5, ls=":", label="Percentile bootstrap 95%")
        ax.plot(h, tab["pct_hi"], color="#1baf7a", lw=1.5, ls=":")
        ax.plot(h, tab["point"], color="#2a78d6", lw=2, label="Point estimate")
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Months after shock")
        ax.set_xticks(range(0, 13, 2))
    axes[0].set_ylabel("Response of Mag7 log return")
    axes[0].legend(fontsize=8, loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig("bootstrap_irf_comparison.png", dpi=300)
    plt.show()
