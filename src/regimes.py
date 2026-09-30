"""Step 2 — Regime detection with a Gaussian Hidden Markov Model (HMM).

Step 1 showed that stress comes in (at least) two flavours:
  - growth shocks (2008, 2020): stocks crash, bonds rally
  - rate shocks (2022): stocks AND bonds fall together
So the model sees both stock and bond behaviour, plus the stock-bond correlation,
not stock volatility alone.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR, RANDOM_SEED

FEATURE_WINDOW = 21   # ~1 trading month
CORR_WINDOW = 63      # ~3 trading months
MODEL_PATH = MODEL_DIR / "hmm.pkl"


# ---------------------------------------------------------------- features
def build_features(returns, window=FEATURE_WINDOW, corr_window=CORR_WINDOW):
    """Rolling features describing the market's 'mood' on each day.

    spy_ret  : SPY return over the last month        (trend in stocks)
    spy_vol  : SPY annualized volatility, last month (fear level)
    tlt_ret  : TLT return over the last month        (what bonds are doing)
    stock_bond_corr : SPY-TLT correlation, last 3 months
                      negative = bonds hedge stocks, positive = they fall together
    Every feature uses only past data up to that day (no lookahead).
    """
    feats = pd.DataFrame(index=returns.index)
    feats["spy_ret"] = returns["SPY"].rolling(window).sum()
    feats["spy_vol"] = returns["SPY"].rolling(window).std() * np.sqrt(252)
    feats["tlt_ret"] = returns["TLT"].rolling(window).sum()
    feats["stock_bond_corr"] = returns["SPY"].rolling(corr_window).corr(returns["TLT"])
    return feats.dropna()


# ---------------------------------------------------------------- model
def _n_params(n_states, n_features):
    """Free parameters of a full-covariance Gaussian HMM (for BIC)."""
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    covs = n_states * n_features * (n_features + 1) / 2
    return start + trans + means + covs


def fit_hmm(features, n_states=3, n_restarts=10, seed=RANDOM_SEED):
    """Fit a Gaussian HMM on standardized features.

    EM can get stuck in a bad local optimum, so we fit several times with
    different seeds and keep the model with the highest log-likelihood.
    Returns (model, scaler, log_likelihood, bic).
    """
    from hmmlearn.hmm import GaussianHMM

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    best_model, best_ll = None, -np.inf
    for i in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=500,
            tol=1e-4,
            random_state=seed + i,
        )
        model.fit(X)
        ll = model.score(X)
        if ll > best_ll:
            best_model, best_ll = model, ll

    bic = -2 * best_ll + _n_params(n_states, X.shape[1]) * np.log(len(X))
    return best_model, scaler, best_ll, bic


def compare_n_states(features, candidates=(2, 3, 4), n_restarts=5):
    """Fit models with different numbers of regimes. Lower BIC = better trade-off."""
    rows = []
    for k in candidates:
        _, _, ll, bic = fit_hmm(features, n_states=k, n_restarts=n_restarts)
        rows.append({"n_states": k, "log_likelihood": ll, "bic": bic})
    return pd.DataFrame(rows).set_index("n_states")


# ---------------------------------------------------------------- regimes
def predict_regimes(model, scaler, features):
    """Most likely regime for each day, relabeled 0, 1, 2... by SPY volatility
    (0 = calmest). HMM state numbers are arbitrary, so without this the labels
    could swap between runs.

    Note: predict() uses the Viterbi path, which looks at the whole sample,
    including the future. Fine for studying history; the backtest (Step 4)
    will refit using only past data.
    """
    X = scaler.transform(features.values)
    raw = model.predict(X)

    vol_by_state = pd.Series(features["spy_vol"].values).groupby(raw).mean()
    order = vol_by_state.sort_values().index
    remap = {old: new for new, old in enumerate(order)}

    regimes = pd.Series([remap[s] for s in raw], index=features.index, name="regime")
    return regimes, remap


def regime_profile(features, regimes):
    """Average feature values in each regime: what each regime 'looks like'."""
    prof = features.groupby(regimes).mean()
    prof["share_of_days"] = regimes.value_counts(normalize=True).sort_index()
    return prof


def asset_performance_by_regime(returns, regimes, periods_per_year=252):
    """Annualized return, vol and Sharpe of each asset inside each regime.
    Uses next-day returns so we measure what happens AFTER the regime is observed."""
    fwd = returns.shift(-1).loc[regimes.index].dropna()
    reg = regimes.loc[fwd.index]
    ann_ret = fwd.groupby(reg).mean() * periods_per_year
    ann_vol = fwd.groupby(reg).std() * np.sqrt(periods_per_year)
    return ann_ret, ann_vol, ann_ret / ann_vol


def regime_durations(regimes):
    """Average length (in days) of an uninterrupted stay in each regime."""
    run_id = (regimes != regimes.shift()).cumsum()
    runs = regimes.groupby(run_id).agg(["first", "size"])
    return runs.groupby("first")["size"].mean().rename("avg_days").rename_axis("regime")


def transition_matrix(model, remap):
    """Daily probability of moving from regime i (row) to regime j (column)."""
    n = len(remap)
    inv = {new: old for old, new in remap.items()}
    T = np.array([[model.transmat_[inv[i], inv[j]] for j in range(n)] for i in range(n)])
    return pd.DataFrame(T, index=range(n), columns=range(n))


# ---------------------------------------------------------------- save / load
def save_model(model, scaler, remap, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "remap": remap}, f)


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["scaler"], obj["remap"]
