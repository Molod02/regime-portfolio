"""Step 4 — Walk-forward (out-of-sample) backtest.

Every year we pretend it is January 1st and we only know the past:
  1. fit the scaler and the HMM on features up to Dec 31 of the previous year
  2. estimate each regime's mean/covariance on that same training window
  3. optimize one set of weights per regime
  4. during the next year, estimate TODAY's regime with the forward filter,
     which uses only data up to today (Viterbi / predict() would peek at the future)
The training window expands every year.
"""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from src.regimes import build_features, fit_hmm
from src.portfolio import METHODS, regime_moments, simple_returns, PERIODS_PER_YEAR

MIN_DAYS_PER_REGIME = 60  # below this, a regime's statistics are too noisy to trust


# ---------------------------------------------------------------- filtering
def filtered_probabilities(model, X):
    """P(regime at day t | data up to day t), via the HMM forward algorithm.

    Unlike model.predict() or model.predict_proba(), which use the whole
    sequence (past AND future), this only looks backwards. Computed in log
    space for numerical stability.
    """
    n_states = len(model.means_)
    log_emission = np.column_stack([
        multivariate_normal.logpdf(X, model.means_[k], model.covars_[k], allow_singular=True)
        for k in range(n_states)
    ])
    log_trans = np.log(model.transmat_ + 1e-300)

    log_alpha = np.empty((len(X), n_states))
    a = np.log(model.startprob_ + 1e-300) + log_emission[0]
    log_alpha[0] = a - logsumexp(a)
    for t in range(1, len(X)):
        a = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_emission[t]
        log_alpha[t] = a - logsumexp(a)
    return np.exp(log_alpha)


# ---------------------------------------------------------------- one refit
def _unconditional_weights(train_returns, optimizer, max_weight):
    r = simple_returns(train_returns)
    return pd.Series(
        optimizer(r.mean() * PERIODS_PER_YEAR, r.cov() * PERIODS_PER_YEAR, max_weight),
        index=train_returns.columns,
    )


def _fit_one_year(returns, features, train_end, n_states, method, max_weight, n_restarts):
    """Fit on data up to train_end. Returns the model pieces and a weight table per raw state."""
    optimizer = METHODS[method]
    train_feats = features.loc[:train_end]
    train_rets = returns.loc[:train_end]

    model, scaler, ll, _ = fit_hmm(train_feats, n_states=n_states, n_restarts=n_restarts)
    train_states = pd.Series(model.predict(scaler.transform(train_feats.values)),
                             index=train_feats.index)  # Viterbi is fine INSIDE the training window

    fallback = _unconditional_weights(train_rets, optimizer, max_weight)
    moments = regime_moments(train_rets, train_states)
    table = {}
    for k in range(n_states):
        n_days = int((train_states == k).sum())
        if k in moments and n_days >= MIN_DAYS_PER_REGIME:
            mu, cov = moments[k]
            table[k] = pd.Series(optimizer(mu, cov, max_weight), index=returns.columns)
        else:
            table[k] = fallback
    return model, scaler, pd.DataFrame(table).T, ll


# ---------------------------------------------------------------- walk-forward
def walk_forward(returns, first_test_year=2010, n_states=3, method="max_sharpe",
                 max_weight=0.8, mode="soft", n_restarts=5, verbose=True):
    """Out-of-sample daily target weights for the regime strategy.

    mode="soft": weights = probability-weighted mix of each regime's weights (smoother)
    mode="hard": weights of the single most likely regime
    Returns (weights, regime_probs, log) where log has one row per refit.
    """
    features = build_features(returns)
    last_year = features.index[-1].year

    all_weights, all_probs, log = [], [], []
    for year in range(first_test_year, last_year + 1):
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        test_mask = features.index.year == year
        if not test_mask.any():
            continue

        model, scaler, table, ll = _fit_one_year(
            returns, features, train_end, n_states, method, max_weight, n_restarts)

        # run the filter from the start up to the end of this test year (past data only)
        feats_to_date = features.loc[: features.index[test_mask][-1]]
        probs = filtered_probabilities(model, scaler.transform(feats_to_date.values))
        probs = pd.DataFrame(probs, index=feats_to_date.index).loc[features.index[test_mask]]

        if mode == "hard":
            probs = pd.get_dummies(probs.idxmax(axis=1)).reindex(columns=table.index, fill_value=0).astype(float)
        weights = probs.values @ table.values
        all_weights.append(pd.DataFrame(weights, index=probs.index, columns=returns.columns))

        # store probabilities under a stable label: state with highest SPY vol in training = "stress"
        vol_rank = features.loc[:train_end].groupby(
            model.predict(scaler.transform(features.loc[:train_end].values)))["spy_vol"].mean().rank()
        all_probs.append(probs.rename(columns=lambda k: int(vol_rank.get(k, k + 1)) - 1))

        log.append({"test_year": year, "train_days": int((features.index <= train_end).sum()),
                    "log_likelihood": round(ll, 1),
                    **{f"w_stress_{a}": round(table.loc[vol_rank.idxmax(), a], 2) for a in returns.columns}})
        if verbose:
            print(f"{year}: trained on {log[-1]['train_days']} days")

    return pd.concat(all_weights), pd.concat(all_probs).sort_index(axis=1), pd.DataFrame(log).set_index("test_year")


def walk_forward_static(returns, first_test_year=2010, method="max_sharpe", max_weight=0.8):
    """Same optimizer and yearly refits, but NO regimes: one set of weights per year.
    Comparing against this isolates what the regime model itself adds."""
    optimizer = METHODS[method]
    index = build_features(returns).index
    out = []
    for year in range(first_test_year, index[-1].year + 1):
        days = index[index.year == year]
        if len(days) == 0:
            continue
        w = _unconditional_weights(returns.loc[:f"{year - 1}-12-31"], optimizer, max_weight)
        out.append(pd.DataFrame([w.values] * len(days), index=days, columns=returns.columns))
    return pd.concat(out)


# ---------------------------------------------------------------- analysis
def annual_turnover(weights):
    """Average fraction of the portfolio traded per year."""
    return weights.diff().abs().sum(axis=1).mean() * PERIODS_PER_YEAR


def calendar_year_returns(port_rets):
    return pd.DataFrame(port_rets).groupby(pd.DataFrame(port_rets).index.year).apply(
        lambda r: (1 + r).prod() - 1)


def period_returns(port_rets, periods):
    """Total return of each strategy over named date ranges."""
    rows = {}
    for name, (start, end) in periods.items():
        r = port_rets.loc[start:end]
        rows[name] = (1 + r).prod() - 1
    return pd.DataFrame(rows).T
