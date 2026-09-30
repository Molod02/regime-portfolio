"""Step 3 — Portfolio construction: one set of weights per regime.

Idea: estimate how the assets behave inside each regime (mean returns and
covariance), then pick the best long-only weights for that regime.
The portfolio holds the weights of the regime observed today and earns
tomorrow's return (no lookahead in the timing).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PERIODS_PER_YEAR = 252
BENCHMARK_6040 = {"SPY": 0.6, "TLT": 0.4, "GLD": 0.0}


# ---------------------------------------------------------------- helpers
def simple_returns(log_returns):
    """Portfolios add up SIMPLE returns across assets, not log returns."""
    return np.exp(log_returns) - 1


def regime_moments(returns, regimes):
    """Annualized mean and covariance of NEXT-DAY simple returns in each regime."""
    fwd = simple_returns(returns).shift(-1).loc[regimes.index].dropna()
    reg = regimes.loc[fwd.index]
    moments = {}
    for k in sorted(reg.unique()):
        r = fwd[reg == k]
        moments[k] = (r.mean() * PERIODS_PER_YEAR, r.cov() * PERIODS_PER_YEAR)
    return moments


# ---------------------------------------------------------------- optimizers
def _optimize(objective, n, max_weight):
    """Long-only, fully invested: weights between 0 and max_weight, summing to 1."""
    w0 = np.full(n, 1 / n)
    res = minimize(
        objective, w0, method="SLSQP",
        bounds=[(0, max_weight)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
    )
    return res.x


def min_variance(mu, cov, max_weight=1.0):
    """Lowest-risk portfolio. Ignores expected returns (they are the noisiest input)."""
    return _optimize(lambda w: w @ cov.values @ w, len(mu), max_weight)


def max_sharpe(mu, cov, max_weight=1.0):
    """Highest return per unit of risk. Uses expected returns, so less stable."""
    def neg_sharpe(w):
        vol = np.sqrt(w @ cov.values @ w)
        return -(w @ mu.values) / vol
    return _optimize(neg_sharpe, len(mu), max_weight)


def risk_parity(mu, cov, max_weight=1.0):
    """Each asset contributes the same share of total risk."""
    def objective(w):
        port_var = w @ cov.values @ w
        contrib = w * (cov.values @ w) / port_var
        return ((contrib - 1 / len(w)) ** 2).sum()
    return _optimize(objective, len(mu), max_weight)


METHODS = {"min_variance": min_variance, "max_sharpe": max_sharpe, "risk_parity": risk_parity}


def regime_weights(returns, regimes, method="max_sharpe", max_weight=0.8):
    """Table of weights: one row per regime, one column per asset.
    max_weight caps any single asset so the optimizer can't go all-in."""
    optimizer = METHODS[method]
    rows = {}
    for k, (mu, cov) in regime_moments(returns, regimes).items():
        rows[k] = pd.Series(optimizer(mu, cov, max_weight), index=mu.index)
    table = pd.DataFrame(rows).T.round(4)
    table.index.name = "regime"
    return table


# ---------------------------------------------------------------- simulation
def daily_weights(regimes, weight_table):
    """Weights held at the end of each day, based on that day's regime."""
    w = weight_table.loc[regimes.values]
    w.index = regimes.index
    return w


def static_weights(index, weights=BENCHMARK_6040):
    return pd.DataFrame([weights] * len(index), index=index)


def portfolio_returns(returns, weights, cost_bps=0.0):
    """Daily portfolio simple returns.

    Weights decided at the close of day t earn the return of day t+1 (shift(1)).
    Portfolio is rebalanced to target weights daily; cost_bps is charged on
    turnover (sum of absolute weight changes) whenever the target changes.
    """
    rets = simple_returns(returns).loc[weights.index]
    held = weights.shift(1).dropna()
    gross = (held * rets.loc[held.index]).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0)
    return gross - turnover * cost_bps / 10_000


def performance(port_rets, name=None):
    """Annualized return, volatility, Sharpe (rf = 0) and max drawdown."""
    wealth = (1 + port_rets).cumprod()
    years = len(port_rets) / PERIODS_PER_YEAR
    ann_ret = wealth.iloc[-1] ** (1 / years) - 1
    ann_vol = port_rets.std() * np.sqrt(PERIODS_PER_YEAR)
    max_dd = (wealth / wealth.cummax() - 1).min()
    return pd.Series(
        {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": ann_ret / ann_vol,
         "max_drawdown": max_dd, "final_wealth": wealth.iloc[-1]},
        name=name,
    )
