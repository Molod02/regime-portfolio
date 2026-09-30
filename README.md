# Regime-Adaptive Portfolio

Can switching a portfolio's allocation according to the market's **regime** (calm vs. stressed) beat a static 60/40 portfolio?

This project detects market regimes with a Hidden Markov Model (HMM) fitted on daily returns and volatility, then allocates between **SPY** (US stocks), **TLT** (long-term Treasuries) and **GLD** (gold) depending on the detected regime. Performance is evaluated with a walk-forward backtest that avoids lookahead bias.

**Status:** Step 1 (data pipeline) complete. Regime detection, portfolio construction and backtest in progress.

## Project structure

```
regime-portfolio/
├── data/
│   ├── raw/            # downloaded prices (re-created by src/data.py, not committed)
│   └── processed/      # log returns
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── config.py       # paths, seed, settings loaded from .env
│   └── data.py         # download prices, compute returns, summary stats
├── model/              # saved models (pickled HMM)
├── reports/            # charts and final summary
├── requirements.txt
├── .env.example
└── README.md
```

## How to rerun

Python 3.11.

```bash
git clone https://github.com/Molod02/regime-portfolio.git
cd regime-portfolio
python -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in values if needed
jupyter notebook notebooks/
```

Run the notebooks in numerical order. The first run downloads data from Yahoo Finance; later runs load the saved CSV.

## Assumptions & risks

- **Data:** Yahoo Finance adjusted close prices, 2005-01-01 to 2026-09-01. Free data may contain errors or be revised.
- **Universe:** only three ETFs; results may not generalize to other assets.
- **Regimes are latent:** the HMM infers them statistically; they are not observed, and labels can shift between fits.
- **Lookahead bias:** regimes must be estimated using only past data at each point in the backtest.
- **Costs:** transaction costs and slippage will be modeled explicitly; frequent regime switches can erase gains.

## Lifecycle mapping

| Stage | Code | Notebook | Output |
|---|---|---|---|
| 1. Data | `src/data.py` | `01_data_exploration.ipynb` | `data/raw/prices.csv`, `data/processed/returns.csv` |
| 2. Regime detection | `src/regimes.py` *(next)* | `02_regime_detection.ipynb` | `model/hmm.pkl` |
| 3. Portfolio construction | `src/portfolio.py` | `03_portfolio.ipynb` | regime weights |
| 4. Backtest | `src/backtest.py` | `04_backtest_results.ipynb` | `reports/` charts, metrics |
