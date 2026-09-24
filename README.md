# Financial Portfolio Risk Modelling and Optimisation

A Python and SQL pipeline that pulls historical stock data, stores it in a SQL database, calculates risk and return metrics with SQL window functions, builds an optimised portfolio using Markowitz mean-variance optimisation, and tests whether a Random Forest model can predict future gold returns.

## Overview

The project takes a small, deliberately diversified basket of assets and asks three questions:

1. **How has each asset performed, and how risky is it?** Cumulative return, annualised volatility, Sharpe ratio and Value-at-Risk (VaR) per asset.
2. **Can portfolio weights be chosen more intelligently than splitting equally?** Markowitz optimisation finds the maximum Sharpe ratio portfolio and maps the efficient frontier.
3. **Can past returns predict future returns?** A Random Forest regressor is trained to predict next-month GLD (gold) returns from the previous month's returns across all assets.

### Assets analysed

| Ticker | Asset | Category |
|--------|-------|----------|
| GLD | SPDR Gold Shares | Commodity |
| CL=F | Crude Oil Futures | Commodity |
| TSLA | Tesla | Tech / automotive |
| AMZN | Amazon | Tech |
| LLOY.L | Lloyds Banking Group | Finance |
| NG.L | National Grid | Utilities |

Monthly data from January 2020 onwards, downloaded from Yahoo Finance.

## Pipeline

```
Yahoo Finance (yfinance)
        │
        ▼
SQLite database (Mo_Stk_Prices table)
        │
        ▼
SQL window functions (LAG / OVER) → monthly % returns per asset
        │
        ├──► Cumulative return plots
        ├──► Risk metrics (volatility, Sharpe, VaR)
        │
        ▼
Covariance matrix + mean returns
        │
        ├──► Max Sharpe optimisation (SciPy SLSQP)
        ├──► Efficient frontier (50 target-return points)
        │
        ▼
Backtest: optimised vs equal-weighted portfolio
        
Lagged returns ──► Random Forest regression (GLD forecast)
```

## Methodology

**Data collection and storage.** Monthly closing prices are downloaded with `yfinance`, flattened into single-level column names, and written to a SQLite table using `pandas.to_sql`.

**Return calculation in SQL.** Monthly percentage returns are computed directly in SQL using the `LAG()` window function ordered by date, rather than in pandas. Cumulative growth of £1 is then built up from those returns.

**Risk metrics (per asset).**
- *Annualised volatility*: monthly standard deviation × √12
- *Sharpe ratio*: (mean monthly return − monthly risk-free rate) / monthly standard deviation, using a UK 3-month bill yield of 3.977% as the risk-free rate
- *Value-at-Risk*: 5th percentile of monthly returns (historical VaR)

**Markowitz optimisation.** The covariance matrix and mean returns feed a portfolio performance function. `scipy.optimize.minimize` (SLSQP) minimises the negative Sharpe ratio subject to:
- weights summing to 1
- long-only weights, each between 0 and 1

The efficient frontier is traced by minimising volatility at 50 evenly spaced target returns between the lowest and highest individual asset mean returns.

**Machine learning.** A `RandomForestRegressor` predicts GLD's monthly return from the previous month's returns of all six assets. The data is split 80/20 chronologically (`shuffle=False`) to avoid shuffling future data into the training set, and evaluated with MSE and R².

**Backtest.** The optimised weights and an equal-weighted baseline are both applied to the full return history and compared as cumulative growth of £1.

## Outputs

- Cumulative return plot for each asset
- Table of per-asset risk metrics
- Covariance matrix and mean returns
- Table of optimal portfolio weights
- Efficient frontier plot, with the max-Sharpe and equal-weighted portfolios marked
- Random Forest MSE, R² and an actual-vs-predicted table
- Backtest plot and final portfolio values for optimised vs equal-weighted

## Getting started

### Requirements

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `yfinance`
- `sqlite3` (included in the Python standard library)

```bash
pip install pandas numpy matplotlib scipy scikit-learn yfinance
```

### Run

```bash
python portfolio_analysis.py
```

Replace `portfolio_analysis.py` with your script's filename. A `stock_portfolio.db` SQLite file is created in the working directory. Re-running the script replaces the existing table.

To analyse different assets, edit the `stocks` list at the top of the script.

## Limitations and future work

This is a learning project, and these are the known limitations:

- **In-sample optimisation and backtest.** Weights are optimised on the same period they are then tested on, so the backtest flatters the optimised portfolio. A stronger approach is a rolling or walk-forward optimisation, or fitting weights on an earlier window and testing on a later one.
- **Small sample.** Monthly data since 2020 gives roughly 70 observations, which is thin for estimating a six-asset covariance matrix and for training a Random Forest. Expect the ML model to have limited predictive power.
- **Historical mean returns are noisy inputs.** Max-Sharpe optimisation is sensitive to expected return estimates and tends to produce concentrated portfolios. Constraints, shrinkage estimators or a minimum-variance objective could help.
- **Currency mixing.** LLOY.L and NG.L are quoted in pence on the London Stock Exchange while the other assets are in US dollars, and no currency conversion is applied. The UK risk-free rate is used across all assets.
- **Simplifications.** No transaction costs, taxes or rebalancing. The first month's return is filled with 0 as there is no prior price.
- **ML scope.** No hyperparameter tuning, cross-validation or feature engineering beyond one-month lagged returns. Comparing against a naive baseline (for example predicting the historical mean) would show whether the model adds anything.

## Skills demonstrated

Python · SQL (window functions) · pandas · NumPy · SciPy optimisation · scikit-learn · financial risk metrics · data visualisation
