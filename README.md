# Multi-Asset Monte Carlo Portfolio Simulator

Python Monte Carlo portfolio simulator with GARCH volatility, multi-asset correlations, and risk analysis including VaR, CVaR, and probability-of-loss metrics. Simulates thousands of price paths, aggregates portfolio values, and visualizes return distributions.

## Installation

Download the repository and install the required packages:

```bash
pip3 install numpy pandas matplotlib scipy yfinance arch numba

## Usage

Edit the configuration in main.py:

CONFIG = {
    "tickers": ['^GSPC', '^DJI'],
    "weights": [0.5, 0.5],
    "start_date": '2025-02-05',
    "end_date": '2026-02-08',
    "paths": 1000,
    "df_tails": 15,
    "vol_window": 30,
    "max_daily_return": 0.2,
    "return_type": "log",
    "db_file": "stocks.db"
}


Run the main script:

python main.py


Optionally, use the plotting functions interactively:

plot_portfolio(portfolio_prices, actual_prices, tickers=CONFIG['tickers'], weights=CONFIG['weights'])
plot_growth_distribution(portfolio_prices, actual_prices, tickers=CONFIG['tickers'], weights=CONFIG['weights'])

Features

Monte Carlo simulation for multi-asset portfolios

GARCH(1,1) volatility modeling with Student-t shocks

Correlated asset simulations using Cholesky decomposition

Portfolio aggregation with custom weights

Risk metrics: VaR, CVaR, probability of loss

Visualization of simulated price paths and return distributions
