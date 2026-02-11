# Multi-Asset Monte Carlo Portfolio Simulator

Python Monte Carlo portfolio simulator with GARCH volatility, multi-asset correlations, and risk analysis including VaR, CVaR, and probability-of-loss metrics. Simulates thousands of price paths, aggregates portfolio values, and visualizes return distributions.

## Features

Multi-Asset Support: Simulate portfolios with any number of assets and configurable weights.

GARCH Volatility Modeling: Capture time-varying volatility with GARCH(1,1) models.

Fat-Tailed Shocks: Use Student-t distributions to model heavy-tailed market shocks.

Correlated Assets: Estimate residual correlations across assets and simulate correlated price paths.

SQLite Database Integration: Store and retrieve historical stock prices efficiently.

Monte Carlo Simulation: Generate thousands of stochastic paths for portfolio valuation.

Risk Metrics: Compute 5–95% VaR bands, Conditional VaR (CVaR), chance of net loss, and growth distributions.

Visualization: Plot portfolio price paths, mean trajectory, VaR bands, and terminal return distributions.

# Installation

Download and run py file

# Install dependencies
pip install numpy pandas matplotlib yfinance scipy arch numba

# Usage
1. Configuration

The simulation parameters are defined in CONFIG:

CONFIG = {
    "tickers": ['^GSPC','^DJI'],  # Portfolio assets
    "weights": np.array([0.5,0.5]),  # Portfolio weights
    "start_date": '2025-02-05',  # Simulation start date
    "end_date": '2026-02-05',    # Simulation end date
    "paths": 1000,               # Number of Monte Carlo paths
    "df_tails": 15,              # Student-t degrees of freedom
    "vol_window": 30,            # Rolling window for initial volatility
    "max_daily_return": None,    # Optional cap on daily returns
    "return_type": "simple",     # "log" or "simple"
    "db_file": "stocks.db"       # SQLite database
}

2. Download Historical Data

The framework automatically downloads historical prices from Yahoo Finance and stores them in a SQLite database:

from your_module import download_and_store

download_and_store(CONFIG['tickers'], CONFIG['db_file'], CONFIG['end_date'])

3. Run Portfolio Simulation
from your_module import PortfolioSimulator

portfolio_sim = PortfolioSimulator(**CONFIG)
portfolio_prices, actual_prices = portfolio_sim.simulate()


portfolio_prices: Simulated Monte Carlo paths of the portfolio.

actual_prices: Actual historical prices for comparison.

4. Visualization
Plot Monte Carlo Price Paths
from your_module import plot_portfolio

plot_portfolio(
    portfolio_prices, 
    actual_prices, 
    tickers=CONFIG['tickers'], 
    weights=CONFIG['weights']
)

Plot Terminal Return Distribution
from your_module import plot_growth_distribution

plot_growth_distribution(
    portfolio_prices, 
    actual_prices, 
    tickers=CONFIG['tickers'], 
    weights=CONFIG['weights']
)


Visualizes the final portfolio return distribution.

Shows mean return, VaR 95%, CVaR 95%, chance of net loss, and actual performance.

Core Components

StockDB – SQLite database interface to store and retrieve historical stock prices.

StockSimulator – Simulates a single asset using GARCH(1,1) Monte Carlo.

PortfolioSimulator – Combines multiple StockSimulators into a weighted portfolio and accounts for correlated residuals.

Monte Carlo Engine – Efficient JIT-compiled simulation of GARCH paths with optional return clipping.

Plotting Module – Visualize paths, VaR bands, and terminal return distributions.

Dependencies

Python 3.9+

numpy

pandas

matplotlib

yfinance

scipy

arch

numba

sqlite3 (built-in)

Example Output

Monte Carlo portfolio price paths with 5–95% VaR bands.

Comparison to actual historical portfolio performance.

Terminal portfolio return distribution with risk metrics.

Notes

Correlated Shocks: Residual correlations are estimated from historical GARCH residuals for realistic multi-asset simulations.

Return Types: Supports both log and simple returns; log returns are recommended for long-term projections.

Fat-Tailed Risk: The Student-t distribution models extreme shocks better than Gaussian assumptions.

Performance: Core Monte Carlo loops are JIT-compiled using numba for speed.

License

MIT License – free to use, modify, and distribute.
