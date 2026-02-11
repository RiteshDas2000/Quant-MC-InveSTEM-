# Multi-Asset Monte Carlo Portfolio Simulation

A Python-based framework to simulate multi-asset portfolio price paths using **Monte Carlo simulations with GARCH(1,1) volatility and Student-t shocks**.  
It allows modeling fat-tailed shocks, correlated residuals, and visualizing portfolio risk metrics versus actual historical performance.

---

## Features

- **Multi-Asset Support**: Simulate portfolios with any number of assets and configurable weights.  
- **GARCH Volatility Modeling**: Capture time-varying volatility with GARCH(1,1) models.  
- **Fat-Tailed Shocks**: Use Student-t distributions to model extreme market events.  
- **Correlated Assets**: Estimate residual correlations and simulate correlated paths.  
- **SQLite Database Integration**: Store and retrieve historical stock prices efficiently.  
- **Monte Carlo Simulation**: Generate thousands of stochastic portfolio paths.  
- **Risk Metrics**: Compute 5–95% VaR bands, Conditional VaR (CVaR), chance of net loss, and growth distributions.  
- **Visualization**: Plot portfolio paths, mean trajectory, VaR bands, and terminal return distributions.  

---

## Installation

Download and run .py file.

```bash
# Clone the repository
git clone <repository-url>
cd multi-asset-monte-carlo

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip3 install numpy pandas matplotlib yfinance scipy arch numba
