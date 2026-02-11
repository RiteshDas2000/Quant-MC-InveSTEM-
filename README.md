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


## Install dependencies
Install via pip by running the following line in terminal
```bash
pip3 install numpy pandas matplotlib yfinance scipy arch numba
```
## Installation

Download and run inveSTEM Multi asset MC portfolio simulator.py.


## Configuration

Define your simulation parameters:

```python
CONFIG = {
    "tickers": ["^GSPC", "^DJI"],      # Portfolio asset tickers
    "weights": [0.5, 0.5],             # Portfolio weights
    "start_date": "2025-02-05",        # Simulation start date
    "end_date": "2026-02-05",          # Simulation end date
    "paths": 1000,                      # Number of Monte Carlo paths
    "df_tails": 15,                     # Student-t degrees of freedom
    "vol_window": 30,                   # Rolling window for initial volatility
    "max_daily_return": None,           # Optional cap on daily returns
    "return_type": "simple",            # "log" or "simple"
    "db_file": "stocks.db"              # SQLite database
}
```

# Monte Carlo Portfolio Plot Elements and Results Description

1. **Monte Carlo Paths (Blue, faint lines)**
   - Each thin blue line represents one simulated portfolio path over the trading period.
   - Generated using GARCH-based volatility and correlated asset returns.
   - Shows the range of possible portfolio outcomes under the stochastic model.

2. **Simulated Mean (Green line)**
   - Represents the average value of all Monte Carlo paths at each trading day.
   - Expected portfolio trajectory.

3. **5–95% Value-at-Risk (VaR) Band (Gold shaded area)**
   - Covers the 5th to 95th percentile of simulated paths.
   - Represents the central 90% range of possible outcomes, showing portfolio uncertainty.
   - Helps visualize the likely range of portfolio values.

4. **5% and 95% VaR Lines (Gold dashed lines)**
   - Lower dashed line: 5% quantile, representing downside extreme scenarios.
   - Upper dashed line: 95% quantile, representing the upside extreme.

5. **Actual Portfolio Prices (Red line)**
   - Shows the real historical portfolio values, calculated from actual market data.
   - Useful for comparing model predictions with actual performance.

6. **Terminal Return Distribution (Histogram in Growth Distribution Plot)**
   - Displays the distribution of final portfolio returns across all Monte Carlo paths.
   - Helps assess probabilities of gains, losses, and extreme outcomes.

7. **Mean Terminal Return (Vertical Green Line in Histogram)**
   - Marks the average final return across all simulations.
   - Serves as the expected final outcome.

8. **5% VaR and 5% CVaR (Vertical Purple Lines in Histogram)**
   - 5% VaR: worst 5% outcomes.
   - 5% CVaR: average of the worst 5% outcomes, highlighting extreme downside risk.

9. **Probability of Loss (Calculated Metric)**
   - Percentage of paths ending below the initial portfolio value.
   - Not plotted directly, but key for risk assessment.

