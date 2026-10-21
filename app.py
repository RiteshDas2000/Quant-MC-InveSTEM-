"""
Flask frontend for inve-STEM — portfolio simulation dashboard.

Runs the Monte Carlo simulation in simulation.py from a web page, so the
portfolio and parameters can be set without editing the code. The charts
are drawn in the browser with Plotly, so they can be zoomed, hovered and
explored with the sliders without running the simulation again.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import traceback
from datetime import date

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request, jsonify
from plotly.offline import get_plotlyjs

# simulation.py must be in the same directory as this file.
import simulation as sim


app = Flask(__name__)

# Percentile levels sent to the browser. With these the sliders can show any
# confidence level and any day of the horizon without another server call.
QUANTILE_LEVELS = np.round(np.arange(0.5, 100, 0.5), 1)
HIST_BINS = 60
SAMPLE_PATHS = 40


def summarise(portfolio_prices, actual_prices, start_date):
    """Turn the full (days x paths) price matrix into the compact set of
    numbers the dashboard needs. Everything is computed on all paths."""
    n_days, n_paths = portfolio_prices.shape
    s0 = float(portfolio_prices[0, 0])
    returns = (portfolio_prices / s0 - 1.0) * 100.0

    quantiles = np.percentile(returns, QUANTILE_LEVELS, axis=1)
    mean = returns.mean(axis=1)
    p_loss = (returns < 0).mean(axis=1) * 100.0

    hist_counts = np.zeros((n_days, HIST_BINS), dtype=int)
    hist_lo = np.zeros(n_days)
    hist_hi = np.zeros(n_days)
    for d in range(n_days):
        lo, hi = returns[d].min(), returns[d].max()
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        hist_counts[d], _ = np.histogram(returns[d], bins=HIST_BINS, range=(lo, hi))
        hist_lo[d], hist_hi[d] = lo, hi

    rng = np.random.default_rng(0)
    pick = rng.choice(n_paths, size=min(SAMPLE_PATHS, n_paths), replace=False)
    sample = returns[:, pick].T

    actual = None
    if actual_prices is not None and len(actual_prices) > 0:
        a = np.asarray(actual_prices, dtype=float)[:n_days]
        actual = [None if np.isnan(v) else round(float(v), 3) for v in (a / a[0] - 1.0) * 100.0]

    dates = pd.bdate_range(start=start_date, periods=n_days).strftime("%Y-%m-%d").tolist()

    def r(x, nd=3):
        return np.round(np.asarray(x, dtype=float), nd).tolist()

    return {
        "s0": s0,
        "n_days": n_days,
        "n_paths": n_paths,
        "dates": dates,
        "levels": QUANTILE_LEVELS.tolist(),
        "quantiles": r(quantiles),
        "mean": r(mean),
        "p_loss": r(p_loss, 2),
        "hist": {"counts": hist_counts.tolist(), "lo": r(hist_lo), "hi": r(hist_hi)},
        "sample": r(sample),
        "actual": actual,
    }


@app.route("/")
def index():
    return render_template("index.html", today=date.today().isoformat())


@app.route("/plotly.js")
def plotly_js():
    # Served from the installed plotly package, so the dashboard works without a CDN.
    return Response(get_plotlyjs(), mimetype="application/javascript",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        data = request.get_json(force=True)
        tickers = [t.strip().upper() for t in data["tickers"] if t.strip()]
        weights = np.array([float(w) for w in data["weights"]], dtype=float)

        if len(tickers) != len(weights):
            return jsonify({"error": "Number of tickers and weights must match."}), 400
        if len(tickers) == 0:
            return jsonify({"error": "Add at least one ticker."}), 400
        if abs(weights.sum() - 1.0) > 1e-6:
            weights = weights / weights.sum()

        start_date = data["start_date"]
        end_date = data["end_date"]
        paths = int(data.get("paths", 1000))
        df_tails = int(data.get("df_tails", 15))
        vol_window = int(data.get("vol_window", 30))
        return_type = data.get("return_type", "simple")
        max_daily_return = data.get("max_daily_return")
        if max_daily_return in ("", None):
            max_daily_return = None
        else:
            max_daily_return = float(max_daily_return)

        db_file = "stocks.db"

        sim.download_and_store(tickers, db_file, end_date)

        portfolio_sim = sim.PortfolioSimulator(
            tickers=tickers,
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            paths=paths,
            df_tails=df_tails,
            vol_window=vol_window,
            max_daily_return=max_daily_return,
            return_type=return_type,
            db_file=db_file,
        )
        portfolio_prices, actual_prices = portfolio_sim.simulate()

        result = summarise(portfolio_prices, actual_prices, start_date)
        result["tickers"] = tickers
        result["weights"] = [float(w) for w in weights]
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
