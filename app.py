"""
Flask frontend for inve-STEM — portfolio simulation UI.

This wraps the user's existing simulation.py (their original script, unmodified)
and exposes it through a web UI.


Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import io
import base64
import traceback
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify

# Import the user's unmodified script.
# We import the pieces we need; their script must be named simulation.py
# and sit in the same directory as this file.
import simulation as sim


app = Flask(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def fig_to_base64(fig):
    """Serialize a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def make_portfolio_figure(portfolio_prices, actual_prices, tickers, weights):
    """Recreates the user's plot_portfolio chart but returns the figure
    instead of calling plt.show(), so we can send it to the browser."""
    from matplotlib.lines import Line2D

    VaR_lower = np.percentile(portfolio_prices, 5, axis=1)
    VaR_upper = np.percentile(portfolio_prices, 95, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.plot(portfolio_prices, color="#4A90E2", alpha=0.03)
    ax.plot(portfolio_prices.mean(axis=1), color="#2ECC71", lw=2,
            label="Simulated Mean")

    ax.fill_between(
        range(len(portfolio_prices)),
        VaR_lower, VaR_upper,
        color="#F5B041", alpha=0.35, label="5–95% VaR Range"
    )
    ax.plot(VaR_lower, linestyle="--", color="#B9770E")
    ax.plot(VaR_upper, linestyle="--", color="#B9770E")

    if actual_prices is not None:
        ax.plot(actual_prices[:len(portfolio_prices)],
                color="#E74C3C", lw=2, label="Actual Price")

    ticker_weights = [f"{t} ({w*100:.0f}%)" for t, w in zip(tickers, weights)]
    ax.set_title(f"inve-STEM · Portfolio Simulation ({', '.join(ticker_weights)})")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Price")

    mc_proxy = Line2D([0], [0], color="#4A90E2", lw=2, alpha=0.3,
                      label="Monte Carlo Paths")
    handles, labels = ax.get_legend_handles_labels()
    handles = [mc_proxy] + handles
    labels = ["Monte Carlo Paths"] + labels
    ax.legend(handles, labels, loc="upper left")

    return fig


def make_distribution_figure(portfolio_prices, actual_prices, tickers, weights):
    """Recreates the user's plot_growth_distribution chart as a figure."""
    final_prices = portfolio_prices[-1]
    final_returns = (final_prices / portfolio_prices[0, 0] - 1) * 100

    if actual_prices is not None:
        last_idx = min(len(actual_prices), len(portfolio_prices)) - 1
        actual_return = (actual_prices[last_idx] / actual_prices[0] - 1) * 100
    else:
        actual_return = None

    var95 = np.percentile(final_returns, 5)
    cvar95 = final_returns[final_returns <= var95].mean()
    chance_of_loss = float(np.mean(final_returns < 0) * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_returns, bins=60, alpha=0.7, edgecolor="black",
            color="#4A90E2")
    ax.axvline(final_returns.mean(), color="#2ECC71", lw=2,
               label=f"Mean: {final_returns.mean():.2f}%")
    ax.axvline(var95, color="#8E44AD", lw=2, linestyle="--",
               label=(f"VaR 95%: {var95:.2f}%, CVaR 95%: {cvar95:.2f}%\n"
                      f"Chance of Net Loss: {chance_of_loss:.2f}%"))
    if actual_return is not None:
        ax.axvline(actual_return, color="#E74C3C", lw=2, linestyle="--",
                   label=f"Actual return: {actual_return:.2f}%")

    ticker_weights = [f"{t} ({w*100:.0f}%)" for t, w in zip(tickers, weights)]
    ax.set_title(f"Portfolio % Growth Distribution ({', '.join(ticker_weights)})")
    ax.set_xlabel("% Growth")
    ax.set_ylabel("Frequency")
    ax.legend()

    metrics = {
        "mean_return": float(final_returns.mean()),
        "median_return": float(np.median(final_returns)),
        "var95": float(var95),
        "cvar95": float(cvar95),
        "chance_of_loss": chance_of_loss,
        "best_case": float(final_returns.max()),
        "worst_case": float(final_returns.min()),
        "actual_return": float(actual_return) if actual_return is not None else None,
    }
    return fig, metrics


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", today=date.today().isoformat())


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
            # Auto-normalize rather than failing
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

        # 1) Download price history into the local SQLite DB
        sim.download_and_store(tickers, db_file, end_date)

        # 2) Run the simulation using the user's PortfolioSimulator
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

        # 3) Build the two charts and return them as base64 images
        fig1 = make_portfolio_figure(portfolio_prices, actual_prices, tickers, weights)
        paths_img = fig_to_base64(fig1)

        fig2, metrics = make_distribution_figure(
            portfolio_prices, actual_prices, tickers, weights
        )
        dist_img = fig_to_base64(fig2)

        return jsonify({
            "paths_img": paths_img,
            "dist_img": dist_img,
            "metrics": metrics,
            "n_paths": int(portfolio_prices.shape[1]),
            "n_days": int(portfolio_prices.shape[0]),
            "tickers": tickers,
            "weights": [float(w) for w in weights],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
