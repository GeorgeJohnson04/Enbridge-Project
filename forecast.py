"""
Price forecast through April 2027.

For each crude type, fits the Long-term war model (current state: Israel-Iran
ongoing conflict, well past the 6-month threshold) and projects forward by:
  - holding exogenous inputs (production, storage, dollar, etc.) at the most
    recently observed value (carried forward per column), with seasonality
    recomputed for each future month
  - recursively feeding each month's prediction back as the lag for the next

Outputs:
  output/forecast.csv             - tabular forecast (light + heavy)
  output/charts/07_forecast.png   - chart used in the presentation
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config

FORECAST_END = pd.Timestamp("2027-04-01")  # next fiscal year end

CRUDE_SPECS = [
    # (label, dep, lag_var, level_col, color)
    ("Light", "log_wti_real", "log_wti_lag1", "wti_real", "#16697A"),
    ("Heavy", "log_wcs_real", "log_wcs_lag1", "wcs_real", "#E8893E"),
]

EXO_COLS = ["log_production", "log_inventory", "net_exports",
            "refinery_util", "log_dxy", "gpr", "crack_spread",
            "hormuz_threat"]


def _resolve_regressors(lag_var: str) -> list[str]:
    return [lag_var if x == "log_price_lag1" else x for x in config.REGRESSORS]


def _fit(df: pd.DataFrame, dep: str, regressors: list[str]):
    cols = [dep] + regressors
    sub = df[cols].dropna()
    X = sm.add_constant(sub[regressors])
    y = sub[dep]
    return sm.OLS(y, X).fit()


def _last_known(df: pd.DataFrame, col: str) -> float:
    return float(df[col].dropna().iloc[-1])


def _future_inputs(df: pd.DataFrame, start: pd.Timestamp,
                   end: pd.Timestamp) -> pd.DataFrame:
    """Future feature matrix: exogenous carried forward (per column), seasonality recomputed."""
    future_dates = pd.date_range(start=start, end=end, freq="MS")
    out = pd.DataFrame(index=future_dates)
    for col in EXO_COLS:
        out[col] = _last_known(df, col)
    m = future_dates.month
    out["month_sin"] = np.sin(2 * np.pi * m / 12)
    out["month_cos"] = np.cos(2 * np.pi * m / 12)
    return out


def forecast_one(df: pd.DataFrame, dep: str, lag_var: str,
                 end: pd.Timestamp = FORECAST_END):
    """Recursive forecast through `end` for one crude type."""
    regressors = _resolve_regressors(lag_var)
    sub = df[df["war_long"] == 1].copy()  # Long-term war regime
    res = _fit(sub, dep, regressors)

    last_price_series = df[dep].dropna()
    last_price_date = last_price_series.index.max()
    last_log_price = float(last_price_series.iloc[-1])

    start = last_price_date + pd.DateOffset(months=1)
    future = _future_inputs(df, start, end)
    sigma = np.sqrt(res.mse_resid)  # residual std error in log space

    rows = []
    current_lag = last_log_price
    for t, idx in enumerate(future.index):
        x = future.loc[idx].copy()
        x[lag_var] = current_lag
        X = pd.DataFrame([x[regressors].values], columns=regressors)
        X = sm.add_constant(X, has_constant="add")
        pred = float(res.predict(X)[0])
        # Forecast variance grows with horizon (random walk in the lag chain)
        se = sigma * np.sqrt(t + 1)
        rows.append({
            "date":         idx,
            "log_pred":     pred,
            "price_pred":   float(np.exp(pred)),
            "price_lo_95":  float(np.exp(pred - 1.96 * se)),
            "price_hi_95":  float(np.exp(pred + 1.96 * se)),
        })
        current_lag = pred

    fcst = pd.DataFrame(rows).set_index("date")
    return fcst, res, last_price_date


def chart_forecast(df: pd.DataFrame, forecasts: dict, last_dates: dict):
    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Historical: last 5 years for context
    cutoff = pd.Timestamp("2027-04-01") - pd.DateOffset(years=6)
    hist = df.loc[df.index >= cutoff]

    ax.plot(hist.index, hist["wti_real"], color="#16697A", lw=1.8,
            label="Light oil (WTI) - actual")
    ax.plot(hist.index, hist["wcs_real"], color="#E8893E", lw=1.8,
            label="Heavy oil - actual")

    for crude, color, lvl in [("Light", "#16697A", "wti_real"),
                              ("Heavy", "#E8893E", "wcs_real")]:
        fcst = forecasts[crude]
        last_date = last_dates[crude]
        last_val = float(df[lvl].dropna().iloc[-1])
        # Connect last actual point to first forecast point
        connect_x = [last_date] + list(fcst.index)
        connect_y = [last_val] + list(fcst["price_pred"].values)
        ax.plot(connect_x, connect_y, color=color, lw=2.2, ls="--",
                label=f"{crude} oil - forecast")
        ax.fill_between(fcst.index, fcst["price_lo_95"], fcst["price_hi_95"],
                        color=color, alpha=0.15, lw=0)

    # Vertical line at the latest known price (use Light, the more recent of the two)
    today = max(last_dates.values())
    ax.axvline(today, color="#555560", lw=0.9, ls=":", alpha=0.7)
    y_top = max(hist["wti_real"].max(),
                forecasts["Light"]["price_hi_95"].max()) * 1.05
    ax.annotate("forecast >",
                xy=(today, y_top * 0.96),
                xytext=(8, 0), textcoords="offset points",
                fontsize=9, color="#555560", style="italic")

    ax.set_title("Oil price forecast through April 2027\n"
                 "Solid = actual prices; dashed = model projection; shaded = 95% confidence range")
    ax.set_ylabel("Price per barrel (USD, 2025 dollars)")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out = config.CHARTS_DIR / "07_forecast.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  {out.name}")


def main():
    df = pd.read_csv(config.FEATURES_CSV, index_col="date", parse_dates=True)
    print(f"Forecast end: {FORECAST_END.date()}")
    print()

    forecasts = {}
    last_dates = {}
    for crude, dep, lag, lvl, _ in CRUDE_SPECS:
        fcst, res, last_date = forecast_one(df, dep, lag)
        forecasts[crude] = fcst
        last_dates[crude] = last_date
        last_actual = float(df[lvl].dropna().iloc[-1])
        end_pred = fcst["price_pred"].iloc[-1]
        end_lo = fcst["price_lo_95"].iloc[-1]
        end_hi = fcst["price_hi_95"].iloc[-1]
        print(f"  {crude:5s}  last actual {last_date.date()} ${last_actual:6.2f}  "
              f"->  Apr 2027 ${end_pred:6.2f}  "
              f"(95% range: ${end_lo:.2f} - ${end_hi:.2f})")

    pieces = []
    for crude in ["Light", "Heavy"]:
        f = forecasts[crude].copy()
        f.columns = [f"{crude.lower()}_{c}" for c in f.columns]
        pieces.append(f)
    combined = pd.concat(pieces, axis=1, sort=False)
    combined.to_csv(config.OUTPUT_DIR / "forecast.csv")
    print(f"\nSaved: {config.OUTPUT_DIR / 'forecast.csv'}")

    chart_forecast(df, forecasts, last_dates)
    return forecasts


if __name__ == "__main__":
    main()
