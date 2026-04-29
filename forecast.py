"""
Forecast oil prices through August 2026 using:

  - LEVELS model (long-conflict regime) — primary point forecast
  - WCS-WTI differential model        — Enbridge-relevant spread

Improvements over the previous version:
  - AR(1) projection of each exogenous (replaces frozen "last value" naive)
  - Monte Carlo simulation: 5000 paths combining residual noise on the price
    equation AND on each exogenous's AR(1) projection, giving honest fan
    charts.
  - Three scenarios (base / bull / bear) overlaying explicit assumption shocks
    on top of the AR(1) baseline.

Outputs:
  output/forecast.csv                  base scenario point + bands
  output/forecast_scenarios.csv        all three scenarios
  output/charts/07_forecast.png        scenario chart
  output/charts/09_diff_forecast.png   WCS-WTI spread forecast
"""
from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
from run_models import _resolve_levels, _drop_constant_cols

warnings.filterwarnings("ignore", category=Warning)

FORECAST_END = pd.Timestamp("2026-08-01")
N_SIM = 5000   # Monte Carlo paths

CRUDE_SPECS = [
    # (label, dep, lag_var, level_col, color)
    ("Light", "log_wti_real", "log_wti_lag1", "wti_real", "#16697A"),
    ("Heavy", "log_wcs_real", "log_wcs_lag1", "wcs_real", "#E8893E"),
]

# Scenarios: shocks added each month to the AR(1)-projected exogenous.
# Values are in the same units as the underlying series (so e.g. log-DXY shocks
# are in log points, refinery_util in percentage points).
SCENARIOS = {
    "base":  {},
    "bull": {  # bullish for price = lower production, weaker dollar, higher demand
        "log_production": -0.02,   # ~2% lower production
        "log_dxy":        -0.02,   # weaker dollar
        "refinery_util":   1.0,    # +1 pp refining demand
    },
    "bear": {  # bearish for price
        "log_production":  0.02,
        "log_dxy":         0.02,
        "refinery_util":  -1.0,
    },
}


def _fit_ar1(s: pd.Series) -> tuple[float, float, float]:
    """Fit AR(1) on a series. Returns (intercept, slope, residual_sd)."""
    s = s.dropna()
    y = s.iloc[1:].values; x = s.iloc[:-1].values
    if len(y) < 6:
        return 0.0, 1.0, float(s.std() if len(s) > 2 else 0.0)
    res = sm.OLS(y, sm.add_constant(x)).fit()
    sd = float(np.sqrt(res.mse_resid))
    return float(res.params[0]), float(res.params[1]), sd


def _project_exogenous(df: pd.DataFrame, regressors: list[str],
                       horizon: pd.DatetimeIndex, scenario_shocks: dict) -> dict:
    """For each exogenous regressor, return:
       - mean_path: deterministic AR(1) projection (with optional level shock)
       - resid_sd:  AR(1) residual sd for Monte Carlo perturbation
    Series like seasonality and dummies are computed deterministically.
    """
    out = {}
    for r in regressors:
        if r in ("month_sin", "month_cos"):
            m = horizon.month
            out[r] = {
                "mean": (np.sin if r == "month_sin" else np.cos)(2 * np.pi * m / 12),
                "sd":   0.0,
            }
            continue
        if r.startswith("regime_") or r in ("opec_shock", "spr_drawdown"):
            # Deterministic carryforward of last observed value
            last = float(df[r].dropna().iloc[-1]) if df[r].notna().any() else 0.0
            out[r] = {"mean": np.full(len(horizon), last), "sd": 0.0}
            continue
        s = df[r].dropna()
        if len(s) == 0:
            out[r] = {"mean": np.zeros(len(horizon)), "sd": 0.0}
            continue
        intercept, slope, sd = _fit_ar1(s)
        last_val = float(s.iloc[-1])
        # Apply scenario shock to the level (persistent, not one-off)
        shock = scenario_shocks.get(r, 0.0)
        path = np.empty(len(horizon))
        prev = last_val
        for t in range(len(horizon)):
            prev = intercept + slope * prev + shock
            path[t] = prev
        out[r] = {"mean": path, "sd": sd}
    return out


def _forecast_one(df: pd.DataFrame, dep: str, lag_var: str, level_col: str,
                  scenario: str = "base") -> pd.DataFrame:
    """Monte Carlo recursive forecast for one crude under one scenario."""
    regressors = _resolve_levels(lag_var)
    sub = df[df["war_long"] == 1][[dep] + regressors].dropna()
    regressors = _drop_constant_cols(sub, regressors)
    X = sm.add_constant(sub[regressors])
    res = sm.OLS(sub[dep], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    sigma_price = float(np.sqrt(res.mse_resid))

    last_log_price = float(df[dep].dropna().iloc[-1])
    last_date = df[dep].dropna().index.max()
    horizon = pd.date_range(start=last_date + pd.DateOffset(months=1),
                            end=FORECAST_END, freq="MS")
    if len(horizon) == 0:
        return pd.DataFrame()

    exo = _project_exogenous(df, [r for r in regressors if r != lag_var],
                             horizon, SCENARIOS[scenario])

    # Pre-allocate paths
    paths = np.empty((N_SIM, len(horizon)))
    rng = np.random.default_rng(42 if scenario == "base" else None)
    coef = res.params

    for s_i in range(N_SIM):
        prev_log = last_log_price
        for t in range(len(horizon)):
            x = {lag_var: prev_log}
            for r in regressors:
                if r == lag_var:
                    continue
                e = exo[r]
                shock = rng.normal(0, e["sd"]) if e["sd"] > 0 else 0.0
                x[r] = e["mean"][t] + shock
            mean_log = coef.get("const", 0.0) + sum(coef[r] * x[r] for r in regressors)
            mean_log += rng.normal(0, sigma_price)  # price equation noise
            paths[s_i, t] = mean_log
            prev_log = mean_log

    levels = np.exp(paths)
    return pd.DataFrame({
        "date":         horizon,
        "price_pred":   np.exp(paths.mean(axis=0)),
        "price_p10":    np.percentile(levels, 10, axis=0),
        "price_p25":    np.percentile(levels, 25, axis=0),
        "price_p50":    np.percentile(levels, 50, axis=0),
        "price_p75":    np.percentile(levels, 75, axis=0),
        "price_p90":    np.percentile(levels, 90, axis=0),
        # 95% band — kept under the legacy column names so build_presentation
        # and any other downstream consumer keeps working.
        "price_lo_95":  np.percentile(levels, 2.5, axis=0),
        "price_hi_95":  np.percentile(levels, 97.5, axis=0),
    }).set_index("date")


def _forecast_differential(df: pd.DataFrame) -> pd.DataFrame:
    """Forecast WCS-WTI differential through FORECAST_END with Monte Carlo."""
    regs = list(config.REGRESSORS_DIFFERENTIAL)
    if "apportionment_lag1" in df.columns and df["apportionment_lag1"].notna().any():
        regs.append("apportionment_lag1")
    regs = [r for r in regs if r in df.columns]
    sub = df[["wcs_wti_diff"] + regs].dropna()
    regs = _drop_constant_cols(sub, regs)
    X = sm.add_constant(sub[regs])
    res = sm.OLS(sub["wcs_wti_diff"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    sigma = float(np.sqrt(res.mse_resid))
    last_diff = float(df["wcs_wti_diff"].dropna().iloc[-1])
    last_date = df["wcs_wti_diff"].dropna().index.max()
    horizon = pd.date_range(last_date + pd.DateOffset(months=1), FORECAST_END, freq="MS")
    exo = _project_exogenous(df, [r for r in regs if r != "wcs_wti_diff_lag1"],
                             horizon, {})

    paths = np.empty((N_SIM, len(horizon)))
    rng = np.random.default_rng(42)
    coef = res.params
    for s_i in range(N_SIM):
        prev = last_diff
        for t in range(len(horizon)):
            x = {"wcs_wti_diff_lag1": prev}
            for r in regs:
                if r == "wcs_wti_diff_lag1":
                    continue
                e = exo[r]
                shock = rng.normal(0, e["sd"]) if e["sd"] > 0 else 0.0
                x[r] = e["mean"][t] + shock
            val = coef.get("const", 0.0) + sum(coef[r] * x[r] for r in regs)
            val += rng.normal(0, sigma)
            paths[s_i, t] = val
            prev = val
    return pd.DataFrame({
        "date":   horizon,
        "diff_pred": paths.mean(axis=0),
        "diff_p10":  np.percentile(paths, 10, axis=0),
        "diff_p90":  np.percentile(paths, 90, axis=0),
    }).set_index("date")


def _chart_scenarios(df: pd.DataFrame, scenarios: dict, last_dates: dict, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    cutoff = FORECAST_END - pd.DateOffset(years=6)
    hist = df.loc[df.index >= cutoff]
    for ax_i, (crude, color, lvl) in enumerate([("Light", "#16697A", "wti_real"),
                                                 ("Heavy", "#E8893E", "wcs_real")]):
        ax = axes[ax_i]
        ax.plot(hist.index, hist[lvl], color=color, lw=1.8, label=f"{crude} - actual")
        last_date = last_dates[crude]
        last_val = float(df[lvl].dropna().iloc[-1])
        for sc, color_sc, ls in [("base", color, "--"),
                                  ("bull", "#2A8B5E", ":"),
                                  ("bear", "#B83A4B", ":")]:
            f = scenarios[crude][sc]
            xs = [last_date] + list(f.index)
            ys = [last_val] + list(f["price_pred"].values)
            ax.plot(xs, ys, color=color_sc, lw=1.6, ls=ls, label=f"{crude} - {sc}")
        # Fan chart for base scenario
        f_base = scenarios[crude]["base"]
        ax.fill_between(f_base.index, f_base["price_p10"], f_base["price_p90"],
                        color=color, alpha=0.15, lw=0, label=f"{crude} - 80% MC band")
        ax.fill_between(f_base.index, f_base["price_p25"], f_base["price_p75"],
                        color=color, alpha=0.20, lw=0)
        today = last_date
        ax.axvline(today, color="#555560", lw=0.9, ls=":", alpha=0.7)
        ax.set_title(f"{crude} oil — base/bull/bear through {FORECAST_END.strftime('%b %Y')}")
        ax.set_ylabel("Price ($/bbl, real 2025 USD)")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Oil price forecast — Monte Carlo bands + scenarios", y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _chart_differential(df: pd.DataFrame, fcst: pd.DataFrame, out_path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cutoff = FORECAST_END - pd.DateOffset(years=6)
    hist = df.loc[df.index >= cutoff, "wcs_wti_diff"].dropna()
    ax.plot(hist.index, hist.values, color="#16697A", lw=1.6, label="WCS - WTI (actual)")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    last_date = hist.index.max()
    last_val = float(hist.iloc[-1])
    xs = [last_date] + list(fcst.index)
    ys = [last_val] + list(fcst["diff_pred"].values)
    ax.plot(xs, ys, color="#E8893E", lw=2.2, ls="--", label="Differential model forecast")
    ax.fill_between(fcst.index, fcst["diff_p10"], fcst["diff_p90"],
                    color="#E8893E", alpha=0.15, lw=0, label="80% MC band")
    ax.set_title(f"WCS - WTI differential — forecast through {FORECAST_END.strftime('%b %Y')}\n"
                 "Negative = WCS trades below WTI (the normal state); narrowing differential helps Enbridge mainline economics")
    ax.set_ylabel("WCS - WTI ($/bbl, real)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    df = pd.read_csv(config.FEATURES_CSV, index_col="date", parse_dates=True)
    print(f"Forecast end: {FORECAST_END.date()}  |  Monte Carlo paths: {N_SIM}\n")

    scenarios = {}
    last_dates = {}
    for crude, dep, lag, lvl, _color in CRUDE_SPECS:
        scenarios[crude] = {}
        last_actual = float(df[lvl].dropna().iloc[-1])
        last_date = df[lvl].dropna().index.max()
        last_dates[crude] = last_date
        for sc in ["base", "bull", "bear"]:
            f = _forecast_one(df, dep, lag, lvl, scenario=sc)
            scenarios[crude][sc] = f
        end = scenarios[crude]["base"].iloc[-1]
        print(f"  {crude:5s}  last {last_date.date()} ${last_actual:6.2f}  ->  "
              f"{FORECAST_END.strftime('%b %Y')}: "
              f"base ${end['price_pred']:.2f}  "
              f"(p10-p90 ${end['price_p10']:.2f} - ${end['price_p90']:.2f})")
        for sc in ("bull", "bear"):
            v = scenarios[crude][sc].iloc[-1]["price_pred"]
            print(f"           {sc:4s}: ${v:.2f}")

    # Save base scenario (back-compat)
    pieces = []
    for crude in ["Light", "Heavy"]:
        f = scenarios[crude]["base"].copy()
        f.columns = [f"{crude.lower()}_{c}" for c in f.columns]
        pieces.append(f)
    pd.concat(pieces, axis=1).to_csv(config.OUTPUT_DIR / "forecast.csv")
    print(f"\nSaved: {config.OUTPUT_DIR / 'forecast.csv'}  (base scenario)")

    # Save all scenarios
    rows = []
    for crude in ["Light", "Heavy"]:
        for sc in ["base", "bull", "bear"]:
            f = scenarios[crude][sc].copy()
            f["crude"] = crude; f["scenario"] = sc
            rows.append(f.reset_index())
    pd.concat(rows, ignore_index=True).to_csv(
        config.OUTPUT_DIR / "forecast_scenarios.csv", index=False)
    print(f"Saved: {config.OUTPUT_DIR / 'forecast_scenarios.csv'}")

    out = config.CHARTS_DIR / "07_forecast.png"
    _chart_scenarios(df, scenarios, last_dates, out)
    print(f"Saved: {out}")

    # Differential
    print("\nDifferential (WCS - WTI)...")
    diff_f = _forecast_differential(df)
    end_d = diff_f.iloc[-1]
    last_d = float(df["wcs_wti_diff"].dropna().iloc[-1])
    print(f"  last ${last_d:+.2f}  ->  {FORECAST_END.strftime('%b %Y')}: "
          f"${end_d['diff_pred']:+.2f}  (p10-p90 ${end_d['diff_p10']:+.2f} to ${end_d['diff_p90']:+.2f})")
    diff_f.to_csv(config.OUTPUT_DIR / "forecast_differential.csv")
    print(f"Saved: {config.OUTPUT_DIR / 'forecast_differential.csv'}")
    out = config.CHARTS_DIR / "09_diff_forecast.png"
    _chart_differential(df, diff_f, out)
    print(f"Saved: {out}")

    return scenarios, diff_f


if __name__ == "__main__":
    main()
