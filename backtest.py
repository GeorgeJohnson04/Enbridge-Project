"""
Walk-forward out-of-sample backtest for the oil price models.

For each test month t in [TEST_START .. last_observed]:
  1. Train each candidate model on data strictly before t.
  2. Predict price at t using only information available up to t-1.
  3. Compare forecast vs actual.

Models compared:
  Naive baselines:
    - random_walk     - price_t = price_{t-1}
    - ar1             - simple AR(1) on log price
    - seasonal_naive  - price_t = price_{t-12}

  Our models:
    - levels   - LEVELS family fit on the relevant war segment
    - returns  - RETURNS family fit on the relevant war segment

Metrics (lower is better unless noted):
  - RMSE        root mean squared error in $/bbl
  - MAPE        mean absolute percentage error
  - MAE         mean absolute error in $/bbl
  - DA          directional accuracy (% of months with correct sign of change) — HIGHER IS BETTER
  - Theil_U     Theil's U statistic (Theil's U2) vs random walk — <1 means we beat the naive

Output:
  output/backtest_results.csv     metrics per (model, target)
  output/backtest_predictions.csv per-month predictions for chart
  output/charts/08_backtest.png   visual: predicted vs actual + RMSE comparison
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import config
from run_models import (
    _resolve_levels, _resolve_returns, _ensure_lag_returns, _drop_constant_cols,
    CRUDE_SPECS,
)

warnings.filterwarnings("ignore", category=Warning)

# Backtest holdout: the last N months of the sample become the test set.
# Walk-forward refits monthly within this window.
TEST_MONTHS = 36   # 3 years of out-of-sample testing


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _baseline_random_walk(history: pd.Series) -> float:
    return float(history.dropna().iloc[-1])


def _baseline_ar1(history: pd.Series) -> float:
    """Simple AR(1) on the log series, refit each step."""
    s = history.dropna()
    if len(s) < 12:
        return float(s.iloc[-1])
    log_s = np.log(s)
    y = log_s.iloc[1:].values
    x = log_s.iloc[:-1].values
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    pred_log = float(res.params[0] + res.params[1] * log_s.iloc[-1])
    return float(np.exp(pred_log))


def _baseline_seasonal(history: pd.Series, target_date: pd.Timestamp) -> float:
    """price_t = price_{t-12} (last year's same month)."""
    one_year_ago = target_date - pd.DateOffset(months=12)
    if one_year_ago in history.index and pd.notna(history.loc[one_year_ago]):
        return float(history.loc[one_year_ago])
    return float(history.dropna().iloc[-1])


def _baseline_ar1_returns(history: pd.Series) -> float:
    """AR(1) on log returns, then chain back to a price. Often the strongest
    naive baseline for asset prices."""
    s = history.dropna()
    if len(s) < 12:
        return float(s.iloc[-1])
    rets = np.log(s).diff().dropna()
    y = rets.iloc[1:].values; x = rets.iloc[:-1].values
    if len(y) < 5:
        return float(s.iloc[-1])
    res = sm.OLS(y, sm.add_constant(x)).fit()
    pred_ret = float(res.params[0] + res.params[1] * rets.iloc[-1])
    return float(s.iloc[-1] * np.exp(pred_ret))


# ---------------------------------------------------------------------------
# Our models — refit + one-step predict
# ---------------------------------------------------------------------------
def _levels_predict(train: pd.DataFrame, test_row: pd.Series, dep: str, lag_var: str) -> float | None:
    regs = _resolve_levels(lag_var)
    cols = [dep] + regs
    sub = train[cols].dropna()
    regs = _drop_constant_cols(sub, regs)
    if len(sub) < len(regs) + 5:
        return None
    X = sm.add_constant(sub[regs])
    res = sm.OLS(sub[dep], X).fit()
    # Build the test row from columns we used; require all present
    if not all(pd.notna(test_row.get(r, np.nan)) for r in regs):
        return None
    x_pred = pd.DataFrame([[test_row[r] for r in regs]], columns=regs)
    x_pred = sm.add_constant(x_pred, has_constant="add")
    pred_log = float(res.predict(x_pred)[0])
    return float(np.exp(pred_log))


def _returns_predict(train: pd.DataFrame, test_row: pd.Series, dep_ret: str,
                     lag_ret: str, level_col: str) -> float | None:
    """Predict next price by predicting log return then chaining."""
    train_ext = _ensure_lag_returns(train, lag_ret)
    regs = _resolve_returns(lag_ret)
    regs = [r for r in regs if r in train_ext.columns]
    cols = [dep_ret] + regs
    sub = train_ext[cols].dropna()
    regs = _drop_constant_cols(sub, regs)
    if len(sub) < len(regs) + 5:
        return None
    X = sm.add_constant(sub[regs])
    res = sm.OLS(sub[dep_ret], X).fit()
    # Need all regressors at test_row time
    if not all(pd.notna(test_row.get(r, np.nan)) for r in regs):
        return None
    x_pred = pd.DataFrame([[test_row[r] for r in regs]], columns=regs)
    x_pred = sm.add_constant(x_pred, has_constant="add")
    pred_dlog = float(res.predict(x_pred)[0])
    last_price = train[level_col].dropna().iloc[-1]
    return float(last_price * np.exp(pred_dlog))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(actual: pd.Series, pred: pd.Series, prev: pd.Series) -> dict:
    """Compute RMSE, MAPE, MAE, directional accuracy, Theil's U vs random walk.

    `prev` is the price at t-1 (used for both directional accuracy and Theil's U)."""
    a, p, q = actual.align(pred, join="inner")[0], actual.align(pred, join="inner")[1], prev.reindex(actual.index)
    mask = a.notna() & p.notna() & q.notna()
    a, p, q = a[mask], p[mask], q[mask]
    if len(a) == 0:
        return {"n": 0}
    err = a - p
    rmse = float(np.sqrt((err ** 2).mean()))
    mae  = float(err.abs().mean())
    mape = float(((err / a).abs()).mean() * 100)
    # Directional accuracy: did the model get the sign of (a - q) right?
    actual_dir = np.sign(a - q)
    pred_dir   = np.sign(p - q)
    da = float((actual_dir == pred_dir).mean() * 100)
    # Theil's U2: ratio of model RMSE to random-walk RMSE
    rw_err = a - q
    rw_rmse = float(np.sqrt((rw_err ** 2).mean()))
    theil = rmse / rw_rmse if rw_rmse > 0 else float("nan")
    return {
        "n": len(a),
        "RMSE_$/bbl": round(rmse, 3),
        "MAE_$/bbl":  round(mae, 3),
        "MAPE_%":     round(mape, 2),
        "DA_%":       round(da, 1),
        "Theil_U":    round(theil, 3),
    }


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------
def walk_forward(df: pd.DataFrame, level_col: str, dep_lvl: str, lag_lvl: str,
                 dep_ret: str, lag_ret: str) -> pd.DataFrame:
    """Run walk-forward backtest for one crude type. Returns per-date predictions
    indexed by date with columns [actual, random_walk, ar1, seasonal_naive,
    levels_model, returns_model]."""
    test_idx = df[level_col].dropna().index[-TEST_MONTHS:]
    rows = []
    for target_date in test_idx:
        if target_date <= df.index[0]:
            continue
        actual = df.loc[target_date, level_col]
        if pd.isna(actual):
            continue
        # Training window: everything strictly BEFORE target_date
        train = df[df.index < target_date]
        if len(train) < 36:
            continue
        history = train[level_col]
        prev_price = history.dropna().iloc[-1] if len(history.dropna()) else np.nan

        # War-state for target month determines which segment to fit on
        target_war = df.loc[target_date, "war_long"] if "war_long" in df.columns else 0
        seg_train = train[train["war_long"] == target_war]
        # Fall back to all data if segment too small
        if len(seg_train) < 30:
            seg_train = train

        test_row = df.loc[target_date]

        rw    = _baseline_random_walk(history)
        ar1   = _baseline_ar1(history)
        ar1r  = _baseline_ar1_returns(history)
        sea   = _baseline_seasonal(history, target_date)
        lvl   = _levels_predict(seg_train, test_row, dep_lvl, lag_lvl)
        ret   = _returns_predict(seg_train, test_row, dep_ret, lag_ret, level_col)

        rows.append({
            "date":           target_date,
            "actual":         actual,
            "prev":           prev_price,
            "random_walk":    rw,
            "ar1":            ar1,
            "ar1_returns":    ar1r,
            "seasonal_naive": sea,
            "levels_model":   lvl,
            "returns_model":  ret,
        })
    return pd.DataFrame(rows).set_index("date")


def _metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics on the COMMON SUBSET where every model has a prediction —
    so RMSE / Theil's U are apples-to-apples. (Models can occasionally drop a
    row when a regressor is NaN; comparing on the union gives misleading
    headline RMSE.)"""
    cols_models = ["random_walk", "ar1", "ar1_returns", "seasonal_naive",
                   "levels_model", "returns_model"]
    common = predictions[["actual", "prev"] + cols_models].dropna()
    actual = common["actual"]; prev = common["prev"]
    rows = []
    for col in cols_models:
        m = _metrics(actual, common[col], prev)
        m["model"] = col
        rows.append(m)
    return pd.DataFrame(rows)[["model", "n", "RMSE_$/bbl", "MAE_$/bbl", "MAPE_%", "DA_%", "Theil_U"]]


def _chart_backtest(all_preds: dict, all_metrics: dict, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    # Top row: prediction lines vs actual, one per crude
    for j, crude in enumerate(["Light", "Heavy"]):
        ax = axes[0][j]
        p = all_preds[crude]
        ax.plot(p.index, p["actual"],         color="black", lw=2.0, label="Actual")
        ax.plot(p.index, p["random_walk"],    color="gray",  lw=1.0, ls="--", label="Random walk")
        ax.plot(p.index, p["ar1_returns"],    color="#888",  lw=1.0, ls=":", label="AR(1) returns")
        ax.plot(p.index, p["levels_model"],   color="#16697A", lw=1.4, label="Levels model")
        ax.plot(p.index, p["returns_model"],  color="#E8893E", lw=1.4, label="Returns model")
        ax.set_title(f"{crude} crude — walk-forward predictions")
        ax.set_ylabel("Price ($/bbl, real 2025 USD)")
        ax.legend(loc="best", framealpha=0.9, fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Bottom row: RMSE bars (lower=better) and Theil's U (lower=better, <1 beats RW)
    for j, crude in enumerate(["Light", "Heavy"]):
        ax = axes[1][j]
        m = all_metrics[crude]
        labels = m["model"].tolist()
        rmse = m["RMSE_$/bbl"].tolist()
        theil = m["Theil_U"].tolist()
        x = np.arange(len(labels))
        width = 0.4
        ax.bar(x - width/2, rmse, width, label="RMSE ($/bbl)", color="#16697A")
        ax2 = ax.twinx()
        ax2.bar(x + width/2, theil, width, label="Theil's U", color="#E8893E", alpha=0.85)
        ax2.axhline(1.0, color="red", ls=":", lw=1, alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("RMSE ($/bbl)", color="#16697A")
        ax2.set_ylabel("Theil's U (vs random walk; <1 = beats RW)", color="#E8893E")
        ax.set_title(f"{crude} — out-of-sample accuracy ({len(all_preds[crude])} months)")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Walk-forward backtest — last {TEST_MONTHS} months refit monthly", y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _differential_predict(train: pd.DataFrame, test_row: pd.Series) -> float | None:
    regs = list(config.REGRESSORS_DIFFERENTIAL)
    if "apportionment_lag1" in train.columns and train["apportionment_lag1"].notna().any():
        regs.append("apportionment_lag1")
    regs = [r for r in regs if r in train.columns]
    cols = ["wcs_wti_diff"] + regs
    sub = train[cols].dropna()
    regs = _drop_constant_cols(sub, regs)
    if len(sub) < len(regs) + 5:
        return None
    X = sm.add_constant(sub[regs])
    res = sm.OLS(sub["wcs_wti_diff"], X).fit()
    if not all(pd.notna(test_row.get(r, np.nan)) for r in regs):
        return None
    x_pred = pd.DataFrame([[test_row[r] for r in regs]], columns=regs)
    x_pred = sm.add_constant(x_pred, has_constant="add")
    return float(res.predict(x_pred)[0])


def walk_forward_differential(df: pd.DataFrame) -> pd.DataFrame:
    """Backtest the WCS-WTI differential model. Baselines:
       - persistence (diff_t = diff_{t-1})
       - rolling-12-month mean
    """
    test_idx = df["wcs_wti_diff"].dropna().index[-TEST_MONTHS:]
    rows = []
    for target_date in test_idx:
        actual = df.loc[target_date, "wcs_wti_diff"]
        if pd.isna(actual):
            continue
        train = df[df.index < target_date]
        if len(train) < 36:
            continue
        history = train["wcs_wti_diff"].dropna()
        if len(history) == 0:
            continue
        prev = float(history.iloc[-1])
        rolling12 = float(history.tail(12).mean())
        model_pred = _differential_predict(train, df.loc[target_date])
        rows.append({
            "date":      target_date,
            "actual":    actual,
            "prev":      prev,
            "persistence":     prev,
            "rolling_12m_mean": rolling12,
            "differential_model": model_pred,
        })
    return pd.DataFrame(rows).set_index("date")


def main():
    df = pd.read_csv(config.FEATURES_CSV, index_col="date", parse_dates=True)
    print(f"Walk-forward backtest: last {TEST_MONTHS} months\n")

    all_preds = {}
    all_metrics = {}
    metrics_combined = []

    for crude_label, b1, dep_lvl, lag_lvl, dep_ret, lag_ret in CRUDE_SPECS:
        level_col = "wti_real" if crude_label == "Light" else "wcs_real"
        print(f"--- {crude_label} crude ({level_col}) ---")
        preds = walk_forward(df, level_col, dep_lvl, lag_lvl, dep_ret, lag_ret)
        if len(preds) == 0:
            print("  No valid test rows."); continue
        m = _metrics_table(preds)
        m.insert(0, "crude", crude_label)
        print(m.to_string(index=False))
        print()
        all_preds[crude_label] = preds
        all_metrics[crude_label] = m
        metrics_combined.append(m)

    if not metrics_combined:
        print("No predictions produced — check data sample.")
        return

    metrics_df = pd.concat(metrics_combined, ignore_index=True)
    metrics_df.to_csv(config.OUTPUT_DIR / "backtest_results.csv", index=False)
    print(f"Saved: {config.OUTPUT_DIR / 'backtest_results.csv'}")

    preds_combined = pd.concat({k: v for k, v in all_preds.items()}, axis=1)
    preds_combined.to_csv(config.OUTPUT_DIR / "backtest_predictions.csv")
    print(f"Saved: {config.OUTPUT_DIR / 'backtest_predictions.csv'}")

    chart_path = config.CHARTS_DIR / "08_backtest.png"
    _chart_backtest(all_preds, all_metrics, chart_path)
    print(f"Saved: {chart_path}")

    # Differential backtest
    print("\n--- Differential (WCS - WTI) ---")
    diff_preds = walk_forward_differential(df)
    if len(diff_preds):
        common = diff_preds.dropna()
        actual = common["actual"]; prev = common["prev"]
        rows = []
        for col in ["persistence", "rolling_12m_mean", "differential_model"]:
            m = _metrics(actual, common[col], prev)
            m["model"] = col
            rows.append(m)
        diff_metrics = pd.DataFrame(rows)[["model", "n", "RMSE_$/bbl", "MAE_$/bbl", "MAPE_%", "DA_%", "Theil_U"]]
        diff_metrics.insert(0, "crude", "WCS-WTI")
        print(diff_metrics.to_string(index=False))
        # Append to combined metrics file
        full = pd.concat([metrics_df, diff_metrics], ignore_index=True)
        full.to_csv(config.OUTPUT_DIR / "backtest_results.csv", index=False)
        diff_preds.to_csv(config.OUTPUT_DIR / "backtest_predictions_diff.csv")

    # Summary verdict
    print("\n=== VERDICT ===")
    for crude, m in all_metrics.items():
        rw_rmse = float(m[m["model"] == "random_walk"]["RMSE_$/bbl"].iloc[0])
        best = m.sort_values("RMSE_$/bbl").iloc[0]
        beats = m[m["RMSE_$/bbl"] < rw_rmse]["model"].tolist()
        beats = [b for b in beats if b != "random_walk"]
        # Best directional accuracy (excluding RW which is meaningless on DA)
        m_da = m[m["model"] != "random_walk"]
        best_da = m_da.sort_values("DA_%", ascending=False).iloc[0]
        print(f"  {crude}: best RMSE = {best['model']} @ ${best['RMSE_$/bbl']:.2f}; "
              f"best direction = {best_da['model']} @ {best_da['DA_%']:.0f}%")
        print(f"           beats random walk on RMSE: {beats or '(none)'}")
    if len(diff_preds):
        print(f"  WCS-WTI: best RMSE = {diff_metrics.sort_values('RMSE_$/bbl').iloc[0]['model']} "
              f"@ ${diff_metrics['RMSE_$/bbl'].min():.2f}")


if __name__ == "__main__":
    main()
