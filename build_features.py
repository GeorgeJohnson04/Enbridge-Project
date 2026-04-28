"""
Builds the feature matrix for the four OLS models.

Reads data/panel_raw.csv (produced by fetch_data.py) and writes
data/panel_features.csv. The four models share the same RHS regressor list;
B1 (crude type) and B2 (war length) act as segmenters, not regressors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _war_state_per_month(index: pd.DatetimeIndex) -> pd.DataFrame:
    """For each month, classify the war regime AS OF that month.

    war_long is time-varying within a single war:
      - First LONG_WAR_MONTHS months of any war  -> war_long = 0 (initial phase)
      - Month LONG_WAR_MONTHS+1 onward            -> war_long = 1 (sustained phase)
      - Wars that end before reaching threshold   -> war_long = 0 throughout
      - Peacetime months                          -> war_long = 0
    """
    rows = []
    for d in index:
        active = None
        for start, end, label, _iran in config.WAR_WINDOWS:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
            if s <= d <= e:
                months_elapsed = (d - s).days / 30.44
                active = (label, months_elapsed >= config.LONG_WAR_MONTHS)
                break
        if active is None:
            rows.append({"in_war": False, "war_long": False, "war_label": ""})
        else:
            rows.append({"in_war": True, "war_long": active[1], "war_label": active[0]})
    return pd.DataFrame(rows, index=index)


def _hormuz_threat_per_month(index: pd.DatetimeIndex) -> pd.Series:
    flag = pd.Series(0, index=index, name="hormuz_threat")
    for start, end in config.HORMUZ_THREAT_WINDOWS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        flag.loc[(flag.index >= s) & (flag.index <= e)] = 1
    return flag


def build_features() -> pd.DataFrame:
    panel = pd.read_csv(config.RAW_PANEL_CSV, index_col="date", parse_dates=True)
    df = panel.copy()

    # Inflation-adjust prices to REAL_PRICE_BASE_YEAR USD
    base_cpi = df.loc[df.index.year == config.REAL_PRICE_BASE_YEAR, "cpi"].mean()
    if pd.isna(base_cpi):
        base_cpi = df["cpi"].dropna().iloc[-12:].mean()  # fallback: latest year average
    df["wti_real"]  = df["wti"]  * (base_cpi / df["cpi"])
    df["wcs_real"]  = df["wcs"]  * (base_cpi / df["cpi"])

    # B2 + B4 binaries
    war = _war_state_per_month(df.index)
    df["war_long"]      = war["war_long"].astype(int)        # B2
    df["in_war"]        = war["in_war"].astype(int)
    df["war_label"]     = war["war_label"]
    df["hormuz_threat"] = _hormuz_threat_per_month(df.index) # B4

    # Net exports already computed in fetch_data; rescale to mbbl/day for readability
    df["net_exports"] = df["net_exports"] / 1000.0

    # Logs of strictly-positive series
    df["log_production"] = np.log(df["production_us"])
    df["log_inventory"]  = np.log(df["inventory_us"])
    df["log_dxy"]        = np.log(df["dxy"])
    df["log_wti_real"]   = np.log(df["wti_real"])
    df["log_wcs_real"]   = np.log(df["wcs_real"])

    # Lagged log price (B6) - per crude type
    df["log_wti_lag1"] = df["log_wti_real"].shift(1)
    df["log_wcs_lag1"] = df["log_wcs_real"].shift(1)

    # Crack spread proxy: Brent-WTI spread (inter-crude differential, $/bbl)
    # True 3-2-1 crack would need gasoline + distillate retail; this is a working proxy.
    df["crack_spread"] = df["brent"] - df["wti"]

    # Seasonality (annual cycle)
    m = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)

    # Save
    df.index.name = "date"
    df.to_csv(config.FEATURES_CSV)

    print(f"Built features panel: {config.FEATURES_CSV.name}  ({len(df)} rows)")
    print(f"  Real price base year: {config.REAL_PRICE_BASE_YEAR}  (CPI base = {base_cpi:.2f})")
    print(f"  Months in war:         {df['in_war'].sum()}")
    print(f"  Months in long war:    {df['war_long'].sum()}")
    print(f"  Months Hormuz threat:  {df['hormuz_threat'].sum()}")
    return df


if __name__ == "__main__":
    build_features()
